"""Dataset loader and data utilities.

Author:
    Shrey Desai and Yasumasa Onoe
"""

import collections
import itertools
import torch

from torch.utils.data import Dataset
from random import shuffle
from utils import cuda, load_dataset


PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'


class Vocabulary:
    """
    This class creates two dictionaries mapping:
        1) words --> indices,
        2) indices --> words.

    Args:
        samples: A list of training examples stored in `QADataset.samples`.
        vocab_size: Int. The number of top words to be used.

    Attributes:
        words: A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
            All words will be lowercased.
        encoding: A dictionary mapping words (string) to indices (int).
        decoding: A dictionary mapping indices (int) to words (string).
    """
    def __init__(self, samples, vocab_size):
        self.words = self._initialize(samples, vocab_size)
        self.encoding = {word: index for (index, word) in enumerate(self.words)}
        self.decoding = {index: word for (index, word) in enumerate(self.words)}

    def _initialize(self, samples, vocab_size):
        """
        Counts and sorts all tokens in the data, then it returns a vocab
        list. `PAD_TOKEN and `UNK_TOKEN` are added at the beginning of the
        list. All words are lowercased.

        Args:
            samples: A list of training examples stored in `QADataset.samples`.
            vocab_size: Int. The number of top words to be used.

        Returns:
            A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
        """
        vocab = collections.defaultdict(int)
        for (_, passage, question, _, _) in samples:
            for token in itertools.chain(passage, question):
                vocab[token.lower()] += 1
        top_words = [
            word for (word, _) in
            sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        ][:vocab_size]
        words = [PAD_TOKEN, UNK_TOKEN] + top_words
        return words
    
    def __len__(self):
        return len(self.words)


class Tokenizer:
    """
    This class provides two methods converting:
        1) List of words --> List of indices,
        2) List of indices --> List of words.

    Args:
        vocabulary: An instantiated `Vocabulary` object.

    Attributes:
        vocabulary: A list of top words (string) sorted by frequency.
            `PAD_TOKEN` (at position 0) and `UNK_TOKEN` (at position 1) are
            prepended.
        pad_token_id: Index of `PAD_TOKEN` (int).
        unk_token_id: Index of `UNK_TOKEN` (int).
    """
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.pad_token_id = self.vocabulary.encoding[PAD_TOKEN]
        self.unk_token_id = self.vocabulary.encoding[UNK_TOKEN]

    def convert_tokens_to_ids(self, tokens):
        """
        Converts words to corresponding indices.

        Args:
            tokens: A list of words (string).

        Returns:
            A list of indices (int).
        """
        return [
            self.vocabulary.encoding.get(token, self.unk_token_id)
            for token in tokens
        ]

    def convert_ids_to_tokens(self, token_ids):
        """
        Converts indices to corresponding words.

        Args:
            token_ids: A list of indices (int).

        Returns:
            A list of words (string).
        """
        return [
            self.vocabulary.decoding.get(token_id, UNK_TOKEN)
            for token_id in token_ids
        ]

skip_pos_tags = ["PUNCT", "DET", "AUX", "SCONJ", "ADP", "PART", "VERB"]
skip_dep_tags = ["prep", "relcl", "appos", "cc", "conj"]

def get_children(tok):
    if not tok.children:
        return []
    child_toks = []
    for child_tok in tok.children:
        if child_tok.pos_ not in skip_pos_tags and child_tok.dep_ not in skip_dep_tags:
            child_toks += [(child_tok.text, child_tok.idx)]
        if child_tok.dep_ not in skip_dep_tags:
            child_toks += get_children(child_tok)
    return child_toks

def get_final_set(s, idx_to_sent):
    final_set = None
    if len(s) == len(idx_to_sent): 
        final_set = set()
    elif len(s) > 1 and max(s) == len(idx_to_sent):
        final_set = set([max(s)])
    elif len(s) == 1 and max(s) == len(idx_to_sent):
        final_set = s
    else:
        final_set = set()
    return final_set

def get_sent_idx_maps(nlp_passage):
    sent_to_idx = dict()
    sent_idx = 0
    for token in nlp_passage:
        if token.sent not in sent_to_idx:
            sent_idx += 1 
        sent_to_idx[token.sent] = sent_idx

    idx_to_sent = {v: k for k, v in sent_to_idx.items()}
    return sent_to_idx, idx_to_sent

def get_objects(nlp_passage, sent_to_idx): 
    objects = dict()
    for token in nlp_passage: 
        sent_idx = sent_to_idx[token.sent]
        if sent_idx not in objects:
            objects[sent_idx] = []
        if ("obj" in token.dep_ or "oprd" in token.dep_ or (token.dep_ == "conj" and token.pos_ == "NOUN")) and token.pos_ not in skip_pos_tags:
            obj = get_children(token) 
            obj.append((token.text, token.idx))
            obj.sort(key=lambda x:x[1])
            objects[sent_idx].append(" ".join(v[0] for v in obj))
    return objects

def get_subjects(nlp_passage, sent_to_idx):
    subjects = dict()
    for token in nlp_passage:
        sent_idx = sent_to_idx[token.sent]
        if sent_idx not in subjects:
            subjects[sent_idx] = []
        if ("ROOT" in token.dep_ and token.pos_ == "NOUN") or ("nsubj" in token.dep_ and token.pos_ not in skip_pos_tags):
            subj = get_children(token) 
            subj.append((token.text, token.idx))
            subj.sort(key=lambda x:x[1])
            subjects[sent_idx].append(" ".join(v[0] for v in subj))
    return subjects

def get_attributes(nlp_passage, sent_to_idx):
    attrs = dict()
    for token in nlp_passage:
        sent_idx = sent_to_idx[token.sent]
        if sent_idx not in attrs:
            attrs[sent_idx] = []
        if "attr" in token.dep_ and token.pos_ not in skip_pos_tags:
            attr = get_children(token) 
            attr.append((token.text, token.idx))
            attr.sort(key=lambda x:x[1])
            attrs[sent_idx].append(" ".join(v[0] for v in attr))
    return attrs

def get_sent_vals_map(nlp_passage, sent_to_idx, idx_to_sent):
    objects = get_objects(nlp_passage, sent_to_idx)
    subjects = get_subjects(nlp_passage, sent_to_idx)
    attrs = get_attributes(nlp_passage, sent_to_idx)

    sent_vals_map = dict()
    for sent_idx in idx_to_sent:
        if sent_idx not in sent_vals_map:
            sent_vals_map[sent_idx] = set()
        for val in subjects[sent_idx] + objects[sent_idx] + attrs[sent_idx]:
            sent_vals_map[sent_idx].add(val)
    return sent_vals_map

def get_ner_tags(nlp_passage, sent_to_idx):
    sent_ner_tags_map = dict()
    for ent in nlp_passage.ents:
        sent_idx = sent_to_idx[ent.sent]
        if sent_idx not in sent_ner_tags_map:
            sent_ner_tags_map[sent_idx] = set()
        sent_ner_tags_map[sent_idx].add(ent.text)
    return sent_ner_tags_map

def get_dep_relevant_sent_idxs(sent_vals_map, idx_to_sent):
    dep_relevant_sent_idxs = set()
    for sent_idx, sent_vals in sent_vals_map.items():
        sent_vals_matched = dict()
        sent_vals_unmatched = set()
        for sent_val in sent_vals:
            is_matched = False
            matching_sent_idxs = set()
            for target_sent_idx, target_sent_vals in sent_vals_map.items():
                if sent_idx == target_sent_idx: 
                    continue
                sent_toks = sent_val.split(" ")
                target_sent_toks = set([tok  for val in target_sent_vals for tok in val.split(" ")])

                if all(sent_tok in target_sent_toks for sent_tok in sent_toks):
                    is_matched = True
                    matching_sent_idxs.add(target_sent_idx)
            if is_matched:
                sent_vals_matched[sent_val] = matching_sent_idxs
            else:
                sent_vals_unmatched.add(sent_val)
            
        #print("sentence %s %s" % (sent_idx, idx_to_sent[sent_idx]))
        #print("sent_dep_matched: %s" % sent_vals_matched)
        #print("sent_dep_unmatched: %s" % sent_vals_unmatched)

        for sent_val, matching_sent_idxs in sent_vals_matched.items():
            dep_relevant_sent_idxs |= {sent_idx}
            dep_relevant_sent_idxs |= matching_sent_idxs

    return dep_relevant_sent_idxs

def get_ques_dep_relevant_sent_idxs(sent_vals_map, idx_to_sent, ques_vals):
    #print("ques_vals: %s" % ques_vals)
    dep_relevant_sent_idxs = set()
    ques_vals_matched = dict()
    ques_vals_unmatched = set()
    for ques_val in ques_vals:
        is_matched = False
        matching_sent_idxs = set()
        for sent_idx, sent_vals in sent_vals_map.items():
            sent_toks = set([tok for val in sent_vals for tok in val.split(" ")])
            ques_toks = ques_val.split(" ")
            if all(ques_tok in sent_toks for ques_tok in ques_toks):
                is_matched = True
                matching_sent_idxs.add(sent_idx)
        if is_matched:
            ques_vals_matched[ques_val] = matching_sent_idxs
        else:
            ques_vals_unmatched.add(ques_val)

    #print("ques_dep_matched: %s" % ques_vals_matched)
    #print("ques_dep_unmatched: %s" % ques_vals_unmatched)

    for ques_val, matching_sent_idxs in ques_vals_matched.items():
        dep_relevant_sent_idxs |= matching_sent_idxs

    return dep_relevant_sent_idxs

def get_ner_relevant_sent_idxs(sent_ner_tags_map, idx_to_sent):
    ner_relevant_sent_idxs = set()
    for sent_idx, sent_ner_tags in sent_ner_tags_map.items():
        sent_ner_tags_matched = dict()
        sent_ner_tags_unmatched = set()
        for sent_ner_tag in sent_ner_tags:
            is_matched = False
            matching_sent_idxs = set()
            for target_sent_idx, target_sent_ner_tags in sent_ner_tags_map.items():
                if sent_idx == target_sent_idx:
                    continue
                sent_ner_tag_toks = sent_ner_tag.split(" ")
                target_sent_ner_tags_toks = set([ner_tag_tok for target_sent_ner_tag in target_sent_ner_tags for ner_tag_tok in target_sent_ner_tag.split(" ")])
                if all(sent_ner_tag_tok in target_sent_ner_tags_toks for sent_ner_tag_tok in sent_ner_tag_toks):
                    is_matched = True
                    matching_sent_idxs.add(target_sent_idx)

            if is_matched:
                sent_ner_tags_matched[sent_ner_tag] = matching_sent_idxs
            else: 
                sent_ner_tags_unmatched.add(sent_ner_tag)

        #print("sentence %s %s" % (sent_idx, idx_to_sent[sent_idx]))
        #print("sent_ner_matched: %s" % sent_ner_tags_matched)
        #print("sent_ner_unmatched: %s" % sent_ner_tags_unmatched)

        for sent_ner_tag, matching_sent_idxs in sent_ner_tags_matched.items():
            ner_relevant_sent_idxs |= {sent_idx}
            ner_relevant_sent_idxs |= matching_sent_idxs

    return ner_relevant_sent_idxs


def get_ques_ner_relevant_sent_idxs(sent_ner_tags_map, idx_to_sent, ques_ner_tags):
    #print("ques_ner_tags: %s" % ques_ner_tags)
    ner_relevant_sent_idxs = set()
    ques_ner_tags_matched = dict()
    ques_ner_tags_unmatched = set()
    for ques_ner_tag in ques_ner_tags:
        is_matched = False
        matching_sent_idxs = set()
        for sent_idx, sent_ner_tags in sent_ner_tags_map.items():
            ques_ner_tag_toks = ques_ner_tag.split(" ")
            sent_ner_tags_toks = set([ner_tag_tok for sent_ner_tag in sent_ner_tags for ner_tag_tok in sent_ner_tag.split(" ")])
            if all(ques_ner_tag_tok in sent_ner_tags_toks for ques_ner_tag_tok in ques_ner_tag_toks):
                is_matched = True
                matching_sent_idxs.add(sent_idx)

        if is_matched:
            ques_ner_tags_matched[ques_ner_tag] = matching_sent_idxs
        else: 
            ques_ner_tags_unmatched.add(ques_ner_tag)

    #print("ques_ner_matched: %s" % ques_ner_tags_matched)
    #print("ques_ner_unmatched: %s" % ques_ner_tags_unmatched)

    for sent_ner_tag, matching_sent_idxs in ques_ner_tags_matched.items():
        ner_relevant_sent_idxs |= matching_sent_idxs

    return ner_relevant_sent_idxs

def get_base_adversarial_set(dataset):
    base_examples = set()
    base_examples_to_adversial_examples = dict()
    for idx, elem in enumerate(dataset.elems):
        passage = [
            token.lower() for (token, offset) in elem['context_tokens']
        ][:dataset.args.max_context_length]
        t_passage = elem["context"]

        is_base_example = True
        for base_example in base_examples:
            if t_passage.startswith(base_example[1]):
                if base_example[0] not in base_examples_to_adversial_examples: 
                    base_examples_to_adversial_examples[base_example[0]] = set()
                base_examples_to_adversial_examples[base_example[0]].add(idx) 
                is_base_example = False
                break
        if is_base_example:
            base_examples.add((idx, t_passage))
    base_example_idxs = set(val[0] for val in base_examples)

    return base_example_idxs, base_examples_to_adversial_examples


class Status:
    num_ques_base_correct = 0
    num_ques_base_incorrect = 0
    num_ques_adv_correct = 0
    num_ques_adv_incorrect = 0

class QADataset(Dataset):
    """
    This class creates a data generator.

    Args:
        args: `argparse` object.
        path: Path to a data file (.gz), e.g. "datasets/squad_dev.jsonl.gz".

    Attributes:
        args: `argparse` object.
        meta: Dataset metadata (e.g. dataset name, split).
        elems: A list of raw examples (jsonl).
        samples: A list of preprocessed examples (tuple). passages and
            questions are shortened to max sequence length.
        tokenizer: `Tokenizer` object.
        batch_size: Int. The number of example in a mini batch.
    """
    def __init__(self, args, path):
        self.args = args
        self.meta, self.elems = load_dataset(path)
        self.samples = self._create_samples()
        self.tokenizer = None
        self.batch_size = args.batch_size if 'batch_size' in args else 1
        self.pad_token_id = self.tokenizer.pad_token_id \
            if self.tokenizer is not None else 0

    def _create_samples(self):
        """
        Formats raw examples to desired form. Any passages/questions longer
        than max sequence length will be truncated.

        Returns:
            A list of words (string).
        """
        import spacy
        nlp = spacy.load("en_core_web_sm")

        base_example_idxs, base_examples_to_adversial_examples = get_base_adversarial_set(self)

        print("No. samples: %s" % len(self.elems))
        #print("No. base samples: %s" % len(base_example_idxs))
        #print("No. adversarial samples: %s" % (len(self.elems) - len(base_example_idxs)))

        all_state = dict()
        last_sent_correct_answers = set() 

        samples = []
        for idx, elem in enumerate(self.elems):

            t_passage = elem['context']
            #print("***************************************************************************************************")
            #print("BASE" if idx in base_example_idxs else "ADVERSARIAL")
            #print("%s: %s" % (idx, t_passage))

            nlp_passage = nlp(t_passage)
            sent_to_idx, idx_to_sent = get_sent_idx_maps(nlp_passage)
            last_sent = idx_to_sent[max(idx_to_sent)]
            last_sent_start_tok = last_sent.start
            last_start_sent_char = last_sent.start_char

            # NER Tagging
            sent_ner_tags_map = get_ner_tags(nlp_passage, sent_to_idx)
            ner_relevant_sent_idxs = get_ner_relevant_sent_idxs(sent_ner_tags_map, idx_to_sent)
            ner_irrelevant_sent_idxs = set([idx for idx in idx_to_sent if idx not in ner_relevant_sent_idxs])
            final_ner_irrelevant_sent_idxs = get_final_set(ner_irrelevant_sent_idxs, idx_to_sent)
          
            # Dependency Parsing
            sent_vals_map = get_sent_vals_map(nlp_passage, sent_to_idx, idx_to_sent)
            dep_relevant_sent_idxs = get_dep_relevant_sent_idxs(sent_vals_map, idx_to_sent)
            dep_irrelevant_sent_idxs = set([idx for idx in idx_to_sent if idx not in dep_relevant_sent_idxs])
            final_dep_irrelevant_sent_idxs = get_final_set(dep_irrelevant_sent_idxs, idx_to_sent)
            
            dep_ner_irrelevant_sent_idxs = set(final_dep_irrelevant_sent_idxs) & set(final_ner_irrelevant_sent_idxs)
            final_dep_ner_irrelevant_sent_idxs = get_final_set(dep_ner_irrelevant_sent_idxs, idx_to_sent)
           
            for qa in elem['qas']:
                qid = qa['qid']
                question = [
                    token for (token, offset) in qa['question_tokens']
                ][:self.args.max_question_length]
                answers = qa['detected_answers']
                answer_start, answer_end = answers[0]['token_spans'][0]

                t_question = qa['question']
                #print("*")
                #print("question: %s" % t_question)

                nlp_question = nlp(t_question)
                ques_to_idx, idx_to_ques = get_sent_idx_maps(nlp_question)

                # NER Tagging
                ques_ner_tags_map = get_ner_tags(nlp_question, ques_to_idx)
                ques_ner_tags = list(ques_ner_tags_map.values())[0] if ques_ner_tags_map else []
                ques_ner_relevant_sent_idxs = get_ques_ner_relevant_sent_idxs(sent_ner_tags_map, idx_to_sent, ques_ner_tags)
                ques_ner_irrelevant_sent_idxs = set([idx for idx in idx_to_sent if idx not in ques_ner_relevant_sent_idxs])
                final_ques_ner_irrelevant_sent_idxs = get_final_set(ques_ner_irrelevant_sent_idxs, idx_to_sent)

                # Dependency Parsing
                ques_vals_map = get_sent_vals_map(nlp_question, ques_to_idx, idx_to_ques)
                ques_vals = list(ques_vals_map.values())[0] if ques_vals_map else []
                ques_dep_relevant_sent_idxs = get_ques_dep_relevant_sent_idxs(sent_vals_map, idx_to_sent, ques_vals)
                ques_dep_irrelevant_sent_idxs = set([idx for idx in idx_to_sent if idx not in ques_dep_relevant_sent_idxs])
                final_ques_dep_irrelevant_sent_idxs = get_final_set(ques_dep_irrelevant_sent_idxs, idx_to_sent)

                final_ques_dep_ner_irrelevant_sent_idxs = final_ques_ner_irrelevant_sent_idxs & final_ques_dep_irrelevant_sent_idxs


                passage_irrelevant = [  #("P_NONE", set(idx_to_sent.keys())),\
                                        ("P_NER", final_ner_irrelevant_sent_idxs),\
                                        #("P_DEP", final_dep_irrelevant_sent_idxs),\
                                        #("P_DEP+NER", final_dep_ner_irrelevant_sent_idxs)
                                    ]
                question_irrelevant = [ #("Q_NONE", set(idx_to_sent.keys())),\
                                        ("Q_NER", final_ques_ner_irrelevant_sent_idxs),\
                                        #("Q_DEP", final_ques_dep_irrelevant_sent_idxs),\
                                        #("Q_DEP+NER", final_ques_dep_ner_irrelevant_sent_idxs)
                                    ]

                class_set = None
                for (p_tag, p_set) in passage_irrelevant:
                    for (q_tag, q_set) in question_irrelevant:
                        if p_tag == "P_NONE" and q_tag == "Q_NONE":
                            continue
                        tag = "%s_%s" % (p_tag, q_tag)
                        if tag not in all_state:
                            all_state[tag] = Status()
                        state = all_state[tag]

                        class_set = p_set & q_set

                        if idx in base_example_idxs:
                            if answer_start > last_sent_start_tok:
                                if class_set: 
                                    assert(len(class_set) == 1 and min(class_set) == max(idx_to_sent.keys()))
                                    state.num_ques_base_incorrect += 1
                                else:
                                    state.num_ques_base_correct += 1
                        else: 
                            if class_set:
                                assert(len(class_set) == 1 and min(class_set) == max(idx_to_sent.keys()))
                                state.num_ques_adv_correct += 1
                            else: 
                                state.num_ques_adv_incorrect += 1

                
                if class_set:
                    passage = [
                        token.lower() for (token, offset) in elem['context_tokens'] if offset < last_start_sent_char
                    ][:self.args.max_context_length]
                else: 
                    passage = [
                        token.lower() for (token, offset) in elem['context_tokens']
                    ][:self.args.max_context_length]
 
                samples.append(
                    (qid, passage, question, answer_start, answer_end)
                )

                if answer_start > last_sent_start_tok:
                    last_sent_correct_answers.add(t_question)
               
        #print("**********")
        #print("RESULTS")      
        #print("num_last_sentence_correct_ansers: %s" % len(last_sent_correct_answers))
      
        #for tag, state  in all_state.items():
        #    print("*")
        #    print("TAG: %s" % tag)
        #    print("num_ques_base_correct: %s" % state.num_ques_base_correct)
        #    print("num_ques_base_incorrect: %s" % state.num_ques_base_incorrect)
        #    print("num_ques_adv_correct: %s" % state.num_ques_adv_correct)
        #    print("num_ques_adv_incorrect: %s" % state.num_ques_adv_incorrect)

        print("Completed processing samples")
        return samples


    def _create_data_generator(self, shuffle_examples=False):
        """
        Converts preprocessed text data to Torch tensors and returns a
        generator.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A generator that iterates through all examples one by one.
            (Tuple of tensors)
        """
        if self.tokenizer is None:
            raise RuntimeError('error: no tokenizer registered')

        example_idxs = list(range(len(self.samples)))
        if shuffle_examples:
            shuffle(example_idxs)

        passages = []
        questions = []
        start_positions = []
        end_positions = []
        for idx in example_idxs:
            # Unpack QA sample and tokenize passage/question.
            qid, passage, question, answer_start, answer_end = self.samples[idx]

            # Convert words to tensor.
            passage_ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(passage)
            )
            question_ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(question)
            )
            answer_start_ids = torch.tensor(answer_start)
            answer_end_ids = torch.tensor(answer_end)

            # Store each part in an independent list.
            passages.append(passage_ids)
            questions.append(question_ids)
            start_positions.append(answer_start_ids)
            end_positions.append(answer_end_ids)

        return zip(passages, questions, start_positions, end_positions)

    def _create_batches(self, generator, batch_size):
        """
        This is a generator that gives one batch at a time. Tensors are
        converted to "cuda" if necessary.

        Args:
            generator: A data generator created by `_create_data_generator`.
            batch_size: Int. The number of example in a mini batch.

        Yields:
            A dictionary of tensors containing a single batch.
        """
        current_batch = [None] * batch_size
        no_more_data = False
        # Loop through all examples.
        while True:
            bsz = batch_size
            # Get examples from generator
            for i in range(batch_size):
                try:
                    current_batch[i] = list(next(generator))
                except StopIteration:  # Run out examples
                    no_more_data = True
                    bsz = i  # The size of the last batch.
                    break
            # Stop if there's no leftover examples
            if no_more_data and bsz == 0:
                break

            passages = []
            questions = []
            start_positions = torch.zeros(bsz)
            end_positions = torch.zeros(bsz)
            max_passage_length = 0
            max_question_length = 0
            # Check max lengths for both passages and questions
            for ii in range(bsz):
                passages.append(current_batch[ii][0])
                questions.append(current_batch[ii][1])
                start_positions[ii] = current_batch[ii][2]
                end_positions[ii] = current_batch[ii][3]
                max_passage_length = max(
                    max_passage_length, len(current_batch[ii][0])
                )
                max_question_length = max(
                    max_question_length, len(current_batch[ii][1])
                )

            # Assume pad token index is 0. Need to change here if pad token
            # index is other than 0.
            padded_passages = torch.zeros(bsz, max_passage_length)
            padded_questions = torch.zeros(bsz, max_question_length)
            # Pad passages and questions
            for iii, passage_question in enumerate(zip(passages, questions)):
                passage, question = passage_question
                padded_passages[iii][:len(passage)] = passage
                padded_questions[iii][:len(question)] = question

            # Create an input dictionary
            batch_dict = {
                'passages': cuda(self.args, padded_passages).long(),
                'questions': cuda(self.args, padded_questions).long(),
                'start_positions': cuda(self.args, start_positions).long(),
                'end_positions': cuda(self.args, end_positions).long()
            }

            if no_more_data:
                if bsz > 0:
                    # This is the last batch (smaller than `batch_size`)
                    yield batch_dict
                break
            yield batch_dict

    def get_batch(self, shuffle_examples=False):
        """
        Returns a data generator that supports mini-batch.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A data generator that iterates though all batches.
        """
        return self._create_batches(
            self._create_data_generator(shuffle_examples=shuffle_examples),
            self.batch_size
        )

    def register_tokenizer(self, tokenizer):
        """
        Stores `Tokenizer` object as an instance variable.

        Args:
            tokenizer: If `True`, shuffle examples. Default: `False`
        """
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.samples)
