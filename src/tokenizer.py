import tensorflow as tf
import numpy as np
import unicodedata
from tqdm.notebook import tqdm

def normalize_to_ascii(s: str) -> str:
    # 1) Decompose Unicode characters (e.g. é → e +  ́)
    # 2) Drop the non-ASCII combining marks in the encode step
    normalized = unicodedata.normalize('NFKD', s)
    ascii_bytes = normalized.encode('ascii', 'ignore')
    return ascii_bytes.decode('ascii')

def word_split(line):
    
    normalized_line = normalize_to_ascii(line)
    # Split into words
    word_list = normalized_line.strip().split()
    word_list = [list(word) for word in word_list]
    return word_list



class TokenizerChar:
    def __init__(self, corpus):

        # Flatten the list of lists into a single list
        corpus_flatten = [item for sublist in corpus for item in sublist]
    
        self.vocab = sorted(list(set(corpus_flatten)))
        self.vocab_size = len(self.vocab)
        self.token_to_idx = {ch: i for i, ch in enumerate(self.vocab)}

        vocab = list(self.token_to_idx.keys())
        indicies = list(self.token_to_idx.values())

        self.create_hash()

    
    def tokenize(self, text):
        if isinstance(text, list):
            text_list = text
            indices = tf.stack([self.tokenize(text) for text in text_list])
        else:
            text = tf.strings.unicode_split(text, input_encoding="UTF-8")
            indices = self.table_tokenize.lookup(text)
        return indices
    
    def detokenize(self, indices):
        text = self.table_detokenize.lookup(indices)
        text = tf.strings.reduce_join(text, axis=-1, separator="")
        return text
    
    def create_hash(self):
        vocab = list(self.token_to_idx.keys())
        indicies = list(self.token_to_idx.values())

        self.table_tokenize = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(vocab, indicies), 
                                                        default_value=-1)
        self.table_detokenize = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(indicies, vocab), 
                                                          default_value="")
        
    def destroy_hash(self):
        self.table_tokenize = None
        self.table_detokenize = None

from collections import Counter
import itertools

def pair_freq(word_list):
    # Flatten all adjacent pairs at once
    all_pairs = itertools.chain.from_iterable(
        zip(word, word[1:]) for word in word_list
    )
    return Counter(all_pairs)


#def pair_freq(word_list):
#    """Return a dict mapping pairs of words to their counts."""
#    pairs = {}
#    for word in word_list:
#        for i in range(len(word) - 1):
#            pair = (word[i], word[i + 1])
#            pairs[pair] = pairs.get(pair, 0) + 1
#    return pairs


class TokenizerBPE:
    def __init__(self, corpus, num_merges):
        self.tokenizer = TokenizerChar(corpus)
        self.token_to_idx = self.tokenizer.token_to_idx
        self.vocab_size = self.tokenizer.vocab_size

        self.word_list = []
        for line in corpus:
            self.word_list.extend(word_split(line))

        self.merge_list = []
        for i in tqdm(range(num_merges)):
            self.merge()

        vocab = list(self.token_to_idx.keys())
        indicies = list(self.token_to_idx.values())

        self.create_hash()
        self.word_list = None


    def tokenize(self, text):
        indicies = np.array(self.tokenizer.tokenize(text))
        for (idx1, idx2), new_idx in self.merge_list:
            for i in reversed(range(len(indicies) - 1)):
                pair = (indicies[i], indicies[i + 1])
                if pair == (idx1, idx2):
                    indicies[i] = new_idx
                    indicies = np.delete(indicies, i + 1)
        indicies = tf.convert_to_tensor(indicies, dtype=tf.int64)
        return indicies

    def detokenize(self, indices):
        text = self.table_detokenize.lookup(indices)
        text = tf.strings.reduce_join(text, axis=-1, separator="")
        return text

    def merge(self):
        pf = pair_freq(self.word_list)
        key_max = max(pf, key=pf.get)
        token1, token2 = key_max
        new_token = token1 + token2
        self.token_to_idx[new_token] = self.vocab_size

        idx1, idx2 = self.token_to_idx[token1], self.token_to_idx[token2]
        self.merge_list.append([(idx1, idx2), self.vocab_size])

        self.vocab_size += 1

        for word in self.word_list:
            for i in reversed(range(len(word) - 1)):
                pair = (word[i], word[i + 1])
                if pair == key_max:
                    word[i] = new_token
                    word.pop(i + 1)

    def create_hash(self):
        vocab = list(self.token_to_idx.keys())
        indicies = list(self.token_to_idx.values())

        self.tokenizer.create_hash()
        self.table_detokenize = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(indicies, vocab), 
                                                          default_value="")
        
    def destroy_hash(self):
        self.tokenizer.destroy_hash()
        self.table_detokenize = None