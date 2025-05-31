import tensorflow as tf
import numpy as np
import unicodedata
import re
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

        self.create_hash()

    
    def encode(self, text):
        if isinstance(text, list):
            text_list = text
            indices = tf.stack([self.encode(text) for text in text_list])
        else:
            text = tf.strings.unicode_split(text, input_encoding="UTF-8")
            indices = self.table_tokenize.lookup(text)
        return indices
    
    def decode(self, indices):
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

class TokenizerBPE:
    def __init__(self, corpus, num_merges, lowercase=False):
        if lowercase:
            print("Lowercasing corpus")
            corpus = [line.lower() for line in tqdm(corpus)]

        corpus_clean = [normalize_to_ascii(line) for line in corpus]
        corpus_clean = [re.sub(r"\s+", " ", line) for line in corpus_clean]  

        self.tokenizer_char = TokenizerChar(corpus_clean)
        self.token_to_idx = self.tokenizer_char.token_to_idx
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        self.token_freq = {}

        self.vocab_size = self.tokenizer_char.vocab_size
        self.create_hash()

        self.pre_merge_list = []
        self.add_special_tokens(["<sep>"])
        self.sep_token = self.token_to_idx["<sep>"]


        corpus_flatten = " ".join(corpus_clean)
        corpus_flatten = re.findall(r"\s*[\w']+|[^\w]", corpus_flatten)
        # shuffle and sample before merging
        corpus_flatten = "<sep>".join(corpus_flatten)
        
        corpus_tokens = self.tokenizer_char.encode(corpus_flatten)
        corpus_tokens = self.pre_merge(corpus_tokens)  

        print(corpus_tokens)

        corpus_tokens
        

        print("Merging tokens")
        self.merge_list = []
        for i in tqdm(range(num_merges)):
            corpus_tokens = self.merge(corpus_tokens)

        self.create_hash()
        self.word_list = None


    def encode(self, text, verbose=False):
        #text = text.lower()

        if verbose:
            decorator = tqdm
        else:
            decorator = lambda x: x

        indices = np.array(self.tokenizer_char.encode(text))
        if len(self.pre_merge_list) > 0:
            indices = self.pre_merge(indices)

        for (idx1, idx2), new_idx in decorator(self.merge_list):
            slice = np.where(np.logical_and(indices[:-1] == idx1,  indices[1:] == idx2))
            if len(slice[0]) > 0:
                indices[:-1][slice] = new_idx
                indices = np.delete(indices, (slice[0]+1))

        return tf.expand_dims(tf.convert_to_tensor(indices, dtype=tf.int32), axis=0)

    def decode(self, indices):
        text = self.table_detokenize.lookup(indices)
        text = tf.strings.reduce_join(text, axis=-1, separator="")
        return text

    def merge(self, corpus_tokens):
        corpus_tokens = np.array(corpus_tokens)  

        new_idx = self.vocab_size
        (idx1, idx2), counts = pair_freq(corpus_tokens, self.sep_token, self.vocab_size)
        self.merge_list.append([(idx1, idx2), self.vocab_size])

    
        token1 = self.idx_to_token[idx1]
        token2 = self.idx_to_token[idx2]
        print(token1, token2, counts)
        new_token = token1 + token2
        self.token_to_idx[new_token] = new_idx
        self.idx_to_token[new_idx] = new_token
        self.token_freq[new_token] = counts
        self.vocab_size += 1

        slice = np.where(np.logical_and(corpus_tokens[:-1] == idx1, corpus_tokens[1:] == idx2))
        if len(slice[0]) > 0:
            corpus_tokens[:-1][slice] = new_idx
            corpus_tokens = np.delete(corpus_tokens, (slice[0]+1))

        return corpus_tokens
    
    def add_special_tokens(self, special_tokens):
        for token in special_tokens:
            if token not in self.token_to_idx:
                token_indices = self.tokenizer_char.encode(token)
                self.token_to_idx[token] = self.vocab_size
                self.idx_to_token[self.vocab_size] = token
                self.pre_merge_list.append([token_indices, self.vocab_size])
                self.vocab_size += 1


        self.create_hash()
    

    def pre_merge(self, indices):
        indices = np.array(indices)
        for token, value in self.pre_merge_list:
            length = len(token)
            token = np.array(token).reshape(-1,1)
            indices_list = []
            for i in range(length):
                indices_list.append(np.pad(indices, (length - i - 1, i), 'constant', constant_values=(0, 0)))
            indices_list = np.array(indices_list)

            locs = np.where(np.equal(indices_list, token).all(axis=0))[0] - length + 1
            del_idx = []
            for loc in list(reversed(locs)):
                indices[loc] = value
                del_idx.extend(list(range(loc + 1, loc + length)))

            if len(del_idx) > 0:    
                del_idx = np.array(del_idx)
                indices = np.delete(indices, del_idx)

        return indices


    def create_hash(self):
        vocab = list(self.token_to_idx.keys())
        indicies = list(self.token_to_idx.values())

        self.tokenizer_char.create_hash()
        self.table_detokenize = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(indicies, vocab), 
                                                          default_value="")
        
    def destroy_hash(self):
        self.tokenizer_char.destroy_hash()
        self.table_detokenize = None


def pair_freq(indices, stop_token, vocab_size):
    indices = np.array(indices)
    mask = (indices[:-1] == stop_token) + (indices[1:] == stop_token)

    indices_large = indices[:-1] + indices[1:]*vocab_size

    indices_large = indices_large[~mask]
    counts = np.bincount(indices_large)
    temp = np.argmax(counts)
    count = counts[temp]
    idx1 = temp % vocab_size
    idx2 = temp // vocab_size

    return (idx1, idx2), count



def fuse_tokenized_corpus(corpus, tokenizer):
    SOS = tokenizer.token_to_idx["<s>"]
    EOS = tokenizer.token_to_idx["</s>"]
    corpus_list = [SOS]
    for line in tqdm(corpus):
        corpus_list.append(line)
        corpus_list.append(EOS)
        corpus_list.append(SOS)

    corpus = tf.concat(corpus_list[:-1], axis=0)
    return corpus

def chunk_corpus(corpus, chunk_size):
    corpus = tf.data.Dataset.from_tensor_slices(corpus)
    corpus = corpus.batch(chunk_size, drop_remainder=True)
    return corpus