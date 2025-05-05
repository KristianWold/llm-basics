import tensorflow as tf
import unicodedata

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

        default_value = -1

        self.table_tokenize = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(vocab, indicies), 
                                                        default_value=default_value)
        self.table_detokenize = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(indicies, vocab), 
                                                          default_value="")

    
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