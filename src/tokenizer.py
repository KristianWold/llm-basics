import tensorflow as tf

class TokenizerChar:
    def __init__(self, text):
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)
        char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}

        vocab = list(char_to_idx.keys())
        indicies = list(char_to_idx.values())

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
    

class TokenizerBPE:
    def __init__(self, text):
        self.tokenizer = tf.keras.layers.TextVectorization(standardize=None, split=None, ragged=False)
        self.tokenizer.adapt([text])
        self.vocab_size = self.tokenizer.vocabulary_size()
        self.vocab = self.tokenizer.get_vocabulary()
        self.vocab_dict = {word: i for i, word in enumerate(self.vocab)}

    def tokenize(self, text):
        return self.tokenizer(text)

    def detokenize(self, indices):
        return self.tokenizer.get_vocabulary()[indices]