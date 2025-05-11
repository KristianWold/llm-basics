import re

def pair_freq(indicies, stop_token):
    indicies = np.array(indicies)
    
    indicies = np.vstack([indicies[:-1], indicies[1:]]).T
    unique, count = np.unique(indicies, axis=0, return_counts=True)
    mask = (unique == stop_token).any(axis=1)
    unique = unique[~mask]
    count = count[~mask]

    idx = np.argmax(count)
    pair = unique[idx]
    return pair


class TokenizerBPE:
    def __init__(self, corpus, num_merges):
        self.tokenizer = TokenizerChar(corpus)
        self.token_to_idx = self.tokenizer.token_to_idx
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}

        self.vocab_size = self.tokenizer.vocab_size

        self.create_hash()

        self.stop_token = np.array(self.tokenizer.tokenize(" "))[0]
        
        corpus_clean = [normalize_to_ascii(line) for line in corpus]
        corpus_flatten = " ".join(corpus_clean)
        
        corpus_flatten = re.findall(r"[\w']+|[^\w\s]", corpus_flatten)
        corpus_flatten = " ".join(corpus_flatten)
        
        corpus_indices = self.tokenizer.tokenize(corpus_flatten)

        self.merge_list = []
        for i in tqdm(range(num_merges)):
            corpus_indices = self.merge(corpus_indices)

        self.create_hash()
        self.word_list = None


    def tokenize(self, text):
        indices = np.array(self.tokenizer.tokenize(text))
        for (idx1, idx2), new_idx in self.merge_list:
            slice = np.where(np.logical_and(indices[:-1] == idx1,  indices[1:] == idx2))
            if len(slice[0]) > 0:
                indices[:-1][slice] = new_idx
                indices = np.delete(indices, (slice[0]+1))

        return tf.expand_dims(tf.convert_to_tensor(indices, dtype=tf.int32), axis=0)

    def detokenize(self, indices):
        text = self.table_detokenize.lookup(indices)
        text = tf.strings.reduce_join(text, axis=-1, separator="")
        return text

    def merge(self, corpus_indices):
        corpus_indices = np.array(corpus_indices)    

        new_idx = self.vocab_size
        idx1, idx2 = pair_freq(corpus_indices, self.stop_token)
        self.merge_list.append([(idx1, idx2), self.vocab_size])

    
        token1 = self.idx_to_token[idx1]
        token2 = self.idx_to_token[idx2]
        print(token1, token2)
        new_token = token1 + token2
        self.token_to_idx[new_token] = new_idx
        self.idx_to_token[new_idx] = new_token
        self.vocab_size += 1

        slice = np.where(np.logical_and(corpus_indices[:-1] == idx1, corpus_indices[1:] == idx2))
        if len(slice[0]) > 0:
            corpus_indices[:-1][slice] = new_idx
            corpus_indices = np.delete(corpus_indices, (slice[0]+1))

        return corpus_indices

    def create_hash(self):
        vocab = list(self.token_to_idx.keys())
        indicies = list(self.token_to_idx.values())

        self.tokenizer.create_hash()
        self.table_detokenize = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(indicies, vocab), 
                                                          default_value="")
        
    def destroy_hash(self):
        self.tokenizer.destroy_hash()
        self.table_detokenize = None


