import unicodedata
import os
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm


def cosine_similarity(embed_a, embed_b, normalize=True):
    """
    Compute the cosine similarity between two vectors.
    """
    if normalize:
        embed_a = tf.nn.l2_normalize(embed_a, axis=1)
        embed_b = tf.nn.l2_normalize(embed_b, axis=1)
    dot_product = embed_a@tf.transpose(embed_b)


    return dot_product


def cluster(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    inertia = kmeans.inertia_
    labels = kmeans.labels_
    clusters = kmeans.cluster_centers_

    return inertia, labels, clusters


class EmbeddingClustering:
    def __init__(self, tokenizer, n_clusters=10):
        
        self.tokenizer = tokenizer
        self.n_clusters = n_clusters

    def fit(self, word_embed):
        inertia, labels, clusters = cluster(word_embed, self.n_clusters)
        self.word_embed = word_embed
        self.inertia = inertia
        self.labels = labels
        self.clusters = tf.convert_to_tensor(clusters, dtype=tf.float32)

        cos_sim = cosine_similarity(self.clusters, self.word_embed)
        idx =  tf.argsort(cos_sim, axis=-1, direction='DESCENDING', stable=False, name=None)

        print(idx.shape)


class Inference:
    def __init__(self, transformer, tokenizer):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.context = ""

    def next_token(self, tokens, T=1):

        if T != 0:
            # 1) Add temperature
            logits = self.transformer(tokens)[0,-1:]
            next_token = tf.cast(tf.random.categorical(logits/T, num_samples=1), tf.int32)
        else:
            # 2) Greedy sampling
            logits = self.transformer(tokens)[0,-1:]
            next_token = tf.argmax(logits, axis=-1, output_type=tf.int32)

        return next_token
    
    def prompt(self, prompt, max_length=None, T=1):
        self.context += prompt

        wrapper = textwrap.TextWrapper(width=80)

        # create a read-only text area
        ta = widgets.Textarea(
            value="",
            layout=widgets.Layout(width='80ch', height='20em'),
            disabled=True
        )
        display(ta)

        for i in range(max_length):
            # 1) Tokenize the context
            tokens = self.tokenizer.encode(self.context)
            tokens = tf.convert_to_tensor([tokens], dtype=tf.int32)

            # 2) Get the next token
            next_token = self.next_token(tokens, T=T)

            # 3) Append the new token to the context
            self.context += self.tokenizer.decode(next_token.numpy()[0])

            context_clean = self.context.decode('utf-8').replace("\n", " ")
            ta.value = wrapper.fill(context_clean)

            if next_token[0, 0] == self.tokenizer.token_to_idx["</s>"]:
                break
        
    def question(self, question, max_length=None, T=1):
        q = self.tokenizer.token_to_idx["<q>"]
        a = self.tokenizer.token_to_idx["<a>"]
        prompt = q + question + a

        self.prompt(prompt, max_length=max_length, T=T)
