import unicodedata
import os
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm


def cosine_similarity(embed_a, embed_b):
    """
    Compute the cosine similarity between two vectors.
    """
    dot_product = embed_a@tf.transpose(embed_b)
    
    norm_a = tf.linalg.norm(embed_a, axis=1, keepdims=True)
    norm_b = tf.linalg.norm(embed_b, axis=1, keepdims=True)

    return dot_product / (norm_a * norm_b)


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
        word_embed, inertia, labels, clusters = cluster(word_embed, self.n_clusters)
        self.word_embed = word_embed
        self.inertia = inertia
        self.labels = labels
        self.clusters = tf.convert_to_tensor(clusters, dtype=tf.float32)

        cos_sim = cosine_similarity(self.clusters, self.word_embed)
        idx =  tf.argsort(cos_sim, axis=-1, direction='DESCENDING', stable=False, name=None)

        print(idx.shape)
