import unicodedata
import os
import numpy as np
import tensorflow as tf

def normalize_to_ascii(s: str) -> str:
    # 1) Decompose Unicode characters (e.g. é → e +  ́)
    # 2) Drop the non-ASCII combining marks in the encode step
    normalized = unicodedata.normalize('NFKD', s)
    ascii_bytes = normalized.encode('ascii', 'ignore')
    return ascii_bytes.decode('ascii')


def read_first_n(directory_path, n):
    # List all entries in the directory
    filenames = os.listdir(directory_path)
    # Filter to only .txt files
    txt_files = [f for f in filenames if f.lower().endswith('.story')]
    # Sort alphabetically (or by any other criteria you like)
    #txt_files.sort()
    # Take the first n
    first_n = txt_files[:n]
    
    contents = []
    for fname in first_n:
        full_path = os.path.join(directory_path, fname)
        with open(full_path, 'r', encoding='utf-8') as f:
            contents.append(normalize_to_ascii(f.read()))
    return contents


def word_split(line):
    
    normalized_line = normalize_to_ascii(line)
    # Split into words
    word_list = normalized_line.strip().split()
    word_list = [list(word) for word in word_list]
    return word_list


def sample_batch(contents, 
                 batch_size, 
                 tokenizer, 
                 max_seq_len):
    
    samples = len(contents)
    indicies_list = []
    for b in range(batch_size):
        
        idx = np.random.randint(0, samples)
        text = contents[idx]
        length = text.shape[1]
        if length < max_seq_len:
            continue
        else:
            start = np.random.randint(0, length - max_seq_len)
            indicies = text[:,start:start + max_seq_len]
            indicies_list.append(indicies)
    
    indicies_list = tf.cast(tf.concat(indicies_list, axis=0), tf.int32)
    y_true = tf.cast(indicies_list, tf.int32)

    return indicies_list, y_true

import os

