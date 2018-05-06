import tensorflow as tf
import data_helper
import numpy as np

def sentence2id(sentences, word2idx, seq_length):
    idx = []
    for words in sentences:
        if len(words) > seq_length:
            words = words[:seq_length]
        elif len(words) < seq_length:
            for i in range(len(words), seq_length):
                words.append('<0>')
        id = words2id(words, word2idx)
        idx.append(id)
    return idx

def words2id(words, word2idx):
    id = []
    for word in words:
        id.append(word2idx.get(word, word2idx['<0>']))
    return id


def get_batch_data(path, batch_size, seq_length, word2idx):
    sentences, labels = data_helper.load_data(path)
    sentences = sentence2id(sentences, word2idx, seq_length)
    train_num = (len(labels) // batch_size) + 1
    data_queues = tf.train.slice_input_producer([sentences, labels])
    inputs, labels = tf.train.shuffle_batch(data_queues, num_threads=8,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=True)
    return inputs, labels, train_num

def one_hot(indices, depth):
    one_hot_indices = []
    for id in indices:
        idx = np.zeros(depth, dtype=np.int32)
        idx[id] = 1
        one_hot_indices.append(idx)
    return one_hot_indices