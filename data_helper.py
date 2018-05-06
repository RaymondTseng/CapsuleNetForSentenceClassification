# -*- coding:utf-8 -*-

import numpy as np
from nltk.tokenize import WordPunctTokenizer
import re
import word2vec
import random
punc = u"[\s+\.\!\/_,\-\?$%^*()+\"\']+|[+——！，。？、~@#￥%……&*（）]+"

def load_data(path):
    sentences = []
    labels = []
    lines = open(path, 'r').readlines()
    label_num = int(lines[0].strip())
    lines = lines[1:]
    random.shuffle(lines)
    for i, line in enumerate(lines):
        temp = line.strip().decode('utf-8').split('\t')
        sentence = temp[0]
        label = int(temp[1])
        labels.append(label)

        sentence = re.sub(punc, u' ', sentence).strip()
        words = WordPunctTokenizer().tokenize(sentence)
        words = [word.lower() for word in words]
        sentences.append(words)
    return sentences, labels

def load_embedding(path):
    wv = word2vec.load(path)
    vocab = wv.vocab
    word2idx = {}
    word_embedding = wv.vectors
    for i in range(1, len(vocab) + 1):
        word2idx[vocab[i-1]] = i
    word2idx['<0>'] = 0
    word_zero = np.zeros(len(word_embedding[0]))
    word_embedding = np.vstack([word_zero, word_embedding])
    return word2idx, word_embedding







