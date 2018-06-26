# -*- coding: utf-8 -*-
""" ------------------------------------------------- 
File Name: data_helper
Description : 
Author : arlen
date：18-6-22
------------------------------------------------- """
import re
import pickle
import numpy as np
from gensim import models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# 读取数据
def read_data(file_path):
    with open(file_path, encoding='utf-8') as f:
        data = f.readlines()
    data = [re.split('\t', i) for i in data]
    q1 = [i[1] for i in data]
    q2 = [i[2] for i in data]
    label = [float(i[3]) for i in data]
    return q1, q2, label


train_q1, train_q2, train_label = read_data('./ch_input/train.txt')
test_q1, test_q2, test_label = read_data('./ch_input/test.txt')
dev_q1, dev_q2, dev_label = read_data('./ch_input/dev.txt')


# 构造训练word2vec的语料库
corpus = train_q1 + train_q2 + test_q1 + test_q2 + dev_q1 + dev_q2
w2v_corpus = [i.split() for i in corpus]
word_set = set(' '.join(corpus).split())

MAX_SEQUENCE_LENGTH = 30  # sequence最大长度为30个词
EMB_DIM = 300  # 词向量为300维

# 训练word2vec模型
w2v_model = models.Word2Vec(w2v_corpus, size=EMB_DIM, window=5, min_count=1, sg=1, workers=4, seed=1234, iter=25)
w2v_model.save('./w2v/w2v_model.pkl')

tokenizer = Tokenizer(num_words=len(word_set))
tokenizer.fit_on_texts(corpus)

train_q1 = tokenizer.texts_to_sequences(train_q1)
train_q2 = tokenizer.texts_to_sequences(train_q2)

test_q1 = tokenizer.texts_to_sequences(test_q1)
test_q2 = tokenizer.texts_to_sequences(test_q2)

dev_q1 = tokenizer.texts_to_sequences(dev_q1)
dev_q2 = tokenizer.texts_to_sequences(dev_q2)

train_pad_q1 = pad_sequences(train_q1, maxlen=MAX_SEQUENCE_LENGTH)
train_pad_q2 = pad_sequences(train_q2, maxlen=MAX_SEQUENCE_LENGTH)

test_pad_q1 = pad_sequences(test_q1, maxlen=MAX_SEQUENCE_LENGTH)
test_pad_q2 = pad_sequences(test_q2, maxlen=MAX_SEQUENCE_LENGTH)

dev_pad_q1 = pad_sequences(dev_q1, maxlen=MAX_SEQUENCE_LENGTH)
dev_pad_q2 = pad_sequences(dev_q2, maxlen=MAX_SEQUENCE_LENGTH)

embedding_matrix = np.zeros([len(tokenizer.word_index) + 1, EMB_DIM])

for word, idx in tokenizer.word_index.items():
    embedding_matrix[idx, :] = w2v_model.wv[word]


def save_pickle(fileobj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(fileobj, f)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        fileobj = pickle.load(f)
    return fileobj


model_data = {'train_q1': train_pad_q1, 'train_q2': train_pad_q2, 'train_label': train_label,
              'test_q1': test_pad_q1, 'test_q2': test_pad_q2, 'test_label': test_label,
              'dev_q1': dev_pad_q1, 'dev_q2': dev_pad_q2, 'dev_label': dev_label}

save_pickle(corpus, './w2v/corpus.pkl')
save_pickle(model_data, './w2v/model_data.pkl')
save_pickle(embedding_matrix, './w2v/embedding_matrix.pkl')
save_pickle(tokenizer, './w2v/tokenizer.pkl')
