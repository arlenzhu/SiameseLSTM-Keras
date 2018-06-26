# -*- coding: utf-8 -*-
""" ------------------------------------------------- 
File Name: siamese_lstm
Description : 
Author : arlen
date：18-6-25
------------------------------------------------- """
import sys

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Embedding, Input, GRU, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

import data_helper

input_dim = data_helper.MAX_SEQUENCE_LENGTH
emb_dim = data_helper.EMB_DIM
model_path = './model/siameselstm.hdf5'
tensorboard_path = './model/ensembling'

embedding_matrix = data_helper.load_pickle('./w2v/embedding_matrix.pkl')

embedding_layer = Embedding(embedding_matrix.shape[0],
                            emb_dim,
                            weights=[embedding_matrix],
                            input_length=input_dim,
                            trainable=False)


def euclidean_distance(vects):
    x, y = vects
    return K.exp(-K.sum(K.abs(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


def create_base_network(input_shape):
    input = Input(shape=input_shape, dtype='int32')
    x = embedding_layer(input)
    x = GRU(300, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(x)
    return Model(input, x, name='GRU')


def compute_accuracy(y_true, y_pred):
    pred = np.array([round(i[0]) for i in y_pred])
    y_true = np.array(y_true)
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    return K.mean(K.cast(K.equal(K.round(y_true), K.round(y_pred)), K.floatx()))


def siamese_model():
    input_shape = (input_dim, )

    base_network = create_base_network(input_shape)

    input_q1 = Input(shape=input_shape, dtype='int32', name='sequence1')
    input_q2 = Input(shape=input_shape, dtype='int32', name='sequence2')

    processed_q1 = base_network(input_q1)
    processed_q2 = base_network(input_q2)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name='distance')([processed_q1, processed_q2])

    model = Model([input_q1, input_q2], [distance])
    adm = Adam(lr=0.001)
    model.compile(loss=contrastive_loss, optimizer=adm, metrics=['accuracy'])

    return model


def train():
    data = data_helper.load_pickle('./w2v/model_data.pkl')

    train_q1 = data['train_q1']
    train_q2 = data['train_q2']
    train_y = data['train_label']

    dev_q1 = data['dev_q1']
    dev_q2 = data['dev_q2']
    dev_y = data['dev_label']

    model = siamese_model()
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=10)
    tensorboard = TensorBoard(log_dir=tensorboard_path)
    callbackslist = [checkpoint, tensorboard]
    model.fit([train_q1, train_q2], train_y,
              batch_size=128,
              epochs=200,
              validation_data=([dev_q1, dev_q2], dev_y),
              callbacks=callbackslist)


def pred(q1, q2):
    tokenizer = data_helper.load_pickle('./w2v/tokenizer.pkl')
    q1_ = tokenizer.texts_to_sequences([q1.split()])
    q2_ = tokenizer.texts_to_sequences([q2.split()])
    q1_ = pad_sequences(q1_, input_dim)
    q2_ = pad_sequences(q2_, input_dim)

    model = siamese_model()
    model.load_weights(model_path)

    pred_ = model.predict([q1_, q2_])

    print('q1:{}, q2:{}, sim:{}'.format(q1, q2, pred_))


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()

    else:
        pred('为什么 配股 后 价格 下跌 了 这么 多', '配股 之后 没 之前 值钱 了')