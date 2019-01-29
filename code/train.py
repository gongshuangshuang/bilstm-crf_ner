import codecs
import pickle
import sys
import numpy as np
import tensorflow as tf
from Batch import BatchGenerator
from bilstm_crf import Model
from utils import train
with open('../data/renmindata.pkl', 'rb') as inp:
	word2id = pickle.load(inp)
	id2word = pickle.load(inp)
	tag2id = pickle.load(inp)
	id2tag = pickle.load(inp)
	x_train = pickle.load(inp)
	y_train = pickle.load(inp)
	x_test = pickle.load(inp)
	y_test = pickle.load(inp)
	x_valid = pickle.load(inp)
	y_valid = pickle.load(inp)
print("train len:", len(x_train))
print("test len:", len(x_test))
print("word2id len", len(word2id))
print('Creating the data generator ...')
data_train = BatchGenerator(x_train, y_train, shuffle=True)
data_valid = BatchGenerator(x_valid, y_valid, shuffle=False)
data_test = BatchGenerator(x_test, y_test, shuffle=False)
print('Finished creating the data generator.')

epochs = 50
batch_size = 32

config = {}
config["lr"] = 0.001
config["embedding_dim"] = 100
config["sen_len"] = len(x_train[0])
config["batch_size"] = batch_size
config["embedding_size"] = len(word2id)+1
config["tag_size"] = len(tag2id)
config["pretrained"] = False
embedding_pre = []
print("begin to train...")
model = Model(config, embedding_pre, dropout_keep=0.5)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    train(model, sess, saver, epochs, batch_size, data_train, data_test, id2word, id2tag)
