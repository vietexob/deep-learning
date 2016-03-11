'''
Created on Mar 9, 2016

To train a LSTM character model over [Text8](http://mattmahoney.net/dc/textdata) data.

@author: trucvietle
'''

import os
import random
import string
import zipfile
import numpy as np
import tensorflow as tf

def read_data(filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return f.read(name)
    f.close()

filename = '../../data/text8.zip'
text = read_data(filename)
print 'Data size', len(text)

## Create a small validation set
valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print train_size, train_text[:64]
print valid_size, valid_text[:64]

## Utility functions to map characters to vocabulary IDs and back
vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])

def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print 'Unexpected character:', char
        return 0

def id2char(dictid):
    if dictid > 0:
        return chr(dictid + first_letter - 1)
    else:
        return ' '

print char2id('a'), char2id('z'), char2id(' ')
print id2char(1), id2char(26), id2char(0)

## Function to generate training batch for the LSTM model
batch_size = 64
num_unrollings = 10
class BatchGenerator (object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size / batch_size
        self._cursor = [offset * segment for offset in xrange(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        '''
        Generates a single batch from the current cursor position in the data.
        '''
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in xrange(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        '''
        Generates the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        '''
        batches = [self._last_batch]
        for step in xrange(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches

    def characters(self, probabilities):
        '''
        Turns a one-hot encoding or a probability distribution over the possible
        characters back into its (most likely) character representation.
        '''
        return [id2char(c) for c in np.argmax(probabilities, 1)]

    def batches2string(self, batches):
        '''
        Coverts a sequence of batches back into their (most likely) string representation.
        '''
        s = [''] * batches[0].shape[0]
        for b in batches:
            s = [''.join(x) for x in zip(s, self.characters(b))]
        return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

print train_batches.batches2string(train_batches.next())
print train_batches.batches2string(train_batches.next())
print valid_batches.batches2string(valid_batches.next())
print valid_batches.batches2string(valid_batches.next())

def logprob(predictions, labels):
    '''
    Log probability of the true labels in a predicted batch.
    '''
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
    '''
    Sample one element from the distribution assumed to be an array of normalized probabilities.
    '''
    r = random.uniform(0, 1)
    s = 0
    for i in xrange(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1

def sample(prediction):
    '''
    Turns a (column) prediction into one-hot encoded samples.
    '''
    p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p

def random_distribution():
    '''
    Generates a random column of probabilities.
    '''
    b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
    return b / np.sum(b, 1)[:, None]

## Simple LSTM model
num_nodes = 64

graph = tf.Graph()
with graph.as_default():
    ## Parameters
    ## Input gate: input, previous output, and bias
    ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, num_nodes]))
    ## Forget gate: input, previous output, and bias
    fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, num_nodes]))
    ## Memory cell: input, state and bias
    cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, num_nodes]))
    ## Output gate: input, previous output, and bias
    ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, num_nodes]))
    ## Variables saving state across unrollings
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    ## Classifier weights and biases
    w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([vocabulary_size]))
    









