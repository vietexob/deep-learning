'''
Created on Feb 5, 2016

@author: trucvietle
'''

import cPickle as pickle
import numpy as np
import tensorflow as tf

## Load the notMNIST data generated from Lesson 1
pickle_file = '../../data/notMNIST.pickle'
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save # hit to help gc save up memory
    print 'Training set', train_dataset.shape, train_labels.shape
    print 'Validation set', valid_dataset.shape, valid_labels.shape
    print 'Test set', test_dataset.shape, test_labels.shape

## Reformat into shape that's more adapted to the models we're going to train
## - Data as flat matrix
## - Labels as float 1-hot encodings
image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size*image_size)).astype(np.float32)
    ## Map 0 to [1.0, 0.0, 0.0, ...], 1 to [0.0, 1.0, 0.0, ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print 'Training set', train_dataset.shape, train_labels.shape
print 'Validation set', valid_dataset.shape, valid_labels.shape
print 'Test set', test_dataset.shape, test_labels.shape

## Train multinomial logit using simple gradient descent
## With gradient descent training, even this much data is prohibitive.
## Subset the training data for faster turnaround
train_subset = 10000

graph = tf.Graph()
with graph.as_default():
    ## Input data
    ## Load the training, validation and test data into constants that are attached to the graph
    tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
    tf_train_labels = tf.constant(train_labels[:train_subset])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    ## Variables
    ## These are the params that we are going to train. The weight matrix will be initialized
    ## using random values followed by a (truncated) normal distribution. The biases are initialized to zero.
    weights = tf.Variable(tf.truncated_normal([image_size*image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))
    
    ## Training computation
    ## Multiply the inputs with the weight matrix, and add biases
    ## Compute the softmax and cross-entropy (it's one operation in TensorFlow because it's very common
    ## and can be optimized. Take average of this cross-entropy across all training examples: that's the loss.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) 
    
    ## Optimizer
    ## Find the minimum of this loss using gradient descent
    optimizer = tf.train.GradientDescentOptimizer(0.50).minimize(loss)
    
    ## Prediction for the training, validation and test data
    ## These are not part of training, but merely here so we can report accuracy figures as we train
    train_pred = tf.nn.softmax(logits)
    valid_pred = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_pred = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
    
## Run the computation and iterate
num_steps = 801

def accuracy(pred, labels):
    return(100.0 * np.sum(np.argmax(pred, 1) == np.argmax(labels, 1))
           / pred.shape[0])

# with tf.Session(graph=graph) as session:
#     ## This is a one-time operation that ensures the params get initialized as described in the graph:
#     ## Random weights for the matrix and zeros for the biases
#     tf.initialize_all_variables().run()
#     print 'Initialized'
#     for step in xrange(num_steps):
#         ## Run the computations. We tell .run() that we want to run the optimizer and get the loss value
#         ## and the training predictions returned as numpy array.
#         _, l, pred = session.run([optimizer, loss, train_pred])
#         if (step % 100 == 0):
#             print 'Loss at step', step, ':', l
#             print 'Training accuracy: %.1f%%' % accuracy(pred, train_labels[:train_subset, :])
#             ## Calling .eval() on valid_pred is basically like calling run(), but just to get that one
#             ## numpy array. Note that it recomputes all its graph dependencies.
#             print 'Validation accuracy: %.1f%%' % accuracy(valid_pred.eval(), valid_labels)
#     print 'Test accuracy: %.1f%%' % accuracy(test_pred.eval(), test_labels) 

## NOW: Switch to SGD instead, which is much faster 
batch_size = 128

graph = tf.Graph()
with graph.as_default():
    ## Input data. For the training data, use a placeholder that will be fed at runtime with
    ## a training mini-batch
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    ## Variables
    weights = tf.Variable(tf.truncated_normal([image_size*image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))
    
    ## Training computation
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
    ## Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.50).minimize(loss)
    
    ## Prediction for training, validation and test data
    train_pred = tf.nn.softmax(logits)
    valid_pred = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_pred = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

## Let's run it
num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print 'Initialized'
    for step in xrange(num_steps):
        ## Pick an offset within the training data, which has been randomized
        ## Note: We could use better randomization across the epochs
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        ## Generate a mini-batch
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        ## Prepare a dict telling the session where to feed the mini-batch
        ## The key of the dict is the placeholder node of the graph to be fed
        ## and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, pred = session.run([optimizer, loss, train_pred], feed_dict=feed_dict)
        if (step % 500 == 0):
            print 'Mini-batch loss at step', step, ':', l
            print 'Mini-batch accuracy: %.1f%%' % accuracy(pred, batch_labels)
            print 'Validation accuracy: %.1f%%' % accuracy(valid_pred.eval(), valid_labels)
    print 'Test accuracy: %.1f%%' % accuracy(test_pred.eval(), test_labels)







