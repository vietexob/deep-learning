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

def accuracy(pred, labels):
    return(100.0 * np.sum(np.argmax(pred, 1) == np.argmax(labels, 1)) / pred.shape[0])

## Introduce and tune L2 regularization for both logistic and NN models.
batch_size = 128
hidden_layer_size = 1024
has_regularization = True
has_dropout = True
## NOTE:
## Has none: Test accuracy: 89.1%
## Has regularization and no dropout:
## Has dropout and no regularization:
## Has both: 
graph = tf.Graph()

with graph.as_default():
    ## Input data. For the training data, use a placeholder that will be fed at runtime with
    ## a training mini-batch
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
#     ## Variables
#     weights = tf.Variable(tf.truncated_normal([image_size*image_size, num_labels]))
#     biases = tf.Variable(tf.zeros([num_labels]))
#      
#     ## Training computation
#     logits = tf.matmul(tf_train_dataset, weights) + biases
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
#     ## Add regularizer
#     loss += tf.nn.l2_loss(weights)
    
    ## Now: Change to one-layer NN
    ## Variables
    weights_h = tf.Variable(tf.truncated_normal([image_size*image_size, hidden_layer_size]))
    biases_h = tf.Variable(tf.zeros([hidden_layer_size]))
    hidden = tf.nn.relu(tf.matmul(tf_train_dataset, weights_h) + biases_h)
    
    ## Output layer
    weights_o = tf.Variable(tf.truncated_normal([hidden_layer_size, num_labels]))
    biases_o = tf.Variable(tf.zeros([num_labels]))
    if has_dropout:
        ## Add dropout
        hidden_dropout = tf.nn.dropout(hidden, keep_prob=0.50)
        logits = tf.matmul(hidden_dropout, weights_o) + biases_o
    else:
        logits = tf.matmul(hidden, weights_o) + biases_o
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    if has_regularization:
        ## Add regularization
        loss += tf.nn.l2_loss(weights_h) + tf.nn.l2_loss(weights_o)
    ## END CHANGE
    
    ## Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.50).minimize(loss)
    
    ## Prediction for training, validation and test data
    train_pred = tf.nn.softmax(logits)
#     valid_pred = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
#     test_pred = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
    
    valid_hidden = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_h) + biases_h)
    valid_logits = tf.matmul(valid_hidden, weights_o) + biases_o
    valid_pred = tf.nn.softmax(valid_logits)
     
    test_hidden = tf.nn.relu(tf.matmul(tf_test_dataset, weights_h) + biases_h)
    test_logits = tf.matmul(test_hidden, weights_o) + biases_o
    test_pred = tf.nn.softmax(test_logits)

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
