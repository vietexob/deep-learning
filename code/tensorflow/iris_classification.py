'''
Created on 21 Mar 2017

    A multi-layer perceptron model for classical classification problem.
    Three hidden layers, and output has 3 classes. Accuracy ~ 0.97.
    
@author: trucvietle
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import shutil

## Load the training and test sets (there are 4 features)
iris_training = '../../data/iris/iris_training.csv'
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=iris_training,
                                                                   target_dtype=np.int,
                                                                   features_dtype=np.float32)

iris_test = '../../data/iris/iris_test.csv'
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=iris_test,
                                                               target_dtype=np.int,
                                                               features_dtype=np.float32)

## Specify that all features have real-valued
feature_columns = [tf.contrib.layers.real_valued_column('', dimension=4)]

## Build a 3-layer ANN with 10, 20, 10 units, respectively
classifier = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
                                            hidden_units = [10, 20, 10],
                                            n_classes = 3, model_dir = '../../data/iris_model')

## Fit the model to the training data
classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

## Evaluate model accuracy
accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)['accuracy']
print('Accuracy: {0:f}'.format(accuracy_score))

## Remove the temp directory
shutil.rmtree('../../data/iris_model')
