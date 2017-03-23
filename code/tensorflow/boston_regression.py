'''
Created on 21 Mar 2017

@author: trucvietle
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import shutil

tf.logging.set_verbosity(tf.logging.INFO)

columns = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "medv"]
features = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"]
label = ['medv']

training_set = pd.read_csv('../../data/boston/boston_train.csv', skipinitialspace=True, skiprows=1, names=columns)
test_set = pd.read_csv('../../data/boston/boston_test.csv', skipinitialspace=True, skiprows=1, names=columns)
pred_set = pd.read_csv('../../data/boston/boston_predict.csv', skipinitialspace=True, skiprows=1, names=columns)

feature_cols = [tf.contrib.layers.real_valued_column(k) for k in features]
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, hidden_units=[10, 10],
                                          model_dir='../../data/boston_model')

def input_fn(dataset):
    '''
    Function that accepts pandas data frame and returns feature columns and
    label values as tensor.
    '''
    
    feature_cols = {k: tf.constant(dataset[k].values, shape=[dataset[k].size, 1]) for k in features}
    labels = tf.constant(dataset[label].values)
    return feature_cols, labels

## Train the regressor
regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)

## Evaluate the model using the test set
ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
loss_score = ev['loss']
print('Loss: {0:f}'.format(loss_score))

## Make predictions
## (1) On the test set
y_test = regressor.predict(input_fn=lambda: input_fn(test_set))
predictions = list(itertools.islice(y_test, test_set.shape[0]))

mse = mean_squared_error(y_true=test_set[label], y_pred=predictions)
rmse = np.sqrt(mse)
print('RMSE = {0:f}'.format(rmse))

mae = mean_absolute_error(y_true=test_set[label], y_pred=predictions)
print('MAE = {0:f}'.format(mae))

## (2) On the 'prediction' set
y = regressor.predict(input_fn=lambda: input_fn(pred_set))
predictions = list(itertools.islice(y, 6))
print('Predictions: {}'.format(str(predictions)))

## Remove the temp directory
shutil.rmtree('../../data/boston_model')
