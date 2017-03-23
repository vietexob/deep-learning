'''
Created on 23 Mar 2017

@author: trucvietle
'''

import tempfile
import pandas as pd
import tensorflow as tf

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
           "relationship", "race", "gender", "capital_gain", "capital_loss", "hours_per_week",
           "native_country", "income_bracket"]

## Read the training and test data
train_file = '../../data/census/adult.data'
test_file = '../../data/census/adult.test'
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

## Construct a label column named "label" whose value is 1 if the income is over 50K, and 0 otherwise
LABEL_COL = 'label'
df_train[LABEL_COL] = (df_train['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
df_test[LABEL_COL] = (df_test['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)

## Categorical variables
CAT_COL = ["workclass", "education", "marital_status", "occupation",
           "relationship", "race", "gender", "native_country"]
## Continuous variables
CON_COL = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

def input_fn(df):
    ## Create a dict mapping from each continuous feature col name (k) to the values
    ## of that col stored in a constant tensor. 
    con_cols = {k: tf.constant(df[k].values) for k in CON_COL}
    
    ## Create a dict mapping from each categorical feature col name (k) to the values
    ## of that col stored in a sparse tensor
    cat_cols = {k: tf.SparseTensor(
        indices = [[i, 0] for i in range(df[k].size)],
        values = df[k].values, dense_shape = [df[k].size, 1]) for k in CAT_COL}
    
    ## Merge two dictionaries into one
    feature_cols = dict(con_cols.items() + cat_cols.items())
    ## Convert label column into constant tensor
    label = tf.constant(df[LABEL_COL].values)
    ## Return the feature COLUMNS and the label
    return feature_cols, label

def train_input_fn():
    return input_fn(df_train)

def eval_input_fn():
    return input_fn(df_test)

## Feature engineering
gender = tf.contrib.layers.sparse_column_with_keys(column_name='gender', keys=['Female', 'Male'])
education = tf.contrib.layers.sparse_column_with_hash_bucket('education', hash_bucket_size=1000)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket('relationship', hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket('workclass', hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket('occupation', hash_bucket_size=1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket('native_country', hash_bucket_size=1000)
marital_status = tf.contrib.layers.sparse_column_with_hash_bucket('marital_status', hash_bucket_size=100)
race = tf.contrib.layers.sparse_column_with_hash_bucket('race', hash_bucket_size=100)

## Continuous variables
age = tf.contrib.layers.real_valued_column('age')
education_num = tf.contrib.layers.real_valued_column('education_num')
capital_gain = tf.contrib.layers.real_valued_column('capital_gain')
capital_loss = tf.contrib.layers.real_valued_column('capital_loss')
hours_per_week = tf.contrib.layers.real_valued_column('hours_per_week')

## Make continuous features categorical
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

## Interactions among features
education_x_occupation = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))
age_buckets_x_education_x_occupation = tf.contrib.layers.crossed_column([age_buckets, education, occupation],
                                                                        hash_bucket_size=int(1e6))

## Define the logistic regression model
model_dir = tempfile.mkdtemp()
feature_columns = [gender, native_country, education, occupation, workclass, marital_status, race,
                   age_buckets, education_x_occupation, age_buckets_x_education_x_occupation]
# model = tf.contrib.learn.LinearClassifier(feature_columns=feature_columns, model_dir = model_dir)
## Add regularization
model = tf.contrib.learn.LinearClassifier(
    feature_columns=feature_columns,
    optimizer = tf.train.FtrlOptimizer(
        learning_rate = 0.1, l1_regularization_strength=1., l2_regularization_strength=1.),
                                          model_dir=model_dir)

## Train and evaluate the model
model.fit(input_fn=train_input_fn, steps=200)
results = model.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print '%s: %s' % (key, results[key])
    