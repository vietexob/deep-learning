'''
Created on 23 Mar 2017

    A wide-deep NN model for binary classification (prediction of income level).
    Wide part is suitable for sparse categorical/interaction features.
    Deep part is suitable for continuous features. The 'deep' NN has to layers:
    first 100 neurons, second 50 neurons. There is one output unit (regression).
    Accuracy ~ 0.85. Precision ~ 0.74. Recall ~ 0.57. AUC ~ 0.88.
    
@author: trucvietle
'''

import tensorflow as tf
import pandas as pd
import tempfile

## Categorical base columns
gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["Female", "Male"])
race = tf.contrib.layers.sparse_column_with_keys(column_name="race",keys=["Amer-Indian-Eskimo",
                                                                          "Asian-Pac-Islander",
                                                                          "Black", "Other", "White"])
education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)

## Continuous base columns
age = tf.contrib.layers.real_valued_column("age")
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

## The wide model (with interaction terms)
wide_columns = [gender, native_country, education, occupation, workclass, relationship, age_buckets,
                tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),
                tf.contrib.layers.crossed_column([native_country, occupation], hash_bucket_size=int(1e4)),
                tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size=int(1e6))]

## The deep model: feed-forward NN
## Each of the sparse, high-dimensional categorical features are first converted into a low-dimensional and
## dense real-valued vector, referred to as an embedding vector
deep_columns = [tf.contrib.layers.embedding_column(workclass, dimension=8),
                tf.contrib.layers.embedding_column(education, dimension=8),
                tf.contrib.layers.embedding_column(gender, dimension=8),
                tf.contrib.layers.embedding_column(relationship, dimension=8),
                tf.contrib.layers.embedding_column(native_country, dimension=8),
                tf.contrib.layers.embedding_column(occupation, dimension=8),
                age, education_num, capital_gain, capital_loss, hours_per_week]

## Combine wide and deep models
model_dir = tempfile.mkdtemp()
model = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=model_dir, linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns, dnn_hidden_units=[100, 50])

## Read the training and test data
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
           "relationship", "race", "gender", "capital_gain", "capital_loss", "hours_per_week",
           "native_country", "income_bracket"]

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

## Train and evaluate the model
model.fit(input_fn=train_input_fn, steps=500)
results = model.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print '%s: %s' % (key, results[key])
