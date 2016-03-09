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








