'''
Created on 5 Mar 2017

@author: trucvietle
'''

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import urllib
import struct
import gzip
import os

def download_data(url, force_download=True):
    fname = url.split('/')[-1]
    if force_download or not os.path.exists(fname):
        urllib.urlretrieve(url, fname)
    return fname

def read_data(label_url, image_url):
    with gzip.open(download_data(label_url)) as flbl:
        magic, num = struct.unpack('>II', flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_data(image_url), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack('>IIII', fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)

## Download the images and their labels
# path = 'http://yann.lecun.com/exdb/mnist/'
path = '../../data/'
(train_lbl, train_img) = read_data(path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
(val_lbl, val_img) = read_data(path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz')

## Plot the first 10 images and print their labels
# for i in range(10):
#     plt.subplot(1, 10, i+1)
#     plt.imshow(train_img[i], cmap='Greys_r')
#     plt.axis('off')
# plt.show()
# print('label: %s' % (train_lbl[0:10], ))

## Create data iterators for mxnet
def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

batch_size = 100
train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)

## Multilayer perceptron


