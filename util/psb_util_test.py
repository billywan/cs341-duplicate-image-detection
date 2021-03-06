'''
This script is for loading preprocessd images

Usage:
    This script is considered to reside under PROJECT_DIR/util (PROJECT_DIR is the outer most level)

Input:
    Input data is default to residing in "/mnt/data/data_batches". 
    
    Each file contains data that is packed in dictionary as:
    {'X1': X1, 'X2': X2, 'y': y}

    X1 and X2 are 4d numpy arrays with dimension (BATCH_SIZE, length, width, 3), BATCH_SIZE=5000

    y is a 1d numpy array with dimension (BATCH_SIZE,), BATCH_SIZE=5000


Ouput:
    List of data dictionary of each batch
'''
#from __future__ import print_function
import pickle
import os
import sys
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input #resnet uses same preprocess_input method as vgg16 & vgg19.
import numpy as np
import random

DATA_DIR = "/Users/EricX/Desktop/CS341/data_batches"




def load_data_file(file_path, expect_label=False):
    print('\nLoading file {}...'.format(file_path))
    with open(file_path, 'rb') as handle:
        data_dict = pickle.load(handle)
        X0, X1, y = data_dict['X1'], data_dict['X2'], None
        if expect_label:
            y = data_dict['y']
            y.astype(np.float32)
        X0, X1 = X0.astype(np.float32), X1.astype(np.float32)
        #X0, X1, y = unison_shuffled_data(X0, X1, y)
        #print(X0.shape, X1.shape, y.shape)
        #print('Preprocessing input...')
        X0, X1 = preprocess_input(X0), preprocess_input(X1)
    return [X0, X1], y

def unison_shuffled_data(a, b, c):
    assert a.shape[0] == b.shape[0] 
    assert b.shape[0] == c.shape[0]
    p = np.random.permutation(a.shape[0])
    return a[p], b[p], c[p]

def batch_generator(data_dir="/Users/EricX/Desktop/CS341/data_batches", batch_size=50, shuffle_files=True):
    data_files = [file for file in sorted(os.listdir(data_dir)) if 'data_batch_' in file]
    # if shuffle_files:
    #     random.shuffle(data_files)
    while True:
        if shuffle_files:
            random.shuffle(data_files)
        for file in data_files:
            print('\nLoading file {}'.format(file))
            with open(os.path.join(data_dir, file), 'rb') as handle:
                data_dict = pickle.load(handle)
                X0, X1, y = data_dict['X1'], data_dict['X2'], data_dict['y']
                X0, X1, y = X0.astype(np.float32), X1.astype(np.float32), y.astype(np.float32)
                #X0, X1, y = unison_shuffled_data(X0, X1, y)
                #print(X0.shape, X1.shape, y.shape)
                #print('Preprocessing input...')
                X0, X1 = preprocess_input(X0), preprocess_input(X1)
                #print('Done preprocessing.')
                for i in range(0, X0.shape[0], batch_size):
                    X0_batch = X0[i:i+batch_size]
                    X1_batch = X1[i:i+batch_size]
                    y_batch = y[i:i+batch_size]
                    #print "yielding next batch..."
                    yield [X0_batch, X1_batch], y_batch

def make_binary(X0, X1, y):
    X0 = X0[y!=0.5]
    X1 = X1[y!=0.5]
    y = y[y!=0.5]
    return X0, X1, y

def batch_generator_binary(data_dir="/Users/EricX/Desktop/CS341/data_batches", batch_size=50):
    while True:
        for fn in os.listdir(data_dir):
            if fn.startswith('data_batch_'):
                print('loading file {} ...'.format(fn))
                with open(os.path.join(data_dir, fn), 'rb') as handle:
                    data_dict = pickle.load(handle)
                    X0, X1, y = data_dict['X1'], data_dict['X2'], data_dict['y']
                    X0, X1, y = X0.astype(np.float32), X1.astype(np.float32), y.astype(np.float32)
                    X0, X1, y = make_binary(X0, X1, y)
                    #X0, X1, y = unison_shuffled_data(X0, X1, y)
                    #print(X0.shape, X1.shape, y.shape)
                    #print('Preprocessing input...')
                    assert len(y[y==0.5]) == 0
                    X0, X1 = preprocess_input(X0), preprocess_input(X1)
                    #print('Done preprocessing')
                    for i in range(0, X0.shape[0], batch_size):
                        X0_batch = X0[i:i+batch_size]
                        X1_batch = X1[i:i+batch_size]
                        y_batch = y[i:i+batch_size]
                        #print "yielding next batch..."
                        yield [X0_batch, X1_batch], y_batch

#load_data()
def test():
    for [a, b], c in batch_generator_binary():
        print(a.shape, b.shape, c.shape)


#test()
