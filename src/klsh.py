'''
Script to perform Kernelized Locality Sensitive Hashing
Some code based on https://github.com/jakevdp/klsh and https://github.com/emchristiansen/CharikarLSH
'''

import os
import argparse
import numpy as np
import leargist
from PIL import Image
from sklearn.metrics.pairwise import rbf_kernel

import util

def kernelize(X, Y):
    '''
    Kernelize input matrices X of shape [n_examples_X, n_features] and Y of shape [n_examples_Y, n_features]

    :return K: [n_examples_X, n_examples_Y] where K[i, j] = kernel(X[i,:], Y[j,:]), centered
    '''
    K = rbf_kernel(X, Y)
    K = K - np.mean(K, axis=0) - np.mean(K, axis=1) + np.mean(K)
    return K

def raise_to_neg_half(X):
    '''
    Raise a square matrix X to the power of -1/2
    '''
    w, V = np.linalg.eigh(X)
    w[w > 1e-8] = w[w > 1e-8] ** -0.5
    return np.dot(V, np.dot(np.diag(w), V.T))

def compute_weight_matrix(K_neg_half, p, t, b):
    '''
    Compute the weight matrix for hashing new examples
    '''
    e_s = np.zeros((p, b))
    # [t, b]
    indices = np.array([np.random.choice(p, t) for i in range(b)]).T
    e_s[indices, np.arange(self.nbits)] = 1
    # [p, b]
    return np.dot(K_neg_half, e_s)

def compute_hash_table(input, W, sample):
    '''
    Compute hash table for input of shape [n_examples, n_features] with weight matrix W and sample used to compute
    the kernel matrix

    :return hash table, each row is the hash for one example
    '''
    # question: subtract same means when kernelizing sample to center?
    K_input = kernelize(input, sample)
    # [n_examples, b]
    return (np.dot(K_input, W) > 0).astype(np.uint8)

def klsh(X, p=300, t=30, b=300):
    '''
    Perform kernelized locality sensitive hashing with input X (numpy array) of dimension
    [n, gist descriptor dimension (default 960)]

    :param p: amount of data to use when approximating the data distribution in the kernel subspace (p in paper)
    :param t: number of random objects to use when choosing kernel-space hyperplanes (t in paper)
    :param b: number of hash bits (number of hash function to create, b in paper)
    :return H: the hash table for all inputs of shape [n, b]
    :return W: the hash table weights used to compute new hashes of shape [p, b]
    '''
    indices = np.random.choice(X.shape[0], p, replace=False)
    # [p, 960]
    sample = X[indices]
    # [p, p]
    K = kernelize(sample, sample)
    # [p, p]
    K_neg_half = raise_to_neg_half(K)
    # [p, b]
    W = compute_weight_matrix(K_neg_half, p, t, b)
    # [n, b]
    H = compute_hash_table(X, W, sample)
    return H, W, sample

def main():
    parser = argparse.ArgumentParser(description='Kernelized Locality Sensitive Hashing')
    parser.add_argument('-p', dest='p', nargs='?', default=300, type=int,
            help='amount of data to use when approximating the data distribution in the kernel subspace (p in paper).')
    parser.add_argument('-t', dest='t', nargs='?', default=30, type=int,
            help='number of random objects to use when choosing kernel-space hyperplanes (t in paper)')
    parser.add_argument('-b', dest='b', nargs='?', default=300, type=int,
            help='number of hash bits (number of hash function to create, b in paper)')
    parser.add_argument('-np', dest='np', nargs='?', default=250, type=int,
            help='number of permutations in nearest neighbor search. For epsilon-approx, use 2n^(1/(1+epsilon)) permutations')
    parser.add_argument('-q', dest='query', nargs='?', default='../data',
            help='Specify path for query image(s)')
    (options, args) = parser.parse_known_args()
    PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(PROJECT_DIR, "../data")
    QUERY_DIR = os.path.join(PROJECT_DIR, options.query)
    X = []
    for file in sorted(os.listdir(DATA_DIR)):
        im = Image.open(file)
        X.append(leargist.color_gist(im))
    H, W, sample = klsh(np.array(X), options.p, options.t, options.b)
    Q = []
    for file in os.listdir(QUERY_DIR):
        im = Image.open(file)
        Q.append(leargist.color_gist(im))
    # [q, b]
    H_Q = compute_hash_table(np.array(Q), W, sample)
    # nearest neighbor search
    permutations = util.generate_permutations(H, options.np)
    candidates = util.lookup(permutations, H_Q)


if __name__ == "__main__":
    main()
