import os
import argparse
import numpy as np
import leargist
from PIL import Image
from sklearn.metrics.pairwise import rbf_kernel

def kernelize(X, Y):
    '''
    Kernelize input matrices X of shape [n_samples_X, n_features] and Y of shape [n_samples_Y, n_features]

    :return K: [n_samples_X, n_samples_Y] where K[i, j] = kernel(X[i,:], Y[j,:]), centered
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

def hash(X, p=300, t=30, b=300):
    '''
    Perform kernelized locality sensitive hashing with input X (numpy array) of dimension
    [n_examples, gist descriptor dimension (default 960)]

    :param p: amount of data to use when approximating the data distribution in the kernel subspace (p in paper)
    :param t: number of random objects to use when choosing kernel-space hyperplanes (t in paper)
    :param b: number of hash bits (number of hash function to create, b in paper)
    :return h: the hash table
    :return w: the hash table weights used to compute new hashes
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
    return W

def main():
    parser = argparse.ArgumentParser(description='Kernelized Locality Sensitive Hashing')
    parser.add_argument('-p', dest='p', nargs='?', default=300, type=int,
            help='amount of data to use when approximating the data distribution in the kernel subspace (p in paper).')
    parser.add_argument('-t', dest='t', nargs='?', default=30, type=int,
            help='number of random objects to use when choosing kernel-space hyperplanes (t in paper)')
    parser.add_argument('-b', dest='b', nargs='?', default=300, type=int,
            help='number of hash bits (number of hash function to create, b in paper)')
    (options, args) = parser.parse_known_args()
    DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data")
    X = []
    for file in sorted(os.listdir(DATA_DIR)):
        im = Image.open(file)
        X.append(leargist.color_gist(im))
    hash(np.array(X), options['p'], options['t'], options['b'])


if __name__ == "__main__":
    main()
