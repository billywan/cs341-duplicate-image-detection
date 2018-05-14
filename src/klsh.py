'''
Script to perform Kernelized Locality Sensitive Hashing
Some code based on https://github.com/jakevdp/klsh and https://github.com/emchristiansen/CharikarLSH
'''

import os
import argparse
import random
import numpy as np
import leargist
import pickle
from PIL import Image
from sklearn.metrics.pairwise import rbf_kernel

import util

def raise_to_neg_half(X):
    '''
    Raise a square matrix X to the power of -1/2
    '''
    w, V = np.linalg.eigh(X)
    w[w > 1e-8] = w[w > 1e-8] ** -0.5
    return np.dot(V, np.dot(np.diag(w), V.T))

class KLSH():
    '''
    Perform kernelized locality sensitive hashing with input X (numpy array) of dimension
    [n, gist descriptor dimension (default 960)]

    :param p: amount of data to use when approximating the data distribution in the kernel subspace (p in paper)
    :param t: number of random objects to use when choosing kernel-space hyperplanes (t in paper)
    :param b: number of hash bits (number of hash function to create, b in paper)
    :return H: the hash table for all inputs of shape [n, b]
    :return W: the hash table weights used to compute new hashes of shape [p, b]
    '''
    def __init__(self, X=None, p=300, t=30, b=300, W=None, sample=None, KMean0=None, KMean=None):
        self.X = X
        self.p = p
        self.t = t
        self.b = b
        self.W = W
        self.sample = sample
        self.KMean0 = KMean0
        self.KMean = KMean
        if W is None:
            self._build_weight_matrix()

    def _build_weight_matrix(self):
        indices = np.random.choice(self.X.shape[0], self.p, replace=False)
        # [p, 960]
        self.sample = self.X[indices]
        print "Kernelizing a sample of {} gist vectors...".format(self.p)
        self._kernelize_sample()
        print "Computing hashing weight matrix..."
        self._compute_weight_matrix()
        print "Hashing input gist vectors..."

    def _kernelize_sample(self):
        '''
        Kernelize input matrices X of shape [n_examples_X, n_features] and Y of shape [n_examples_Y, n_features]

        :return K: [n_examples_X, n_examples_Y] where K[i, j] = kernel(X[i,:], Y[j,:]), centered
        '''
        K = rbf_kernel(self.sample, self.sample)
        self.KMean0 = np.mean(K, axis=0)
        self.KMean = np.mean(K)
        # [p, p]
        self.K = K - self.KMean0 - np.mean(K, axis=1)[:, None] + self.KMean

    def _compute_weight_matrix(self):
        '''
        Compute the weight matrix for hashing new examples
        '''
        K_neg_half = raise_to_neg_half(self.K)
        e_s = np.zeros((self.p, self.b))
        # [t, b]
        indices = np.array([np.random.choice(self.p, self.t) for i in range(self.b)]).T
        e_s[indices, np.arange(self.b)] = 1
        # [p, b]
        self.W = np.dot(K_neg_half, e_s)

    def _kernelize(self, X):
        '''
        Kernelize input matrices X of shape [n_examples_X, n_features] and Y of shape [n_examples_Y, n_features]

        :return K: [n_examples_X, n_examples_Y] where K[i, j] = kernel(X[i,:], Y[j,:]), centered
        '''
        K = rbf_kernel(X, self.sample)
        K = K - self.KMean0 - np.mean(K, axis=1)[:, None] + self.KMean
        return K

    def compute_hash_table(self, input):
        '''
        Compute hash table for input of shape [n_examples, n_features] with weight matrix W and sample used to compute
        the kernel matrix

        :return hash table, each row is the hash for one example
        '''
        K_input = self._kernelize(input)
        # [n_examples, b]
        return (np.dot(K_input, self.W) > 0).astype(np.uint8)

    def save_params(self, dir):
        np.save(os.path.join(dir, 'W.npy'), self.W)
        np.save(os.path.join(dir, 'sample.npy'), self.sample)
        np.save(os.path.join(dir, 'KMean0.npy'), self.KMean0)
        np.save(os.path.join(dir, 'KMean.npy'), self.KMean)

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
    parser.add_argument('--param', dest='param', nargs='?', default='../param',
            help='Specify path for LSH parameters')
    parser.add_argument('--input', dest='input', nargs='?', default='../data',
            help='Specify path for LSH processed input data')
    (options, args) = parser.parse_known_args()

    PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = "/mnt/data/photoshopbattle_images"
    PARAM_DIR = os.path.join(PROJECT_DIR, options.param)
    INPUT_DIR = os.path.join(PROJECT_DIR, options.input)

    if os.path.exists(PARAM_DIR):
        print "Found existing parameters, loading..."
        W = np.load(os.path.join(PARAM_DIR, 'W.npy'))
        sample = np.load(os.path.join(PARAM_DIR, 'sample.npy'))
        KMean0 = np.load(os.path.join(PARAM_DIR, 'KMean0.npy'))
        KMean = np.load(os.path.join(PARAM_DIR, 'KMean.npy'))
        klsh = KLSH(W=W, sample=sample, KMean0=KMean0, KMean=KMean)
        H = np.load(os.path.join(PARAM_DIR, 'H.npy'))
        submissionList = pickle.load(open(os.path.join(INPUT_DIR, 'submissions'), 'rb'))
    else:
        os.mkdir(PARAM_DIR)
        if os.path.exists(os.path.join(INPUT_DIR, 'X.npy')):
            print "Found input gist vectors, loading..."
            X = np.load(os.path.join(INPUT_DIR, 'X.npy'))
            submissionList = pickle.load(open(os.path.join(INPUT_DIR, 'submissions'), 'rb'))
        else:
            print "Existing input gist vectors not found, computing..."
            X = []
            submissionList = []
            for dirName, _, fileList in os.walk(DATA_DIR):
                if len(fileList) > 1:
                    submission = sorted(fileList)[0]
                    fullPath = os.path.join(dirName, submission)
                    print "computing gist vector for {}...".format(fullPath)
                    try:
                        im = Image.open(fullPath)
                        X.append(leargist.color_gist(im, orientations=(4,4,2)))
                        submissionList.append(submission.rsplit('.', 1)[0])
                        if len(submissionList) % 500 == 0:
                            print "computed gist vectors for {} images".format(len(submissionList))
                            print "=" * 50
                    except:
                        print "Unable to open image {}".format(fullPath)
            X = np.array(X)
            np.save(os.path.join(INPUT_DIR, 'X.npy'), X)
            pickle.dump(submissionList, open(os.path.join(INPUT_DIR, 'submissions'), 'wb'))
        klsh = KLSH(X, p=options.p, t=options.t, b=options.b)
        klsh.save_params(PARAM_DIR)
        H = klsh.compute_hash_table(np.array(X))
        np.save(os.path.join(PARAM_DIR, 'H.npy'), H)

    # print H[22]
    Q = []
    queryList = []
    print "Randomly pick 100 query images..."
    dirAndFiles = random.sample([(dirName, fileList) for dirName, _, fileList in os.walk(DATA_DIR) if len(fileList) > 1], 100)
    for dir, files in dirAndFiles:
        query = random.choice(sorted(files)[1:])
        queryList.append(query)
        im = Image.open(os.path.join(dir, query))
        Q.append(leargist.color_gist(im))

    print "Hashing query gist vectors..."
    # [q, b]
    H_Q = klsh.compute_hash_table(np.array(Q))
    # print H_Q[0]
    # nearest neighbor search
    print "Permuting hash table..."
    permutations = util.generate_permutations(H, options.np)
    print "Searching for query and generating candidates..."
    candidates = util.lookup(permutations, H_Q)
    positive_count = 0
    for i, candidate in enumerate(candidates):
        print "Found {} candidates for query {}: ".format(len(candidate), i)
        try:
            idx = submissionList.index(queryList[i].rsplit('_', 1))
            if idx in candidate:
                positive_count += 1
        except ValueError:
            print "Unexpected error: query's original not found in submission list"
    print "Original found in candidates for {}%% of images".format(1.0 * positive_count)

if __name__ == "__main__":
    main()
