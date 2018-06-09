'''
Script to perform Kernelized Locality Sensitive Hashing with the "bands" technique
Assumes input and query gist vectors have already been computed

Outputs generated query and candidate batches for Siamese network

Some code based on https://github.com/jakevdp/klsh
'''

import os
import sys
import argparse
import random

import numpy as np
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
        np.save(os.path.join(dir, 'W-reddit.npy'), self.W)
        np.save(os.path.join(dir, 'sample-reddit.npy'), self.sample)
        np.save(os.path.join(dir, 'KMean0-reddit.npy'), self.KMean0)
        np.save(os.path.join(dir, 'KMean-reddit.npy'), self.KMean)

def main():
    parser = argparse.ArgumentParser(description='Kernelized Locality Sensitive Hashing')
    parser.add_argument('-p', dest='p', nargs='?', default=300, type=int,
            help='amount of data to use when approximating the data distribution in the kernel subspace (p in paper).')
    parser.add_argument('-t', dest='t', nargs='?', default=30, type=int,
            help='number of random objects to use when choosing kernel-space hyperplanes (t in paper)')
    # all: 192 rows with 8 rows per band: 320 candidates, 50% recall
    # bin12: 225 rows with 15 rows per band: 5749 candidates
    parser.add_argument('-b', dest='b', nargs='?', default=180, type=int,
            help='number of hash bits (number of hash function to create, b in paper)')
    parser.add_argument('-r', dest='r', nargs='?', default=9, type=int,
            help='number of columns (rows as described in 246) per band, shoud divide b evenly')
    parser.add_argument('--param', dest='param', nargs='?', default='../param',
            help='Specify path for LSH parameters')
    parser.add_argument('--input', dest='input', nargs='?', default='../data',
            help='Specify path for LSH processed input data')
    parser.add_argument('--output', dest='output', required=True,
            help='Specify filename for generated candidate batches, relative to this file')
    (options, args) = parser.parse_known_args()

    PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
    PARAM_DIR = os.path.join(PROJECT_DIR, options.param)
    INPUT_DIR = os.path.join(PROJECT_DIR, options.input)
    CANDIDATE_DIR = "/mnt/data/reddit_images_2016_06_preprocessed"
    QUERY_DIR = "/mnt/data/reddit_images_2016_06_preprocessed_sampled"
    BATCH_SIZE = 10000

    if os.path.exists(PARAM_DIR):
        print "Found existing parameters, loading..."
        W = np.load(os.path.join(PARAM_DIR, 'W-reddit.npy'))
        sample = np.load(os.path.join(PARAM_DIR, 'sample-reddit.npy'))
        KMean0 = np.load(os.path.join(PARAM_DIR, 'KMean0-reddit.npy'))
        KMean = np.load(os.path.join(PARAM_DIR, 'KMean-reddit.npy'))
        klsh = KLSH(W=W, sample=sample, KMean0=KMean0, KMean=KMean)
        H = np.load(os.path.join(PARAM_DIR, 'H-reddit.npy'))

    else:
        os.mkdir(PARAM_DIR)
        if os.path.exists(os.path.join(INPUT_DIR, 'X-reddit.npy')):
            print "Found input gist vectors, loading..."
            X = np.load(os.path.join(INPUT_DIR, 'X-reddit.npy'))
            with open(os.path.join(INPUT_DIR, 'submissions-reddit'), 'rb') as file:
                submissionList = pickle.load(file)
        else:
            print "Existing input gist vectors not found, please compute them first"
            sys.exit()

        klsh = KLSH(X, p=options.p, t=options.t, b=options.b)
        klsh.save_params(PARAM_DIR)
        H = klsh.compute_hash_table(np.array(X))
        np.save(os.path.join(PARAM_DIR, 'H-reddit.npy'), H)

    if os.path.exists(os.path.join(INPUT_DIR, 'Q-reddit.npy')):
        print "Found query gist vectors, loading..."
        Q = np.load(os.path.join(INPUT_DIR, 'Q-reddit.npy')).tolist()
        with open(os.path.join(INPUT_DIR, 'queries-reddit'), 'rb') as file:
            queries = pickle.load(file)
    else:
        print "Existing query gist vectors not found, please compute them first"
        sys.exit()

    print "Hashing query gist vectors..."
    # [q, b]
    H_Q = klsh.compute_hash_table(np.array(Q))

    # nearest neighbor search
    numBands = options.b / options.r
    print "Hashing database LSH bits into buckets..."
    bucketsOfBands = util.generate_buckets(H, numBands)
    print "Hashing query LSH bits into buckets and generating candidates..."
    candidates = util.generate_candidates(bucketsOfBands, H_Q, numBands)

    # statistics
    numQueries = len(queries)
    positive_count = 0
    success_count = 0
    numCandidates = []
    for i, candidate in enumerate(candidates):
        numCandidate = len(candidate)
        queryName = queries[i]
        print "Found {} candidates for query {}: {}".format(numCandidate, i, queryName)
        if numCandidate > 0:
            positive_count += 1
            numCandidates.append(numCandidate)
    print "=" * 50
    print "{}% of images have candidates".format(100.0 * positive_count / numQueries)
    print "Average number of candidates {}".format(np.mean(numCandidates))

    # generate batches for Siamese evaluation
    # print "Generating batches..."
    # X1 = []
    # X2 = []
    # batchCounter = 0
    # # each element in index: (startBatch, startBatchIdx, endBatch, endBatchIdx, [relative indices of true candidates])
    # index = []
    # fileStem = os.path.join(PROJECT_DIR, options.output)
    # for i, candidateList in enumerate(candidates):
    #     queryName = queries[i]
    #     print "=" * 50
    #     print "Query {}: {}".format(i, queryName)
    #     if len(candidateList) > 0:
    #         startBatch = batchCounter
    #         startBatchIdx = len(X1)
    #         relIndices = []
    #         query = np.array(Image.open(os.path.join(QUERY_DIR, queryName)))
    #         for j, candidateIdx in enumerate(candidateList):
    #             # get candidate submission dir name
    #             candidate = submissionList[candidateIdx]
    #             print "Candidate: {}".format(candidate)
    #             X1.append(np.array(Image.open(os.path.join(CANDIDATE_DIR, candidate))))
    #             X2.append(query)
    #             if len(X1) == BATCH_SIZE:
    #                 print "Dumping batch {}...".format(batchCounter)
    #                 with open(fileStem + "_" + str(batchCounter).zfill(2), 'wb') as file:
    #                     pickle.dump({'X1': np.array(X1), 'X2': np.array(X2)}, file)
    #                 del X1[:]
    #                 del X2[:]
    #                 batchCounter += 1
    #         # sanity check: only 1 parent
    #         assert len(relIndices) <= 1
    #         print "Appending index: ({}, {}, {}, {}, {})".format(startBatch, startBatchIdx, batchCounter, len(X1), relIndices)
    #         index.append((startBatch, startBatchIdx, batchCounter, len(X1), relIndices))
    # print "Dumping final batch..."
    # with open(fileStem + "_" + str(batchCounter).zfill(2), 'wb') as file:
    #     pickle.dump({'X1': np.array(X1), 'X2': np.array(X2)}, file)
    # with open(os.path.join(INPUT_DIR, 'index-reddit'), 'wb') as file:
    #     pickle.dump(index, file)

if __name__ == "__main__":
    main()
