'''
Script to perform baseline LSH to compare our model with.
Assumes input and query gist vectors have already been computed
Baseline LSH using https://github.com/kayzhu/LSHash
'''

import os
import sys
import argparse

import pickle
import numpy as np
from lshash import LSHash


def main():
    parser = argparse.ArgumentParser(description='Baseline Locality Sensitive Hashing')
    parser.add_argument('-b', dest='b', nargs='?', default=300, type=int,
            help='number of hash bits (number of hash function to create)')
    parser.add_argument('--input', dest='input', nargs='?', default='../data',
            help='Specify path for LSH processed input data')
    (options, args) = parser.parse_known_args()

    PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
    INPUT_DIR = os.path.join(PROJECT_DIR, options.input)

    if os.path.exists(os.path.join(INPUT_DIR, 'X.npy')):
        print "Found input gist vectors, loading..."
        X = np.load(os.path.join(INPUT_DIR, 'X.npy'))
        submissionList = pickle.load(open(os.path.join(INPUT_DIR, 'submissions'), 'rb'))
    else:
        print "Existing input gist vectors not found, please compute them first"
        sys.exit()

    input_dim = X.shape[1]
    X = X.tolist()
    lsh = LSHash(options.b, input_dim)

    print "Hashing input vectors..."
    cnt = 0
    for row in X:
        lsh.index(row)
        cnt += 1
        if cnt % 5000 == 0:
            print "Hashed {} vectors".format(cnt)

    if os.path.exists(os.path.join(INPUT_DIR, 'Q.npy')):
        print "Found query gist vectors, loading..."
        Q = np.load(os.path.join(INPUT_DIR, 'Q.npy')).tolist()
        queryList = pickle.load(open(os.path.join(INPUT_DIR, 'queries'), 'rb'))
    else:
        print "Existing query gist vectors not found, please compute them first"
        sys.exit()

    positive_count = 0
    for i, row in enumerate(Q):
        candidates = lsh.query(row, num_results=5, distance_func=hamming)
        try:
            candidate_indices = [X.index(list(candidate)) for (candidate, dist) in candidates]
        except ValueError:
            print "Unexpected error: candidate not found in X"
        try:
            true_idx = submissionList.index(queryList[i].rsplit('_', 1)[0])
            if true_idx in candidate_indices:
                positive_count += 1
        except ValueError:
            print "Unexpected error: query's original not found in submission list"
    print "Original found in candidates for {}%% of images".format(100.0 * positive_count / len(Q))


if __name__ == "__main__":
    main()
