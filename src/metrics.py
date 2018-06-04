'''
Script to compute Mean Reciprocal Rank and Precision at 3 given predictions and an index file

Predictions is a 2D numpy array, each row contains predictions of a batch.
NOTE: this is probably not rectangular as the last batch is not a full batch - each element of predictions is actually 2D [10000, 1]
Index is an array of num_queries tuples (startBatch, startBatchIdx, endBatch, endBatchIdx, [relative indices of true candidates])
'''

import os
import sys
import argparse

import numpy as np
import pickle

def main():
    parser = argparse.ArgumentParser(description='Compute Mean Reciprocal Rank and Precision at 3')
    parser.add_argument('-p', dest='predictions', required=True,
            help='the prediction file relative to ../data')
    parser.add_argument('-i', dest='index', required=True,
            help='the index file relative to ../data')
    parser.add_argument('-m', dest='mode', required=True,
            help='the mode indicating which is candidate and which is query (pc/cp)')
    (options, args) = parser.parse_known_args()

    if options.mode != 'pc' and options.mode != 'cp':
        print "Incorrect mode: {}".format(options.mode)
        sys.exit()

    PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
    INPUT_DIR = os.path.join(PROJECT_DIR, '../data')
    DATA_DIR = "/mnt/data2/photoshopbattle_images_samples"
    predictions = np.load(os.path.join(INPUT_DIR, options.predictions))
    with open(os.path.join(INPUT_DIR, options.index), 'rb') as file:
        index = pickle.load(file)

    # Computations
    MRR = []
    AMRR = []
    HR = []
    for i, tuple in enumerate(index):
        startBatch, startBatchIdx, endBatch, endBatchIdx, relIndices = tuple
        if len(relIndices) == 0:
            MRR.append(0)
            AMRR.append(0)
            HR.append(0)
        else:
            if startBatch == endBatch:
                # within a batch
                predictionsOfQuery = predictions[startBatch].flatten()[startBatchIdx:endBatchIdx]
            elif startBatch == endBatch - 1:
                # across 2 batches
                predictionsOfQuery = np.concatenate((predictions[startBatch].flatten()[startBatchIdx:], predictions[endBatch].flatten()[:endBatchIdx]))
            else:
                print "Impossible! Candidates more than 1 batch."
                sys.exit()
            # indices that sort in descending order
            sortedIndices = np.argsort(-predictionsOfQuery)
            # MRR
            for i, idx in enumerate(sortedIndices):
                if idx in relIndices:
                    rank = i + 1
                    print "Rank {}".format(rank)
                    MRR.append(1.0 / rank)
                    break
            # Precision at 3
            if options.mode == 'cp':
                # NMRR
                MRRs = []
                for i, idx in enumerate(sortedIndices):
                    if idx in relIndices:
                        rank = i + 1
                        MRRs.append(1.0 / rank)
                AMRR.append(np.mean(MRRs))
                hits = 0
                for idx in sortedIndices[:3]:
                    if idx in relIndices:
                        hits += 1
                precision = 1.0 * hits / 3
                print "Precision at 3: {}".format(precision)
                HR.append(precision)
    print "Mean Reciprocal Rank: {}".format(np.mean(MRR))
    print "MRR all children: {}".format(np.mean(AMRR))
    print "Mean Precision at 3: {}".format(np.mean(HR))


if __name__ == "__main__":
    main()
