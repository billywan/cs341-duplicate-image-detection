'''
Script to compute Mean Reciprocal Rank and Hit Rate at 10 given predictions and an index file

Predictions is a 2D numpy array, each row contains predictions of a batch
Index is an array of num_queries tuples (startBatch, startBatchIdx, endBatch, endBatchIdx, [relative indices of true candidates])
'''

import os
import sys
import argparse

import numpy as np
import pickle

def main():
    parser = argparse.ArgumentParser(description='Compute Mean Reciprocal Rank and Hit Rate at 10')
    parser.add_argument('-p', dest='predictions', required=True,
            help='the prediction file relative to ../data')
    parser.add_argument('-i', dest='index', required=True,
            help='the index file relative to ../data')
    parser.add_argument('-q', dest='queryNames', required=True,
            help='the query names file relative to ../data')
    parser.add_argument('-m', dest='mode', required=True,
            help='the mode indicating which is candidate and which is query (pc/cp)')
    (options, args) = parser.parse_known_args()

    if options.mode != 'pc' and options.mode != 'cp':
        print "Incorrect mode: {}".format(options.mode)
        sys.exit()

    PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
    INPUT_DIR = os.path.join(PROJECT_DIR, '../data')
    DATA_DIR = "/mnt/data2/photoshopbattle_images_samples"
    predictions = np.load(os.path.join(INPUT_DIR, options.p))
    with open(os.path.join(INPUT_DIR, options.i), 'rb') as file:
        index = pickle.load(file)
    with open(os.path.join(INPUT_DIR, options.q), 'rb') as file:
        queryNames = pickle.load(file)

    # Computations
    MRR = []
    HR = []
    for i, tuple in enumerate(index):
        startBatch, startBatchIdx, endBatch, endBatchIdx, relIndices = tuple
        if len(relIndices) == 0:
            MRR.append(0)
            HR.append(0)
        else:
            if startBatch == endBatch:
                # within a batch
                predictionsOfQuery = predictions[startBatch, startBatchIdx:endBatchIdx]
            else if startBatch == endBatch - 1:
                # across 2 batches
                predictionsOfQuery = np.concatenate((predictions[startBatch, startBatchIdx:], predictions[endBatch, :endBatchIdx]))
            # indices that sort in descending order
            sortedIndices = np.argsort(-predictionsOfQuery)
            # MRR
            for i, idx in enumerate(sortedIndices):
                if idx in relIndices:
                    print "Rank {}".format(i)
                    MRR.append(1.0 / i)
                    break
            # Hit Rate
            if options.mode == 'pc':
                # only 1 possible parent
                assert len(relIndices) == 1
                if relIndices[0] in sortedIndices[:10]:
                    print "Hit Rate at 10: 1"
                    HR.append(1)
                else:
                    print "Hit Rate at 10: 0"
                    HR.append(0)
            else:
                hits = 0
                for idx in sortedIndices[:10]:
                    if idx in relIndices:
                        hits += 1
                queryName = queryNames[i]
                stem = queryName.rsplit('.', 1)[0]
                numChildren = len(os.listdir(os.path.join(DATA_DIR, stem))) - 1
                hitRate = 1.0 * hits / numChildren
                print "Hit Rate at 10: {}".format(hitRate)
                HR.append(hitRate)
    print "Mean Reciprocal Rank: {}".format(np.mean(MRR))
    print "Mean Hit Rate at 10: {}".format(np.mean(HR))    


if __name__ == "__main__":
    main()
