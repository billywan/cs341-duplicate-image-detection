'''
Script to output retrieved images ranked by similarity scores for reddit dataset

Predictions is a 2D numpy array, each row contains predictions of a batch.

NOTE: this is probably not rectangular as the last batch is not a full batch - each element of predictions is actually 2D [10000, 1]
Index is an array of num_queries tuples (startBatch, startBatchIdx, endBatch, endBatchIdx, [relative indices of true candidates])
'''

import os
import sys
import argparse
import numpy as np
import pickle
import shutil
from PIL import Image

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = "/mnt/data/data_batches_lsh_reddit"
INPUT_DIR = os.path.join(PROJECT_DIR, '../data')
OUTPUT_DIR = "/mnt/data/reddit_output"

queryCounter = 0
candidateCounter = 0

def makeDir(dirName):
    if os.path.exists(dirName):
        print("{} already exists! Cleaning....".format(dirName))
        shutil.rmtree(dirName)
    print("Creating new folder at {}".format(dirName))
    os.makedirs(dirName)

def saveImgs(startBatch, startBatchIdx, endBatch, endBatchIdx, sortedIndices):
    global queryCounter, candidateCounter
    assert startBatch == endBatch

    filePath = os.path.join(DATA_DIR, "data_batch_" + str(startBatch).zfill(2))
    outDir = os.path.join(OUTPUT_DIR, "test_" + str(queryCounter).zfill(2))
    makeDir(outDir)

    print('\nLoading file {}...'.format(filePath))
    with open(filePath, 'rb') as handle:
        dict = pickle.load(handle)
        X1, X2 = dict['X1'], dict['X2']
        queryArr = np.array(X2[startBatchIdx])
        candidateArrs = []
        for i in sortedIndices:
            candidateArrs.append(np.array(X1[startBatchIdx + i]))
    # Save images
    queryOutPath = os.path.join(outDir, str(queryCounter).zfill(2)+".jpg")
    Image.fromarray(queryArr).save(queryOutPath)
    print("Writing output to: {}".format(queryOutPath))
    queryCounter += 1

    for arr in candidateArrs:
        candidateOutPath = os.path.join(outDir, str(candidateCounter).zfill(3)+".jpg")
        Image.fromarray(arr).save(candidateOutPath)
        print("Writing output to: {}".format(candidateOutPath))
        candidateCounter += 1


def main():
    parser = argparse.ArgumentParser(description='Compute Mean Reciprocal Rank and Precision at 3')
    parser.add_argument('-p', dest='predictions', required=True,
            help='the prediction file relative to ../data')
    parser.add_argument('-i', dest='index', required=True,
            help='the index file relative to ../data')
    (options, args) = parser.parse_known_args()

    makeDir(OUTPUT_DIR)
    predictions = np.load(os.path.join(INPUT_DIR, options.predictions))
    with open(os.path.join(INPUT_DIR, options.index), 'rb') as file:
        index = pickle.load(file)

    # Computations
    for i, tuple in enumerate(index):
        print "Processing index #{}: {}".format(i, tuple)
        startBatch, startBatchIdx, endBatch, endBatchIdx, relIndices = tuple

        if startBatch == endBatch:
            # within a batch
            predictionsOfQuery = predictions[startBatch].flatten()[startBatchIdx:endBatchIdx]
        elif startBatch == endBatch - 1:
            continue
        else:
            print "Impossible! Candidates more than 1 batch."
            continue
        # top 5 indices that sort in descending order
        sortedIndices = np.argsort(-predictionsOfQuery)[:10]
        print sortedIndices
        saveImgs(startBatch, startBatchIdx, endBatch, endBatchIdx, sortedIndices)



if __name__ == "__main__":
    main()
