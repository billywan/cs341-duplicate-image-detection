'''
Script to compute gist vectors for input images and save to disk
Here, input/candidate is children, queries are parents.
'''

import os
import sys
import random
import numpy as np
import leargist
import pickle
from PIL import Image


def main():
    PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = "/mnt/data2/photoshopbattle_images_samples"
    INPUT_DIR = os.path.join(PROJECT_DIR, '../data')

    if os.path.exists(os.path.join(INPUT_DIR, 'X.npy')) or os.path.exists(os.path.join(INPUT_DIR, 'Q.npy')):
        print "Input/query gist vectors already exist, skipping..."
    else:
        X = []
        Q = []
        candidateList = []
        queryList = []
        for dirName, _, fileList in os.walk(DATA_DIR):
            if len(fileList) > 1:
                sortedFileList = sorted(fileList)
                parent = sortedFileList[0]
                # if no parent, skip
                if parent.count("_") > 1:
                    print "Skipping {} with files {}".format(dirName, fileList)
                else:
                    # children as candidates
                    for child in sortedFileList[1:]:
                        childFullPath = os.path.join(dirName, child)
                        print "Computing gist vector for child {}...".format(childFullPath)
                        try:
                            im = Image.open(childFullPath)
                            X.append(leargist.color_gist(im, orientations=(4,4,2)))
                            candidateList.append(child.rsplit('_', 1)[0])
                        except:
                            print "Unable to open image {}!!!".format(fullPath)
                    # parent as query
                    parentFullPath = os.path.join(dirName, parent)
                    print "Computing gist vector for parent {}...".format(parentFullPath)
                    try:
                        im = Image.open(parentFullPath)
                        Q.append(leargist.color_gist(im, orientations=(4,4,2)))
                        queryList.append(parent.rsplit('.', 1)[0])
                    except:
                        print "Unable to open image {}".format(fullPath)
        X = np.array(X)
        print "Saving input gist vectors and submission list..."
        np.save(os.path.join(INPUT_DIR, 'X.npy'), X)
        with open(os.path.join(INPUT_DIR, 'candidates'), 'wb') as fileOut:
            pickle.dump(candidateList, fileOut)

        Q = np.array(Q)
        print "Saving query gist vectors and query list..."
        np.save(os.path.join(INPUT_DIR, 'Q.npy'), Q)
        with open(os.path.join(INPUT_DIR, 'queries'), 'wb') as fileOut:
            pickle.dump(queryList, fileOut)


if __name__ == "__main__":
    main()
