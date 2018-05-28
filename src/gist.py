'''
Script to compute gist vectors for input images and save to disk
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
    DATA_DIR = "/mnt/data/photoshopbattle_images_preprocessed"
    INPUT_DIR = os.path.join(PROJECT_DIR, '../data')

    skipDirs = []
    if os.path.exists(os.path.join(INPUT_DIR, 'X.npy')):
        print "Input gist vectors already exist, skipping..."
    else:
        X = []
        submissionList = []
        for dirName, _, fileList in os.walk(DATA_DIR):
            if len(fileList) > 1:
                submission = sorted(fileList)[0]
                # if no parent, skip
                if submission.count("_") > 1:
                    skipDirs.append(dirName)
                else:
                    fullPath = os.path.join(dirName, submission)
                    print "Computing gist vector for {}...".format(fullPath)
                    try:
                        im = Image.open(fullPath)
                        X.append(leargist.color_gist(im, orientations=(4,4,2)))
                        submissionList.append(submission.rsplit('.', 1)[0])
                        if len(submissionList) % 500 == 0:
                            print "Computed gist vectors for {} images".format(len(submissionList))
                            print "=" * 50
                    except:
                        print "Unable to open image {}".format(fullPath)
        X = np.array(X)

        print "Saving input gist vectors and submission list"
        np.save(os.path.join(INPUT_DIR, 'X.npy'), X)
        pickle.dump(submissionList, open(os.path.join(INPUT_DIR, 'submissions'), 'wb'))

    if os.path.exists(os.path.join(INPUT_DIR, 'Q.npy')):
        print "Query gist vectors already exist, exiting..."
        sys.exit()

    Q = []
    queryList = []
    print "=" * 50
    for dirName, _, fileList in os.walk(DATA_DIR):
        if len(fileList) > 1 and dirName not in skipDirs:
            print "Computing gist vectors for query images..."
            for query in sorted(fileList)[1:]:
                fullPath = os.path.join(dirName, query)
                print "Computing gist vector for query {}...".format(fullPath)
                try:
                    im = Image.open(fullPath)
                    Q.append(leargist.color_gist(im, orientations=(4,4,2)))
                    queryList.append(query)
                except:
                    print "Unable to open image {}".format(fullPath)
    Q = np.array(Q)

    print "Saving query gist vectors and query list"
    np.save(os.path.join(INPUT_DIR, 'Q.npy'), Q)
    pickle.dump(queryList, open(os.path.join(INPUT_DIR, 'queries'), 'wb'))


if __name__ == "__main__":
    main()
