'''
Script to compute gist vectors for input images and save to disk
Here, input/candidate is parent, queries are children.
'''

import os
import sys
import random
import numpy as np
import leargist
from PIL import Image


def main():
    PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
    CANDIDATE_DIR = "/mnt/data/reddit_images_2016_06_preprocessed"
    QUERY_DIR = "/mnt/data/reddit_images_2016_06_preprocessed_sampled"
    INPUT_DIR = os.path.join(PROJECT_DIR, '../data')

    skipDirs = []
    if os.path.exists(os.path.join(INPUT_DIR, 'X-reddit.npy')):
        print "Input gist vectors already exist, skipping..."
    else:
        X = []
        for file in os.listdir(CANDIDATE_DIR):
            fullPath = os.path.join(CANDIDATE_DIR, file)
            print "Computing gist vector for {}".format(fullPath)
            try:
                im = Image.open(fullPath)
                X.append(leargist.color_gist(im, orientations=(4,4,2)))
            except:
                print "Unable to open image {}".format(fullPath)
        X = np.array(X)

        print "Saving input gist vectors"
        np.save(os.path.join(INPUT_DIR, 'X-reddit.npy'), X)

    if os.path.exists(os.path.join(INPUT_DIR, 'Q-reddit.npy')):
        print "Query gist vectors already exist, exiting..."
        sys.exit()

    Q = []
    print "=" * 50
    for file in os.listdir(QUERY_DIR):
        fullPath = os.path.join(QUERY_DIR, file)
        print "Computing gist vector for query {}...".format(fullPath)
        try:
            im = Image.open(fullPath)
            Q.append(leargist.color_gist(im, orientations=(4,4,2)))
        except:
            print "Unable to open image {}".format(fullPath)
    Q = np.array(Q)

    print "Saving query gist vectors and query list"
    np.save(os.path.join(INPUT_DIR, 'Q-reddit.npy'), Q)


if __name__ == "__main__":
    main()
