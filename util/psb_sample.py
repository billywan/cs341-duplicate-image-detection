'''
Randomly sample directories from photoshopbattle_images and copy to photoshopbattle_images_samples

Usage: python PROJ_DIR/util/psb_sample.py
'''

import os
import random
import shutil
import sys

# Some constants
DISK_DIR = "/mnt/data2"
DATA_DIR = os.path.join(DISK_DIR, "photoshopbattle_images_samples")
OUTPUT_DIR = os.path.join(DISK_DIR, "photoshopbattle_images_small_samples")
SAMPLE_SIZE = 50

def make_dir(dir_name):
    if os.path.exists(dir_name):
        print("{} already exists! Cleaning....".format(dir_name))
        shutil.rmtree(dir_name)
    print("Creating new folder at {}".format(dir_name))
    os.makedirs(dir_name)

make_dir(OUTPUT_DIR)
rand_dirs = random.sample(os.listdir(DATA_DIR), SAMPLE_SIZE)
counter = 0
for dir in rand_dirs:
	dest_dir = os.path.join(OUTPUT_DIR, dir)
	src_dir = os.path.join(DATA_DIR, dir)
	shutil.copytree(src_dir, dest_dir)

	counter += 1
	print("Sampling #{} directory".format(counter))
