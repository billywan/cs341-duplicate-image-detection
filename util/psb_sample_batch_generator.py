'''
Script for generate sample batches 

Usage: python PROJ_DIR/util/psb_sample_batch_generator.py

'''
import numpy as np
import os
from PIL import Image
import pickle
import random
import shutil
import sys

# Some constants
DISK_DIR = "/mnt/data2"
DATA_DIR = os.path.join(DISK_DIR, "photoshopbattle_images_small_samples")
PARENT_DIR = os.path.join(DISK_DIR, "photoshopbattle_images_small_samples_parents")
OUTPUT_DIR = os.path.join(DISK_DIR, "data_batches_small_samples")

BATCH_SIZE = 2000
SCORE_POS = np.float32(1.0)
SCORE_NEG = np.float32(0.0)

result_counter = 0
batch_counter = 0
_X1 = []
_X2 = []
_y = []


def make_dir(dir_name):
    if os.path.exists(dir_name):
        print("{} already exists! Cleaning....".format(dir_name))
        shutil.rmtree(dir_name)
    print("Creating new folder at {}".format(dir_name))
    os.makedirs(dir_name)

# Open images and save as np array
def processOne(file_path):
	try:
		img = Image.open(file_path)		
		processed_img = np.array(img).astype(np.float32)
		return True, processed_img
		
	except:
		print("Unable to open image at {} for unknwon reason".format(file_path))
		return False, None

# Process images in a directory
def processImg(dirName, fileList):
	submission_fn = fileList[0]
	comments_fn = fileList[1:]

	# Process images and save as np array
	submission_succeed, submission_img = processOne(os.path.join(dirName,submission_fn))
	comment_imgs = []
	comments_succeed = False
	for fn in comments_fn:
		succeed, comment_img = processOne(os.path.join(dirName,fn))
		comments_succeed = comments_succeed or succeed
		if succeed:
			comment_imgs.append(comment_img)
	return submission_succeed and comments_succeed, submission_img, comment_imgs

# Pack data as dictionary
def writeOutput():
	global _X1, _X2, _y
	X1 = np.array(_X1[:BATCH_SIZE]).copy()
	X2 = np.array(_X2[:BATCH_SIZE]).copy()
	y = np.array(_y[:BATCH_SIZE]).copy()

	data_dict = {'X1' : X1, 'X2' : X2, 'y' : y}
	file_path = os.path.join(OUTPUT_DIR, "data_batch_"+str(batch_counter))
	out = open(file_path, "wb")
	pickle.dump(data_dict, out)
	print("Writing output to: {}".format(file_path))
	out.close()


def cleanUp():
	global batch_counter, _X1, _X2, _y
	del _X1[:]
	del _X2[:]
	del _y[:]
	batch_counter += 1

def processParentImgs(parent_dir):
	parent_imgs= []
	parent_fns = os.listdir(parent_dir)
	for fn in parent_fns:
		succeed, img = processOne(os.path.join(parent_dir, fn))
		if succeed:
			parent_imgs.append(img)
	return parent_imgs


if not os.path.exists(DATA_DIR):
	sys.exit("Directory photoshopbattle_images_small_samples does not exists. Ending...")
make_dir(OUTPUT_DIR)

parent_imgs = processParentImgs(PARENT_DIR)

# Main Logic
# Loop through each subdirectory (corresponding to submission and comment images with the same post_id)
for dirName, subDirList, fileList in os.walk(DATA_DIR):
	print("Processing directory: {}. Total number of results: {}".format(dirName, result_counter))
	print("=" * 50)

	# Write output
	if len(_X1) >= BATCH_SIZE:
		writeOutput()
		cleanUp()

	# Filter and sort fileList
	fileList = list(filter(lambda fn: fn.endswith(".jpg") or fn.endswith(".png"), fileList))
	if len(fileList) == 0 or len(fileList) == 1:
		continue
	fileList.sort()
	if fileList[0].count("_") != 1:
		continue

	succeeded, submission_img, comment_imgs = processImg(dirName, fileList)
	
	if not succeeded:
		continue
	# Filter out images without 3 dimensions
	if np.ndim(submission_img) != 3 or submission_img.shape[2] != 3:
		continue
	comment_imgs = list(filter(lambda x: np.ndim(x) == 3 and x.shape[2] == 3, comment_imgs))
	
	# Generate (s, c, 1) pair
	for i in range(len(comment_imgs)):
		_X1.append(submission_img)
		_X2.append(comment_imgs[i])
		_y.append(SCORE_POS)
		result_counter += 1

	# Generate (c, r, 0) pair
	for i in range(len(comment_imgs)):
		for j in range(len(parent_imgs)):
			_X1.append(comment_imgs[i])
			_X2.append(parent_imgs[j])
			_y.append(SCORE_NEG)
			result_counter += 1


# Write to output the rest of results
writeOutput()
cleanUp()
print("Total number of results: {}".format(result_counter))



	





