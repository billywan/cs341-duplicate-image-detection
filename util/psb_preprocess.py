'''
This script is for preprocessing crawled images of photoshopbattle submissions and comments 

Usage:
    This script is considered to reside under PROJECT_DIR/util (PROJECT_DIR is the outer most level)
    python <path-to-this-script>


Input:
    Input data is default to residing in "/mnt/data/photoshopbattle_images/<post_id>".
    
    Each post_id has its own directory containing both submission images with name format "<post_id>.ext" and comment images with name format "<post_id_index>.ext"

Ouput:
	Files containing data that is packed in dictionary as:
    {'X1': X1, 'X2': X2, 'y': y}

    X1 and X2 are 4d numpy arrays with dimension (BATCH_SIZE, length, width, 3), BATCH_SIZE=5000

    y is a 1d numpy array with dimension (BATCH_SIZE,), BATCH_SIZE=5000
'''

import numpy as np
import os
from PIL import Image
import pickle
import random
import shutil

# Some constants
DISK_DIR = "/mnt/data"
DATA_DIR = os.path.join(DISK_DIR, "photoshopbattle_images")
OUTPUT_DIR = os.path.join(DISK_DIR, "data_batches")

BATCH_SIZE = 5000
IMG_SIZE = 256, 256
SCORE_POS = 1
SCORE_INT = 0.5
SCORE_NEG = 0

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

# Open and resize images and save as np array
def resizeOne(file_path):
	try:
		img = Image.open(file_path)
		processed_img = np.array(img.resize(IMG_SIZE, Image.ANTIALIAS))
		return True, processed_img
	except:
		print("Unable to open image at {} for unknwon reason".format(file_path))
		return False, None

# Resize images in a directory
def resizeImg(dirName, fileList):
	submission_fn = fileList[0]
	comments_fn = fileList[1:]

	# Resize images and save as np array
	try:
		submission_img = np.array(Image.open(os.path.join(dirName, submission_fn)).resize(IMG_SIZE, Image.ANTIALIAS))
		comment_imgs = [np.array(Image.open(os.path.join(dirName, comment_fn)).resize(IMG_SIZE, Image.ANTIALIAS)) for comment_fn in comments_fn]
		return True, submission_img, comment_imgs
	except:
		print("Unable to open image in {} for unknown reason".format(dirName))
		return False, None, None

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

if not os.path.exists(DATA_DIR):
	sys.exit("Directory photoshopbattle_images does not exists. Ending...")
make_dir(OUTPUT_DIR)

# Main Logic
# Loop through each subdirectory (corresponding to submission and comment images with the same post_id)
for dirName, subDirList, fileList in os.walk(DATA_DIR):
	print("Processing directory: {}. Total number of results: {}".format(dirName, result_counter))
	print("=" * 50)

	# Write output
	if len(_X1) >= BATCH_SIZE:
		writeOutput()
		cleanUp()

    # Filter out post_ids without comments
	if len(fileList) == 0 or len(fileList) == 1:
		continue

	succeeded, submission_img, comment_imgs = resizeImg(dirName, fileList)
	
	if not succeeded:
		continue
	# Filter out images without 3 dimensions
	if np.ndim(submission_img) != 3 or submission_img.shape[2] != 3:
		continue

	comment_imgs = list(filter(lambda x: np.ndim(x) == 3 and x.shape[2] == 3, comment_imgs))
	
	num_comments = len(comment_imgs)

	if num_comments == 1:
	# Generate only (s, c, 1) pair
		_X1.append(submission_img)
		_X2.append(comment_imgs[0])
		_y.append(SCORE_POS)
		result_counter += 1

	else:
		# Generate both (s, c, 1) and (c1, c2, 0.5) pair with equal amount
		# Generate (s, c, 1) pair
		for i in range(num_comments):
			_X1.append(submission_img)
			_X2.append(comment_imgs[i])
			_y.append(SCORE_POS)
			result_counter += 1
		# Generate (c1, c2, 0.5) pair
		# Randomly sample c1 and c2 from comment images
		for i in range(num_comments):
			i1 = 0
			i2 = 0
			while True:
				i1 = random.randint(0, num_comments-1)
				i2 = random.randint(0, num_comments-1)
				if i1 != i2:
					break
			_X1.append(comment_imgs[i1])
			_X2.append(comment_imgs[i2])
			_y.append(SCORE_INT)
			result_counter += 1
	# Generate (s, _, 0) pair
	for i in range(num_comments):
		while True:
		# Randomly sample a image from photoshopbattle_images
			rand_dir = os.path.join(DATA_DIR, random.choice(os.listdir(DATA_DIR)))
			rand_file_path = os.path.join(rand_dir, os.listdir(rand_dir)[0])
			succeeded, rand_img = resizeOne(rand_file_path)
			if succeeded and np.ndim(rand_img)==3 and rand_img.shape[2]==3:
				_X1.append(submission_img)
				_X2.append(rand_img)
				_y.append(SCORE_NEG)
				result_counter += 1
				break

# Write to output the rest of results
writeOutput()
print("Total number of results: {}".format(result_counter))



	





