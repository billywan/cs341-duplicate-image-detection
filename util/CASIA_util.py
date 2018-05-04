import os
import math
import numpy as np
from PIL import Image

SCORE = 1
src_in_path = "/Users/zhangyun/Documents/Stanford/2018Spring/CS341/cs341-duplicate-image-detection/Dataset/CASIA_DATASET/CASIA2/Au_out"
tar_in_path = "/Users/zhangyun/Documents/Stanford/2018Spring/CS341/cs341-duplicate-image-detection/Dataset/CASIA_DATASET/CASIA2/Sp_out"

# Load source images and target images as lists of numpy arrays
def load_data():
	# Housekeeping
	if not os.path.exists(src_in_path) or not os.path.exists(tar_in_path):
		raise Exception("Input path does not exist")
	X_src, X_tar = [],[]
	for file_name in os.listdir(src_in_path):
		image = Image.open(os.path.join(src_in_path, file_name))
		X_src.append(np.array(image))
	for file_name in os.listdir(tar_in_path):
		image = Image.open(os.path.join(tar_in_path, file_name))
		X_tar.append(np.array(image))
	assert len(X_src) == len(X_tar)
	print "Loading data completed"
	return X_src, X_tar

# Generate a list of tuple (X_src, X_tar, y) batches
# Return value is in the form of list of lists, with each inner list as a batch
def batch_generator(batch_size=100):
	batches = []
	X_src, X_tar = load_data()
	num_batches = int(math.ceil(float(len(X_src))/batch_size))
	for i in range(num_batches):
		batch = []
		for j in range(i*batch_size, min((i+1)*batch_size, len(X_src))):
			batch.append((X_src[j], X_tar[j], SCORE))
		batches.append(batch)
	return batches
