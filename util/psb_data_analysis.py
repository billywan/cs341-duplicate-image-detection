'''
This script is for calculating the euclidean distance bewteen parent and corresponding children images

Usage:
    This script is considered to reside under PROJECT_DIR/util (PROJECT_DIR is the outer most level)
    python <path-to-this-script>

Input:
    Input data is default to residing in "/mnt/data/data_batches".

Output:
	List of euclidean distance

'''
import os
import sys
from PIL import Image
import numpy as np
import shutil
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Some constants
DISK_DIR = "/mnt/data2"
DATA_DIR = os.path.join(DISK_DIR, "data_batches")
OUTPUT_DIR = "/home/zhangyun/output"
SCORE_POS = np.float32(1.0)
SCORE_INT = np.float32(0.5)
SCORE_NEG = np.float32(0.0)

def make_dir(dir_name):
    if os.path.exists(dir_name):
        print("{} already exists! Cleaning....".format(dir_name))
        shutil.rmtree(dir_name)
    print("Creating new folder at {}".format(dir_name))
    os.makedirs(dir_name)

# Return euclidean distance bewteen two numpy array
def calculateEuclideanDist(a1, a2):
	return np.sum((a1-a2)**2)


if not os.path.exists(DATA_DIR):
	sys.exit("Directory data_batches does not exists. Ending...")
make_dir(OUTPUT_DIR)

dist_list = []
result_count = 0

# Main Logic
# Loop through each subdirectory (parent and children images with the same post_id)
for dirName, subDirList, fileList in os.walk(DATA_DIR):
	for fn in fileList:
		if not fn.startswith("data_batch"):
			continue
		print("Processing data batch: {}".format(fn))
		print("=" * 50)

		with open(os.path.join(dirName, fn), "rb") as handle:
			data_dict = pickle.load(handle)
			X1, X2, y = data_dict["X1"], data_dict["X2"], data_dict["y"]
			batch_size = X1.shape[0]
			for i in range(batch_size):
				# Calculate distance between parent and child
				if y[i] == SCORE_NEG or y[i] == SCORE_INT:
					continue
				dist = calculateEuclideanDist(X1[i], X2[i])
				dist_list.append(dist)
				print("Processing result {}".format(result_count))
				result_count += 1
				print("=" * 50)

file_path = os.path.join(OUTPUT_DIR, "dist")	
file_out = open(file_path, "wb")
pickle.dump(dist_list, file_out)
print("Writing output to: {}".format(DATA_DIR))
file_out.close()

# Density plot
with open(file_path, "rb") as handle:
	dist = np.array(pickle.load(handle))
	percentile_33 = np.percentile(dist, 33)
	percentile_66 = np.percentile(dist, 66)
	print("33% percentile of the distance is {}. 66% percentile of the distance is {}". format(percentile_33, percentile_66))
	weights = np.ones_like(dist)/float(len(dist))
	plt.figure()
	plt.hist(dist, bins=3, normed=False, weights=weights)
	plt.title("Distribution of Euclidean distance between parent and children images")
	plt.xlabel("distance")
	plt.ylabel("probability")
	plt.savefig(os.path.join(OUTPUT_DIR, "dist_fig.jpg"))