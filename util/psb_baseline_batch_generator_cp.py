'''
Script for generating baseline data batch. Child images are used are query and parent images are used as candidates. 

Assume all parent images have been collected into a single folder.

Output:

Data batches (X1, X2, y) with batch_size = 10000

Index files storing the relative indices of real candidates. Will later be used in metrics.py

Usage:

python util/psb_baseline_batch_generator_cp.py

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
CANDIDATE_DIR = os.path.join(DISK_DIR, "photoshopbattle_images_small_samples_parents")
PROJECT_DIR = os.path.join("..", os.path.abspath(os.path.dirname(__file__)))
OUTPUT_DIR = os.path.join(DISK_DIR, "data_batches_baseline_cp")
INDEX_DIR = os.path.join(PROJECT_DIR, "index")

BATCH_SIZE = 10000
SCORE_POS = 1.0
SCORE_NEG = 0.0

query_counter, batch_counter = 0, 0
X1 = []
X2 = []
y = []
index = []  # each element in index: (start_batch, start_batch_idx, end_batch, end_batch_idx, [relative indices of true candidates])


def make_dir(dir_name):
    if os.path.exists(dir_name):
        print("{} already exists! Cleaning....".format(dir_name))
        shutil.rmtree(dir_name)
    print("Creating new folder at {}".format(dir_name))
    os.makedirs(dir_name)


# Pack data as dictionary
def writeOutput():
	global X1, X2, y, batch_counter
	print "Dumping batch {}...".format(batch_counter)
 	with open(os.path.join(OUTPUT_DIR, "data_batch_" + str(batch_counter).zfill(2)), 'wb') as out:
 		pickle.dump({'X1': np.array(X1), 'X2': np.array(X2), 'y': np.array(y)}, out)
        del X1[:]
        del X2[:]
        del y[:]
        batch_counter += 1
	out.close()


if not os.path.exists(DATA_DIR):
	sys.exit("Directory photoshopbattle_images_small_samples does not exists. Ending...")
if not os.path.exists(CANDIDATE_DIR):
	sys.exit("Directory photoshopbattle_images_small_samples_parents does not exists. Ending...")
make_dir(OUTPUT_DIR)
make_dir(INDEX_DIR)
candidate_fns = sorted(os.listdir(CANDIDATE_DIR))
candidate_arrs = [np.array(Image.open(os.path.join(CANDIDATE_DIR, candidate_fn))) for candidate_fn in candidate_fns]

# Main Logic
# Loop through each subdirectory (corresponding to submission and comment images with the same post_id)
for dir_name, subdir_list, file_list in os.walk(DATA_DIR):
	print("Processing directory: {}." .format(dir_name))
	print("=" * 50)
	if len(file_list) <= 1:
		continue
	file_list = sorted(file_list)
	query_fns = file_list[1:]

	# Generate (c, p) pair for each query
	for i, query_fn in enumerate(query_fns):
		start_batch = batch_counter
		start_batch_idx = len(X1)
		rel_indices = []
		print("Generating pairs for query #{}:{}".format(query_counter, query_fn))
		query_counter += 1
		query_arr = np.array(Image.open(os.path.join(dir_name, query_fn)))

		for j, candidate_fn in enumerate(candidate_fns):
			X1.append(candidate_arrs[j])
			X2.append(query_arr)
			if candidate_fn.rsplit('.', 1)[0] in query_fn:
				# parent child with same post_id
				print("Found query {} and candidate {} with the same post_id. idx={}".format(query_fn, candidate_fn, j))
				y.append(SCORE_POS)
				rel_indices.append(j)
			else:
				y.append(SCORE_NEG)
			# Dump data batch
			if len(X1) == BATCH_SIZE:
				writeOutput()
	
		index.append((start_batch, start_batch_idx, batch_counter, len(X1), rel_indices))
		print "Appending index: ({}, {}, {}, {}, {}) for query {}".format(start_batch, start_batch_idx, batch_counter, len(X1), rel_indices, query_fn)


# Dump final batch and index file
print "Dumping final batch..."
writeOutput()
with open(os.path.join(os.path.join(PROJECT_DIR, "output"), 'index-cp'), 'wb') as out:
	pickle.dump(index, out)
	out.close()






	





