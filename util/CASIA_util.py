'''
This script load processed CASIA dataset to batches

Usage:
    This script is considered to reside under PROJECT_DIR/util (PROJECT_DIR is the outer most level)

Input: 
	1. Processed authentic images under directory src_in_path

	2. Processed spliced images under directory tar_in_path

Output:
	List of tuples(X_src, X_tar, score) with length batch_size

	X_src and X_tar are 3d numpy arrays with dimension (length, width, 3)
	
	score is an integer
'''
import os
import math
import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from PIL import Image

SCORE = 1
src_in_path = "/Users/EricX/Desktop/CS341/cs341-duplicate-image-detection/Preprocessed_CASIA_Dataset/CASIA_DATASET/CASIA2/Au_out"
tar_in_path = "/Users/EricX/Desktop/CS341/cs341-duplicate-image-detection/Preprocessed_CASIA_Dataset/CASIA_DATASET/CASIA2/Sp_out"


def get_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


# Load source images and target images as lists of numpy arrays
def load_data(img_size=[224, 224]):
    # Housekeeping
    print(os.path.exists(src_in_path), os.path.exists(tar_in_path))
    if not os.path.exists(src_in_path) or not os.path.exists(tar_in_path):
        raise Exception("Input path does not exist")
        
    X_src, X_tar = [],[]
    for file_name in os.listdir(src_in_path):
        img = image.load_img(os.path.join(src_in_path, file_name), target_size=img_size)
        img = image.img_to_array(img)
        X_src.append(img)
    for file_name in os.listdir(tar_in_path):
        img = image.load_img(os.path.join(tar_in_path, file_name), target_size=img_size)
        img = image.img_to_array(img)
        X_tar.append(img)
    assert len(X_src) == len(X_tar)
    #X_src, X_tar = preprocess_input(np.stack(X_src, 0)), preprocess_input(np.stack(X_tar, 0))
    print "Loading data completed"
    return X_src[:1000], X_tar[:1000]

# Generate a list of tuple (X_src, X_tar, y) batches
# Return value is in the form of list of lists, with each inner list as a batch
def batch_generator(img_size=[224, 224], batch_size=50):
    batches = []
    X_src, X_tar = load_data(img_size)
    num_batches = int(math.ceil(float(len(X_src))/batch_size))
    for i in range(num_batches):
        X_src_batch = X_src[i*batch_size:i*batch_size+batch_size]
        X_tar_batch = X_tar[i*batch_size:i*batch_size+batch_size]
        score_batch = [SCORE]*batch_size
        batch = (preprocess_input(np.stack(X_src_batch, 0)), preprocess_input(np.stack(X_tar_batch, 0)), np.array(score_batch))
        batches.append(batch)
    return batches
