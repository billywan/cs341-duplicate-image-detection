import os
import sys
import logging
import shutil
import numpy as np
from PIL import Image
from pyspark import SparkConf, SparkContext

DISK_DIR = "/mnt/data"
DATA_DIR = os.path.join(DISK_DIR, "reddit_images_2016_06")
OUTPUT_DIR = os.path.join(DISK_DIR, "reddit_images_2016_06_preprocessed")
IMG_SIZE = 224, 224


def quiet_logs(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

def make_dir(dir_name):
    if os.path.exists(dir_name):
        print("{} already exists! Cleaning....".format(dir_name))
        shutil.rmtree(dir_name)
    print("Creating new folder at {}".format(dir_name))
    os.makedirs(dir_name)

# Open and resize images and save as np array
def resize_one(filename):
	in_path = os.path.join(DATA_DIR, filename)
	out_path = os.path.join(OUTPUT_DIR, filename)
	try:
		img = Image.open(in_path)		
		img_arr = np.array(img.resize(IMG_SIZE, Image.ANTIALIAS))
		if np.ndim(img_arr) != 3 or img_arr.shape[2] != 3:
			return False
		Image.fromarray(img_arr).save(out_path)
		print "Writing output to {}".format(out_path)
		return True
	except:
		print("Unable to open image at {} for unknown reason".format(in_path))
		return False


conf = SparkConf()
sc = SparkContext(conf=conf)

# Quiet down the logger
quiet_logs(sc)

# Main Logic
if not os.path.exists(DATA_DIR):
	sys.exit("Directory reddit_images_2016_06 does not exists. Ending...")
make_dir(OUTPUT_DIR)

file_list = list(filter(lambda fn: fn.endswith(".jpg") or fn.endswith(".png"), os.listdir(DATA_DIR)))
file_list_rdd = sc.parallelize(file_list)
count = file_list_rdd.filter(resize_one).count()
print('Total resized: {}'.format(count))