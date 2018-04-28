'''
This script resize and process CASIA dataset

Usage:
    This script is considered to reside under PROJECT_DIR/util (PROJECT_DIR is the outer most level)

Input: 
	1. Orignal authentic images under directory au_in_path

	2. Original spliced images under directory sp_in_path

Output:
	1. Processed authentic images under directory au_out_path

	2. Processed spliced images under directory sp_out_path

Note:
	au_int_path is an intermediate directory for storing partially processed images
'''
from PIL import Image
import os
import shutil
import numpy as np
from numpy import array

au_in_path = "/Users/zhangyun/Documents/Stanford/2018Spring/CS341/cs341-duplicate-image-detection/Dataset/CASIA_DATASET/CASIA2/Au"
sp_in_path = "/Users/zhangyun/Documents/Stanford/2018Spring/CS341/cs341-duplicate-image-detection/Dataset/CASIA_DATASET/CASIA2/Sp"
au_int_path = "/Users/zhangyun/Documents/Stanford/2018Spring/CS341/cs341-duplicate-image-detection/Dataset/CASIA_DATASET/CASIA2/Au_int"
sp_out_path = "/Users/zhangyun/Documents/Stanford/2018Spring/CS341/cs341-duplicate-image-detection/Dataset/CASIA_DATASET/CASIA2/Sp_out"
au_out_path = "/Users/zhangyun/Documents/Stanford/2018Spring/CS341/cs341-duplicate-image-detection/Dataset/CASIA_DATASET/CASIA2/Au_out"

for path in [au_int_path, sp_out_path, au_out_path]:
	if os.path.exists(path):
		shutil.rmtree(path)
		os.makedirs(path)
	else:
		os.makedirs(path)


# Return euclidean distance bewteen two images
def calculateDist(i1, i2):
	return np.sum((array(i1)-array(i2))**2)

# Process authentic images
au_dict = {}
au_count = 0
for file_name in os.listdir(au_in_path):
	au_count += 1
	print ("Converting number %i au image: %s " % (au_count, file_name))
	image = Image.open(os.path.join(au_in_path, file_name))
	size = 256, 256
	image = image.resize(size, Image.ANTIALIAS)
	li = file_name.split('_')
	new_file_name = os.path.join(au_int_path, (li[1]+li[2])[:-4]+".jpg")
	au_dict[(li[1]+li[2])[:-4]] = image
	image.save(new_file_name, "JPEG")

# Process spliced images
sp_dict = {}
sp_count = 0
for file_name in os.listdir(sp_in_path):
	sp_count += 1
	print ("Converting number %i sp image: %s " % (sp_count, file_name))
	image = Image.open(os.path.join(sp_in_path, file_name))
	size = 256, 256
	image = image.resize(size, Image.ANTIALIAS)
	_, _, ch = array(image).shape
	if ch != 3:
		continue
	s1 = file_name.split('_')[5]
	s2 = file_name.split('_')[6]
	if s1 not in au_dict or s2 not in au_dict:
		continue
	d1 = calculateDist(au_dict[s1], image)
	d2 = calculateDist(au_dict[s2], image)
	au_file_name = s1 if d1 < d2 else s2
	if au_file_name in sp_dict:
		sp_dict[au_file_name] += 1
	else:
		sp_dict[au_file_name] = 1
	new_file_name = os.path.join(sp_out_path, au_file_name+"_"+str(sp_dict[au_file_name])+".jpg")
	image.save(new_file_name, "JPEG")

# Filter out authentic images without spliced images and form training tuples
for au_file_name in sp_dict:
	count = sp_dict[au_file_name]
	image = au_dict[au_file_name]
	for i in range(count):
		new_file_name = os.path.join(au_out_path, au_file_name+"_"+str(i+1)+".jpg")
		image.save(new_file_name, "JPEG")
