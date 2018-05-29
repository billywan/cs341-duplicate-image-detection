'''
This script is for crawling the top level submissions of photoshopbattle dataset.

Usage:
    <path to spark-submit> <path to this script>
    e.g. spark-2.2.1-bin-hadoop2.7/bin/spark-submit image_downloader_psb_submissions_spark.py

    This script is considered to reside under PROJECT_DIR (outer most level)

Input:
    Input data is default to reside in PROJECT_DIR/data/

Ouput: Downloaded images are default to reside in /mnt/data/photoshopbattle_images/<post_id>. Each post_id has its own directory.

Note:
    Requires Python 3.
    The output of this scripts is downloaded images with name format "<post_fullname>.jpg"
    After crawling is completed, run PROJECT_DIR/util/filter_dir.sh to filter out empty directories caused by downloading failure.
'''


import urllib.request
from imgurdownloader import ImgurDownloader, ImgurException
import os
import sys
import argparse
import time
import random
import datetime
import logging
import numpy as np
import pandas as pd
import shutil
from collections import defaultdict
from pyspark import SparkConf, SparkContext


def has_extension(link_tuple):
    link, post_id = link_tuple
    for ext in IMAGE_EXTENSIONS:
        if link.endswith(ext):
            return True
    return False

def fix_typo_extension(link_tuple):
    link, post_id = link_tuple
    for ext in TYPO_IMAGE_EXTENSIONS:
        if link.endswith(ext):
            print("replacing {} with .png".format(ext))
            link.replace(ext, ".png")
    return (link, post_id)

def not_gif(link_tuple):
    link, post_id = link_tuple
    return not (link.endswith('gif') or link.endswith('gifv'))

def is_album_link(link_tuple):
    link, post_id = link_tuple
    album_tags = ['/a/', '/r/', '/gallery']
    for tag in album_tags:
        if tag in link:
            return True
    return False

def validate_imgur_no_ext_link(link_tuple):
    link, post_id = link_tuple
    key = link.split('/')[-1]
    new_link = 'http://i.imgur.com/{key}.jpg'.format(key=key)
    return (new_link, post_id)

def download_one(link_tuple):
    link_type, post_id, link = link_tuple
    if link == "http://i.imgur.com/ET4VfDd.jpg":
        return False
    download_dir = os.path.join(OUTPUT_DIR, post_id)
    make_dir(download_dir)
    msg = "Downloading {} of type {} with post id {}".format(link, link_type, post_id)
    print('='*50, msg, sep='\n')
    
    if link_type == 'imgur_album':
        try:
            ImgurDownloader(link, download_dir).save_images(post_id=post_id)
            return True
        except (KeyboardInterrupt, SystemExit):
            raise
        except ImgurException as e:
            print("Failed to download image(s) from album at: {}".format(link), e.msg)
        except Exception as e:
            print("Failed to download image(s) from album at: {} for unknown reasons".format(link))

    if link_type == 'image':
        try:
            urllib.request.urlretrieve(link, os.path.join(download_dir, post_id+'.jpg'))
            return True
        except ConnectionResetError:
            print("Failed to download image at {} due to connection reset".format(link))
        except urllib.error.URLError as e:
            print("Failed to download image at: {}".format(link), e.reason)
        except:
            print("Failed to download image at: {} for unknown reasons".format(link))
    return False

def make_dir(dir_name):
    if os.path.exists(dir_name):
        print("{} already exists! Cleaning....".format(dir_name))
        shutil.rmtree(dir_name)
    print("Creating new folder at {}".format(dir_name))
    os.makedirs(dir_name)

def quiet_logs(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

def download(data_file):
    msg = 'Processing {}...'.format(data_file)
    print('='*len(msg), msg, sep='\n')
    data_frame = pd.read_csv(data_file, names=COL_NAMES, usecols=['post_fullname', 'post_target_url'])
    # filter images from inappropriate subreddits
    post_ids = data_frame['post_fullname'].tolist()
    links = data_frame['post_target_url'].tolist()
    links_tuple = list(zip(links, post_ids))
    links_tuple = list(filter(lambda t: isinstance(t[0], str), links_tuple))

    # main logic of the spark job
    links_rdd = sc.parallelize(links_tuple)
    imgur_links = links_rdd.filter(lambda t : 'imgur' in t[0])
    imgur_image_links = imgur_links.filter(has_extension)
    imgur_album_links = imgur_links.subtract(imgur_image_links).filter(is_album_link)  # '/a/, /r/, /gallery/'
    imgur_no_ext_links = imgur_links.subtract(imgur_image_links).subtract(imgur_album_links).filter(not_gif).map(validate_imgur_no_ext_link)

    normal_image_links = links_rdd.filter(has_extension).map(fix_typo_extension)
    reddituploads_links = links_rdd.filter(lambda t: 'i.reddituploads.com' in t[0]).map(lambda t: (t[0]+'.jpg', t[1]))

    image_links = imgur_no_ext_links + normal_image_links + reddituploads_links
    image_links_dict = dict(image_links.collect()) # {link: post_id}
    image_links = image_links.map(lambda t: t[0]).distinct()

    album_links_dict = dict(imgur_album_links.collect()) # {link: post_id}
    album_links = imgur_album_links.map(lambda t: t[0]).distinct()

    processed_image_links = image_links.filter(lambda l: l in image_links_dict).map(lambda l: ('image', image_links_dict[l], l))
    processed_album_links = album_links.filter(lambda l: l in album_links_dict).map(lambda l: ('imgur_album', album_links_dict[l], l))
    processed_links = (processed_image_links + processed_album_links).distinct()
    print('Total valid links: {}'.format(processed_links.count()))

    count = processed_links.filter(download_one).count()
    print('Total Downloaded: {}'.format(count))    


conf = SparkConf()
sc = SparkContext(conf=conf)

# Quiet down the logger
quiet_logs(sc)

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join("/mnt/data2", "photoshopbattle_images")
COL_NAMES = ['timestamp', 'user_id', 'subreddit_name', 'post_fullname', 'post_type', 'post_title', 'post_target_url', 'post_body']
# for now don't download .gif and .gifv (mp4) files
IMAGE_FORMATS = ['png', 'jpg', 'jpeg']
TYPO_IMAGE_FORMATS = ['com', 'img', 'jgp', 'jph', 'jpd', 'jpp', 'jpq', 'jog', 'kpg', 'pbg']
# IMAGE_FORMATS = ['bmp', 'dib', 'eps', 'ps', 'gif', 'im', 'jpg', 'jpe', 'jpeg',
#                  'pcd', 'pcx', 'png', 'pbm', 'pgm', 'ppm', 'psd', 'tif',
#                  'tiff', 'xbm', 'xpm', 'rgb', 'rast', 'svg']
IMAGE_EXTENSIONS = ['.' + format for format in (IMAGE_FORMATS + TYPO_IMAGE_FORMATS)]
TYPO_IMAGE_EXTENSIONS = ['.' + format for format in TYPO_IMAGE_FORMATS]

make_dir(OUTPUT_DIR)
data_dir = os.fsencode(DATA_DIR)
data_files = []
for file in os.listdir(data_dir):
    filename = os.fsdecode(file)
    if filename != "photoshopbattles_submissions.csv":
        continue
    else:
        data_files.append(os.path.join(DATA_DIR, filename))

for data_file in data_files:
    download(data_file)

sc.stop()