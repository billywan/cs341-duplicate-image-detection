import urllib.request
import threading
from imgurdownloader import ImgurDownloader, ImgurException
import os, sys, argparse, time, random, datetime, logging
import numpy as np
import pandas as pd
from pyspark import SparkConf, SparkContext



def has_extension(link):
    for ext in IMAGE_EXTENSION:
        if link.endswith(ext):
            return True
    return False

def is_album_link(link):
    album_tags = ['/a/', '/r/', '/gallery']
    for tag in album_tags:
        if tag in link:
            return True
    return False

def validate_imgur_no_ext_link(link):
    key = link.split('/')[-1]
    new_link = 'http://i.imgur.com/{key}.jpeg'.format(key=key)
    return new_link


def download_one(link_tuple):
    link_type, link = link_tuple
    if link_type == 'imgur_album':
        try:
            ImgurDownloader(link, DOWNLOAD_DIR).save_images()
        #except (ImgurException, TimeoutError):
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print("Failed to download image(s) at:", link)
    if link_type == 'image':
        try:
            file_path = os.path.join(DOWNLOAD_DIR, link.split('/')[-1])
            urllib.request.urlretrieve(link, file_path)
        except urllib.error.URLError as e:
            print(e.reason, link)

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        print('Creating new folder at {}'.format(dir_name))
        os.makedirs(dir_name)
    else:
        raise ValueError("dir_name already exists!")


def quiet_logs(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )


def download(data_file):
    msg = 'Processing {}...'.format(data_file)
    print('='*len(msg), msg, sep='\n')
    data_frame = pd.read_csv(data_file, names=COL_NAMES, usecols=['post_type', 'post_target_url'])
    links = data_frame['post_target_url'].tolist()
    links = list(filter(lambda l: isinstance(l, str), links))

    #main logic of the spark job
    links_rdd = sc.parallelize(links)  
    imgur_links = links_rdd.filter(lambda l: 'imgur' in l) 
    imgur_image_links = imgur_links.filter(has_extension) 
    imgur_album_links = imgur_links.subtract(imgur_image_links).filter(is_album_link) # '/a/, /r/, /gallery/'
    imgur_no_ext_links = imgur_links.subtract(imgur_image_links).subtract(imgur_album_links).map(validate_imgur_no_ext_link) 

    normal_image_links = links_rdd.filter(has_extension) 
    reddituploads_links = links_rdd.filter(lambda l: 'i.reddituploads.com' in l).map(lambda l: l+'.jpeg') 

    image_links = (imgur_image_links + imgur_no_ext_links + normal_image_links + reddituploads_links).distinct() 
    processed_links = image_links.map(lambda l: ('image', l)) + imgur_album_links.map(lambda l: ('imgur_album', l)) 
    print('Total valid links: {}'.format(processed_links.count()))

    count = processed_links.map(download_one).count()
    print('Total Downloaded: {}'.format(count))





conf = SparkConf()
sc = SparkContext(conf=conf)

# Quite down the logger
quiet_logs(sc)

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DOWNLOAD_DIR = os.path.join(DATA_DIR, 'download/'+str(datetime.datetime.now()))
COL_NAMES = ['timestamp', 'user_id', 'subreddit_name', 'post_fullname', 'post_type', 'post_title', 'post_target_url', 'post_body']
IMAGE_FORMATS = ['png', 'jpg', 'gif', 'jpe', 'jpeg']
# IMAGE_FORMATS = ['bmp', 'dib', 'eps', 'ps', 'gif', 'im', 'jpg', 'jpe', 'jpeg',
#                  'pcd', 'pcx', 'png', 'pbm', 'pgm', 'ppm', 'psd', 'tif',
#                  'tiff', 'xbm', 'xpm', 'rgb', 'rast', 'svg']
IMAGE_EXTENSION = ['.'+format for format in IMAGE_FORMATS]
 # for now don't download .gifv (mp4) files


make_dir(DOWNLOAD_DIR)
data_dir = os.fsencode(DATA_DIR)
data_files = []
for file in os.listdir(data_dir):
    filename = os.fsdecode(file)
    if not filename.endswith(".csv"):
        continue
    else:
        #print(os.path.join(DATA_DIR, filename))
        data_files.append(os.path.join(DATA_DIR, filename))

for data_file in data_files:
    download(data_file)









