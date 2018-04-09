import urllib.request
from imgurdownloader import ImgurDownloader, ImgurException
import os
import random
import pandas as pd


PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
#print(PROJECT_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
#print(DATA_DIR)

data_dir = os.fsencode(DATA_DIR)
data_files = []
for file in os.listdir(data_dir):
    filename = os.fsdecode(file)
    if not filename.endswith(".csv"):
        continue
    else:
        print(os.path.join(DATA_DIR, filename))
        data_files.append(os.path.join(DATA_DIR, filename))


COL_NAMES = ['timestamp', 'user_id', 'subreddit_name', 'post_fullname', 'post_type', 'post_title', 'post_target_url', 'post_body']

data_frame = pd.read_csv(data_files[0], names=COL_NAMES, usecols=['post_type', 'post_target_url'])
links = data_frame['post_target_url'].tolist()
links = list(filter(lambda l: isinstance(l, str), links))
#print(list(links))

IMAGE_FORMATS = ['png', 'jpg', 'gif']
# grep '\.[a-z][a-z][a-z]$' imagelinks_submissions_2016_06.csv | grep -v jpg | grep -v png | grep -v gif | grep -v self,
# grep '\.[a-z][a-z][a-z],$' imagelinks_submissions_2016_06.csv | grep -v jpg | grep -v png | grep -v gif | grep -v self, | grep -v imgur.com,
TYPO_IMAGE_FORMATS = ['com', 'img', 'jgp', 'jph', 'jpd', 'jpp', 'jpq', 'jog', 'kpg', 'pbg', 'giv']
# IMAGE_FORMATS = ['bmp', 'dib', 'eps', 'ps', 'gif', 'im', 'jpg', 'jpe', 'jpeg',
#                  'pcd', 'pcx', 'png', 'pbm', 'pgm', 'ppm', 'psd', 'tif',
#                  'tiff', 'xbm', 'xpm', 'rgb', 'rast', 'svg']
IMAGE_EXTENSION = ['.'+format for format in IMAGE_FORMATS]
 # for now don't download .gifv (mp4) files



def get_imgur_links(links):
    return filter(lambda l: 'imgur' in l, links)

def get_image_links(links):
    def has_extension(link):
        for ext in IMAGE_EXTENSION:
            if link.endswith(ext):
                return True
        return False

    filtered = filter(lambda l: 'imgur' not in l, links)
    filtered = filter(lambda l: has_extension(l), filtered)
    return filtered

def get_reddituploads_links(links):
    filtered = filter(lambda l: 'i.reddituploads.com' in l, links)
    mapped = map(lambda l: l+'.jpeg', filtered)
    return mapped

def download(links, dest_dir=None):
    if not os.path.exists(dest_dir):
        print('Creating new folder at {} to hold images'.format(dest_dir))
        os.makedirs(dest_dir)
    else:
        raise ValueError("dest_dir already exists!")
    count = 0
    imgur_links = list(get_imgur_links(links))
    image_links = list(get_reddituploads_links(links)) + list(get_image_links(links))
    imgur_links_count = len(imgur_links)
    image_links_count = len(image_links)
    print("total: {0}, imgur: {1}, image:{2}, combined valid: {3}".format(len(links), imgur_links_count, image_links_count, imgur_links_count+image_links_count))

    # random.shuffle(imgur_links)
    # random.shuffle(image_links)


    for link in imgur_links:
        img_name = "{0:010d}".format(count)
        try:
            ImgurDownloader(link, dest_dir, img_name).save_images()
            count+=1
        #except (ImgurException, TimeoutError):
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print("Failed to download image at link", link)


    for link in image_links:
        img_name = "{0:010d}.png".format(count)
        try:
            #print(link)
            urllib.request.urlretrieve(link, os.path.join(dest_dir, img_name))
            count+=1
        except urllib.error.URLError as e:
            print(e.reason, link)



# print(os.path.join(DATA_DIR, 'download'))
download(links, os.path.join(DATA_DIR, 'download'))

# import timeit
# print(timeit.timeit('urllib.request.urlretrieve("http://i.imgur.com/H9GRMiT.jpg", "00000001.jpg")', number=1, setup='import urllib.request'))
# print(timeit.timeit('ImgurDownloader("http://i.imgur.com/H9GRMiT.jpg", "/Users/EricX/Desktop/CS341").save_images()', number=1, setup='from imgurdownloader import ImgurDownloader'))

#urllib.request.urlretrieve("http://i.imgur.com/H9GRMiT.jpg", "00000001.jpg")
