'''
This script is for loading preprocessd images

Usage:
    This script is considered to reside under PROJECT_DIR/util (PROJECT_DIR is the outer most level)

Input:
    Input data is default to residing in "/mnt/data/data_batches". 
    
    Each file contains data that is packed in dictionary as:
    {'X1': X1, 'X2': X2, 'y': y}

    X1 and X2 are 4d numpy arrays with dimension (BATCH_SIZE, length, width, 3), BATCH_SIZE=5000

    y is a 1d numpy array with dimension (BATCH_SIZE,), BATCH_SIZE=5000


Ouput:
	List of data dictionary of each batch
'''
import pickle
import os

DATA_DIR = "/mnt/data/data_batches"

def load_data():
  data_list = []
  for fn in os.listdir(DATA_DIR):
    with open(os.path.join(DATA_DIR, fn), 'rb') as handle:
      data_dict = pickle.load(handle)
      for k, v in data_dict.items():
        print('%s: ' % k, v.shape)
        data_list.append(data_dict)
  return data_list

if not os.path.exists(DATA_DIR):
  sys.exit("Directory data_batches does not exists. Ending...")
    
load_data()
