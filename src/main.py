from __future__ import absolute_import
from __future__ import division

import os
import io
import json
import sys
import logging

import tensorflow as tf
import keras
from keras.models import load_model

import siamese


MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
#DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir

# High-level options
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / eval")
tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train. 0 means train indefinitely")

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on dense layers.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use")

# # How often to print, save, eval
# tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
# tf.app.flags.DEFINE_integer("save_every", 500, "How many iterations to do per save.")
# tf.app.flags.DEFINE_integer("eval_every", 500, "How many iterations to do per calculating loss/f1/em on dev set. Warning: this is fairly time-consuming so don't do it too often.")
# tf.app.flags.DEFINE_integer("keep", 1, "How many checkpoints to keep. 0 indicates keep all (you shouldn't need to do keep all though - it's very storage intensive).")

# # Reading and saving data
# tf.app.flags.DEFINE_string("train_dir", "", "Training directory to save the model parameters and other info. Defaults to experiments/{experiment_name}")
# tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "Where to find preprocessed SQuAD data for training. Defaults to data/")
# tf.app.flags.DEFINE_string("ckpt_load_dir", "", "For official_eval mode, which directory to load the checkpoint fron. You need to specify this for official_eval mode.")
# tf.app.flags.DEFINE_string("json_in_path", "", "For official_eval mode, path to JSON input file. You need to specify this for official_eval_mode.")
# tf.app.flags.DEFINE_string("json_out_path", "predictions.json", "Output path for official_eval mode. Defaults to predictions.json")



FLAGS = tf.app.flags.FLAGS

def initialize_model(expect_exists=False):
    # Setup experiment dir
    # if expect_exists:
    train_dir = os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)
    model_file_path = os.path.join(train_dir, "model.h5")
    if not os.path.exists(model_file_path):
        if expect_exists:
            raise Exception("No existing model found at %s"%model_file_path)
        else:
            if not os.path.exists(train_dir):
                os.makedirs(train_dir)
            model = siamese.build_model(FLAGS)
    else:
        print "Trying to load existing model at %s" %model_file_path
        try:
            model = load_model(model_file_path)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            raise Exception("Failed to load model at %s"%model_file_path)
    return model


def main(unused_argv):
    # Print an error message if you've entered flags incorrectly
    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: %s" % unused_argv)

    # Check for Python 2
    if sys.version_info[0] != 2:
        raise Exception("ERROR: You must use Python 2 but you are running Python %i" % sys.version_info[0])

    if not FLAGS.experiment_name:
        raise Exception("ERROR: You need to specify --experiment_name") 

    if FLAGS.mode == "train":
        model = initialize_model(expect_exists=False)
        #model.train()
    elif FLAGS.mode == "eval":
        model = initialize_model(expect_exists=True)
        #model.predict()






if __name__ == "__main__":
    main()















