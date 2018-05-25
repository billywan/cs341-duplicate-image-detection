from __future__ import absolute_import
from __future__ import division

import os
import io
import json
import sys
import logging
import argparse
import tensorflow as tf
import keras
from keras.models import load_model

import siamese


MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir
#MODEL_CHECKPOINT_NAME = 'model_weights.{epoch:02d}-{val_mean_absolute_error:.4f}.hdf5'
MODEL_CHECKPOINT_NAME = 'model_weights.hdf5'

DATA_DIR = "/mnt/data2/data_batches_01_12"
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")
EVAL_DATA_DIR = os.path.join(DATA_DIR, "eval")

####################################################################################################################################
# High-level options
tf.app.flags.DEFINE_integer("gpu", 4, "How many GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / eval")
tf.app.flags.DEFINE_string("base_model", "resnet50" , "base model for feature extraction. Currently support resnet50 and vgg16")
tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train. 0 means train indefinitely")
tf.app.flags.DEFINE_integer("steps_per_epoch", 200, "batch_size")
tf.app.flags.DEFINE_integer("validation_steps", 20, "batch_size")
####################################################################################################################################
# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on dense layers.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use")
tf.app.flags.DEFINE_float("reg_rate", 0.0, "Rate of regularization for each dense layers.")
tf.app.flags.DEFINE_float("loss_scale", 10, "Scale factor to apply on prediction loss; used to make the prediction loss comparable to l2 weight regularization")
tf.app.flags.DEFINE_boolean("batch_norm", True , "whether or not to use batch normalization on each dense layer")
####################################################################################################################################
# Data paths
tf.app.flags.DEFINE_string("train_data_dir", DATA_DIR, "Default training data path")
tf.app.flags.DEFINE_string("test_data_dir", TEST_DATA_DIR, "Default testing data path")
tf.app.flags.DEFINE_string("eval_data_dir", EVAL_DATA_DIR, "Default evaluation data path same as default testing data path")
####################################################################################################################################



def get_flags():

    parser = argparse.ArgumentParser(description='Siamese Neural Network')
    parser.add_argument('--gpu', dest='gpu', nargs='?', default=4, type=int,
                                help='How many GPU to use, if you have multiple.')
    parser.add_argument('--mode', dest='mode', nargs='?', default='train', type=str,
                                help='Available modes: train / eval')
    parser.add_argument('--experiment_name', dest='experiment_name', nargs='?', default='', type=str,
                                help='Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment')
    parser.add_argument('--num_epochs', dest='num_epochs', nargs='?', default=0, type=int,
                                help='Number of epochs to train. 0 means train indefinitely')
    parser.add_argument('--steps_per_epoch', dest='steps_per_epoch', nargs='?', default=200, type=int,
                                help='steps_per_epoch')
    parser.add_argument('--validation_steps', dest='validation_steps', nargs='?', default=20, type=int,
                                help='validation_steps')
    parser.add_argument('--learning_rate', dest='learning_rate', nargs='?', default=0.01, type=float,
                                help='learning_rate')
    parser.add_argument('--dropout', dest='dropout', nargs='?', default=0.25, type=float,
                                help='dropout')
    parser.add_argument('--batch_size', dest='batch_size', nargs='?', default=100, type=int,
                                help='batch_size')

    parser.add_argument('--reg_rate', dest='reg_rate', nargs='?', default=0.0, type=float,
                                help='reg_rate')
    parser.add_argument('--loss_scale', dest='loss_scale', nargs='?', default=10, type=int,
                                help='loss_scale')
    parser.add_argument('--base_model', dest='base_model', nargs='?', default='resnet50', type=str,
                                help='base_model')
    parser.add_argument('--batch_norm', dest='batch_norm', nargs='?', default=True, type=bool,
                                help='batch_norm')


    parser.add_argument('--train_data_dir', dest='train_data_dir', nargs='?', default=DATA_DIR, type=str,
                                help='Default training data path')
    parser.add_argument('--test_data_dir', dest='test_data_dir', nargs='?', default=TEST_DATA_DIR, type=str,
                                help='Default testing data path')
    parser.add_argument('--eval_data_dir', dest='eval_data_dir', nargs='?', default=EVAL_DATA_DIR, type=str,
                                help='eval_data_dir')

    FLAGS = parser.parse_args()
    return FLAGS


#############################################################################
FLAGS = tf.app.flags.FLAGS
#############################################################################


def initialize_model(FLAGS, expect_exists=False):
    
    train_dir = os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)
    model_file_path = os.path.join(train_dir, MODEL_CHECKPOINT_NAME)
    if not os.path.exists(model_file_path):
        if expect_exists:
            raise Exception("No existing model found at %s"%model_file_path)
        else:
            if not os.path.exists(train_dir):
                print("Making training directory at {}".format(train_dir))
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

    #FLAGS = get_flags()

    # Print an error message if you've entered flags incorrectly
    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: %s" % unused_argv)

    # Check for Python 2
    if sys.version_info[0] != 2:
        raise Exception("ERROR: You must use Python 2 but you are running Python %i" % sys.version_info[0])

    if not FLAGS.experiment_name:
        raise Exception("ERROR: You need to specify --experiment_name") 

    if FLAGS.mode == "train":
        model = initialize_model(FLAGS, expect_exists=False)
        siamese.train(model, FLAGS)
    elif FLAGS.mode == "eval":
        model = initialize_model(FLAGS, expect_exists=True)
        siamese.predict(model, FLAGS)





if __name__ == "__main__":
    #main()
    tf.app.run()















