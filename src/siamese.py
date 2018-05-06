#from __future__ import print_function

#miscellaneous imports
import os, sys
import random
import pickle
import numpy as np
import matplotlib.pyplot
from matplotlib.pyplot import imshow
#from sklearn.decomposition import PCA
#from scipy.spatial import distance
from tqdm import tqdm
from functools import partial
#adding parent/util directory to the system path, so that any file in the util package can be imported
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'util'))
import psb_util_test as psb_util
#from main import MAIN_DIR, EXPERIMENTS_DIR, MODEL_CHECKPOINT_NAME

#keras related imports
import tensorflow as tf
import keras
from keras.preprocessing import image
# from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout, concatenate, Lambda
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras import regularizers

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
#DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir
MODEL_CHECKPOINT_NAME = 'model.hdf5'

#GLOB_FLAGS = {}
IMG_SHAPE = [224, 224, 3]
VGG_MODEL = keras.applications.VGG16(weights='imagenet', include_top=False)
#VGG_MODEL.summary()
FEAT_LAYERS = ['block4_pool', 'block5_pool']
SCORE_WEIGHTS = [10., 10.]
#infer how many dense layers used for prediction
PREDICTION_DENSE_DIMS = [1024, 1024]
drop_rate = 0.15




def get_feature_model(base_model=VGG_MODEL, output_layer_names=FEAT_LAYERS):
    assert type(output_layer_names) == list
    outputs = [base_model.get_layer(name).output for name in output_layer_names]
    feature_model = Model(inputs=base_model.input, outputs=outputs, name = 'Feature_Model')
    #feature_model.summary()
    #feature model is kept frozen
    for layer in feature_model.layers:
        layer.trainable=False
    return feature_model



def flatten_dense(feat_tensor, FLAGS, out_dim=1024, activation='relu', batch_norm=True):
    feat_tensor = Flatten()(feat_tensor)
    feat_tensor = dense_with_bn(feat_tensor, FLAGS, out_dim, activation, batch_norm)
    return feat_tensor

def dense_with_bn(feat_tensor, FLAGS, out_dim=1024, activation='relu', batch_norm=True, l2_reg=False):
    kernel_regularizer=None
    if l2_reg:
        kernel_regularizer = regularizers.l2(FLAGS.reg_rate)
    feat_tensor = Dense(out_dim, activation = 'linear', kernel_regularizer=kernel_regularizer)(feat_tensor)
    #use bn before activation
    if batch_norm: 
        feat_tensor = BatchNormalization()(feat_tensor)
    feat_tensor = Activation(activation)(feat_tensor)
    if FLAGS.dropout != 0:
        print "dropout is {}".format(FLAGS.dropout)
        feat_tensor = Dropout(FLAGS.dropout)(feat_tensor)
    return feat_tensor

def get_prediction(src_feat, tar_feat, FLAGS, name="", dense_dims=PREDICTION_DENSE_DIMS):
    combined_feat = concatenate([src_feat, tar_feat], name='merge_features'+name)
    for dense_dims in PREDICTION_DENSE_DIMS:
        combined_feat = dense_with_bn(combined_feat, FLAGS, out_dim=dense_dims, l2_reg=True)
    #A trick for bounded output range is to scale the target values between (0,1) and use sigmoid output + binary cross-entropy loss.
    prediction = Dense(1, activation = 'sigmoid')(combined_feat)
    return prediction

def aggregate_predictions(predictions):
    def weighted_average(a, weights):
        assert len(a) == len(weights)
        res = 0.0
        for m, n in zip(a, weights):
            res += m*n
        return res

    score = Lambda(weighted_average, arguments={'weights':SCORE_WEIGHTS})(predictions)
    return score


def build_model(FLAGS):
    #assign flags to global flag so other part of the code can use
    #GLOB_FLAGS = FLAGS
    src_in = Input(shape = IMG_SHAPE, name = 'src_input')
    tar_in = Input(shape = IMG_SHAPE, name = 'tar_input')
    feature_model = get_feature_model() 
    src_feats = feature_model(src_in) #list of features from all layers in FEAT_LAYERS
    tar_feats = feature_model(tar_in)
    assert len(src_feats) == len(FEAT_LAYERS)
    assert len(tar_feats) == len(FEAT_LAYERS)
    feat_pairs_by_layer = zip(src_feats, tar_feats)
    feat_pairs_dense = [(flatten_dense(src_feat, FLAGS, 1024, 'relu', True), flatten_dense(tar_feat, FLAGS, 1024, 'relu', True))\
                        for (src_feat, tar_feat) in feat_pairs_by_layer]
    predictions_by_layer = [get_prediction(src_feat, tar_feat, FLAGS, str(i)) for i, (src_feat, tar_feat) in enumerate(feat_pairs_dense)]
    assert len(predictions_by_layer) == len(FEAT_LAYERS)

    final_score = aggregate_predictions(predictions_by_layer)

    siamese_model = Model(inputs=[src_in, tar_in], outputs = [final_score], name = 'Similarity_Model')
    siamese_model.summary()
    return siamese_model



# src_feat = feature_model(src_in)
# tar_feat = feature_model(tar_in)

# src_feat = Flatten()(src_feat)
# tar_feat = Flatten()(tar_feat)
# src_feat = Dense(1024, activation = 'linear')(src_feat)
# src_feat = BatchNormalization()(src_feat)
# src_feat = Activation('relu')(src_feat)
# tar_feat = Dense(1024, activation = 'linear')(tar_feat)
# tar_feat = BatchNormalization()(tar_feat)
# tar_feat = Activation('relu')(tar_feat)

# combined_features = concatenate([src_feat, tar_feat], name = 'merge_features')
# combined_features = Dense(1024, activation = 'linear')(combined_features)
# combined_features = BatchNormalization()(combined_features)
# combined_features = Activation('relu')(combined_features)
# combined_features = Dense(1024, activation = 'linear')(combined_features)
# combined_features = BatchNormalization()(combined_features)
# combined_features = Activation('relu')(combined_features)
# #A trick for bounded output range is to scale the target values between (0,1) and use sigmoid output + binary cross-entropy loss.
# combined_features = Dense(1, activation = 'sigmoid')(combined_features)
# similarity_model = Model(inputs = [src_in, tar_in], outputs = [combined_features], name = 'Similarity_Model')
# similarity_model.summary()


# for layer in feature_model.layers:
#     layer.trainable=False
# # setup the optimization process
# #A trick for bounded output range is to scale the target values between (0,1) and use sigmoid output + binary cross-entropy loss.


tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs to train. 0 means train indefinitely")
tf.app.flags.DEFINE_integer("batch_size", 200, "batch_size")
tf.app.flags.DEFINE_integer("steps_per_epoch", 700, "batch_size")
#tf.app.flags.DEFINE_integer("validation_steps", 100, "batch_size")
tf.app.flags.DEFINE_float("dropout", 0.25, "Fraction of units randomly dropped on dense layers.")
tf.app.flags.DEFINE_float("reg_rate", 0.01, "Rate of regularization for each dense layers.")


FLAGS = tf.app.flags.FLAGS


def train(model, FLAGS):
    if FLAGS.gpu > 1: #utilize multiple gpus
        siamese_model = multi_gpu_model(model, gpus=FLAGS.gpu)
    else:
        siamese_model = model
    siamese_model.compile(optimizer='adam', loss = 'mean_squared_error', metrics = ['mae'])
    train_batch_generator = psb_util.batch_generator(data_dir="/mnt/data/data_batches", batch_size=FLAGS.batch_size)
    test_batch_generator = psb_util.batch_generator(data_dir="/mnt/data/data_batches/test", batch_size=FLAGS.batch_size)
    #steps_per_epoch = 28*5000/FLAGS.batch_size
    #validation_steps = 4*5000/FLAGS.batch_size
    #test set currently has 15,375 pairs

    train_dir = os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)
    assert os.path.exists(train_dir)

    checkpointer = ModelCheckpoint(filepath=os.path.join(train_dir, MODEL_CHECKPOINT_NAME), verbose=1, save_best_only=True)
    loss_history = siamese_model.fit_generator(train_batch_generator, 
                                                validation_data = test_batch_generator,
                                                steps_per_epoch = FLAGS.steps_per_epoch,
                                                validation_steps = FLAGS.validation_steps,
                                                epochs = FLAGS.num_epochs,
                                                verbose = True,
                                                callbacks = [checkpointer])

def main():
    siamese_model = build_model(FLAGS)
    siamese_model = multi_gpu_model(siamese_model, gpus=4)
    siamese_model.compile(optimizer='adam', loss = 'mean_squared_error', metrics = ['mae'])
    train_batch_generator = psb_util.batch_generator(data_dir="/mnt/data/data_batches", batch_size=FLAGS.batch_size)
    test_batch_generator = psb_util.batch_generator(data_dir="/mnt/data/data_batches/test", batch_size=FLAGS.batch_size)
    #steps_per_epoch = 28*5000/FLAGS.batch_size
    validation_steps = (3*5000+375)/FLAGS.batch_size
    loss_history = siamese_model.fit_generator(train_batch_generator, 
                                                validation_data = test_batch_generator,
                                                steps_per_epoch = FLAGS.steps_per_epoch,
                                                validation_steps = validation_steps,
                                                epochs = FLAGS.num_epochs,
                                                verbose = True)





if __name__ == "__main__":
    print("num_epochs is {}".format(FLAGS.num_epochs))
    print("batch_size is {}".format(FLAGS.batch_size))
    print("steps_per_epoch is {}".format(FLAGS.steps_per_epoch))
    #print("validation_steps is {}".format(FLAGS.validation_steps))

    main()


























