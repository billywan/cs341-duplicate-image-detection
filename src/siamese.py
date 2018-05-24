#from __future__ import print_function

#miscellaneous imports
import os, sys, time
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
from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout, concatenate, Lambda, GlobalAveragePooling2D, Dot
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import regularizers
import keras.backend as K

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
#DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir
MODEL_CHECKPOINT_NAME = 'model.hdf5'

#GLOB_FLAGS = {}
IMG_SHAPE = [224, 224, 3]
VGG_MODEL = keras.applications.VGG16(weights='imagenet', include_top=False)
#VGG_MODEL.summary()
RESNET_MODEL = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
FEAT_LAYERS = ['block4_pool', 'block5_pool']
SCORE_WEIGHTS = [0.5, 0.5]
#infer how many dense layers used for prediction
PREDICTION_DENSE_DIMS = [1024, 1024] #[1024, 1024]

def get_feat_layers(FLAGS):
    if FLAGS.base_model == "vgg16":
        return ['block4_pool', 'block5_pool']
    elif FLAGS.base_model == "resnet50":
        #(None, 55, 55, 64), (None, 55, 55, 256), (None, 28, 28, 512), (None, 14, 14, 1024), (None, 7, 7, 2048)
        return ['max_pooling2d_1', 'activation_10', 'activation_22', 'activation_40', 'activation_49']
    else:
        raise Exception("base_model {} invalid".format(FLAGS.base_model))

def get_feat_weights(FLAGS):
    if FLAGS.base_model == "vgg16":
        weights = [0.5, 0.5]
    elif FLAGS.base_model == "resnet50":
        weights = [0.0, 0.1, 0.2, 0.3, 0.4]
    else:
        raise Exception("base_model {} invalid".format(FLAGS.base_model))
    assert np.sum(weights) == 1.0
    return weights



#val_loss: 0.4423 - val_acc: 0.8237 - val_mean_absolute_error: 0.2716  (epoch 12/20)
#        weights = [0.1, 0.2, 0.2, 0.2, 0.3]

#loss: 0.1325 - acc: 0.9639 - mean_absolute_error: 0.0990 - val_loss: 0.4953 - val_acc: 0.8235 - val_mean_absolute_error: 0.2106
# (epoch 40/40)     dropout=0.5       weights = [0.0, 0.1, 0.2, 0.0, 0.7]

#weights = [0.0, 0.1, 0.2, 0.3, 0.4]
#loss: 0.1757 - acc: 0.9410 - mean_absolute_error: 0.1332 - val_loss: 0.4796 - val_acc: 0.8323 - val_mean_absolute_error: 0.2183



#loss: 0.0720 - acc: 0.9784 - mean_absolute_error: 0.0560 
#- val_loss: 0.3495 - val_acc: 0.8894 - val_mean_absolute_error: 0.1477
#Epoch 38/40

# def get_resnet_feat_layers():
#     #channel dims: 64, 256, 512, 1024, 2048
#     layers = ['max_pooling2d_1', 'activation_10', 'activation_22', 'activation_40', 'activation_49']
#     return layers

# def get_resnet_feat_weights():
#     weights = [0.1, 0.1, 0.1, 0.2, 0.5]
#     #channel dims: 64, 256, 512, 1024, 2048
#     return weights    

# def get_vgg_feat_layers():
#     layers = ['block4_pool', 'block5_pool']
#     return layers

# def get_vgg_feat_weights():
#     weights = [0.5, 0.5]
#     return weights


def get_base_model(FLAGS):
    if FLAGS.base_model == "vgg16":
        return VGG_MODEL
    elif FLAGS.base_model == "resnet50":
        return RESNET_MODEL
    else:
        raise Exception("base_model {} invalid".format(FLAGS.base_model))

def get_feature_model(FLAGS):
    output_layer_names = get_feat_layers(FLAGS)
    assert type(output_layer_names) == list
    base_model = get_base_model(FLAGS)
    outputs = [base_model.get_layer(name).output for name in output_layer_names]
    feature_model = Model(inputs=base_model.input, outputs=outputs, name = 'Feature_Model_'+FLAGS.base_model)
    #feature_model.summary()
    #feature model is kept frozen
    for layer in feature_model.layers:
        layer.trainable=False
    feature_model.summary()
    return feature_model



def flatten_dense(feat_tensor, FLAGS, out_dim=1024, activation='relu'):
    feat_tensor = Flatten()(feat_tensor)
    feat_tensor = dense_with_bn(feat_tensor, FLAGS, out_dim, activation)
    return feat_tensor

def dense_with_bn(feat_tensor, FLAGS, out_dim=1024, activation='relu', l2_reg=False):
    kernel_regularizer=None
    if l2_reg:
        kernel_regularizer = regularizers.l2(FLAGS.reg_rate)
    feat_tensor = Dense(out_dim, activation = 'linear', kernel_regularizer=kernel_regularizer)(feat_tensor)
    
    if FLAGS.dropout != 0:
        print "dropout is {}".format(FLAGS.dropout)
        feat_tensor = Dropout(FLAGS.dropout)(feat_tensor)
    
    #feat_tensor = Activation(activation)(feat_tensor)

    #use bn before activation, just as resnet
    print("Batch Norm {}".format(FLAGS.batch_norm))
    if FLAGS.batch_norm:
        print("Batch Norm True")
        feat_tensor = BatchNormalization()(feat_tensor)
    
    feat_tensor = Activation(activation)(feat_tensor)
    
    # if FLAGS.dropout != 0:
    #     print "dropout is {}".format(FLAGS.dropout)
    #     feat_tensor = Dropout(FLAGS.dropout)(feat_tensor)
    return feat_tensor

def get_prediction(src_feat, tar_feat, FLAGS, name="", dense_dims=PREDICTION_DENSE_DIMS):
    combined_feat = concatenate([src_feat, tar_feat], name='merge_features'+name)
    for dense_dims in PREDICTION_DENSE_DIMS:
        combined_feat = dense_with_bn(combined_feat, FLAGS, out_dim=dense_dims, l2_reg=True)
    #A trick for bounded output range is to scale the target values between (0,1) and use sigmoid output + binary cross-entropy loss.
    prediction = Dense(1, activation = 'sigmoid')(combined_feat)
    return prediction

def aggregate_predictions(FLAGS, predictions):
    # def weighted_average(a, weights):
    #     assert len(a) == len(weights)
    #     res = 0.0
    #     for m, n in zip(a, weights):
    #         res += m*n
    #     return res

    def weighted_average(a):
        weights = get_feat_weights(FLAGS)
        assert len(a) == len(weights)
        res = 0.0
        for m, n in zip(a, weights):
            res += m*n
        return res


    # def test(a, weights):
    #     return Dot(1)(a, weights)
    #score = Lambda(weighted_average, arguments={'weights':get_feat_weights(FLAGS)})(predictions)
    score = Lambda(weighted_average)(predictions)

    return score

def get_loss_function(FLAGS):
    def scaled_mse_loss(yTrue, yPred):
        return FLAGS.loss_scale*K.mean(K.square(yTrue - yPred))
    return scaled_mse_loss


def resnet_flatten_dense(feat_tensor, FLAGS, out_dim=1024, activation='relu'):
    feat_tensor = GlobalAveragePooling2D()(feat_tensor)
    #feat_tensor = Flatten()(feat_tensor)
    print feat_tensor.shape
    feat_tensor = dense_with_bn(feat_tensor, FLAGS, out_dim, activation)
    return feat_tensor

def build_model(FLAGS):
    #assign flags to global flag so other part of the code can use
    #GLOB_FLAGS = FLAGS
    src_in = Input(shape = IMG_SHAPE, name = 'src_input')
    tar_in = Input(shape = IMG_SHAPE, name = 'tar_input')
    feature_model = get_feature_model(FLAGS)
    src_feats = feature_model(src_in) #list of features from all layers in FEAT_LAYERS
    tar_feats = feature_model(tar_in)
    assert len(src_feats) == len(get_feat_layers(FLAGS))
    assert len(tar_feats) == len(get_feat_layers(FLAGS))
    feat_pairs_by_layer = zip(src_feats, tar_feats)
    if FLAGS.base_model == "vgg16":
        feat_pairs_dense = [(flatten_dense(src_feat, FLAGS, 1024, 'relu'), flatten_dense(tar_feat, FLAGS, 1024, 'relu'))\
                            for (src_feat, tar_feat) in feat_pairs_by_layer]
        predictions_by_layer = [get_prediction(src_feat, tar_feat, FLAGS, str(i)) for i, (src_feat, tar_feat) in enumerate(feat_pairs_dense)]
    elif FLAGS.base_model == "resnet50":
        feat_pairs_dense = [(resnet_flatten_dense(src_feat, FLAGS, 1024, 'relu'), resnet_flatten_dense(tar_feat, FLAGS, 1024, 'relu'))\
                            for (src_feat, tar_feat) in feat_pairs_by_layer]
        predictions_by_layer = [get_prediction(src_feat, tar_feat, FLAGS, str(i)) for i, (src_feat, tar_feat) in enumerate(feat_pairs_dense)]
    else:
        raise Exception("base_model {} invalid".format(FLAGS.base_model))
    assert len(predictions_by_layer) == len(get_feat_layers(FLAGS))
    
    # for item in predictions_by_layer:
    #     print item.shape
    # for i, score in enumerate(predictions_by_layer):
    #     predictions_by_layer[i] = K.print_tensor(score, message='score {} = '.format(i))

    final_score = aggregate_predictions(FLAGS, predictions_by_layer)

    siamese_model = Model(inputs=[src_in, tar_in], outputs = [final_score], name = 'Similarity_Model')
    siamese_model.summary()
    return siamese_model



# for layer in feature_model.layers:
#     layer.trainable=False
# # setup the optimization process
# #A trick for bounded output range is to scale the target values between (0,1) and use sigmoid output + binary cross-entropy loss.






# tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs to train. 0 means train indefinitely")
# tf.app.flags.DEFINE_integer("batch_size", 200, "batch_size")
# tf.app.flags.DEFINE_integer("steps_per_epoch", 700, "batch_size")
# #tf.app.flags.DEFINE_integer("validation_steps", 100, "batch_size")
# tf.app.flags.DEFINE_float("dropout", 0.25, "Fraction of units randomly dropped on dense layers.")
# tf.app.flags.DEFINE_float("reg_rate", 0.001, "Rate of regularization for each dense layers.")
# tf.app.flags.DEFINE_float("loss_scale", 20, "Scale factor to apply on prediction loss; used to make the prediction loss comparable to l2 weight regularization")
# tf.app.flags.DEFINE_string("base_model", "resnet50" , "base model for feature extraction. Currently support resnet50 and vgg16")
# tf.app.flags.DEFINE_boolean("batch_norm", True , "whether or not to use batch normalization on each dense layer")


# FLAGS = tf.app.flags.FLAGS




# class PrintScores(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         K.function()
        

    








def compile_model(model, FLAGS):
    loss_func = get_loss_function(FLAGS)
    model.compile(optimizer='adam', loss = loss_func, metrics = ['accuracy', 'mae'])



def train(model, FLAGS):
    if FLAGS.gpu > 1: #utilize multiple gpus
        siamese_model = multi_gpu_model(model, gpus=FLAGS.gpu)
    else:
        siamese_model = model

    compile_model(siamese_model, FLAGS)

    train_batch_generator = psb_util.batch_generator(data_dir=FLAGS.train_data_dir, batch_size=FLAGS.batch_size, shuffle_files=False)
    test_batch_generator = psb_util.batch_generator(data_dir=FLAGS.test_data_dir, batch_size=FLAGS.batch_size)
    #steps_per_epoch = 28*5000/FLAGS.batch_size
    #validation_steps = 4*5000/FLAGS.batch_size
    #test set currently has 15,375 pairs

    train_dir = os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)
    assert os.path.exists(train_dir)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=4, min_lr=0.0001)
    checkpointer = ModelCheckpoint(filepath=os.path.join(train_dir, MODEL_CHECKPOINT_NAME), verbose=1, save_best_only=True)
    #checkpointer.set_model(model) 
    loss_history = siamese_model.fit_generator(train_batch_generator,
                                                validation_data = test_batch_generator,
                                                steps_per_epoch = FLAGS.steps_per_epoch,
                                                validation_steps = FLAGS.validation_steps,
                                                epochs = FLAGS.num_epochs,
                                                verbose = True,
                                                max_queue_size=1,
                                                callbacks = [reduce_lr])#, checkpointer])
    siamese_model.save(os.path.join(train_dir, MODEL_CHECKPOINT_NAME))


def predict(model, FLAGS):
    eval_batch_generator = psb_util.batch_generator(data_dir=FLAGS.eval_data_dir, batch_size=FLAGS.batch_size)
    [X1, X2], y = next(eval_batch_generator)
    predictions = siamese_model.predict([X1, X2], batch_size = FLAGS.batch_size, verbose=1)
    print predictions.shape, y.shape
    for i in range(30):
        print "predictions vs ", predictions[i], y[i]
    # print "predictions[:30]", predictions[:30]
    # print "y[:30]", y[:30]
    # predictions = siamese_model.predict_generator(test_batch_generator, 
    #                                                 steps=21, 
    #                                                 max_queue_size=1, 
    #                                                 workers=4, 
    #                                                 use_multiprocessing=True, 
    #                                                 verbose=1)

def main():
    siamese_model = build_model(FLAGS)
    siamese_model = multi_gpu_model(siamese_model, gpus=4)
    #siamese_model.compile(optimizer='adam', loss = 'mean_squared_error', metrics = ['mae'])
    compile_model(siamese_model, FLAGS)
    data_dir = "/mnt/data2/data_batches_01_12"
    test_dir = os.path.join(data_dir, "test")
    print("data_dir is {}".format(data_dir))
    print("test_dir is {}".format(test_dir))
    train_batch_generator = psb_util.batch_generator(data_dir=data_dir, batch_size=FLAGS.batch_size, shuffle_files=True)
    test_batch_generator = psb_util.batch_generator(data_dir=test_dir, batch_size=FLAGS.batch_size, shuffle_files=False)
    #steps_per_epoch = 28*5000/FLAGS.batch_size
    #validation_steps = (3*5000+375)/FLAGS.batch_size
    validation_steps = 28#43#21#60

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                                    patience=4, min_lr=0.0001)
    loss_history = siamese_model.fit_generator(train_batch_generator,
                                                validation_data = test_batch_generator,
                                                steps_per_epoch = FLAGS.steps_per_epoch,
                                                validation_steps = validation_steps,
                                                epochs = FLAGS.num_epochs,
                                                verbose = True, 
                                                callbacks=[reduce_lr]) 
                                                
    t0 = time.time()
    predictions = siamese_model.predict_generator(test_batch_generator, 
                                                    steps=21, 
                                                    max_queue_size=10, 
                                                    workers=4, 
                                                    use_multiprocessing=True, 
                                                    verbose=1)
    t1 = time.time()
    print("time taken {}".format(t1-t0))
    print predictions.shape
    print predictions[:10]



if __name__ == "__main__":
    print("num_epochs is {}".format(FLAGS.num_epochs))
    print("batch_size is {}".format(FLAGS.batch_size))
    print("steps_per_epoch is {}".format(FLAGS.steps_per_epoch))
    #print("validation_steps is {}".format(FLAGS.validation_steps))

    main()
