#from __future__ import print_function

#miscellaneous imports
import os, sys, time
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot
from matplotlib.pyplot import imshow
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
from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout, concatenate, Lambda, GlobalAveragePooling2D, Dot, Multiply, Add
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import regularizers
import keras.backend as K
from keras.models import model_from_json

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir
#MODEL_CHECKPOINT_NAME = 'model_weights.{epoch:02d}-{val_mean_absolute_error:.4f}.hdf5'
MODEL_CHECKPOINT_NAME = 'model_weights.hdf5'

IMG_SHAPE = [224, 224, 3]
VGG_MODEL = keras.applications.VGG16(weights='imagenet', include_top=False)
RESNET_MODEL = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
FEAT_LAYERS = ['block4_pool', 'block5_pool']
SCORE_WEIGHTS = [0.5, 0.5]
#infer how many dense layers used for prediction
PREDICTION_DENSE_DIMS = [1024, 1024] 

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
    #feature model is kept frozen
    for layer in feature_model.layers:
        layer.trainable=False
    feature_model.summary()
    return feature_model



def flatten_dense(feat_tensor, FLAGS, out_dim=1024, activation='relu'):
    feat_tensor = Flatten()(feat_tensor)
    feat_tensor = dense_with_bn(feat_tensor, FLAGS, out_dim, activation)
    return feat_tensor

def dense_with_bn(feat_tensor, FLAGS, out_dim=1024, activation='relu'):
    kernel_regularizer = regularizers.l2(FLAGS.reg_rate) if FLAGS.reg_rate else None
    feat_tensor = Dense(out_dim, activation = 'linear', kernel_regularizer=kernel_regularizer)(feat_tensor)
    
    if FLAGS.dropout != 0:
        print "dropout is {}".format(FLAGS.dropout)
        feat_tensor = Dropout(FLAGS.dropout)(feat_tensor)
    
    #use bn before activation, just as resnet
    print("Batch Norm {}".format(FLAGS.batch_norm))
    if FLAGS.batch_norm:
        print("Batch Norm True")
        feat_tensor = BatchNormalization()(feat_tensor)
    
    feat_tensor = Activation(activation)(feat_tensor)

    return feat_tensor

def get_prediction(src_feat, tar_feat, FLAGS, name="", dense_dims=PREDICTION_DENSE_DIMS):
    combined_feat = concatenate([src_feat, tar_feat], name='merge_features'+name)
    for dense_dims in PREDICTION_DENSE_DIMS:
        combined_feat = dense_with_bn(combined_feat, FLAGS, out_dim=dense_dims)
    #A trick for bounded output range is to scale the target values between (0,1) and use sigmoid output + binary cross-entropy loss.
    prediction = Dense(1, activation = 'sigmoid')(combined_feat)
    return prediction

def aggregate_predictions(FLAGS, predictions):
    def weighted_average(a):
        weights = get_feat_weights(FLAGS)
        assert len(a) == len(weights)
        res = weights[0]*a[0]
        for i in range(1, len(a)):
            res += weights[i]*a[i]
        return res

    def test(predictions_by_layer):
        weights = get_feat_weights(FLAGS)
        assert len(predictions_by_layer) == len(weights)
        #k_weights = K.variable(weights)
        final_score = weights[0]*predictions_by_layer[0]
        for i in range(1, len(predictions_by_layer)):
            final_score += weights[i]*predictions_by_layer[i]
        return final_score

    #score = Lambda(weighted_average, arguments={'weights':get_feat_weights(FLAGS)})(predictions)
    score = Lambda(weighted_average, name="final_lambda_layer")(predictions)
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
    
    final_score = aggregate_predictions(FLAGS, predictions_by_layer)
    siamese_model = Model(inputs=[src_in, tar_in], outputs = [final_score], name = 'Similarity_Model')
    siamese_model.summary()
    return siamese_model



def compile_model(model, FLAGS):
    loss_func = get_loss_function(FLAGS)
    if FLAGS.gpu > 1: #utilize multiple gpus
        model = ModelMGPU(model , FLAGS.gpu)
        #siamese_model = multi_gpu_model(model, gpus=FLAGS.gpu)
    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy', 'mae'])
    #"binary_crossentropy"
    return model


# A wrapper class for multi-gpu model. Only single GPU model can use ModelCheckpoint call back to save weights
# So overload the __getattribute__ method so that when getting attribute name "load" and "save", we retrieve single GPU model's attrname
class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)



def train(model, FLAGS):

    model = compile_model(model, FLAGS)

    train_batch_generator = psb_util.batch_generator(data_dir=FLAGS.train_data_dir, batch_size=FLAGS.batch_size, shuffle_files=True)
    test_batch_generator = psb_util.batch_generator(data_dir=FLAGS.test_data_dir, batch_size=FLAGS.batch_size, shuffle_files=False)
    #steps_per_epoch = 28*5000/FLAGS.batch_size
    #validation_steps = 4*5000/FLAGS.batch_size
    #test set currently has 15,375 pairs
    #[X1, X2], y = next(test_batch_generator)


    train_dir = os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)
    assert os.path.exists(train_dir)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=4, min_lr=0.00001)
    checkpointer = ModelCheckpoint(filepath=os.path.join(train_dir, MODEL_CHECKPOINT_NAME), monitor='val_mean_absolute_error', verbose=1, save_best_only=True, save_weights_only=True)
    #checkpointer.set_model(model) 
    loss_history = model.fit_generator(train_batch_generator,
                                        validation_data = test_batch_generator,
                                        steps_per_epoch = FLAGS.steps_per_epoch,
                                        validation_steps = FLAGS.validation_steps,
                                        epochs = FLAGS.num_epochs,
                                        verbose = True,
                                        max_queue_size=1,
                                        callbacks = [reduce_lr, checkpointer])
    
    #model.save_weights(os.path.join(train_dir, MODEL_CHECKPOINT_NAME))

    loaded_model = build_model(FLAGS)
    loaded_model.load_weights(os.path.join(train_dir, MODEL_CHECKPOINT_NAME))
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("model.h5")
    # print("Saved model to disk")

    # # -------------- load the saved model --------------
    # # load json and create model
    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights("model.h5")
    # print("Loaded model from disk")

    # evaluate loaded model on test data
    # loaded_model.compile(loss='binary_crossentropy',
    #                         optimizer='adam',
    #                         metrics=['accuracy', 'mae'])
    [X1, X2], y = psb_util.load_data_file("/mnt/data2/data_batches_01_12/test/test_data_batch_000", label=True)
    loaded_model = compile_model(loaded_model, FLAGS)
    scores = loaded_model.evaluate([X1, X2], y, verbose=1)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    print('Test mae:', scores[2])


def predict_data_file(model, file_path, FLAGS):
    print "Predicting data in {} ...".format(file_path)
    data = psb_util.load_data_file(file_path, expect_label=False)
    [X1, X2], _ = data # At prediction time, no labels are available
    predictions = model.predict([X1, X2], batch_size = FLAGS.batch_size, verbose=1)
    print "Done predicting over {} example pairs.".format(len(predictions))
    return predictions

def predict(model, FLAGS):
    model = compile_model(model, FLAGS)
    #eval_batch_generator = psb_util.batch_generator(data_dir=FLAGS.eval_data_dir, batch_size=FLAGS.batch_size)
    #[X1, X2], y = next(eval_batch_generator)

    predictions = {}
    if os.path.isdir(FLAGS.eval_data_path):
        data_files = [os.path.join(FLAGS.eval_data_path, file) for file in os.listdir(FLAGS.eval_data_path)]
        for file in data_files:
            predictions[file] = predict_data_file(model, file, FLAGS)
    else: 
        predictions[FLAGS.eval_data_path] = predict_data_file(model, FLAGS.eval_data_path, FLAGS)

    for k, v in predictions.iteritems():
        print "predicted {num} examples in {file}".format(num=v.shape, file=k)
    print "Prediction finished."
    
    #dump the predictions into a file.
    
    # print "predictions[:30]", predictions[:30]
    # print "y[:30]", y[:30]
    # predictions = siamese_model.predict_generator(test_batch_generator, 
    #                                                 steps=21, 
    #                                                 max_queue_size=1, 
    #                                                 workers=4, 
    #                                                 use_multiprocessing=True, 
    #                                                 verbose=1)



def eval_data_file(model, file_path, FLAGS):
    print "\nEvaluating data in {}...".format(file_path)
    data = psb_util.load_data_file(file_path, expect_label=True)
    [X1, X2], y = data # At evaluation time, labels are provided
    scores = model.evaluate([X1, X2], y, batch_size = FLAGS.batch_size, verbose=1)
    #print "Done evaluating over {} examples(pairs).".format(len(predictions))
    return scores

def eval(model, FLAGS):
    model = compile_model(model, FLAGS)
    evaluations = {}
    if os.path.isdir(FLAGS.eval_data_path):
        print "You supplied the directory {} for evaluation.".format(FLAGS.eval_data_path)
        data_files = [os.path.join(FLAGS.eval_data_path, file) for file in os.listdir(FLAGS.eval_data_path)]
        for file in data_files:
            evaluations[file] = eval_data_file(model, file, FLAGS)
    else: 
        print "You supplied a single file for evaluation"
        evaluations[FLAGS.eval_data_path] = eval_data_file(model, FLAGS.eval_data_path, FLAGS)

    print "Evaluation finished, printing results..."
    print "="*60
    results = []
    index = []
    for filename, scores in evaluations.iteritems():
        results.append(list(scores))
        index.append(os.path.basename(filename))
    columns = list(model.metrics_names)
    results_df = pd.DataFrame(results, columns=columns, index=index)
    print results_df
    print "="*60
    #dump the result into a file.


def main():
    siamese_model = build_model(FLAGS)
    siamese_model = multi_gpu_model(siamese_model, gpus=4)
    #siamese_model.compile(optimizer='adam', loss = 'mean_squared_error', metrics = ['mae'])
    siamese_model = compile_model(siamese_model, FLAGS)
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
                                                
    # t0 = time.time()
    # predictions = siamese_model.predict_generator(test_batch_generator, 
    #                                                 steps=21, 
    #                                                 max_queue_size=10, 
    #                                                 workers=4, 
    #                                                 use_multiprocessing=True, 
    #                                                 verbose=1)
    # t1 = time.time()
    # print("time taken {}".format(t1-t0))
    # print predictions.shape
    # print predictions[:10]



if __name__ == "__main__":
    print("num_epochs is {}".format(FLAGS.num_epochs))
    print("batch_size is {}".format(FLAGS.batch_size))
    print("steps_per_epoch is {}".format(FLAGS.steps_per_epoch))
    #print("validation_steps is {}".format(FLAGS.validation_steps))
    main()
