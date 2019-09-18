# PARAMETERS
learning_rate = 0.001
batch_size = 2
epochs = 2
train_iterations = 100
use_gpu = True

import logging
encoding = "utf-8-sig"  # or utf-8
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler('logger.log', encoding=encoding)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('-' * 20 + 'START OF RUN' + '-' * 20)

# -*- coding: utf-8 -*-
logger.info('Loading modules')
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # Stops tf allocating all memory
session = tf.Session(config=config)

# Sets keras backend to tensorflow
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
if not use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np
import glob
import keras
import json
import os
import random
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Embedding, Activation, BatchNormalization
from keras.layers import Convolution1D, Convolution2D, MaxPooling2D, AveragePooling2D, LSTM
from keras.layers import GlobalAveragePooling2D, UpSampling2D, TimeDistributed, Reshape
from keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.client import device_lib
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
import pdb
import time
import matplotlib
import re

# Needed to change plot position while calculating. NEEDS TO ADDED BEFORE pyplot import
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def log_func(txt):
	logger.info(txt)

def main():
    model = create_model()
    model = compile_model(model)
    all_groups = group_images()
    random.shuffle(all_groups)  # Shuffle data set
    all_training_data = preprocess_data(all_groups)
    fit_model(model, all_training_data)

def preprocess_data(all_groups):
    logger.info('Preprocessing data')
    all_training_data = []
    for idx, group in enumerate(all_groups):
        training_data = []
        img, mask, percent = group
        img_array = img_to_array(img)
        if img_array.shape != (1024, 1024, 3):
            # Note: Perhaps we could make it so that the CNN takes any size, but the output size would be all over the place
            logger.warning(f'Image {img} has shape {img_array.shape}, not shape (1024, 1024, 3). Skipping')
            continue
        mask_array = normalize_mask(mask)
        training_data.extend([img_array, mask_array, percent])
        all_training_data.append(training_data)
        if idx > 40:
            break
        print(f'{idx} done out of {len(all_groups)}', end='\r')
    return all_training_data

def normalize_mask(mask_file):
    mask = cv2.imread(mask_file)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask[mask == 255] = 1
    return mask

def img_to_array(img_file):
    img = cv2.imread(img_file)  # Preprocess image as well and simplify?
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def fit_model(model, all_training_data):
    logger.info('Creating a numpy array')
    X = [all_training_data[i][0] for i in range(len(all_training_data))]
    Y = [all_training_data[i][1] for i in range(len(all_training_data))]
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape)
    for iteration in range(train_iterations):
        logger.info('-' * 10 + f'Iteration {iteration+1} out of {train_iterations}' + '-' * 10 )
        try:
            history = model.fit(X, Y,
                                epochs=epochs, batch_size=batch_size,
                                validation_split=0.1, callbacks=[],
                                shuffle=True, verbose=1)

            logger.info('Loss: ' + str(round(history.history['loss'][-1], 4)) + ' Val Loss: '
                        + str(round(history.history['val_loss'][-1], 4)))
            prediction = model.predict(np.expand_dims(X[0], axis=0))
            predicted_mask = predict_to_mask(prediction)
            predicted_mask = np.squeeze(predicted_mask, axis=0)
            plt.subplot(1,2,1)
            plt.title('Real mask')
            plt.imshow(predict_to_mask(Y[0]))
            plt.subplot(1,2,2)
            plt.title('Predicted mask')
            plt.imshow(predicted_mask)
            plt.show()
        except KeyboardInterrupt:
            model.save('current_weights.h5', overwrite=True)

def predict_to_mask(mask):
    mask[mask > 0.8] = 255
    return mask

def create_model():
    logger.info('Creating model')
    model = Sequential()
    model.add(Convolution2D(128, 20, 20, input_shape=(1024, 1024, 3), activation='relu'))
    model.add(Convolution2D(6, 15, 15, activation='sigmoid'))
    model.add(Convolution2D(6, 5, 5, activation='sigmoid'))
    model.add(AveragePooling2D(3,3))
    model.add(Convolution2D(3, 32, 32, activation='sigmoid'))
    model.add(Convolution2D(3, 40, 40, activation='sigmoid'))
    model.add(Convolution2D(20, 4, 4, activation='sigmoid'))
    model.add(UpSampling2D(2))
    model.add(AveragePooling2D(2,2))
    model.add(UpSampling2D(4))
    model.add(Dense(3, activation='sigmoid'))  # Softmax since classification
    #model.add(Reshape((1024, 1024, 3)))
    logger.info(model.summary(print_fn=log_func))
    logger.info("Input shape: " + str(model.input_shape))
    logger.info("Output shape: " + str(model.output_shape))
    return model

def compile_model(model):
    logger.info('Compiling model')
    adam = Adam(lr=learning_rate) # lr 0.001 --> default adam
    sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True) # lr 0.001 --> default adam
    rmsprop = RMSprop(lr=learning_rate)
    model.compile(loss='mse', optimizer=adam, metrics = ['binary_accuracy'])
    return model

def group_images():
    logger.info('Grouping images')
    images = glob.glob('Training_dataset/Images/*')
    masks = glob.glob('Training_dataset/Masks/all/*')
    percentages = glob.glob('Training_dataset/Percentages/*')

    all_groups = []
    for idx, img in enumerate(images):
        group = []
        group.append(img)
        img_basename = os.path.basename(img).split('.')[0]
        for mask in masks:
            mask_basename = os.path.basename(mask).split('.')[0]
            if mask_basename == img_basename:
                group.append(mask)
                break
        for percent in percentages:
            percent_basename = os.path.basename(percent).split('.')[0]
            if percent_basename == img_basename:
                group.append(percent)
                break
        if len(group) == 3:
            all_groups.append(group)
        else:
            logger.warning(f'Image {img} not matched with mask and/or percent')
        print(f'{idx} done out of {len(images)}', end='\r')
    return all_groups

if __name__ == '__main__':
    main()
