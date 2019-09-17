# PARAMETERS
learning_rate = 0.001
batch_size = 100
train_iterations = 100
multi_gpu = False

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
from keras.utils.training_utils import multi_gpu_model
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
        img = img_to_array(img)
        mask = normalize_mask(mask)
        training_data.extend([img, mask, percent])
        all_training_data.append(training_data)
        print(f'{idx} done out of {len(all_groups)}', end='\r')
        if idx == 10:
            break
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

def simplify_mask(mask_file):
    """Simplify mask from RGB to int array
    0, 0, 0 = Empty = Nothing = 0
    255, 255, 0 = Yellow = building = 1
    255, 0, 255 = Purple = road = 2
    255, 0, 0 = Red = water = 3
    """
    mask = cv2.imread(mask_file)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_array = np.zeros((mask.shape[0], mask.shape[1]))
    for x in range(mask.shape[0]):  # Width
        for y in range(mask.shape[1]):  # Height
            rgb = list(mask[x,y,:])
            if any(value > 0 for value in rgb):
                if rgb == [255, 255, 0]:  # Building
                    mask_array[x,y] = 1
                elif rgb == [255, 0, 255]:  # Road
                    mask_array[x,y] = 2
                elif rgb == [255, 0, 0]:  # Road
                    mask_array[x,y] = 3
                else:
                    print("ERROR")
                    print(mask_file)
                    print(rgb, x, y)
                    quit()

def fit_model(model, all_training_data):
    for i in range(train_iterations):
        for train_data in all_training_data:
            X = np.array(train_data[0])
            Y = np.array(train_data[1])
            X = np.expand_dims(X, axis=0)
            Y = np.expand_dims(Y, axis=0)
            history = model.fit(X, Y,
                                epochs=2)

def create_model():
    logger.info('Creating model')
    model = Sequential()
    model.add(Convolution2D(128, 20, 20, input_shape=(1024, 1024, 3), activation='relu'))
    model.add(Convolution2D(128, 15, 15, activation='sigmoid'))
    model.add(Convolution2D(128, 10, 10, activation='sigmoid'))
    model.add(Convolution2D(128, 5, 5, activation='sigmoid'))
    model.add(AveragePooling2D(3,3))
    model.add(Convolution2D(128, 32, 32, activation='sigmoid'))
    model.add(Convolution2D(128, 40, 40, activation='sigmoid'))
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
    devices = device_lib.list_local_devices()
    gpu_count = 0
    for device in devices:
        if device.device_type == "GPU":
            gpu_count += 1
    logger.info(f"Found {gpu_count} usable gpus")
    if multi_gpu:
        model = multi_gpu_model(model, gpus=gpu_count)
    adam = Adam(lr=learning_rate) # lr 0.001 --> default adam
    sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True) # lr 0.001 --> default adam
    rmsprop = RMSprop(lr=learning_rate)
    model.compile(loss='mse', optimizer=adam, metrics = ['accuracy'])
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
