# PARAMETERS
learning_rate = 0.001
batch_size = 10
epochs = 2
train_iterations = 100
use_gpu = True
tensorflow_debug = False
model_path = 'mask_rcnn_coco.h5'

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
if not tensorflow_debug:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
import multiprocessing
from multiprocessing import freeze_support, RLock
import itertools
import mrcnn
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config as mrcnn_config
import tqdm
import pickle
from sklearn.model_selection import train_test_split
import imgaug

# Needed to change plot position while calculating. NEEDS TO ADDED BEFORE pyplot import
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# https://www.pyimagesearch.com/2019/06/10/keras-mask-r-cnn/
# https://www.analyticsvidhya.com/blog/2019/07/computer-vision-implementing-mask-r-cnn-image-segmentation/
CLASS_NAMES = ['bg',
               'building',
               'road',
               'water']
COLORS = [[0, 0, 0],
          [255, 255, 0],
          [255, 0, 255],
          [255, 0, 0]]
"""
0, 0, 0 = Empty = Nothing = 0
255, 255, 0 = Yellow = building = 1
255, 0, 255 = Purple = road = 2
255, 0, 0 = Red = water = 3
"""
IDS = [0,1,2,3]

class Config(mrcnn_config):
    NAME = "aerial_drone_map"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # batch size
    NUM_CLASSES = len(CLASS_NAMES)  # Needs to include bg
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    STEPS_PER_EPOCH = 2000
    IMAGES_PER_GPU = 1
    # Use smaller anchors because our image and objects are small
    # NOTE Different anchor sizes are suitable for houses/water/roads.
    RPN_ANCHOR_SCALES = (64, 128, 256, 512, 1024)  # anchor side in pixels
    # Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100
    VALIDATION_STEPS = 25

class DroneDataset(utils.Dataset):
    def load_data(self, train_data):
        self.train_data = train_data
        self.image_id_to_path = {}
        for idx, img in enumerate(train_data.keys()):
            self.image_id_to_path[idx] = img
        self.add_class("object", IDS[1], "building")
        self.add_class("object", IDS[2], "road")
        self.add_class("object", IDS[3], "water")
        for idx, file_path in enumerate(train_data.keys()):
            self.add_image("object",
                           image_id=file_path,  # For some reason, this won't work
                           path=file_path)
            #self.train_data[idx+1] = self.train_data.pop(file_path)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        path = self.image_id_to_path[image_id]
        mask = self.train_data[path][0]
        class_ids = self.train_data[path][1]
        return mask, class_ids

    def image_reference(self, image_id):  # Needed
        """Return the path of the image."""
        return self.image_id_to_path[image_id]

def log_func(txt):
	logger.info(txt)

def main():
    if not os.path.isfile('preprocessed_data.pkl'):
        all_groups = group_images()
        random.shuffle(all_groups)  # Shuffle data sets
        pool = multiprocessing.Pool(multiprocessing.cpu_count(),
                                    initializer=tqdm.tqdm.set_lock,
                                    initargs=(RLock(),))
        # NOTE: bad way of doing it. Spikes at 32 Gb or RAM. Perhaps a generator would be better?
        logger.info('Preprocessing data')
        all_groups = list_chunk(all_groups, multiprocessing.cpu_count())
        id_and_groups = [(id+1, group) for id, group in enumerate(all_groups)]
        train_data = pool.map(preprocess_data, id_and_groups)
        pool.close()
        train_data = merge_dicts(train_data)
        with open('preprocessed_data.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        #train_data = preprocess_data([1, all_groups])
    else:
        with open('preprocessed_data.pkl', 'rb') as f:
            train_data = pickle.load(f)
    train_data, val_data = dict_split(train_data, 0.7)
    logger.info(f'Training: {len(list(train_data.keys()))} images, Validation: {len(list(val_data.keys()))} images')
    dataset_train = prepare_dataset(train_data)
    dataset_val = prepare_dataset(val_data)
    model, predict_model = create_model()
    fit_model(model, predict_model, dataset_train, dataset_train)

def prepare_dataset(data_dict):
    dataset = DroneDataset()
    dataset.load_data(data_dict)
    dataset.prepare()
    return dataset

def merge_dicts(dict_list):
    first_dict = dict_list[0]
    for d in dict_list[1:]:
        first_dict.update(d)
    return first_dict

def create_model():
    config = Config()
    config.display()
    model = modellib.MaskRCNN(mode="training",
                              model_dir=os.getcwd(),
                              config=config)
    model.keras_model.metrics_tensors = []
    model.load_weights(model_path,
                       by_name=True,
                       exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    predict_model = modellib.MaskRCNN(mode="inference", config=config,  # Will be used to predict
                                      model_dir=os.getcwd())
    predict_model.keras_model.metrics_tensors = []
    return model, predict_model

def list_chunk(l, n_chunks):
    avg = len(l) / float(n_chunks)
    out = []
    last = 0.0
    while last < len(l):
        out.append(l[int(last):int(last + avg)])
        last += avg
    return out

def dict_split(d, percent_split):
    keys = list(d.keys())
    new_dict = {}
    for idx, key in enumerate(keys):
        new_dict[key] = d[key]
        del d[key]
        if idx >= len(keys)*percent_split:
            break
    return new_dict, d

def preprocess_data(id_and_groups):
    id = id_and_groups[0]
    all_groups = id_and_groups[1]
    progress = tqdm.tqdm(total=len(all_groups),
                         desc=f'Preprocessing data id {id}',
                         position=id)
    data_dict = {}
    for idx, group in enumerate(all_groups):
        training_data = []
        img, full_mask, building_mask, road_mask, water_mask, percent = group
        img_array = img_to_array(img)
        if img_array.shape != (1024, 1024, 3):
            # Note: Perhaps we could make it so that the CNN takes any size, but the output size would be all over the place
            #logger.warning(f'Image {img} has shape {img_array.shape}, not shape (1024, 1024, 3). Skipping')
            continue
        categorical_mask, categories = mask_to_categorical_array(img_array, full_mask)
        data_dict[img] = [categorical_mask, categories]
        progress.update(1)
    return data_dict

def mask_to_categorical_array(img_array, mask):
    if mask is None:
        mask = np.zeros(img_array.shape)
        return mask
    mask = img_to_array(mask)
    if mask.shape != img_array.shape:
        logger.error("Wrong shape!!")
        quit()

    categories = [0]*(len(CLASS_NAMES)-1)
    """
    NOTE: What you should do is create an empty matrix with (Width, Height, Categories)
    However, the mask image is already (Widht, Height, Channels) <-- channels = colors,
    meaning we don't need to create another one, so we just overwrite current mask.
    This is just a funny, but useful coincidence in this case.
    """
    for x in range(0, img_array.shape[0]):
        for y in range(0, img_array.shape[1]):
            channels_xy = mask[y,x]
            if all(channels_xy == COLORS[0]): # BG
                pixel = [False, False, False]
            elif all(channels_xy == COLORS[1]): # building
                pixel = [True, False, False]
                if not IDS[1] in categories:
                    categories[0] = 1
            elif all(channels_xy == COLORS[2]): # road
                pixel = [False, True, False]
                if not IDS[2] in categories:
                    categories[1] = 1
            elif all(channels_xy == COLORS[3]): # water
                pixel = [False, False, True]
                if not IDS[3] in categories:
                    categories[2] = 1
            else:
                print("Whoa, how the hell did you get here???")
            mask[y,x] = pixel  #img_array[y,x]
    categories = np.array(categories, dtype=np.int32)  # Convert to array, or else it crashes
    return mask, categories

def img_to_array(img_file):
    img = cv2.imread(img_file)  # Preprocess image as well and simplify?
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def fit_model(model, predict_model, dataset_train, dataset_val):
    augmentation = imgaug.augmenters.Fliplr(0.5)
    try:
        model.train(dataset_train, dataset_val,
                    epochs=40,
                    learning_rate=learning_rate,
                    layers='heads',
                    augmentation=augmentation)  # Or heads
        model.train(dataset_train, dataset_val,
                    learning_rate=learning_rate,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)
        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=learning_rate / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt. Quitting")
        quit()

def predict_to_mask(mask):
    mask = mask*255
    return mask

def group_images():
    logger.info('Grouping images')
    images = glob.glob('Training_dataset/Images/*')
    all_masks = glob.glob('Training_dataset/Masks/all/*')
    building_masks = glob.glob('Training_dataset/Masks/building/*')
    road_masks = glob.glob('Training_dataset/Masks/road/*')
    water_masks = glob.glob('Training_dataset/Masks/water/*')
    percentages = glob.glob('Training_dataset/Percentages/*')

    all_groups = []
    for idx, img in enumerate(images):
        img_basename = os.path.basename(img).split('.')[0]
        full_mask = group_item_and_image(img_basename, all_masks)
        building_mask = group_item_and_image(img_basename, building_masks)
        road_mask = group_item_and_image(img_basename, road_masks)
        water_mask = group_item_and_image(img_basename, water_masks)
        percent = group_item_and_image(img_basename, percentages)
        group = [img, full_mask, building_mask, road_mask, water_mask, percent]
        all_groups.append(group)
        print(f'{idx} done out ot {len(images)}', end='\r')
    return all_groups

def group_item_and_image(img_basename, mask_list):
    for mask in mask_list:
        mask_basename = os.path.basename(mask).split('.')[0]
        if mask_basename == img_basename:
            return mask
    return None

if __name__ == '__main__':
    freeze_support()  # Windows support
    main()
