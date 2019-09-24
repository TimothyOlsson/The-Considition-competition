# PARAMETERS
learning_rate = 0.001
batch_size = 10
epochs = 2
train_iterations = 100
use_gpu = True
tensorflow_debug = False
model_path = 'mask_rcnn_coco.h5'
annotations_path = 'Training_dataset/Annotations/master_train.json'
images_path = 'Training_dataset/Images'

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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import multiprocessing
from multiprocessing import freeze_support, RLock
import itertools
import mrcnn
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config as mrcnn_config
from mrcnn.visualize import display_images
from mrcnn.model import log
import tqdm
import pickle
from sklearn.model_selection import train_test_split
import imgaug
import json
import math

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

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
IDS = [0,1,2,3],
coco = COCO(annotations_path)

class Config(mrcnn_config):
    NAME = "aerial_drone_map"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # batch size
    NUM_CLASSES = len(CLASS_NAMES)  # Needs to include bg
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    STEPS_PER_EPOCH = 500
    IMAGES_PER_GPU = 1
    # Use smaller anchors because our image and objects are small
    # NOTE Different anchor sizes are suitable for houses/water/roads.
    RPN_ANCHOR_SCALES = (64, 128, 256, 512, 1024)  # anchor side in pixels
    # Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100
    VALIDATION_STEPS = 25

class DroneDataset(utils.Dataset):
    def load_data(self, train_data):
        print(vars(self))
        class_ids = sorted(coco.getCatIds())
        self.train_data = train_data
        for idx, key in enumerate(train_data.keys()):
            data_dict = train_data[key]
            self.add_image('shape',
                           path=data_dict['file_path'],
                           height=data_dict['height'],
                           width=data_dict['width'],
                           image_id=data_dict['image_id'])
            #a = coco.getAnnIds(imgIds=[key], catIds=class_ids, iscrowd=None)

    def define_classes(self, all_annotations):
        # It needs to be sorted in id order
        categories = []
        ids = []
        names = []
        for d in all_annotations['categories']:
            ids.append(d['id'])
            categories.append(d['supercategory'])
            names.append(d['name'])
        print(ids, names, categories)
        z = list(zip(ids,names,categories))
        z.sort()
        ids, names, categories = zip(*z)
        print(ids, names, categories)
        for id, name, cat in zip(ids, names, categories):
            self.add_class(cat, id, name)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        class_ids = self.train_data[image_id]['categorical_vector']
        if class_ids == []:  # If empty, return empty mask
            return super(DroneDataset, self).load_mask(image_id)

        class_ids = np.array(class_ids, dtype=np.int32)
        mask, _ = draw_mask_with_segments(self.train_data[image_id]['segments'],
                                          self.train_data[image_id]['height'],
                                          self.train_data[image_id]['width'])
        return mask, class_ids

    def image_reference(self, image_id):  # Needed
        """Return the path of the image."""
        #print("LOADING IMAGE", image_id, self.train_data[image_id]['file_path'])
        info = self.image_info[image_id]
        print(info)
        return self.train_data[image_id]['file_path']

def log_func(txt):
	logger.info(txt)

def main():
    with open(annotations_path, 'r') as f:
        all_annotations = json.load(f)
    if not os.path.isfile('preprocessed_data.pkl'):
        logger.info('Preprocessing data')
        train_data = preprocess_data(all_annotations)
        with open('preprocessed_data.pkl', 'wb') as f:
            pickle.dump(train_data, f)
    else:
        logger.info('Loading preprocessed data')
        with open('preprocessed_data.pkl', 'rb') as f:
            train_data = pickle.load(f)
    train_data, val_data = dict_split(train_data, 0.7)
    logger.info(f'Training: {len(list(train_data.keys()))} images, Validation: {len(list(val_data.keys()))} images')
    dataset_train = prepare_dataset(train_data, all_annotations)
    dataset_val = prepare_dataset(val_data, all_annotations)
    visualize_training_data(dataset_train, dataset_val)
    model, predict_model, config = create_model()
    fit_model(model, predict_model, config, dataset_train, dataset_train)
    #test_drone_images(predict_model, config, dataset_train, dataset_val)

def visualize_training_data(dataset_train, dataset_val):
    for idx, image_id in enumerate(dataset_train.train_data.keys()):
        print("IMAGE ID", image_id)
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        print("CLASS IDS", class_ids)
        print("CLASS NAMES", dataset_train.class_names)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

def test_drone_images(predict_model, config, dataset_train, dataset_val):
    for image_id in dataset_train.image_id_to_path.keys():
        #image = dataset_train.load_image(image_id)
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_train, config, image_id, use_mini_mask=False)
        print(f"IMG {image.shape}")
        results = predict_model.detect([image], verbose=1)
        ax = get_ax(1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    dataset_train.class_names, r['scores'], ax=ax,
                                    title="Predictions")
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)
        plt.show()

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

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
    return model, predict_model, config

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

def dict_chunk(d, n_chunks):
    keys = list(d.keys())
    dict_list = []
    new_dict = {}
    threshold = math.ceil(len(keys)/n_chunks)
    stop_points = [i for i in range(len(keys)) if i%threshold==0 and i != 0]
    for idx, key in enumerate(keys):
        new_dict[key] = d[key]
        del d[key]
        if idx+1 in stop_points:
            dict_list.append(new_dict)
            new_dict = {}
    if new_dict != {}:
        dict_list.append(new_dict)
    if d != {}:
        dict_list.append(d)
    return dict_list

def prepare_dataset(data_dict, all_annotations):
    dataset = DroneDataset()
    dataset.define_classes(all_annotations)
    dataset.load_data(data_dict)
    dataset.prepare()
    return dataset

def preprocess_data(all_annotations):
     # NOTE: bad way of doing it. Spikes at 32 Gb or RAM. Perhaps a generator would be better?
    id = 1
    progress = tqdm.tqdm(total=len(all_annotations['images']),
                         desc=f'Preprocessing data id {id}',
                         position=id)

    all_preprocessed_data = {}
    n_categories = len(all_annotations['categories']) + 1
    for d_img in all_annotations['images']:
        data_dict = {}
        if 1024 != d_img['height'] or 1024 != d_img['width']:
            tqdm.tqdm.write(f"Skipping image {d_img['file_name']}")
            continue
        data_dict['file_name'] = d_img['file_name']
        data_dict['file_path'] = os.path.join(images_path, d_img['file_name'])
        data_dict['height'] = d_img['height']
        data_dict['width'] = d_img['width']
        data_dict['image_id'] = d_img['id']
        segments = []
        for d_segment in all_annotations['annotations']:
            if d_segment['image_id'] == data_dict['image_id']:
                segments.append(d_segment)
        # Also clears some segments
        mask, segments = draw_mask_with_segments(segments,
                                                 data_dict['height'],
                                                 data_dict['width'])
        categorical_vector = []
        for d_segment in segments:
            categorical_vector.append(d_segment['category_id'])
        data_dict['segments'] = segments   # Load dynamically, else requires loads of memory
        data_dict['categorical_vector'] = categorical_vector
        all_preprocessed_data[d_img['id']] = data_dict
        progress.update(1)
    progress.close()
    return all_preprocessed_data

def draw_mask_with_segments(segments, height, width):
    masks = []
    bad_segments_idx = []
    for idx, d_segment in enumerate(segments):
        m = annToMask(d_segment, height, width)
        if m is None:
            bad_segments_idx.append(idx)
        else:
            masks.append(m)
    for index in sorted(bad_segments_idx, reverse=True):
        del segments[index]
    if len(masks) == 0:
        mask = np.zeros((height, width, 1), dtype=bool)
    else:
        mask = np.stack(masks, axis=2).astype(np.bool)
    return mask, segments

def annToRLE(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        try:
            rles = maskUtils.frPyObjects(segm, height, width)
        except TypeError:
            # Bad segment. Ignore it
            tqdm.tqdm.write("Bad segment found. Removing")
            return None
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann['segmentation']
    return rle

def annToMask(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    if rle is None:
        return None
    m = maskUtils.decode(rle)
    if m.max() < 1:
        tqdm.tqdm.write("Bad segment found. Removing")
        return None
    return m

def fit_model(model, predict_model, config, dataset_train, dataset_val):
    augmentation = imgaug.augmenters.Fliplr(0.5)
    try:
        # Training - Stage 1
        model.train(dataset_train, dataset_val,
                    epochs=40,
                    learning_rate=config.LEARNING_RATE,
                    layers='heads',
                    augmentation=augmentation)
        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)
        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt. Quitting")
        quit()

if __name__ == '__main__':
    freeze_support()  # Windows support
    main()
