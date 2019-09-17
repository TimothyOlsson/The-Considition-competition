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
import matplotlib.pyplot as plt
import keras
import json
import os
import random

def main():
    all_groups = group_images()
    random.shuffle(all_groups)  # Shuffle data set
    training_data = preprocess_data(all_groups)
    fit_data(training_data)

def preprocess_data(all_groups):
    logger.info('Preprocessing data')
    all_training_data = []
    for idx, group in enumerate(all_groups):
        training_data = []
        img, mask, percent = group
        img = cv2.imread(img)  # Preprocess image as well and simplify?
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        simple_mask = simplyfy_mask(mask)
        print(f'{idx} done out of {len(all_groups)}', end='\r')

def simplyfy_mask(mask):
    """Simplify mask from RGB to int array
    0, 0, 0 = Empty = Nothing = 0
    255, 255, 0 = Yellow = building = 1
    255, 0, 255 = Purple = road = 2
    255, 0, 0 = Red = water = 3
    """
    mask = cv2.imread(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

def fit_data(training_data):
    pass

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
