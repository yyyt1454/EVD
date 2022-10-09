import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import PIL
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tqdm
import cv2  

from dataloader import *

# Run imports
import tensorflow_datasets as tfds

from official.vision.configs import video_classification
from official.projects.movinet.configs import movinet as movinet_configs
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_layers
from official.projects.movinet.modeling import movinet_model


model = tf.keras.models.load_model('/home/ahreumseo/research/violence/datasets/MoViNet-TF/models/official/projects/movinet/rwf-2000_a0_scratch')

batch_size = 1
num_frames = 64
frame_stride = 0
resolution = 224



# RWF-2000
    
val_generator = DataGenerator_past(directory='/home/ahreumseo/research/violence/datasets/RWF2000-Video-Database-for-Violence-Detection/Dataset/dataset_npy_224/val',
                              batch_size=batch_size, 
                              data_augmentation=False,
                             shuffle=False)

# Surveillance Fight
val_generator2 = DataGenerator_past(directory='/home/ahreumseo/research/violence/datasets/Surveillance_Fight/npy_224',
                              batch_size=batch_size, 
                              data_augmentation=False,
                             shuffle=False)

# errors = []

# with open('error2.txt', 'w') as f:
#     for i in range(len(val_generator)):
        
#         video = val_generator[i]
#         label, pred = video[1], model(video[0]) 
#         label2 = 'violence' if np.argmax(label,axis=1) == 0 else 'non-violence'
#         pred2 = 'violence' if np.argmax(pred,axis=1) == 0 else 'non-violence'
        
#         if label2 != pred2:
#             print(f'index: {i}, label: {label, label2}, pred: {pred, pred2}\n')
#             f.write(f'index: {i}, label: {label, label2}, pred: {pred, pred2}\n')
#             errors.append(i)
            
#     f.write(f'{errors}')

model.evaluate(val_generator)

model.evaluate(val_generator2)