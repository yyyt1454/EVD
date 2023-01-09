import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
# import PIL
import pandas as pd

import tqdm
import cv2  

import tensorflow as tf
# import tensorflow_datasets as tfds
# import tensorflow_hub as hub
from keras.utils import Sequence
from keras.utils import np_utils


class DataGenerator_past(Sequence):
    """Data Generator inherited from keras.utils.Sequence
    Args: 
        directory: the path of data set, and each sub-folder will be assigned to one class
        batch_size: the number of data points in each batch
        shuffle: whether to shuffle the data per epoch
    Note:
        If you want to load file with other data format, please fix the method of "load_data" as you want
    """
    def __init__(self, directory, batch_size=1, shuffle=True, data_augmentation=True, init_states=None, train=True):
        # Initialize the params
        self.batch_size = batch_size
        self.directory = directory
        self.shuffle = shuffle
        self.data_aug = data_augmentation
        # Load all the save_path of files, and create a dictionary that save the pair of "data:label"
        self.X_path, self.Y_dict = self.search_data() 
        # Print basic statistics information
        self.print_stats()
        self.init_states = init_states
        self.train=train
        return None
        
    def search_data(self):
        X_path = []
        Y_dict = {}
        # list all kinds of sub-folders
        self.dirs = sorted(os.listdir(self.directory))
        one_hots = np_utils.to_categorical(range(len(self.dirs)))
        for i,folder in enumerate(self.dirs):
            folder_path = os.path.join(self.directory,folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path,file)
                # append the each file path, and keep its label  
                X_path.append(file_path)
                Y_dict[file_path] = one_hots[i]
        return X_path, Y_dict
    
    def print_stats(self):
        # calculate basic information
        self.n_files = len(self.X_path)
        self.n_classes = len(self.dirs)
        self.indexes = np.arange(len(self.X_path))
#         np.random.shuffle(self.indexes)
        # Output states
        print("Found {} files belonging to {} classes.".format(self.n_files,self.n_classes))
        for i,label in enumerate(self.dirs):
            print('%10s : '%(label),i)
        return None
    
    def __len__(self):
        # calculate the iterations of each epoch
        steps_per_epoch = np.ceil(len(self.X_path) / float(self.batch_size))
        return int(steps_per_epoch)

    def __getitem__(self, index):
        """Get the data of each batch
        """
        # get the indexs of each batch
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # using batch_indexs to get path of current batch
        batch_path = [self.X_path[k] for k in batch_indexs]
        # get batch data
        batch_x, batch_y = self.data_generation(batch_path)
        if self.init_states:
            # return {**self.init_states, 'image': batch_x}, batch_y
            return (self.init_states, batch_x), batch_y
        return batch_x, batch_y

    def on_epoch_end(self):
        # shuffle the data at each end of epoch
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_path):
        # load data into memory, you can change the np.load to any method you want
        batch_x = [self.load_data(x) for x in batch_path]
        batch_y = [self.Y_dict[x] for x in batch_path]
        # transfer the data format and take one-hot coding for labels
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y
      
    def normalize(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return (data-mean) / std
    
    def random_flip(self, video, prob):
        s = np.random.rand()
        if s < prob:
            video = np.flip(m=video, axis=2)
        return video    
    
    def uniform_sampling(self, video, target_frames=64):
        # get total frames of input video and calculate sampling interval 
        len_frames = int(len(video))
        if len_frames < target_frames:
            target_frames = len_frames
        interval = int(np.ceil(len_frames/target_frames))
        # init empty list for sampled video and 
        sampled_video = []
        for i in range(0,len_frames,interval):
            sampled_video.append(video[i])     
        # calculate numer of padded frames and fix it 
        num_pad = target_frames - len(sampled_video)
        if num_pad>0:
            padding = [video[i] for i in range(-num_pad,0)]
            sampled_video += padding     
        # get sampled video
        return np.array(sampled_video, dtype=np.float32)
    
    def dynamic_crop(self, video):
        # extract layer of optical flow from video
        opt_flows = video[...,3]
        # sum of optical flow magnitude of individual frame
        magnitude = np.sum(opt_flows, axis=0)
        # filter slight noise by threshold 
        thresh = np.mean(magnitude)
        magnitude[magnitude<thresh] = 0
        # calculate center of gravity of magnitude map and adding 0.001 to avoid empty value
        x_pdf = np.sum(magnitude, axis=1) + 0.001
        y_pdf = np.sum(magnitude, axis=0) + 0.001
        # normalize PDF of x and y so that the sum of probs = 1
        x_pdf /= np.sum(x_pdf)
        y_pdf /= np.sum(y_pdf)
        # randomly choose some candidates for x and y 
        x_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=x_pdf)
        y_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=y_pdf)
        # get the mean of x and y coordinates for better robustness
        x = int(np.mean(x_points))
        y = int(np.mean(y_points))
        # avoid to beyond boundaries of array
        x = max(56,min(x,167))
        y = max(56,min(y,167))
        # get cropped video 
        return video[:,x-56:x+56,y-56:y+56,:]  
    
    def color_jitter(self,video):
        # range of s-component: 0-1
        # range of v component: 0-255
        s_jitter = np.random.uniform(-0.2,0.2)
        v_jitter = np.random.uniform(-30,30)
        for i in range(len(video)):
            hsv = cv2.cvtColor(video[i], cv2.COLOR_RGB2HSV)
            s = hsv[...,1] + s_jitter
            v = hsv[...,2] + v_jitter
            s[s<0] = 0
            s[s>1] = 1
            v[v<0] = 0
            v[v>255] = 255
            hsv[...,1] = s
            hsv[...,2] = v
            video[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return video
        
    def load_data(self, path):
        data = np.load(path, mmap_mode='r', allow_pickle=True)
        # data = np.float32(data)
        # # sampling 64 frames uniformly from the entire video
        # data = self.uniform_sampling(video=data, target_frames=64)
        # # whether to utilize the data augmentation
        # if  self.data_aug:
        #     data = self.color_jitter(data)
        #     data = self.random_flip(data, prob=0.5)
        # # normalize
        # data = self.normalize(data)
        if self.train:
            data = preprocess_movinet(data)
        return data
#         return path


def preprocess_movinet(data):
    data = np.float32(data)
    # Normalize
    mean = np.mean(data)
    std = np.std(data)
    return (data-mean) / std


    
class Dataloader_CCTV(Sequence):

    def __init__(self, img_file_path='/home/ahreumseo/research/violence/datasets/CCTV_busan/data_npy/224', 
                 label_file='/home/ahreumseo/research/violence/datasets/CCTV_busan/label/label_motionobj.csv', 
                 target_module = 'motion', batch_size = 1, transform=None, test_mode=False):
        
        self.img_file_path = img_file_path  # npy 파일 path 
        self.label_file = label_file  # annotation 파일 
        self.target_module = target_module
        
        self.batch_size = batch_size
        self.tranform = transform
        self.test_mode = test_mode
        self.total_frame = 540
        
        self._read_df(label_file)

    def __len__(self):
        return len(self.vname_list)


    # batch 단위로 직접 묶어줘야 함
    def __getitem__(self, index):
        video = self.vname_list.iloc[index, -1]
        
        # load video numpy 
        vname = self.vname_list.iloc[index, -1][:-4]
        img = np.load(os.path.join(self.img_file_path, vname + '.npy'))
        
        # load label 
        if self.target_module == 'motion':
            label = np.array(self.df.loc[self.df['file_name']==video, 'movement'])
        elif self.target_module == 'person':
            label = np.array(self.df.loc[self.df['file_name']==video, 'people_count'])
        elif self.target_module == 'car':
            label = np.array(self.df.loc[self.df['file_name']==video, 'car_count'])
        else:
            label = np.array(self.df.loc[self.df['file_name']==video, 'obj_count'])
        # self.indexes = np.arange(len(self.vname_list))
        # batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # videos = self.vname_list.iloc[batch_indexs, -1]
        
        # # load video numpy 
        # # vnames = self.vname_list.iloc[batch_indexs, -1][:-4]
        # # batch_x = [self.load_data(x) for x in batch_path]
        # img = np.array([np.load(os.path.join(self.img_file_path, video[:-4] + '.npy')) for video in videos])
        
        # # load label 
        # if self.target_module == 'motion':
        #     label = np.array([self.df.loc[self.df['file_name']==video, 'movement'] for video in videos])
        # elif self.target_module == 'person':
        #     label = np.array([self.df.loc[self.df['file_name']==video, 'people_count'] for video in videos])
        # elif self.target_module == 'car':
        #     label = np.array([self.df.loc[self.df['file_name']==video, 'car_count'] for video in videos])
        # else:
        #     label = np.array([self.df.loc[self.df['file_name']==video, 'obj_count'] for video in videos])
        
        # preprocess data
        if self.tranform:
            img = self._tranform(img)
        
        # test mode: return only image 
        if self.test_mode:
            return img
        else:
            return img, label

    def _read_df(self, label_file):
        self.df = pd.read_csv(label_file)
        self.df = self.df.sort_values(by = ['file_name', 'frame'])
        self.vname_list = self.df['file_name'].drop_duplicates().reset_index().drop(['index'], axis = 1)
        
    def _transform(self, data):
        # preprocess data
        
        return data