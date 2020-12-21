import os
import sys
import math
import json
import random
import tensorflow as tf
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from config import Config
from args_options import parse_train_args
from sklearn.model_selection import train_test_split

class DatasetIterator(tf.keras.utils.Sequence):
    def __init__(self, config, train_set):
        dataset_path = config.project_config['dataset_path']
        self.dataset_json_path = f'{dataset_path}/indexed_data.json'

        train_set
        self.batch_size = config.batch_size
        self.max_pc_size =  int(config.train_config['point_cloud_size'])
        
        if not os.path.isfile(self.dataset_json_path):
            print("File path {} does not exist. Exiting...".format(dataset_json_path))
            sys.exit()

        with open(self.dataset_json_path) as json_file:
            self.data = json.load(json_file)
            self.data_size = len(self.data)
    
        
    def __len__(self):
        return math.ceil(self.data_size / self.batch_size)

    def __getitem__(self, batch_idx):
        curr_batch_floor = batch_idx * self.batch_size
        curr_batch_ceil = min((batch_idx+1) * self.batch_size, self.data_size)
        curr_batch_size =  curr_batch_ceil - curr_batch_floor
    
        transform_true_prev = np.zeros((curr_batch_size, 4, 4), dtype = "float32")
        transform_true_curr = np.zeros((curr_batch_size, 4, 4), dtype = "float32")
        source_point_clouds = []
        target_point_clouds = []
        
        for i, sample_idx in enumerate(range(curr_batch_floor, curr_batch_ceil )):
            transform_true_prev[i] = self.create_transform(self.data[sample_idx]['gt_prev'])
            transform_true_curr[i] = self.create_transform(self.data[sample_idx]['gt_curr'])
            source_point_clouds.append(self.create_PC(self.data[sample_idx]['start']))
            target_point_clouds.append(self.create_PC(self.data[sample_idx]['end']))
            
        self.max_pc_size = self.find_max_pc_size(source_point_clouds, target_point_clouds)
        source_point_clouds_res = np.zeros((curr_batch_size, self.max_pc_size, 3), dtype = "float32")
        target_point_clouds_res = np.zeros((curr_batch_size, self.max_pc_size, 3), dtype = "float32")

        for i in range(len(source_point_clouds_res)):
            source_point_clouds_res[i] = source_point_clouds[i][:self.max_pc_size, :]
            target_point_clouds_res[i] = target_point_clouds[i][:self.max_pc_size, :]

#         print(f'Shape: {source_point_clouds.shape}')
        return [source_point_clouds_res, target_point_clouds_res, transform_true_prev, transform_true_curr], np.zeros((curr_batch_size)).astype('float32')
        
    def find_max_pc_size(self, source_point_clouds, target_point_clouds):
        smallest = self.max_pc_size
        for cloud in source_point_clouds:
            if cloud.shape[0] < smallest:
                smallest = cloud.shape[0]
        for cloud in target_point_clouds:
            if cloud.shape[0] < smallest:
                smallest = cloud.shape[0]
        return smallest        
        
    def create_PC(self, file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        np.random.shuffle(points)
        return points
    
    def create_transform(self, gt):
        x = float(gt['x1']) - float(gt['x2'])
        y = float(gt['y1']) - float(gt['y2'])
        z = float(gt['z1']) - float(gt['z2'])
        t = np.array([[x], [y], [z]], dtype = np.float64)
        r1 = R.from_quat([gt['q_x1'], gt['q_y1'], gt['q_z1'], gt['q_w1']]).inv().as_matrix()
        r2 = R.from_quat([gt['q_x2'], gt['q_y2'], gt['q_z2'], gt['q_w2']]).as_matrix()
        r = np.matmul(r1,r2)
        lastRow = np.array([0,0,0,1], dtype = np.float64)
        return np.append(np.append(r, t, axis = 1), [lastRow], axis = 0)

        
def get_dataset_split(config):
    dataset_json_path = config.project_config['dataset_path'] + '/indexed_data.json'
    train_test_split_prec = 1 - float(config.train_config['train_test_split'])
    train_validate_split_prec = 1 - float(config.train_config['train_validate_split'])
    seed =  int(config.project_config['seed'])
    
    with open(dataset_json_path) as json_file:
        data = json.load(json_file)
        data_size = len(data)
               
    train_to_split_set, test_set = train_test_split(data, test_size = train_test_split_prec, random_state = seed, shuffle = True)
    train_set, valid_set = train_test_split(train_to_split_set, test_size = train_validate_split_prec, random_state = seed, shuffle = True)
    
    return train_set, valid_set, test_set