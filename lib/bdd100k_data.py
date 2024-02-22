#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

import lib.transform_cv2 as T
from lib.base_dataset import BaseDataset, BaseDatasetIm
from lib.cvCudaDataset import ImageBatchDecoderPyTorch, ImageBatchPNGDecoderPyTorch, ImageBatchPNGDecoderPyTorchDist

# Bus、Light、Sign、Person、Bike、Truck、Motor、Car、Train、Rider

labels_info = [
    {"name": "unlabel", "id":255, "color":[0,0,0], "trainId": 255},
    {"name": "road", "id": 0, "color": [0, 0, 0], "trainId": 0},
    {"name": "sidewalk", "id": 1, "color": [0, 0, 0], "trainId": 1},
    {"name": "building", "id": 2, "color": [0, 0, 0], "trainId": 2},
    {"name": "wall", "id": 3, "color": [0, 0, 0], "trainId": 3},
    {"name": "fence", "id": 4, "color": [0, 0, 0], "trainId": 4},
    {"name": "pole", "id": 5, "color": [0, 0, 0], "trainId": 5},
    {"name": "traffic light", "id": 6, "color": [0, 0, 0], "trainId": 6},
    {"name": "traffic sign", "id": 7, "color": [0, 0, 0], "trainId": 7},
    {"name": "vegetation", "id": 8, "color": [0, 0, 0], "trainId": 8},
    {"name": "terrain", "id": 9, "color": [0, 0, 0], "trainId": 9},
    {"name": "sky", "id": 10, "color": [0, 0, 0], "trainId": 10},
    {"name": "person", "id": 11, "color": [0, 0, 0], "trainId": 11},
    {"name": "rider", "id": 12, "color": [0, 0, 0], "trainId": 12},
    {"name": "car", "id": 13, "color": [0, 0, 0], "trainId": 13},
    {"name": "truck", "id": 14, "color": [0, 0, 0], "trainId": 14},
    {"name": "bus", "id": 15, "color": [0, 0, 0], "trainId": 15},
    {"name": "train", "id": 16, "color": [0, 0, 0], "trainId": 16},
    {"name": "motorcycle", "id": 17, "color": [0, 0, 0], "trainId": 17},
    {"name": "bicycle", "id": 18, "color": [0, 0, 0], "trainId": 18},
]
# labels_info_train = labels_info_eval
## CityScapes -> {unify class1, unify class2, ...}
# Wall -> {Wall, fence}

class Bdd100k(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(Bdd100k, self).__init__(
                dataroot, annpath, trans_func, mode)
    
        mode = 'eval'
        self.n_cats = 19
        if mode=='train':
            self.n_cats =20
        
        self.lb_ignore = 255
        # self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        
        self.labels_info = labels_info
            
        for el in self.labels_info:
            if mode=='train' and el['trainId'] == 255:
                self.lb_map[el['id']] = 19
            else:
                self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(0.3038, 0.3383, 0.3034), # city, rgb
            std=(0.2071, 0.2088, 0.2090),
        )

        # self.to_tensor = T.ToTensor(
        #     mean=(0.3257, 0.3690, 0.3223), # city, rgb
        #     std=(0.2112, 0.2148, 0.2115),
        # )

## Only return img without label
class Bdd100kIm(BaseDatasetIm):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(Bdd100kIm, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.n_cats = 19
        self.lb_ignore = -1
        self.lb_map = np.arange(256).astype(np.uint8)
        for el in self.labels_info:
            self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(0.3038, 0.3383, 0.3034), # city, rgb
            std=(0.2071, 0.2088, 0.2090),
        )

class Bdd100kCVCUDA(ImageBatchPNGDecoderPyTorchDist):
    '''
    '''
    def __init__(self, dataroot, annpath, batch_size, device_id, cuda_ctx, mode='train'):
        super(Bdd100kCVCUDA, self).__init__(
                dataroot, annpath, batch_size, device_id, cuda_ctx, mode)
    
        # mode = 'eval'
        self.n_cats = 19
        # if mode=='train':
        #     self.n_cats =20
        
        self.lb_ignore = 255
        # self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        
        self.labels_info = labels_info
            
        for el in self.labels_info:
            # if mode=='train' and el['trainId'] == 255:
            #     self.lb_map[el['id']] = 19
            # else:
            self.lb_map[el['id']] = el['trainId']



# if __name__ == "__main__":
#     from tqdm import tqdm
#     from torch.utils.data import DataLoader
#     ds = CityScapes('./data/', mode='eval')
#     dl = DataLoader(ds,
#                     batch_size = 4,
#                     shuffle = True,
#                     num_workers = 4,
#                     drop_last = True)
#     for imgs, label in dl:
#         print(len(imgs))
#         for el in imgs:
#             print(el.size())
#         break
