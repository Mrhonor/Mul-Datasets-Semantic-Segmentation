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
from PIL import Image

import lib.transform_cv2 as T
from lib.base_dataset import BaseDataset, BaseDatasetIm


labels_info = [
{"id": 0, "trainId": 255, "name": "unlabeled"},
{"id": 1, "trainId": 0, "name": "ego vehicle"},
{"id": 2, "trainId": 255, "name": "rectification border"},
{"id": 3, "trainId": 255, "name": "out of roi"},
{"id": 4, "trainId": 255, "name": "static"},
{"id": 5, "trainId": 255, "name": "dynamic"},
{"id": 6, "trainId": 255, "name": "ground"},
{"id": 7, "trainId": 1, "name": "road"},
{"id": 8, "trainId": 2, "name": "sidewalk"},
{"id": 9, "trainId": 255, "name": "parking"},
{"id": 10, "trainId": 255, "name": "rail track"},
{"id": 11, "trainId": 3, "name": "building"},
{"id": 12, "trainId": 4, "name": "wall"},
{"id": 13, "trainId": 5, "name": "fence"},
{"id": 14, "trainId": 6, "name": "guard rail"},
{"id": 15, "trainId": 255, "name": "bridge"},
{"id": 16, "trainId": 255, "name": "tunnel"},
{"id": 17, "trainId": 7, "name": "pole"},
{"id": 18, "trainId": 255, "name": "polegroup"},
{"id": 19, "trainId": 8, "name": "traffic light"},
{"id": 20, "trainId": 9, "name": "traffic sign"},
{"id": 21, "trainId": 10, "name": "vegetation"},
{"id": 22, "trainId": 11, "name": "terrain"},
{"id": 23, "trainId": 12, "name": "sky"},
{"id": 24, "trainId": 13, "name": "person"},
{"id": 25, "trainId": 14, "name": "rider"},
{"id": 26, "trainId": 15, "name": "car"},
{"id": 27, "trainId": 16, "name": "truck"},
{"id": 28, "trainId": 17, "name": "bus"},
{"id": 29, "trainId": 255, "name": "caravan"},
{"id": 30, "trainId": 255, "name": "trailer"},
{"id": 31, "trainId": 255, "name": "train"},
{"id": 32, "trainId": 18, "name": "motorcycle"},
{"id": 33, "trainId": 19, "name": "bicycle"},
{"id": 34, "trainId": 20, "name": "pickup"},
{"id": 35, "trainId": 21, "name": "van"},
{"id": 36, "trainId": 22, "name": "billboard"},
{"id": 37, "trainId": 23, "name": "street-light"},
{"id": 38, "trainId": 24, "name": "road-marking"},
]

# labels_info_train = labels_info_eval
## CityScapes -> {unify class1, unify class2, ...}
# Wall -> {Wall, fence}

class wd2(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(wd2, self).__init__(
                dataroot, annpath, trans_func, mode)
    
        
        self.n_cats = 25
        
        self.lb_ignore = -1
        # self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        
        self.labels_info = labels_info
            
        for el in self.labels_info:
            self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(0.3038, 0.3383, 0.3034), # city, rgb
            std=(0.2071, 0.2088, 0.2090),
        )
        
 