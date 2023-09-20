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


labels_info = [
{"name": "unlabeled", "id": 0, "trainId": 255},
{"name": "wall", "id": 1, "trainId": 1},
{"name": "floor", "id": 2, "trainId": 2},
{"name": "cabinet", "id": 3, "trainId": 3},
{"name": "bed", "id": 4, "trainId": 4},
{"name": "chair", "id": 5, "trainId": 5},
{"name": "sofa", "id": 6, "trainId": 6},
{"name": "table", "id": 7, "trainId": 7},
{"name": "door", "id": 8, "trainId": 8},
{"name": "window", "id": 9, "trainId": 9},
{"name": "bookshelf", "id": 10, "trainId": 10},
{"name": "picture", "id": 11, "trainId": 11},
{"name": "counter", "id": 12, "trainId": 12},
{"name": "blinds", "id": 13, "trainId": 13},
{"name": "desk", "id": 14, "trainId": 14},
{"name": "shelves", "id": 15, "trainId": 15},
{"name": "curtain", "id": 16, "trainId": 16},
{"name": "dresser", "id": 17, "trainId": 17},
{"name": "pillow", "id": 18, "trainId": 18},
{"name": "mirror", "id": 19, "trainId": 19},
{"name": "floor mat", "id": 20, "trainId": 20},
{"name": "clothes", "id": 21, "trainId": 21},
{"name": "ceiling", "id": 22, "trainId": 22},
{"name": "books", "id": 23, "trainId": 23},
{"name": "refridgerator", "id": 24, "trainId": 24},
{"name": "television", "id": 25, "trainId": 25},
{"name": "paper", "id": 26, "trainId": 26},
{"name": "towel", "id": 27, "trainId": 27},
{"name": "shower curtain", "id": 28, "trainId": 28},
{"name": "box", "id": 29, "trainId": 29},
{"name": "whiteboard", "id": 30, "trainId": 30},
{"name": "person", "id": 31, "trainId": 31},
{"name": "night stand", "id": 32, "trainId": 32},
{"name": "toilet", "id": 33, "trainId": 33},
{"name": "sink", "id": 34, "trainId": 34},
{"name": "lamp", "id": 35, "trainId": 35},
{"name": "bathtub", "id": 36, "trainId": 36},
{"name": "bag", "id": 37, "trainId": 0},
]
# labels_info_train = labels_info_eval
## CityScapes -> {unify class1, unify class2, ...}
# Wall -> {Wall, fence}

class Sunrgbd(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(Sunrgbd, self).__init__(
                dataroot, annpath, trans_func, mode)
    
        
        self.n_cats = 37
        
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

        # self.to_tensor = T.ToTensor(
        #     mean=(0.3257, 0.3690, 0.3223), # city, rgb
        #     std=(0.2112, 0.2148, 0.2115),
        # )

## Only return img without label
class SunrgbdIm(BaseDatasetIm):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(SunrgbdIm, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.n_cats = 37
        self.lb_ignore = -1
        self.lb_map = np.arange(256).astype(np.uint8)
        for el in self.labels_info:
            self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(0.3038, 0.3383, 0.3034), # city, rgb
            std=(0.2071, 0.2088, 0.2090),
        )



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
