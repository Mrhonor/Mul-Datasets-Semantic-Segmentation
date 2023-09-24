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
{"name": "person", "id": 1, "trainId": 0},
{"name": "bicycle", "id": 2, "trainId": 1},
{"name": "car", "id": 3, "trainId": 2},
{"name": "motorcycle", "id": 4, "trainId": 3},
{"name": "airplane", "id": 5, "trainId": 4},
{"name": "bus", "id": 6, "trainId": 5},
{"name": "train", "id": 7, "trainId": 6},
{"name": "truck", "id": 8, "trainId": 7},
{"name": "boat", "id": 9, "trainId": 8},
{"name": "traffic light", "id": 10, "trainId": 9},
{"name": "fire hydrant", "id": 11, "trainId": 10},
{"name": "stop sign", "id": 13, "trainId": 11},
{"name": "parking meter", "id": 14, "trainId": 12},
{"name": "bench", "id": 15, "trainId": 13},
{"name": "bird", "id": 16, "trainId": 14},
{"name": "cat", "id": 17, "trainId": 15},
{"name": "dog", "id": 18, "trainId": 16},
{"name": "horse", "id": 19, "trainId": 17},
{"name": "sheep", "id": 20, "trainId": 18},
{"name": "cow", "id": 21, "trainId": 19},
{"name": "elephant", "id": 22, "trainId": 20},
{"name": "bear", "id": 23, "trainId": 21},
{"name": "zebra", "id": 24, "trainId": 22},
{"name": "giraffe", "id": 25, "trainId": 23},
{"name": "backpack", "id": 27, "trainId": 24},
{"name": "umbrella", "id": 28, "trainId": 25},
{"name": "handbag", "id": 31, "trainId": 26},
{"name": "tie", "id": 32, "trainId": 27},
{"name": "suitcase", "id": 33, "trainId": 28},
{"name": "frisbee", "id": 34, "trainId": 29},
{"name": "skis", "id": 35, "trainId": 30},
{"name": "snowboard", "id": 36, "trainId": 31},
{"name": "sports ball", "id": 37, "trainId": 32},
{"name": "kite", "id": 38, "trainId": 33},
{"name": "baseball bat", "id": 39, "trainId": 34},
{"name": "baseball glove", "id": 40, "trainId": 35},
{"name": "skateboard", "id": 41, "trainId": 36},
{"name": "surfboard", "id": 42, "trainId": 37},
{"name": "tennis racket", "id": 43, "trainId": 38},
{"name": "bottle", "id": 44, "trainId": 39},
{"name": "wine glass", "id": 46, "trainId": 40},
{"name": "cup", "id": 47, "trainId": 41},
{"name": "fork", "id": 48, "trainId": 42},
{"name": "knife", "id": 49, "trainId": 43},
{"name": "spoon", "id": 50, "trainId": 44},
{"name": "bowl", "id": 51, "trainId": 45},
{"name": "banana", "id": 52, "trainId": 46},
{"name": "apple", "id": 53, "trainId": 47},
{"name": "sandwich", "id": 54, "trainId": 48},
{"name": "orange", "id": 55, "trainId": 49},
{"name": "broccoli", "id": 56, "trainId": 50},
{"name": "carrot", "id": 57, "trainId": 51},
{"name": "hot dog", "id": 58, "trainId": 52},
{"name": "pizza", "id": 59, "trainId": 53},
{"name": "donut", "id": 60, "trainId": 54},
{"name": "cake", "id": 61, "trainId": 55},
{"name": "chair", "id": 62, "trainId": 56},
{"name": "couch", "id": 63, "trainId": 57},
{"name": "potted plant", "id": 64, "trainId": 58},
{"name": "bed", "id": 65, "trainId": 59},
{"name": "dining table", "id": 67, "trainId": 60},
{"name": "toilet", "id": 70, "trainId": 61},
{"name": "tv", "id": 72, "trainId": 62},
{"name": "laptop", "id": 73, "trainId": 63},
{"name": "mouse", "id": 74, "trainId": 64},
{"name": "remote", "id": 75, "trainId": 65},
{"name": "keyboard", "id": 76, "trainId": 66},
{"name": "cell phone", "id": 77, "trainId": 67},
{"name": "microwave", "id": 78, "trainId": 68},
{"name": "oven", "id": 79, "trainId": 69},
{"name": "toaster", "id": 80, "trainId": 70},
{"name": "sink", "id": 81, "trainId": 71},
{"name": "refrigerator", "id": 82, "trainId": 72},
{"name": "book", "id": 84, "trainId": 73},
{"name": "clock", "id": 85, "trainId": 74},
{"name": "vase", "id": 86, "trainId": 75},
{"name": "scissors", "id": 87, "trainId": 76},
{"name": "teddy bear", "id": 88, "trainId": 77},
{"name": "hair drier", "id": 89, "trainId": 78},
{"name": "toothbrush", "id": 90, "trainId": 79},
{"name": "banner", "id": 92, "trainId": 80},
{"name": "blanket", "id": 93, "trainId": 81},
{"name": "bridge", "id": 95, "trainId": 82},
{"name": "cardboard", "id": 100, "trainId": 83},
{"name": "counter", "id": 107, "trainId": 84},
{"name": "curtain", "id": 109, "trainId": 85},
{"name": "door-stuff", "id": 112, "trainId": 86},
{"name": "floor-wood", "id": 118, "trainId": 87},
{"name": "flower", "id": 119, "trainId": 88},
{"name": "fruit", "id": 122, "trainId": 89},
{"name": "gravel", "id": 125, "trainId": 90},
{"name": "house", "id": 128, "trainId": 91},
{"name": "light", "id": 130, "trainId": 92},
{"name": "mirror-stuff", "id": 133, "trainId": 93},
{"name": "net", "id": 138, "trainId": 94},
{"name": "pillow", "id": 141, "trainId": 95},
{"name": "platform", "id": 144, "trainId": 96},
{"name": "playingfield", "id": 145, "trainId": 97},
{"name": "railroad", "id": 147, "trainId": 98},
{"name": "river", "id": 148, "trainId": 99},
{"name": "road", "id": 149, "trainId": 100},
{"name": "roof", "id": 151, "trainId": 101},
{"name": "sand", "id": 154, "trainId": 102},
{"name": "sea", "id": 155, "trainId": 103},
{"name": "shelf", "id": 156, "trainId": 104},
{"name": "snow", "id": 159, "trainId": 105},
{"name": "stairs", "id": 161, "trainId": 106},
{"name": "tent", "id": 166, "trainId": 107},
{"name": "towel", "id": 168, "trainId": 108},
{"name": "wall-brick", "id": 171, "trainId": 109},
{"name": "wall-stone", "id": 175, "trainId": 110},
{"name": "wall-tile", "id": 176, "trainId": 111},
{"name": "wall-wood", "id": 177, "trainId": 112},
{"name": "water-other", "id": 178, "trainId": 113},
{"name": "window-blind", "id": 180, "trainId": 114},
{"name": "window-other", "id": 181, "trainId": 115},
{"name": "tree-merged", "id": 184, "trainId": 116},
{"name": "fence-merged", "id": 185, "trainId": 117},
{"name": "ceiling-merged", "id": 186, "trainId": 118},
{"name": "sky-other-merged", "id": 187, "trainId": 119},
{"name": "cabinet-merged", "id": 188, "trainId": 120},
{"name": "table-merged", "id": 189, "trainId": 121},
{"name": "floor-other-merged", "id": 190, "trainId": 122},
{"name": "pavement-merged", "id": 191, "trainId": 123},
{"name": "mountain-merged", "id": 192, "trainId": 124},
{"name": "grass-merged", "id": 193, "trainId": 125},
{"name": "dirt-merged", "id": 194, "trainId": 126},
{"name": "paper-merged", "id": 195, "trainId": 127},
{"name": "food-other-merged", "id": 196, "trainId": 128},
{"name": "building-other-merged", "id": 197, "trainId": 129},
{"name": "rock-merged", "id": 198, "trainId": 130},
{"name": "wall-other-merged", "id": 199, "trainId": 131},
{"name": "rug-merged", "id": 200, "trainId": 132},
{"name": "unlabeled", "id":0, "trainId": 255},
]

# labels_info_train = labels_info

# labels_info_train = labels_info_eval
## CityScapes -> {unify class1, unify class2, ...}
# Wall -> {Wall, fence}

class Coco_data(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(Coco_data, self).__init__(
                dataroot, annpath, trans_func, mode)
            
            
        mode = 'eval'
    
        self.n_cats = 133
        if mode=='train':
            self.n_cats = 134
        
        self.lb_ignore = -1
        # self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        
        self.labels_info = labels_info
            
        for el in self.labels_info:
            if mode=='train' and el['trainId'] == 255:
                self.lb_map[el['id']] = 134
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
class Coco_dataIm(BaseDatasetIm):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(Coco_data, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.n_cats = 133
        self.lb_ignore = -1
        self.lb_map = np.arange(256).astype(np.uint8)
        for el in self.labels_info:
            self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(0.3257, 0.3690, 0.3223), # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )


