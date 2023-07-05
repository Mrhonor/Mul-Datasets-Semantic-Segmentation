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
{"name": "person", "id": 1, "trainId": 1},
{"name": "bicycle", "id": 2, "trainId": 2},
{"name": "car", "id": 3, "trainId": 3},
{"name": "motorcycle", "id": 4, "trainId": 4},
{"name": "airplane", "id": 5, "trainId": 5},
{"name": "bus", "id": 6, "trainId": 6},
{"name": "train", "id": 7, "trainId": 7},
{"name": "truck", "id": 8, "trainId": 8},
{"name": "boat", "id": 9, "trainId": 9},
{"name": "traffic light", "id": 10, "trainId": 10},
{"name": "fire hydrant", "id": 11, "trainId": 11},
{"name": "stop sign", "id": 13, "trainId": 13},
{"name": "parking meter", "id": 14, "trainId": 14},
{"name": "bench", "id": 15, "trainId": 15},
{"name": "bird", "id": 16, "trainId": 16},
{"name": "cat", "id": 17, "trainId": 17},
{"name": "dog", "id": 18, "trainId": 18},
{"name": "horse", "id": 19, "trainId": 19},
{"name": "sheep", "id": 20, "trainId": 20},
{"name": "cow", "id": 21, "trainId": 21},
{"name": "elephant", "id": 22, "trainId": 22},
{"name": "bear", "id": 23, "trainId": 23},
{"name": "zebra", "id": 24, "trainId": 24},
{"name": "giraffe", "id": 25, "trainId": 25},
{"name": "backpack", "id": 27, "trainId": 27},
{"name": "umbrella", "id": 28, "trainId": 28},
{"name": "handbag", "id": 31, "trainId": 31},
{"name": "tie", "id": 32, "trainId": 32},
{"name": "suitcase", "id": 33, "trainId": 33},
{"name": "frisbee", "id": 34, "trainId": 34},
{"name": "skis", "id": 35, "trainId": 35},
{"name": "snowboard", "id": 36, "trainId": 36},
{"name": "sports ball", "id": 37, "trainId": 37},
{"name": "kite", "id": 38, "trainId": 38},
{"name": "baseball bat", "id": 39, "trainId": 39},
{"name": "baseball glove", "id": 40, "trainId": 40},
{"name": "skateboard", "id": 41, "trainId": 41},
{"name": "surfboard", "id": 42, "trainId": 42},
{"name": "tennis racket", "id": 43, "trainId": 43},
{"name": "bottle", "id": 44, "trainId": 44},
{"name": "wine glass", "id": 46, "trainId": 46},
{"name": "cup", "id": 47, "trainId": 47},
{"name": "fork", "id": 48, "trainId": 48},
{"name": "knife", "id": 49, "trainId": 49},
{"name": "spoon", "id": 50, "trainId": 50},
{"name": "bowl", "id": 51, "trainId": 51},
{"name": "banana", "id": 52, "trainId": 52},
{"name": "apple", "id": 53, "trainId": 53},
{"name": "sandwich", "id": 54, "trainId": 54},
{"name": "orange", "id": 55, "trainId": 55},
{"name": "broccoli", "id": 56, "trainId": 56},
{"name": "carrot", "id": 57, "trainId": 57},
{"name": "hot dog", "id": 58, "trainId": 58},
{"name": "pizza", "id": 59, "trainId": 59},
{"name": "donut", "id": 60, "trainId": 60},
{"name": "cake", "id": 61, "trainId": 61},
{"name": "chair", "id": 62, "trainId": 62},
{"name": "couch", "id": 63, "trainId": 63},
{"name": "potted plant", "id": 64, "trainId": 64},
{"name": "bed", "id": 65, "trainId": 65},
{"name": "dining table", "id": 67, "trainId": 67},
{"name": "toilet", "id": 70, "trainId": 70},
{"name": "tv", "id": 72, "trainId": 72},
{"name": "laptop", "id": 73, "trainId": 73},
{"name": "mouse", "id": 74, "trainId": 74},
{"name": "remote", "id": 75, "trainId": 75},
{"name": "keyboard", "id": 76, "trainId": 76},
{"name": "cell phone", "id": 77, "trainId": 77},
{"name": "microwave", "id": 78, "trainId": 78},
{"name": "oven", "id": 79, "trainId": 79},
{"name": "toaster", "id": 80, "trainId": 80},
{"name": "sink", "id": 81, "trainId": 81},
{"name": "refrigerator", "id": 82, "trainId": 82},
{"name": "book", "id": 84, "trainId": 84},
{"name": "clock", "id": 85, "trainId": 85},
{"name": "vase", "id": 86, "trainId": 86},
{"name": "scissors", "id": 87, "trainId": 87},
{"name": "teddy bear", "id": 88, "trainId": 88},
{"name": "hair drier", "id": 89, "trainId": 89},
{"name": "toothbrush", "id": 90, "trainId": 90},
{"name": "banner", "id": 92, "trainId": 92},
{"name": "blanket", "id": 93, "trainId": 93},
{"name": "bridge", "id": 95, "trainId": 95},
{"name": "cardboard", "id": 100, "trainId": 100},
{"name": "counter", "id": 107, "trainId": 107},
{"name": "curtain", "id": 109, "trainId": 109},
{"name": "door-stuff", "id": 112, "trainId": 112},
{"name": "floor-wood", "id": 118, "trainId": 118},
{"name": "flower", "id": 119, "trainId": 119},
{"name": "fruit", "id": 122, "trainId": 122},
{"name": "gravel", "id": 125, "trainId": 125},
{"name": "house", "id": 128, "trainId": 128},
{"name": "light", "id": 130, "trainId": 130},
{"name": "mirror-stuff", "id": 133, "trainId": 133},
{"name": "net", "id": 138, "trainId": 138},
{"name": "pillow", "id": 141, "trainId": 141},
{"name": "platform", "id": 144, "trainId": 144},
{"name": "playingfield", "id": 145, "trainId": 145},
{"name": "railroad", "id": 147, "trainId": 147},
{"name": "river", "id": 148, "trainId": 148},
{"name": "road", "id": 149, "trainId": 149},
{"name": "roof", "id": 151, "trainId": 151},
{"name": "sand", "id": 154, "trainId": 154},
{"name": "sea", "id": 155, "trainId": 155},
{"name": "shelf", "id": 156, "trainId": 156},
{"name": "snow", "id": 159, "trainId": 159},
{"name": "stairs", "id": 161, "trainId": 161},
{"name": "tent", "id": 166, "trainId": 166},
{"name": "towel", "id": 168, "trainId": 168},
{"name": "wall-brick", "id": 171, "trainId": 171},
{"name": "wall-stone", "id": 175, "trainId": 175},
{"name": "wall-tile", "id": 176, "trainId": 176},
{"name": "wall-wood", "id": 177, "trainId": 177},
{"name": "water-other", "id": 178, "trainId": 178},
{"name": "window-blind", "id": 180, "trainId": 180},
{"name": "window-other", "id": 181, "trainId": 181},
{"name": "tree-merged", "id": 184, "trainId": 184},
{"name": "fence-merged", "id": 185, "trainId": 185},
{"name": "ceiling-merged", "id": 186, "trainId": 186},
{"name": "sky-other-merged", "id": 187, "trainId": 187},
{"name": "cabinet-merged", "id": 188, "trainId": 188},
{"name": "table-merged", "id": 189, "trainId": 189},
{"name": "floor-other-merged", "id": 190, "trainId": 190},
{"name": "pavement-merged", "id": 191, "trainId": 191},
{"name": "mountain-merged", "id": 192, "trainId": 192},
{"name": "grass-merged", "id": 193, "trainId": 193},
{"name": "dirt-merged", "id": 194, "trainId": 194},
{"name": "paper-merged", "id": 195, "trainId": 195},
{"name": "food-other-merged", "id": 196, "trainId": 196},
{"name": "building-other-merged", "id": 197, "trainId": 197},
{"name": "rock-merged", "id": 198, "trainId": 198},
{"name": "wall-other-merged", "id": 199, "trainId": 199},
{"name": "rug-merged", "id": 200, "trainId": 200},
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
    
        self.n_cats = 200
        
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
class Coco_dataIm(BaseDatasetIm):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(Coco_data, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.n_cats = 200
        self.lb_ignore = -1
        self.lb_map = np.arange(256).astype(np.uint8)
        for el in self.labels_info:
            self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(0.3257, 0.3690, 0.3223), # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )


