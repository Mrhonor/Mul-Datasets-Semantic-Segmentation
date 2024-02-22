#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
from time import sleep
sys.path.insert(0, '.')
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
from lib.cvCudaDataset import ImageBatchDecoderPyTorch, ImageBatchPNGDecoderPyTorch, ImageBatchPNGDecoderPyTorchDist


# labels_info = [
#     {"name": "person", "id": 0, "color": [0, 0, 0], "trainId": 0},
#     {"name": "truck", "id": 1, "color": [0, 0, 0], "trainId": 1},
#     {"name": "fence", "id": 2, "color": [0, 0, 0], "trainId": 2},
#     {"name": "billboard", "id": 3, "color": [0, 0, 0], "trainId": 3},
#     {"name": "bus", "id": 4, "color": [0, 0, 0], "trainId": 4},
#     {"name": "out of roi", "id": 5, "color": [0, 0, 0], "trainId": 5},
#     {"name": "curb", "id": 6, "color": [0, 0, 0], "trainId": 6},
#     {"name": "obs-str-bar-fallback", "id": 7, "color": [0, 0, 0], "trainId": 7},
#     {"name": "tunnel", "id": 8, "color": [0, 0, 0], "trainId": 8},
#     {"name": "non-drivable fallback", "id": 9, "color": [0, 0, 0], "trainId": 9},
#     {"name": "bridge", "id": 10, "color": [0, 0, 0], "trainId": 10},
#     {"name": "road", "id": 11, "color": [0, 0, 0], "trainId": 11},
#     {"name": "wall", "id": 12, "color": [0, 0, 0], "trainId": 12},
#     {"name": "traffic sign", "id": 13, "color": [0, 0, 0], "trainId": 13},
#     {"name": "trailer", "id": 14, "color": [0, 0, 0], "trainId": 14},
#     {"name": "animal", "id": 15, "color": [0, 0, 0], "trainId": 15},
#     {"name": "building", "id": 16, "color": [0, 0, 0], "trainId": 16},
#     {"name": "sky", "id": 17, "color": [0, 0, 0], "trainId": 17},
#     {"name": "drivable fallback", "id": 18, "color": [0, 0, 0], "trainId": 18},
#     {"name": "guard rail", "id": 19, "color": [0, 0, 0], "trainId": 19},
#     {"name": "bicycle", "id": 20, "color": [0, 0, 0], "trainId": 20},
#     {"name": "traffic light", "id": 21, "color": [0, 0, 0], "trainId": 21},
#     {"name": "polegroup", "id": 22, "color": [0, 0, 0], "trainId": 22},
#     {"name": "motorcycle", "id": 23, "color": [0, 0, 0], "trainId": 23},
#     {"name": "car", "id": 24, "color": [0, 0, 0], "trainId": 24},
#     {"name": "parking", "id": 25, "color": [0, 0, 0], "trainId": 25},
#     {"name": "fallback background", "id": 26, "color": [0, 0, 0], "trainId": 26},
#     {"name": "license plate", "id": 27, "color": [0, 0, 0], "trainId": 255},
#     {"name": "rectification border", "id": 28, "color": [0, 0, 0], "trainId": 27},
#     {"name": "train", "id": 29, "color": [0, 0, 0], "trainId": 28},
#     {"name": "rider", "id": 30, "color": [0, 0, 0], "trainId": 29},
#     {"name": "rail track", "id": 31, "color": [0, 0, 0], "trainId": 30},
#     {"name": "sidewalk", "id": 32, "color": [0, 0, 0], "trainId": 31},
#     {"name": "caravan", "id": 33, "color": [0, 0, 0], "trainId": 32},
#     {"name": "pole", "id": 34, "color": [0, 0, 0], "trainId": 33},
#     {"name": "vegetation", "id": 35, "color": [0, 0, 0], "trainId": 34},
#     {"name": "autorickshaw", "id": 36, "color": [0, 0, 0], "trainId": 35},
#     {"name": "vehicle fallback", "id": 37, "color": [0, 0, 0], "trainId": 36},
#     {"name": "unlabel", "id":255, "color":[0,0,0], "trainId": 255},
# ]

labels_info = [
    {"name": "person", "id": 0, "color": [0, 0, 0], "trainId": 4},
    {"name": "truck", "id": 1, "color": [0, 0, 0], "trainId": 10},
    {"name": "fence", "id": 2, "color": [0, 0, 0], "trainId": 15},
    {"name": "billboard", "id": 3, "color": [0, 0, 0], "trainId": 17},
    {"name": "bus", "id": 4, "color": [0, 0, 0], "trainId": 11},
    {"name": "out of roi", "id": 5, "color": [0, 0, 0], "trainId": 255},
    {"name": "curb", "id": 6, "color": [0, 0, 0], "trainId": 13},
    {"name": "obs-str-bar-fallback", "id": 7, "color": [0, 0, 0], "trainId": 21},
    {"name": "tunnel", "id": 8, "color": [0, 0, 0], "trainId": 23},
    {"name": "non-drivable fallback", "id": 9, "color": [0, 0, 0], "trainId": 3},
    {"name": "bridge", "id": 10, "color": [0, 0, 0], "trainId": 23},
    {"name": "road", "id": 11, "color": [0, 0, 0], "trainId": 0},
    {"name": "wall", "id": 12, "color": [0, 0, 0], "trainId": 14},
    {"name": "traffic sign", "id": 13, "color": [0, 0, 0], "trainId": 18},
    {"name": "trailer", "id": 14, "color": [0, 0, 0], "trainId": 12},
    {"name": "animal", "id": 15, "color": [0, 0, 0], "trainId": 4},
    {"name": "building", "id": 16, "color": [0, 0, 0], "trainId": 22},
    {"name": "sky", "id": 17, "color": [0, 0, 0], "trainId": 25},
    {"name": "drivable fallback", "id": 18, "color": [0, 0, 0], "trainId": 1},
    {"name": "guard rail", "id": 19, "color": [0, 0, 0], "trainId": 16},
    {"name": "bicycle", "id": 20, "color": [0, 0, 0], "trainId": 7},
    {"name": "traffic light", "id": 21, "color": [0, 0, 0], "trainId": 19},
    {"name": "polegroup", "id": 22, "color": [0, 0, 0], "trainId": 20},
    {"name": "motorcycle", "id": 23, "color": [0, 0, 0], "trainId": 6},
    {"name": "car", "id": 24, "color": [0, 0, 0], "trainId": 9},
    {"name": "parking", "id": 25, "color": [0, 0, 0], "trainId": 1},
    {"name": "fallback background", "id": 26, "color": [0, 0, 0], "trainId": 25},
    {"name": "license plate", "id": 27, "color": [0, 0, 0], "trainId": 255},
    {"name": "rectification border", "id": 28, "color": [0, 0, 0], "trainId": 255},
    {"name": "train", "id": 29, "color": [0, 0, 0], "trainId": 255},
    {"name": "rider", "id": 30, "color": [0, 0, 0], "trainId": 5},
    {"name": "rail track", "id": 31, "color": [0, 0, 0], "trainId": 3},
    {"name": "sidewalk", "id": 32, "color": [0, 0, 0], "trainId": 2},
    {"name": "caravan", "id": 33, "color": [0, 0, 0], "trainId": 12},
    {"name": "pole", "id": 34, "color": [0, 0, 0], "trainId": 20},
    {"name": "vegetation", "id": 35, "color": [0, 0, 0], "trainId": 24},
    {"name": "autorickshaw", "id": 36, "color": [0, 0, 0], "trainId": 8},
    {"name": "vehicle fallback", "id": 37, "color": [0, 0, 0], "trainId": 12},
    {"name": "unlabel", "id":255, "color":[0,0,0], "trainId": 255},
]

# labels_info_eval = [
# {"name": "person", "id": 0, "trainId": 0},
# {"name": "truck", "id": 1, "trainId": 1},
# {"name": "fence", "id": 2, "trainId": 2},
# {"name": "billboard", "id": 3, "trainId": 3},
# {"name": "bus", "id": 4, "trainId": 4},
# {"name": "out of roi", "id": 5, "trainId": 5},
# {"name": "curb", "id": 6, "trainId": 6},
# {"name": "obs-str-bar-fallback", "id": 7, "trainId": 7},
# {"name": "tunnel", "id": 8, "trainId": 8},
# {"name": "non-drivable fallback", "id": 9, "trainId": 9},
# {"name": "bridge", "id": 10, "trainId": 10},
# {"name": "road", "id": 11, "trainId": 11},
# {"name": "wall", "id": 12, "trainId": 12},
# {"name": "traffic sign", "id": 13, "trainId": 13},
# {"name": "trailer", "id": 14, "trainId": 255},
# {"name": "animal", "id": 15, "trainId": 15},
# {"name": "building", "id": 16, "trainId": 16},
# {"name": "sky", "id": 17, "trainId": 17},
# {"name": "drivable fallback", "id": 18, "trainId": 18},
# {"name": "guard rail", "id": 19, "trainId": 19},
# {"name": "bicycle", "id": 20, "trainId": 20},
# {"name": "traffic light", "id": 21, "trainId": 21},
# {"name": "polegroup", "id": 22, "trainId": 22},
# {"name": "motorcycle", "id": 23, "trainId": 23},
# {"name": "car", "id": 24, "trainId": 24},
# {"name": "parking", "id": 25, "trainId": 25},
# {"name": "fallback background", "id": 26, "trainId": 26},
# {"name": "license plate", "id": 27, "trainId": 255},
# {"name": "rectification border", "id": 28, "trainId": 255},
# {"name": "train", "id": 29, "trainId": 255},
# {"name": "rider", "id": 30, "trainId": 29},
# {"name": "rail track", "id": 31, "trainId": 255},
# {"name": "sidewalk", "id": 32, "trainId": 31},
# {"name": "caravan", "id": 33, "trainId": 32},
# {"name": "pole", "id": 34, "trainId": 33},
# {"name": "vegetation", "id": 35, "trainId": 34},
# {"name": "autorickshaw", "id": 36, "trainId": 35},
# {"name": "vehicle fallback", "id": 37, "trainId": 36},
# {"name": "unlabel", "id": 255, "trainId": 255},

# ]

# labels_info_train = labels_info_eval
## CityScapes -> {unify class1, unify class2, ...}
# Wall -> {Wall, fence}

class Idd(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(Idd, self).__init__(
                dataroot, annpath, trans_func, mode)
    
        
        mode = 'eval'
        # self.n_cats = 37
        # if mode == 'eval':
        self.n_cats = 26
        if mode == 'train':
            self.n_cats = 27
        
        self.lb_ignore = -1
        # self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        
        self.labels_info = labels_info
            
        for el in self.labels_info:
            if mode=='train' and el['trainId'] == 255:
                self.lb_map[el['id']] = self.n_cats - 1
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

class IddIm(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(IddIm, self).__init__(
                dataroot, annpath, trans_func, mode)
    
        
        mode = 'eval'
        # self.n_cats = 37
        # if mode == 'eval':
        self.n_cats = 26
        if mode == 'train':
            self.n_cats = 27
        
        self.lb_ignore = -1
        # self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        
        self.labels_info = labels_info
            
        for el in self.labels_info:
            if mode=='train' and el['trainId'] == 255:
                self.lb_map[el['id']] = self.n_cats - 1
            else:
                self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(0.3038, 0.3383, 0.3034), # city, rgb
            std=(0.2071, 0.2088, 0.2090),
        )
        
    def __getitem__(self, idx):
        impth = self.img_paths[idx]
        lbpth = self.lb_paths[idx]
        
        # start = time.time()
        img = cv2.imread(impth)[:, :, ::-1]

        # img = cv2.resize(img, (1920, 1280))
        label = np.array(Image.open(lbpth).convert('RGB'))
        # end = time.time()
        # print("idx: {}, cv2.imread time: {}".format(idx, end - start))
        # label = np.array(Image.open(lbpth).convert('RGB').resize((1920, 1280),Image.ANTIALIAS))
        
        # start = time.time()
        # label = self.convert_labels(label, impth)
        label = Image.fromarray(label)
        # end = time.time()
        # print("idx: {}, convert_labels time: {}".format(idx, end - start))

        if not self.lb_map is None:
            label = self.lb_map[label]
            
        if (label == 19).any():
            print(impth)
        im_lb = dict(im=img, lb=label)
        if not self.trans_func is None:
            # start = time.time()
            im_lb = self.trans_func(im_lb)
            # end = time.time()  
            # print("idx: {}, trans time: {}".format(idx, end - start))
        im_lb = self.to_tensor(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        if self.mode == 'ret_path':
            return impth, label.unsqueeze(0).detach()
        
        return img, label
        # return img.detach()

    def __len__(self):
        return self.len

    # def convert_labels(self, label, impth):
    #     mask = np.full(label.shape[:2], 2, dtype=np.uint8)
    #     # mask = np.zeros(label.shape[:2])
    #     for k, v in self.color2id.items():
    #         mask[cv2.inRange(label, np.array(k) - 1, np.array(k) + 1) == 255] = v
            
            
    #         # if v == 30 and cv2.inRange(label, np.array(k) - 1, np.array(k) + 1).any() == True:
    #         #     label[cv2.inRange(label, np.array(k) - 1, np.array(k) + 1) == 255] = [0, 0, 0]
    #         #     cv2.imshow(impth, label)
    #         #     cv2.waitKey(0)
    #     return mask

class IddCVCUDA(ImageBatchPNGDecoderPyTorchDist):
    '''
    '''
    def __init__(self, dataroot, annpath, batch_size, device_id, cuda_ctx, mode='train'):
        super(IddCVCUDA, self).__init__(
                dataroot, annpath, batch_size, device_id, cuda_ctx, mode)
    
        
        # mode = 'eval'
        # self.n_cats = 37
        # if mode == 'eval':
        self.n_cats = 26
        # if mode == 'train':
        #     self.n_cats = 27
        self.imStack = False
        # self.lb_ignore = -1
        self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        
        self.labels_info = labels_info
            
        for el in self.labels_info:
            # if mode=='train' and el['trainId'] == 255:
            #     self.lb_map[el['id']] = self.n_cats - 1
            # else:
            self.lb_map[el['id']] = el['trainId']


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    ds = IddIm('/cpfs01/projects-SSD/pujianxiangmuzu_SSD/pujian/mr/datasets/idd', 'datasets/IDD/val.txt')
    dl = DataLoader(ds,
                    batch_size = 1,
                    shuffle = False,
                    num_workers = 1,
                    drop_last = False)
    for imgs, label in dl:
        continue
        # print(len(imgs))
        # for el in imgs:
        #     print(el.size())
        # break
