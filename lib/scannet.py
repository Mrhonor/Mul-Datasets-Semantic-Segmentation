#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
sys.path.insert(0, '.')
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
{"id": 0, "trainId": 255, "name":   "unlabel"},
{"id": 1, "trainId": 0, "name":	"wall"},
{"id": 2, "trainId": 1, "name":	"floor"},
{"id": 3, "trainId": 2, "name":	"cabinet"},
{"id": 4, "trainId": 3, "name":	"bed"},
{"id": 5, "trainId": 4, "name":	"chair"},
{"id": 6, "trainId": 5, "name":	"sofa"},
{"id": 7, "trainId": 6, "name":	"table"},
{"id": 8, "trainId": 7, "name":	"door"},
{"id": 9, "trainId": 8, "name":	"window"},
{"id": 10, "trainId": 9, "name":	"bookshelf"},
{"id": 11, "trainId": 10, "name":	"picture"},
{"id": 12, "trainId": 11, "name":	"counter"},
{"id": 13, "trainId": 255, "name":	"blinds"},
{"id": 14, "trainId": 12, "name":	"desk"},
{"id": 15, "trainId": 255, "name":	"shelves"},
{"id": 16, "trainId": 13, "name":	"curtain"},
{"id": 17, "trainId": 255, "name":	"dresser"},
{"id": 18, "trainId": 255, "name":	"pillow"},
{"id": 19, "trainId": 255, "name":	"mirror"},
{"id": 20, "trainId": 255, "name":	"floor mat"},
{"id": 21, "trainId": 255, "name":	"clothes"},
{"id": 22, "trainId": 255, "name":	"ceiling"},
{"id": 23, "trainId": 255, "name":	"books"},
{"id": 24, "trainId": 14, "name":	"refridgerator"},
{"id": 25, "trainId": 255, "name":	"television"},
{"id": 26, "trainId": 255, "name":	"paper"},
{"id": 27, "trainId": 255, "name":	"towel"},
{"id": 28, "trainId": 15, "name":	"shower curtain"},
{"id": 29, "trainId": 255, "name":	"box"},
{"id": 30, "trainId": 255, "name":	"whiteboard"},
{"id": 31, "trainId": 255, "name":	"person"},
{"id": 32, "trainId": 255, "name":	"nightstand"},
{"id": 33, "trainId": 16, "name":	"toilet"},
{"id": 34, "trainId": 17, "name":	"sink"},
{"id": 35, "trainId": 255, "name":	"lamp"},
{"id": 36, "trainId": 18, "name":	"bathtub"},
{"id": 37, "trainId": 255, "name":	"bag"},
{"id": 38, "trainId": 255, "name":	"otherstructure"},
{"id": 39, "trainId": 19, "name":	"otherfurniture"},
{"id": 40, "trainId": 255, "name":	"otherprop"},
]


class scannet(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(scannet, self).__init__(
                dataroot, annpath, trans_func, mode)
    

        self.n_cats = 20
        # if mode == 'train':
        #     self.n_cats = 20
        
        self.lb_ignore = -1
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




if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    class TransformationVal(object):

        def __call__(self, im_lb):
            im, lb = im_lb['im'], im_lb['lb']
            return dict(im=im, lb=lb)

    ds = CityScapes('/home/mr/datasets', 'datasets/Cityscapes/train.txt', trans_func=TransformationVal(), mode='ret_path')
    dl = DataLoader(ds,
                    batch_size = 1,
                    shuffle = False,
                    num_workers = 1,
                    drop_last = False)
    for imgs, label in dl:
        if torch.min(label) == 255:
            print(imgs)
