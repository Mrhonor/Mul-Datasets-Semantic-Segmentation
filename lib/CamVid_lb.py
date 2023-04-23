#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
from PIL import Image
import numpy as np

import lib.transform_cv2 as T



labels_info_train = [
    {"name": "Sky", "id": 0, "color": [128, 128, 128], "trainId": 0},
    {"name": "Bridge", "id": 1, "color": [0, 128, 64], "trainId": 1},
    {"name": "Building", "id": 2, "color": [128, 0, 0], "trainId": 1},
    {"name": "Wall", "id": 3, "color": [64, 192, 0], "trainId": 11},
    {"name": "Tunnel", "id": 4, "color": [64, 0, 64], "trainId": 1},
    {"name": "Archway", "id": 5, "color": [192, 0, 128], "trainId": 1},
    {"name": "Column_Pole", "id": 6, "color": [192, 192, 128], "trainId": 2},
    {"name": "TrafficCone", "id": 7, "color": [0, 0, 64], "trainId": 2},
    {"name": "Road", "id": 8, "color": [128, 64, 128], "trainId": 3},
    {"name": "LaneMkgsDriv", "id": 9, "color": [128, 0, 192], "trainId": 3},
    {"name": "LaneMkgsNonDriv", "id": 10, "color": [192, 0, 64], "trainId": 3},
    {"name": "Sidewalk", "id": 11, "color": [0, 0, 192], "trainId": 4},
    {"name": "ParkingBlock", "id": 12, "color": [64, 192, 128], "trainId": 4},
    {"name": "RoadShoulder", "id": 13, "color": [128, 128, 192], "trainId": 4},
    {"name": "Tree", "id": 14, "color": [128, 128, 0], "trainId": 5},
    {"name": "VegetationMisc", "id": 15, "color": [192, 192, 0], "trainId": 5},
    {"name": "SignSymbol", "id": 16, "color": [192, 128, 128], "trainId": 6},
    {"name": "Misc_Text", "id": 17, "color": [128, 128, 64], "trainId": 6},
    {"name": "TrafficLight", "id": 18, "color": [0, 64, 64], "trainId": 6},
    {"name": "Fence", "id": 19, "color": [64, 64, 128], "trainId": 7},
    {"name": "Car", "id": 20, "color": [64, 0, 128], "trainId": 8},
    {"name": "SUVPickupTruck", "id": 21, "color": [64, 128, 192], "trainId": 8},
    {"name": "Truck_Bus", "id": 22, "color": [192, 128, 192], "trainId": 8},
    {"name": "Train", "id": 23, "color": [192, 64, 128], "trainId": 8},
    {"name": "OtherMoving", "id": 24, "color": [128, 64, 64], "trainId": 8},
    {"name": "Pedestrian", "id": 25, "color": [64, 64, 0], "trainId":9},
    {"name": "Child", "id": 26, "color": [192, 128, 64], "trainId":9},
    {"name": "CartLuggagePram", "id": 27, "color": [64, 0, 192], "trainId": 9},
    {"name": "Animal", "id": 28, "color": [64, 128, 64], "trainId": 9},
    {"name": "Bicyclist", "id": 29, "color": [0, 128, 192], "trainId": 10},
    {"name": "MotorcycleScooter", "id": 30, "color": [192, 0, 192], "trainId": 10},
    {"name": "Void", "id": 31, "color": [0, 0, 0], "trainId": 12}
]

labels_info_eval = [
    {"name": "Sky", "id": 0, "color": [128, 128, 128], "trainId": 0},
    {"name": "Bridge", "id": 1, "color": [0, 128, 64], "trainId": 1},
    {"name": "Building", "id": 2, "color": [128, 0, 0], "trainId": 1},
    {"name": "Wall", "id": 3, "color": [64, 192, 0], "trainId": 11},
    {"name": "Tunnel", "id": 4, "color": [64, 0, 64], "trainId": 1},
    {"name": "Archway", "id": 5, "color": [192, 0, 128], "trainId": 1},
    {"name": "Column_Pole", "id": 6, "color": [192, 192, 128], "trainId": 2},
    {"name": "TrafficCone", "id": 7, "color": [0, 0, 64], "trainId": 2},
    {"name": "Road", "id": 8, "color": [128, 64, 128], "trainId": 3},
    {"name": "LaneMkgsDriv", "id": 9, "color": [128, 0, 192], "trainId": 3},
    {"name": "LaneMkgsNonDriv", "id": 10, "color": [192, 0, 64], "trainId": 3},
    {"name": "Sidewalk", "id": 11, "color": [0, 0, 192], "trainId": 4},
    {"name": "ParkingBlock", "id": 12, "color": [64, 192, 128], "trainId": 4},
    {"name": "RoadShoulder", "id": 13, "color": [128, 128, 192], "trainId": 4},
    {"name": "Tree", "id": 14, "color": [128, 128, 0], "trainId": 5},
    {"name": "VegetationMisc", "id": 15, "color": [192, 192, 0], "trainId": 5},
    {"name": "SignSymbol", "id": 16, "color": [192, 128, 128], "trainId": 6},
    {"name": "Misc_Text", "id": 17, "color": [128, 128, 64], "trainId": 6},
    {"name": "TrafficLight", "id": 18, "color": [0, 64, 64], "trainId": 6},
    {"name": "Fence", "id": 19, "color": [64, 64, 128], "trainId": 7},
    {"name": "Car", "id": 20, "color": [64, 0, 128], "trainId": 8},
    {"name": "SUVPickupTruck", "id": 21, "color": [64, 128, 192], "trainId": 8},
    {"name": "Truck_Bus", "id": 22, "color": [192, 128, 192], "trainId": 8},
    {"name": "Train", "id": 23, "color": [192, 64, 128], "trainId": 8},
    {"name": "OtherMoving", "id": 24, "color": [128, 64, 64], "trainId": 8},
    {"name": "Pedestrian", "id": 25, "color": [64, 64, 0], "trainId":9},
    {"name": "Child", "id": 26, "color": [192, 128, 64], "trainId":9},
    {"name": "CartLuggagePram", "id": 27, "color": [64, 0, 192], "trainId": 9},
    {"name": "Animal", "id": 28, "color": [64, 128, 64], "trainId": 9},
    {"name": "Bicyclist", "id": 29, "color": [0, 128, 192], "trainId": 10},
    {"name": "MotorcycleScooter", "id": 30, "color": [192, 0, 192], "trainId": 10},
    {"name": "Void", "id": 31, "color": [0, 0, 0], "trainId": -1}
]

## {unify id} : {unify name} [{city trainId}, {Camvid trainId}]
# 0: road [0, 3], 1: sidewalk [1, 4], 2: building [2, 1], 3: wall [3, 1]
# 4: fence [3, 7], 5: pole [4, 2], 6: traffic light [5, 6] 7: traffic sign [6, 6]
# 8: vegetation [7, 5], 9: terrain [8, 5], 10: sky [9, 0], 11: person [10, 9]
# 12: rider [11, 10], 13: car [12, 8], 14: truck [13,  8], 15: bus [14, 8]
# 16: train [15, 8], 17: motorcycle [16, 10], 18: bicycle [17, 10]

## Camvid -> {unify class1, unify class2, ...}
# Building -> {Building, Wall}, SignSymbol -> {traffic light, traffic sign},
# Tree -> {vegetation, terrain}, Car -> {Car, truck, bus, train}
# Bicyclist -> {motorcycle, bicycle, rider}


class CamVid(Dataset):
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(CamVid, self).__init__()
        # assert mode in ('train', 'eval', 'test')

        self.mode = mode
        self.trans_func = trans_func
        # self.n_cats = 13
        if mode == 'train':
            # self.n_cats = 13
            self.n_cats = 12
            self.labels_info = labels_info_eval
        elif mode == 'eval':
            self.n_cats = 12
            self.labels_info = labels_info_eval
        else:
            self.n_cats = 12
            self.labels_info = labels_info_eval
            
        self.lb_map = np.arange(256).astype(np.uint8)

        for el in self.labels_info:
            self.lb_map[el['id']] = el['trainId']

        self.ignore_lb = -1

        with open(annpath, 'r') as fr:
            pairs = fr.read().splitlines()

        self.img_paths, self.lb_paths = [], []
        for pair in pairs:
            imgpth, lbpth = pair.split(',')
            self.img_paths.append(osp.join(dataroot, imgpth))
            self.lb_paths.append(osp.join(dataroot, lbpth))

        assert len(self.img_paths) == len(self.lb_paths)
        self.len = len(self.img_paths)

        self.to_tensor = T.ToTensor(
            mean=(0.3038, 0.3383, 0.3034),  # city, rgb
            std=(0.2071, 0.2088, 0.2090),
        )

        # self.to_tensor = T.ToTensor(
        #     mean=(0.3257, 0.3690, 0.3223),  # city, rgb
        #     std=(0.2112, 0.2148, 0.2115),
        # )
        
        self.colors = []

        for el in self.labels_info:
            (r, g, b) = el['color']
            self.colors.append((r, g, b))
            
        self.color2id = dict(zip(self.colors, range(len(self.colors))))
        
        # self.detail_net = DetailBranch(n_bn=1)
        # self.segment_net = SegmentBranch(n_bn=1)
        
        
        # self.detail_net.load_state_dict(torch.load('res/detail_precise_bn.pth', map_location='cuda:0'), strict=False)
        # self.detail_net.cuda()
        
        # self.segment_net.load_state_dict(torch.load('res/segment_precise_bn_40.pth', map_location='cuda:0'), strict=False)
        # self.segment_net.cuda()
        # self.detail_net.eval()
        # self.segment_net.eval()
        
        
        

    def __getitem__(self, idx):
        impth = self.img_paths[idx]
        lbpth = self.lb_paths[idx]
        # print(impth)

        img = cv2.imread(impth)[:, :, ::-1]
        if img is None:
            print(impth)
        
        # img = cv2.resize(img, (960, 768))
        label = np.array(Image.open(lbpth).convert('RGB'))
        label = self.convert_labels(label, impth)
        label = Image.fromarray(label)

        if not self.lb_map is None:
            label = self.lb_map[label]
        im_lb = dict(im=img, lb=label)
        if not self.trans_func is None:
            im_lb = self.trans_func(im_lb)
        im_lb = self.to_tensor(im_lb)
        img, label = im_lb['im'], im_lb['lb']

        if self.mode == 'ret_path':
            return impth, label
        
        return img.detach(), label.unsqueeze(0).detach()
        # return img.detach()
    
        # img = img.detach().unsqueeze(0)
        # img = img.cuda()
        # # print(img.shape)
        # fd = self.detail_net(img)
        # fs = self.segment_net(img)

        # # print(fd[0].shape)
        # return fd[0], fs[0]

    def __len__(self):
        return self.len

    def convert_labels(self, label, impth):
        mask = np.full(label.shape[:2], 2, dtype=np.uint8)
        # mask = np.zeros(label.shape[:2])
        for k, v in self.color2id.items():
            mask[cv2.inRange(label, np.array(k) - 1, np.array(k) + 1) == 255] = v
            
            
            # if v == 19 and cv2.inRange(label, np.array(k) - 1, np.array(k) + 1).any() == True:
            #     label[cv2.inRange(label, np.array(k) - 1, np.array(k) + 1) == 255] = [0, 0, 0]
            #     cv2.imshow(impth, label)
            #     cv2.waitKey(0)
        return mask

if __name__ == "__main__":
    pass