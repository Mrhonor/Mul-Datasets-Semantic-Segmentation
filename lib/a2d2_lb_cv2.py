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

labels_info = [
    {"name": "Car 1", "id": 0, "color": [255, 0, 0], "trainId": 0},
    {"name": "Car 2", "id": 1, "color": [200, 0, 0], "trainId": 0},
    {"name": "Car 3", "id": 2, "color": [150, 0, 0], "trainId": 0},
    {"name": "Car 4", "id": 3, "color": [128, 0, 0], "trainId": 0},
    {"name": "Bicycle 1", "id": 4, "color": [182, 89, 6], "trainId": 1},
    {"name": "Bicycle 2", "id": 5, "color": [150, 50, 4], "trainId": 1},
    {"name": "Bicycle 3", "id": 6, "color": [90, 30, 1], "trainId": 1},
    {"name": "Bicycle 4", "id": 7, "color": [90, 30, 30], "trainId": 1},
    {"name": "Pedestrian 1", "id": 8, "color": [204, 153, 255], "trainId": 2},
    {"name": "Pedestrian 2", "id": 9, "color": [189, 73, 155], "trainId": 2},
    {"name": "Pedestrian 3", "id": 10, "color": [239, 89, 191], "trainId": 2},
    {"name": "Truck 1", "id": 11, "color": [255, 128, 0], "trainId": 3},
    {"name": "Truck 2", "id": 12, "color": [200, 128, 0], "trainId": 3},
    {"name": "Truck 3", "id": 13, "color": [150, 128, 0], "trainId": 3},
    {"name": "Small vehicles 1", "id": 14, "color": [0, 255, 0], "trainId": 4},
    {"name": "Small vehicles 2", "id": 15, "color": [0, 200, 0], "trainId": 4},
    {"name": "Small vehicles 3", "id": 16, "color": [0, 150, 0], "trainId": 4},
    {"name": "Traffic signal 1", "id": 17, "color": [0, 128, 255], "trainId": 5},
    {"name": "Traffic signal 2", "id": 18, "color": [30, 28, 158], "trainId": 5},
    {"name": "Traffic signal 3", "id": 19, "color": [60, 28, 100], "trainId": 5},
    {"name": "Traffic sign 1", "id": 20, "color": [0, 255, 255], "trainId": 6},
    {"name": "Traffic sign 2", "id": 21, "color": [30, 220, 220], "trainId": 6},
    {"name": "Traffic sign 3", "id": 22, "color": [60, 157, 199], "trainId": 6},
    {"name": "Utility vehicle 1", "id": 23, "color": [255, 255, 0], "trainId": 7},
    {"name": "Utility vehicle 2", "id": 24, "color": [255, 255, 200], "trainId": 7},
    {"name": "Sidebars", "id": 25, "color": [233, 100, 0], "trainId": 8},
    {"name": "Speed bumper", "id": 26, "color": [110, 110, 0], "trainId": 9},
    {"name": "Curbstone", "id": 27, "color": [128, 128, 0], "trainId": 10},
    {"name": "Solid line", "id": 28, "color": [255, 193, 37], "trainId": 11},
    {"name": "Irrelevant signs", "id": 29, "color": [64, 0, 64], "trainId": 12},
    {"name": "Road blocks", "id": 30, "color": [185, 122, 87], "trainId": 13},
    {"name": "Tractor", "id": 31, "color": [0, 0, 100], "trainId": 14},
    {"name": "Non-drivable street", "id": 32, "color": [139, 99, 108], "trainId": 15},
    {"name": "Zebra crossing", "id": 33, "color": [210, 50, 115], "trainId": 16},
    {"name": "Obstacles / trash", "id": 34, "color": [255, 0, 128], "trainId": 17},
    {"name": "Poles", "id": 35, "color": [255, 246, 143], "trainId": 18},
    {"name": "RD restricted area", "id": 36, "color": [150, 0, 150], "trainId": 19},
    {"name": "Animals", "id": 37, "color": [204, 255, 153], "trainId": 20},
    {"name": "Grid structure", "id": 38, "color": [238, 162, 173], "trainId": 21},
    {"name": "Signal corpus", "id": 39, "color": [33, 44, 177], "trainId": 22},
    {"name": "Drivable cobblestone", "id": 40, "color": [180, 50, 180], "trainId": 23},
    {"name": "Electronic traffic", "id": 41, "color": [255, 70, 185], "trainId": 24},
    {"name": "Slow drive area", "id": 42, "color": [238, 233, 191], "trainId": 25},
    {"name": "Nature object", "id": 43, "color": [147, 253, 194], "trainId": 26},
    {"name": "Parking area", "id": 44, "color": [150, 150, 200], "trainId": 27},
    {"name": "Sidewalk", "id": 45, "color": [180, 150, 200], "trainId": 28},
    {"name": "Ego car", "id": 46, "color": [72, 209, 204], "trainId": 29},
    {"name": "Painted driv. instr.", "id": 47, "color": [200, 125, 210], "trainId": 30},
    {"name": "Traffic guide obj.", "id": 48, "color": [159, 121, 238], "trainId": 31},
    {"name": "Dashed line", "id": 49, "color": [128, 0, 255], "trainId": 32},
    {"name": "RD normal street", "id": 50, "color": [255, 0, 255], "trainId": 33},
    {"name": "Sky", "id": 51, "color": [135, 206, 255], "trainId": 34},
    {"name": "Buildings", "id": 52, "color": [241, 230, 255], "trainId": 35},
    {"name": "Blurred area", "id": 53, "color": [96, 69, 143], "trainId": 36},
    {"name": "Rain dirt", "id": 54, "color": [53, 46, 82], "trainId": 37}
]

class A2D2Data(Dataset):
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(A2D2Data, self).__init__()
        assert mode in ('train', 'val', 'test')

        self.mode = mode
        self.trans_func = trans_func
        self.n_cats = 38
        self.lb_map = np.arange(256).astype(np.uint8)

        for el in labels_info:
            self.lb_map[el['id']] = el['trainId']

        self.ignore_lb = 255

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
            mean=(0.3257, 0.3690, 0.3223),  # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )

        self.colors = []

        for el in labels_info:
            (r, g, b) = el['color']
            self.colors.append((r, g, b))
            
        self.color2id = dict(zip(self.colors, range(len(self.colors))))

    def __getitem__(self, idx):
        impth = self.img_paths[idx]
        lbpth = self.lb_paths[idx]
        img = cv2.imread(impth)[:, :, ::-1]
        label = np.array(Image.open(lbpth).convert('RGB'))
        label = self.convert_labels(label)
        label = Image.fromarray(label)

        if not self.lb_map is None:
            label = self.lb_map[label]
        im_lb = dict(im=img, lb=label)
        if not self.trans_func is None:
            im_lb = self.trans_func(im_lb)
        im_lb = self.to_tensor(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        return img.detach(), label.unsqueeze(0).detach()

    def __len__(self):
        return self.len

    def convert_labels(self, label):
        mask = np.full(label.shape[:2], 2, dtype=np.uint8)
        # mask = np.zeros(label.shape[:2])
        for k, v in self.color2id.items():
            mask[cv2.inRange(label, np.array(k) - 20, np.array(k) + 20) == 255] = v
        return mask

if __name__ == "__main__":
    pass