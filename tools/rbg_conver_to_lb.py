
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
import time


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
    {"name": "Blurred area", "id": 53, "color": [96, 69, 143], "trainId": -1},
    {"name": "Rain dirt", "id": 54, "color": [53, 46, 82], "trainId": -1}
]

labels_info_eval = [
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

# labels_info = labels_info_eval

class A2D2Data(Dataset):
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(A2D2Data, self).__init__()
        # assert mode in ('train', 'eval', 'test')

        self.mode = mode
        self.trans_func = trans_func
        # self.n_cats = 38
        self.n_cats = 36
        self.lb_map = np.arange(256).astype(np.uint8)

        for el in labels_info:
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

        
        self.colors = []

        for el in labels_info:
            (r, g, b) = el['color']
            self.colors.append((r, g, b))
            
        self.color2id = dict(zip(self.colors, range(len(self.colors))))

    def __getitem__(self, idx):
        # impth = self.img_paths[idx]
        lbpth = self.lb_paths[idx]
        
        # start = time.time()
        # img = cv2.imread(impth)[:, :, ::-1]

        # img = cv2.resize(img, (1920, 1280))
        label = np.array(Image.open(lbpth).convert('RGB'))
        # end = time.time()
        # print("idx: {}, cv2.imread time: {}".format(idx, end - start))
        # label = np.array(Image.open(lbpth).convert('RGB').resize((1920, 1280),Image.ANTIALIAS))
        
        # start = time.time()
        label = self.convert_labels(label, None)
        # label = Image.fromarray(label)
        new_lbpth = lbpth.replace(".png", "_L.png")
        cv2.imwrite(new_lbpth, label)
        # im = np.array(cv2.imread(new_lbpth, cv2.IMREAD_GRAYSCALE))
        # print(im.shape)
        # print(label.shape)
        # if (im == label).any() == False:
        #     print("wrong!")
        
        return idx
        # end = time.time()
        # print("idx: {}, convert_labels time: {}".format(idx, end - start))

        
        # return img.detach(), label.unsqueeze(0).detach()
        # return img.detach()

    def __len__(self):
        return self.len

    def convert_labels(self, label, impth):
        mask = np.full(label.shape[:2], 2, dtype=np.uint8)
        # mask = np.zeros(label.shape[:2])
        for k, v in self.color2id.items():
            mask[cv2.inRange(label, np.array(k) - 1, np.array(k) + 1) == 255] = v
            
            
            # if v == 30 and cv2.inRange(label, np.array(k) - 1, np.array(k) + 1).any() == True:
            #     label[cv2.inRange(label, np.array(k) - 1, np.array(k) + 1) == 255] = [0, 0, 0]
            #     cv2.imshow(impth, label)
            #     cv2.waitKey(0)
        return mask

        
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

        
        self.colors = []

        for el in self.labels_info:
            (r, g, b) = el['color']
            self.colors.append((r, g, b))
            
        self.color2id = dict(zip(self.colors, range(len(self.colors))))
        
    def __getitem__(self, idx):
        # impth = self.img_paths[idx]
        lbpth = self.lb_paths[idx]
        
        # start = time.time()
        # img = cv2.imread(impth)[:, :, ::-1]

        # img = cv2.resize(img, (1920, 1280))
        label = np.array(Image.open(lbpth).convert('RGB'))
        # end = time.time()
        # print("idx: {}, cv2.imread time: {}".format(idx, end - start))
        # label = np.array(Image.open(lbpth).convert('RGB').resize((1920, 1280),Image.ANTIALIAS))
        
        # start = time.time()
        label = self.convert_labels(label, None)
        # label = Image.fromarray(label)
        new_lbpth = lbpth.replace(".png", "_L.png")
        cv2.imwrite(new_lbpth, label)

        return idx


    def __len__(self):
        return self.len


    def convert_labels(self, label, impth):
        mask = np.full(label.shape[:2], 2, dtype=np.uint8)
        # mask = np.zeros(label.shape[:2])
        for k, v in self.color2id.items():
            mask[cv2.inRange(label, np.array(k) - 1, np.array(k) + 1) == 255] = v
            
            
            # if v == 30 and cv2.inRange(label, np.array(k) - 1, np.array(k) + 1).any() == True:
            #     label[cv2.inRange(label, np.array(k) - 1, np.array(k) + 1) == 255] = [0, 0, 0]
            #     cv2.imshow(impth, label)
            #     cv2.waitKey(0)
        return mask

labels_info = [
{"name": "Bird", "id": 0, "color": [165, 42, 42], "trainId": 0},
{"name": "Ground Animal", "id": 1, "color": [0, 192, 0], "trainId": 1},
{"name": "Ambiguous Barrier", "id": 2, "color": [250, 170, 31], "trainId": 255},
{"name": "Concrete Block", "id": 3, "color": [250, 170, 32], "trainId": 255},
{"name": "Curb", "id": 4, "color": [196, 196, 196], "trainId": 2},
{"name": "Fence", "id": 5, "color": [190, 153, 153], "trainId": 3},
{"name": "Guard Rail", "id": 6, "color": [180, 165, 180], "trainId": 4},
{"name": "Barrier", "id": 7, "color": [90, 120, 150], "trainId": 5},
{"name": "Road Median", "id": 8, "color": [250, 170, 33], "trainId": 6},
{"name": "Road Side", "id": 9, "color": [250, 170, 34], "trainId": 7},
{"name": "Lane Separator", "id": 10, "color": [128, 128, 128], "trainId": 255},
{"name": "Temporary Barrier", "id": 11, "color": [250, 170, 35], "trainId": 8},
{"name": "Wall", "id": 12, "color": [102, 102, 156], "trainId": 9},
{"name": "Bike Lane", "id": 13, "color": [128, 64, 255], "trainId": 10},
{"name": "Crosswalk - Plain", "id": 14, "color": [140, 140, 200], "trainId": 11},
{"name": "Curb Cut", "id": 15, "color": [170, 170, 170], "trainId": 255},
{"name": "Driveway", "id": 16, "color": [250, 170, 36], "trainId": 12},
{"name": "Parking", "id": 17, "color": [250, 170, 160], "trainId": 13},
{"name": "Parking Aisle", "id": 18, "color": [250, 170, 37], "trainId": 14},
{"name": "Pedestrian Area", "id": 19, "color": [96, 96, 96], "trainId": 15},
{"name": "Rail Track", "id": 20, "color": [230, 150, 140], "trainId": 16},
{"name": "Road", "id": 21, "color": [128, 64, 128], "trainId": 17},
{"name": "Road Shoulder", "id": 22, "color": [110, 110, 110], "trainId": 255},
{"name": "Service Lane", "id": 23, "color": [110, 110, 110], "trainId": 18},
{"name": "Sidewalk", "id": 24, "color": [244, 35, 232], "trainId": 19},
{"name": "Traffic Island", "id": 25, "color": [128, 196, 128], "trainId": 20},
{"name": "Bridge", "id": 26, "color": [150, 100, 100], "trainId": 21},
{"name": "Building", "id": 27, "color": [70, 70, 70], "trainId": 22},
{"name": "Garage", "id": 28, "color": [150, 150, 150], "trainId": 23},
{"name": "Tunnel", "id": 29, "color": [150, 120, 90], "trainId": 24},
{"name": "Person", "id": 30, "color": [220, 20, 60], "trainId": 255},
{"name": "Person Group", "id": 31, "color": [220, 20, 60], "trainId": 25},
{"name": "Bicyclist", "id": 32, "color": [255, 0, 0], "trainId": 26},
{"name": "Motorcyclist", "id": 33, "color": [255, 0, 100], "trainId": 27},
{"name": "Other Rider", "id": 34, "color": [255, 0, 200], "trainId": 28},
{"name": "Lane Marking - Dashed Line", "id": 35, "color": [255, 255, 255], "trainId": 255},
{"name": "Lane Marking - Straight Line", "id": 36, "color": [255, 255, 255], "trainId": 255},
{"name": "Lane Marking - Zigzag Line", "id": 37, "color": [250, 170, 29], "trainId": 255},
{"name": "Lane Marking - Ambiguous", "id": 38, "color": [250, 170, 28], "trainId": 29},
{"name": "Lane Marking - Arrow (Left)", "id": 39, "color": [250, 170, 26], "trainId": 255},
{"name": "Lane Marking - Arrow (Other)", "id": 40, "color": [250, 170, 25], "trainId": 30},
{"name": "Lane Marking - Arrow (Right)", "id": 41, "color": [250, 170, 24], "trainId": 31},
{"name": "Lane Marking - Arrow (Split Left or Straight)", "id": 42, "color": [250, 170, 22], "trainId": 255},
{"name": "Lane Marking - Arrow (Split Right or Straight)", "id": 43, "color": [250, 170, 21], "trainId": 32},
{"name": "Lane Marking - Arrow (Straight)", "id": 44, "color": [250, 170, 20], "trainId": 33},
{"name": "Lane Marking - Crosswalk", "id": 45, "color": [255, 255, 255], "trainId": 255},
{"name": "Lane Marking - Give Way (Row)", "id": 46, "color": [250, 170, 19], "trainId": 34},
{"name": "Lane Marking - Give Way (Single)", "id": 47, "color": [250, 170, 18], "trainId": 35},
{"name": "Lane Marking - Hatched (Chevron)", "id": 48, "color": [250, 170, 12], "trainId": 255},
{"name": "Lane Marking - Hatched (Diagonal)", "id": 49, "color": [250, 170, 11], "trainId": 36},
{"name": "Lane Marking - Other", "id": 50, "color": [255, 255, 255], "trainId": 255},
{"name": "Lane Marking - Stop Line", "id": 51, "color": [255, 255, 255], "trainId": 255},
{"name": "Lane Marking - Symbol (Bicycle)", "id": 52, "color": [250, 170, 16], "trainId": 255},
{"name": "Lane Marking - Symbol (Other)", "id": 53, "color": [250, 170, 15], "trainId": 255},
{"name": "Lane Marking - Text", "id": 54, "color": [250, 170, 15], "trainId": 37},
{"name": "Lane Marking (only) - Dashed Line", "id": 55, "color": [255, 255, 255], "trainId": 255},
{"name": "Lane Marking (only) - Crosswalk", "id": 56, "color": [255, 255, 255], "trainId": 255},
{"name": "Lane Marking (only) - Other", "id": 57, "color": [255, 255, 255], "trainId": 255},
{"name": "Lane Marking (only) - Test", "id": 58, "color": [255, 255, 255], "trainId": 38},
{"name": "Mountain", "id": 59, "color": [64, 170, 64], "trainId": 39},
{"name": "Sand", "id": 60, "color": [230, 160, 50], "trainId": 40},
{"name": "Sky", "id": 61, "color": [70, 130, 180], "trainId": 41},
{"name": "Snow", "id": 62, "color": [190, 255, 255], "trainId": 42},
{"name": "Terrain", "id": 63, "color": [152, 251, 152], "trainId": 43},
{"name": "Vegetation", "id": 64, "color": [107, 142, 35], "trainId": 44},
{"name": "Water", "id": 65, "color": [0, 170, 30], "trainId": 45},
{"name": "Banner", "id": 66, "color": [255, 255, 128], "trainId": 46},
{"name": "Bench", "id": 67, "color": [250, 0, 30], "trainId": 47},
{"name": "Bike Rack", "id": 68, "color": [100, 140, 180], "trainId": 48},
{"name": "Catch Basin", "id": 69, "color": [220, 128, 128], "trainId": 49},
{"name": "CCTV Camera", "id": 70, "color": [222, 40, 40], "trainId": 50},
{"name": "Fire Hydrant", "id": 71, "color": [100, 170, 30], "trainId": 51},
{"name": "Junction Box", "id": 72, "color": [40, 40, 40], "trainId": 52},
{"name": "Mailbox", "id": 73, "color": [33, 33, 33], "trainId": 255},
{"name": "Manhole", "id": 74, "color": [100, 128, 160], "trainId": 53},
{"name": "Parking Meter", "id": 75, "color": [20, 20, 255], "trainId": 54},
{"name": "Phone Booth", "id": 76, "color": [142, 0, 0], "trainId": 55},
{"name": "Pothole", "id": 77, "color": [70, 100, 150], "trainId": 56},
{"name": "Signage - Advertisement", "id": 78, "color": [250, 171, 30], "trainId": 255},
{"name": "Signage - Ambiguous", "id": 79, "color": [250, 172, 30], "trainId": 255},
{"name": "Signage - Back", "id": 80, "color": [250, 173, 30], "trainId": 57},
{"name": "Signage - Information", "id": 81, "color": [250, 174, 30], "trainId": 58},
{"name": "Signage - Other", "id": 82, "color": [250, 175, 30], "trainId": 59},
{"name": "Signage - Store", "id": 83, "color": [250, 176, 30], "trainId": 60},
{"name": "Street Light", "id": 84, "color": [210, 170, 100], "trainId": 61},
{"name": "Pole", "id": 85, "color": [153, 153, 153], "trainId": 255},
{"name": "Pole Group", "id": 86, "color": [153, 153, 153], "trainId": 62},
{"name": "Traffic Sign Frame", "id": 87, "color": [128, 128, 128], "trainId": 63},
{"name": "Utility Pole", "id": 88, "color": [0, 0, 80], "trainId": 64},
{"name": "Traffic Cone", "id": 89, "color": [210, 60, 60], "trainId": 65},
{"name": "Traffic Light - General (Single)", "id": 90, "color": [250, 170, 30], "trainId": 255},
{"name": "Traffic Light - Pedestrians", "id": 91, "color": [250, 170, 30], "trainId": 255},
{"name": "Traffic Light - General (Upright)", "id": 92, "color": [250, 170, 30], "trainId": 255},
{"name": "Traffic Light - General (Horizontal)", "id": 93, "color": [250, 170, 30], "trainId": 255},
{"name": "Traffic Light - Cyclists", "id": 94, "color": [250, 170, 30], "trainId": 255},
{"name": "Traffic Light - Other", "id": 95, "color": [250, 170, 30], "trainId": 66},
{"name": "Traffic Sign - Ambiguous", "id": 96, "color": [192, 192, 192], "trainId": 255},
{"name": "Traffic Sign (Back)", "id": 97, "color": [192, 192, 192], "trainId": 255},
{"name": "Traffic Sign - Direction (Back)", "id": 98, "color": [192, 192, 192], "trainId": 255},
{"name": "Traffic Sign - Direction (Front)", "id": 99, "color": [220, 220, 0], "trainId": 255},
{"name": "Traffic Sign (Front)", "id": 100, "color": [220, 220, 0], "trainId": 255},
{"name": "Traffic Sign - Parking", "id": 101, "color": [0, 0, 196], "trainId": 67},
{"name": "Traffic Sign - Temporary (Back)", "id": 102, "color": [192, 192, 192], "trainId": 68},
{"name": "Traffic Sign - Temporary (Front)", "id": 103, "color": [220, 220, 0], "trainId": 69},
{"name": "Trash Can", "id": 104, "color": [140, 140, 20], "trainId": 70},
{"name": "Bicycle", "id": 105, "color": [119, 11, 32], "trainId": 71},
{"name": "Boat", "id": 106, "color": [150, 0, 255], "trainId": 72},
{"name": "Bus", "id": 107, "color": [0, 60, 100], "trainId": 73},
{"name": "Car", "id": 108, "color": [0, 0, 142], "trainId": 255},
{"name": "Caravan", "id": 109, "color": [0, 0, 90], "trainId": 74},
{"name": "Motorcycle", "id": 110, "color": [0, 0, 230], "trainId": 75},
{"name": "On Rails", "id": 111, "color": [0, 80, 100], "trainId": 76},
{"name": "Other Vehicle", "id": 112, "color": [128, 64, 64], "trainId": 77},
{"name": "Trailer", "id": 113, "color": [0, 0, 110], "trainId": 78},
{"name": "Truck", "id": 114, "color": [0, 0, 70], "trainId": 79},
{"name": "Vehicle Group", "id": 115, "color": [0, 0, 142], "trainId": 80},
{"name": "Wheeled Slow", "id": 116, "color": [0, 0, 192], "trainId": 81},
{"name": "Water Valve", "id": 117, "color": [170, 170, 170], "trainId": 82},
{"name": "Car Mount", "id": 118, "color": [32, 32, 32], "trainId": 83},
{"name": "Dynamic", "id": 119, "color": [111, 74, 0], "trainId": 84},
{"name": "Ego Vehicle", "id": 120, "color": [120, 10, 10], "trainId": 85},
{"name": "Ground", "id": 121, "color": [81, 0, 81], "trainId": 86},
{"name": "Static", "id": 122, "color": [111, 111, 0], "trainId": 87},
{"name": "Unlabeled", "id": 123, "color": [0, 0, 0], "trainId": 255},
]
# labels_info_train = labels_info_eval
## CityScapes -> {unify class1, unify class2, ...}
# Wall -> {Wall, fence}

class Mapi(Dataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(Mapi, self).__init__()
                
    
        self.mode = mode
        self.trans_func = trans_func
        # self.n_cats = 38
        self.n_cats = 88
        self.lb_map = np.arange(256).astype(np.uint8)

        for el in labels_info:
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

        # self.to_tensor = T.ToTensor(
        #     mean=(0.3038, 0.3383, 0.3034), # city, rgb
        #     std=(0.2071, 0.2088, 0.2090),
        # )
        
        self.colors = []

        for el in labels_info:
            (r, g, b) = el['color']
            self.colors.append((r, g, b))
            
        self.color2id = dict(zip(self.colors, range(len(self.colors))))

    def __getitem__(self, idx):
        # impth = self.img_paths[idx]
        lbpth = self.lb_paths[idx]
        label = np.array(Image.open(lbpth).convert('RGB'))

        label = self.convert_labels(label, None)
        new_lbpth = lbpth.replace(".png", "_L.png")
        cv2.imwrite(new_lbpth, label)

        return idx

    def __len__(self):
        return self.len

    def convert_labels(self, label, impth):
        mask = np.full(label.shape[:2], 2, dtype=np.uint8)
        # mask = np.zeros(label.shape[:2])
        for k, v in self.color2id.items():
            mask[cv2.inRange(label, np.array(k), np.array(k)) == 255] = v
            
            
            # if v == 30 and cv2.inRange(label, np.array(k) - 1, np.array(k) + 1).any() == True:
            #     label[cv2.inRange(label, np.array(k) - 1, np.array(k) + 1) == 255] = [0, 0, 0]
            #     cv2.imshow(impth, label)
            #     cv2.waitKey(0)
        return mask


if __name__ == "__main__":
    dataroot = "/home/pujian/mr/datasets/mapi"
    for mode in {'training', 'validation'}:
        annpath = f'/cpfs01/projects-SSD/pujianxiangmuzu_SSD/pujian/mr/Mul-Datasets-Semantic-Segmentation/datasets/mapi/{mode}.txt'
        data_reader = Mapi(dataroot, annpath)
        for i in data_reader:
            if i % 100 == 0:
                print(i)