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
    {"name": "unlabel", "id": 0, "trainId": 255},
    {"name": "flag", "id": 150, "trainId": 0},
    {"name": "wall", "id": 1, "trainId": 1},
    {"name": "building, edifice", "id": 2, "trainId": 2},
    {"name": "sky", "id": 3, "trainId": 3},
    {"name": "floor, flooring", "id": 4, "trainId": 4},
    {"name": "tree", "id": 5, "trainId": 5},
    {"name": "ceiling", "id": 6, "trainId": 6},
    {"name": "road, route", "id": 7, "trainId": 7},
    {"name": "bed ", "id": 8, "trainId": 8},
    {"name": "windowpane, window ", "id": 9, "trainId": 9},
    {"name": "grass", "id": 10, "trainId": 10},
    {"name": "cabinet", "id": 11, "trainId": 11},
    {"name": "sidewalk, pavement", "id": 12, "trainId": 12},
    {"name": "person, individual, someone, somebody, mortal, soul", "id": 13, "trainId": 13},
    {"name": "earth, ground", "id": 14, "trainId": 14},
    {"name": "door, double door", "id": 15, "trainId": 15},
    {"name": "table", "id": 16, "trainId": 16},
    {"name": "mountain, mount", "id": 17, "trainId": 17},
    {"name": "plant, flora, plant life", "id": 18, "trainId": 18},
    {"name": "curtain, drape, drapery, mantle, pall", "id": 19, "trainId": 19},
    {"name": "chair", "id": 20, "trainId": 20},
    {"name": "car, auto, automobile, machine, motorcar", "id": 21, "trainId": 21},
    {"name": "water", "id": 22 , "trainId": 22 },
    {"name": "painting, picture", "id": 23, "trainId": 23},
    {"name": "sofa, couch, lounge", "id": 24 , "trainId": 24 },
    {"name": "shelf", "id": 25 , "trainId": 25 },
    {"name": "house", "id": 26 , "trainId": 26 },
    {"name": "sea", "id": 27 , "trainId": 27 },
    {"name": "mirror", "id": 28, "trainId": 28},
    {"name": "rug, carpet, carpeting", "id": 29, "trainId": 29},
    {"name": "field", "id": 30, "trainId": 30},
    {"name": "armchair", "id": 31, "trainId": 31},
    {"name": "seat", "id": 32, "trainId": 32},
    {"name": "fence, fencing", "id": 33, "trainId": 33},
    {"name": "desk", "id": 34, "trainId": 34},
    {"name": "rock, stone", "id": 35, "trainId": 35},
    {"name": "wardrobe, closet, press", "id": 36, "trainId": 36},
    {"name": "lamp", "id": 37, "trainId": 37},
    {"name": "bathtub, bathing tub, bath, tub", "id": 38, "trainId": 38},
    {"name": "railing, rail", "id": 39, "trainId": 39},
    {"name": "cushion", "id": 40, "trainId": 40},
    {"name": "base, pedestal, stand", "id": 41, "trainId": 41},
    {"name": "box", "id": 42, "trainId": 42},
    {"name": "column, pillar", "id": 43, "trainId": 43},
    {"name": "signboard, sign", "id": 44, "trainId": 44},
    {"name": "chest of drawers, chest, bureau, dresser", "id": 45, "trainId": 45},
    {"name": "counter", "id": 46, "trainId": 46},
    {"name": "sand", "id": 47, "trainId": 47},
    {"name": "sink", "id": 48, "trainId": 48},
    {"name": "skyscraper", "id": 49, "trainId": 49},
    {"name": "fireplace, hearth, open fireplace", "id": 50, "trainId": 50},
    {"name": "refrigerator, icebox", "id": 51, "trainId": 51},
    {"name": "grandstand, covered stand", "id": 52, "trainId": 52},
    {"name": "path", "id": 53, "trainId": 53},
    {"name": "stairs, steps", "id": 54, "trainId": 54},
    {"name": "runway", "id": 55, "trainId": 55},
    {"name": "case, display case, showcase, vitrine", "id": 56, "trainId": 56},
    {"name": "pool table, billiard table, snooker table", "id": 57, "trainId": 57},
    {"name": "pillow", "id": 58, "trainId": 58},
    {"name": "screen door, screen", "id": 59, "trainId": 59},
    {"name": "stairway, staircase", "id": 60, "trainId": 60},
    {"name": "river", "id": 61, "trainId": 61},
    {"name": "bridge, span", "id": 62, "trainId": 62},
    {"name": "bookcase", "id": 63, "trainId": 63},
    {"name": "blind, screen", "id": 64, "trainId": 64},
    {"name": "coffee table, cocktail table", "id": 65, "trainId": 65},
    {"name": "toilet, can, commode, crapper, pot, potty, stool, throne", "id": 66, "trainId": 66},
    {"name": "flower", "id": 67, "trainId": 67},
    {"name": "book", "id": 68, "trainId": 68},
    {"name": "hill", "id": 69, "trainId": 69},
    {"name": "bench", "id": 70, "trainId": 70},
    {"name": "countertop", "id": 71, "trainId": 71},
    {"name": "stove, kitchen stove, range, kitchen range, cooking stove", "id": 72, "trainId": 72},
    {"name": "palm, palm tree", "id": 73, "trainId": 73},
    {"name": "kitchen island", "id": 74, "trainId": 74},
    {"name": "computer, computing machine, computing device, data processor, electronic computer, information processing system", "id": 75, "trainId": 75},
    {"name": "swivel chair", "id": 76, "trainId": 76},
    {"name": "boat", "id": 77, "trainId": 77},
    {"name": "bar", "id": 78, "trainId": 78},
    {"name": "arcade machine", "id": 79, "trainId": 79},
    {"name": "hovel, hut, hutch, shack, shanty", "id": 80, "trainId": 80},
    {"name": "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle", "id": 81, "trainId": 81},
    {"name": "towel", "id": 82, "trainId": 82},
    {"name": "light, light source", "id": 83, "trainId": 83},
    {"name": "truck, motortruck", "id": 84, "trainId": 84},
    {"name": "tower", "id": 85, "trainId": 85},
    {"name": "chandelier, pendant, pendent", "id": 86, "trainId": 86},
    {"name": "awning, sunshade, sunblind", "id": 87, "trainId": 87},
    {"name": "streetlight, street lamp", "id": 88, "trainId": 88},
    {"name": "booth, cubicle, stall, kiosk", "id": 89, "trainId": 89},
    {"name": "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box", "id": 90, "trainId": 90},
    {"name": "airplane, aeroplane, plane", "id": 91, "trainId": 91},
    {"name": "dirt track", "id": 92, "trainId": 92},
    {"name": "apparel, wearing apparel, dress, clothes", "id": 93, "trainId": 93},
    {"name": "pole", "id": 94, "trainId": 94},
    {"name": "land, ground, soil", "id": 95, "trainId": 95},
    {"name": "bannister, banister, balustrade, balusters, handrail", "id": 96, "trainId": 96},
    {"name": "escalator, moving staircase, moving stairway", "id": 97, "trainId": 97},
    {"name": "ottoman, pouf, pouffe, puff, hassock", "id": 98, "trainId": 98},
    {"name": "bottle", "id": 99, "trainId": 99},
    {"name": "buffet, counter, sideboard", "id": 100, "trainId": 100},
    {"name": "poster, posting, placard, notice, bill, card", "id": 101, "trainId": 101},
    {"name": "stage", "id": 102, "trainId": 102},
    {"name": "van", "id": 103, "trainId": 103},
    {"name": "ship", "id": 104, "trainId": 104},
    {"name": "fountain", "id": 105, "trainId": 105},
    {"name": "conveyer belt, conveyor belt, conveyer, conveyor, transporter", "id": 106, "trainId": 106},
    {"name": "canopy", "id": 107, "trainId": 107},
    {"name": "washer, automatic washer, washing machine", "id": 108, "trainId": 108},
    {"name": "plaything, toy", "id": 109, "trainId": 109},
    {"name": "swimming pool, swimming bath, natatorium", "id": 110, "trainId": 110},
    {"name": "stool", "id": 111, "trainId": 111},
    {"name": "barrel, cask", "id": 112, "trainId": 112},
    {"name": "basket, handbasket", "id": 113, "trainId": 113},
    {"name": "waterfall, falls", "id": 114, "trainId": 114},
    {"name": "tent, collapsible shelter", "id": 115, "trainId": 115},
    {"name": "bag", "id": 116, "trainId": 116},
    {"name": "minibike, motorbike", "id": 117, "trainId": 117},
    {"name": "cradle", "id": 118, "trainId": 118},
    {"name": "oven", "id": 119, "trainId": 119},
    {"name": "ball", "id": 120, "trainId": 120},
    {"name": "food, solid food", "id": 121, "trainId": 121},
    {"name": "step, stair", "id": 122, "trainId": 122},
    {"name": "tank, storage tank", "id": 123, "trainId": 123},
    {"name": "trade name, brand name, brand, marque", "id": 124, "trainId": 124},
    {"name": "microwave, microwave oven", "id": 125, "trainId": 125},
    {"name": "pot, flowerpot", "id": 126, "trainId": 126},
    {"name": "animal, animate being, beast, brute, creature, fauna", "id": 127, "trainId": 127},
    {"name": "bicycle, bike, wheel, cycle ", "id": 128, "trainId": 128},
    {"name": "lake", "id": 129, "trainId": 129},
    {"name": "dishwasher, dish washer, dishwashing machine", "id": 130, "trainId": 130},
    {"name": "screen, silver screen, projection screen", "id": 131, "trainId": 131},
    {"name": "blanket, cover", "id": 132, "trainId": 132},
    {"name": "sculpture", "id": 133, "trainId": 133},
    {"name": "hood, exhaust hood", "id": 134, "trainId": 134},
    {"name": "sconce", "id": 135, "trainId": 135},
    {"name": "vase", "id": 136, "trainId": 136},
    {"name": "traffic light, traffic signal, stoplight", "id": 137, "trainId": 137},
    {"name": "tray", "id": 138, "trainId": 138},
    {"name": "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin", "id": 139, "trainId": 139},
    {"name": "fan", "id": 140, "trainId": 140},
    {"name": "pier, wharf, wharfage, dock", "id": 141, "trainId": 141},
    {"name": "crt screen", "id": 142, "trainId": 142},
    {"name": "plate", "id": 143, "trainId": 143},
    {"name": "monitor, monitoring device", "id": 144, "trainId": 144},
    {"name": "bulletin board, notice board", "id": 145, "trainId": 145},
    {"name": "shower", "id": 146, "trainId": 146},
    {"name": "radiator", "id": 147, "trainId": 147},
    {"name": "glass, drinking glass", "id": 148, "trainId": 148},
    {"name": "clock", "id": 149, "trainId": 149},
]
# labels_info_train = labels_info_eval
## CityScapes -> {unify class1, unify class2, ...}
# Wall -> {Wall, fence}

class ade2016(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(ade2016, self).__init__(
                dataroot, annpath, trans_func, mode)
    
        mode = 'eval'

        self.n_cats = 150
        if mode == 'train':
            self.n_cats = 151
        
        
        self.lb_ignore = -1
        # self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        
        self.labels_info = labels_info
            
        for el in self.labels_info:
            if mode == 'train' and el['trainId'] == 255:
                self.lb_map[el['id']] = 150
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
# sun
# wall,floor,cabinet,bed,chair,sofa,table,door,window,bookshelf,picture,counter,blinds,desk,shelves,curtain,dresser,pillow,mirror,floor_mat,clothes,ceiling,books,fridge,tv,paper,towel,shower_curtain,box,whiteboard,person,night_stand,toilet,sink,lamp,bathtub,bag

## Only return img without label
class ade2016Im(BaseDatasetIm):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(ade2016Im, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.n_cats = 150
        self.lb_ignore = -1
        self.lb_map = np.arange(256).astype(np.uint8)
        for el in self.labels_info:
            self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(0.3038, 0.3383, 0.3034), # city, rgb
            std=(0.2071, 0.2088, 0.2090),
        )



if __name__ == "__main__":

    from tqdm import tqdm
    from torch.utils.data import DataLoader
    
    
    class TransformationTrain(object):

        def __init__(self, scales, cropsize):
            self.trans_func = T.Compose([
                T.RandomResizedCrop(scales, cropsize),
                T.RandomHorizontalFlip(),
                T.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4
                ),
            ])

        def __call__(self, im_lb):
            im_lb = self.trans_func(im_lb)
            return im_lb

    trans_func = TransformationTrain([0.5,2.0], [713, 713])
    
    ds = ade2016('/cpfs01/projects-HDD/pujianxiangmuzu_HDD/pujian/mr/datasets/ade/ADEChallengeData2016/', 'datasets/ADE/training.txt', trans_func, mode='train')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 2,
                    drop_last = True)
    i = 0
    index = 0
    for imgs, label in dl:
        
        if index % 10 == 0:
            print(index)
            
        index += 1
        if torch.min(label) == 255:
            print('min_label: 255')
            i += 1
    print(i)
