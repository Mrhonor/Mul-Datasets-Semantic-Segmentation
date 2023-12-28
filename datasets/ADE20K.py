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
    {"id": 1, "trainId": 1, "name": "bed"},
    {"id": 2, "trainId": 2, "name": "windowpane"},
    {"id": 3, "trainId": 3, "name": "cabinet"},
    {"id": 4, "trainId": 4, "name": "person"},
    {"id": 5, "trainId": 5, "name": "door"},
    {"id": 6, "trainId": 6, "name": "table"},
    {"id": 7, "trainId": 7, "name": "curtain"},
    {"id": 8, "trainId": 8, "name": "chair"},
    {"id": 9, "trainId": 9, "name": "car"},
    {"id": 10, "trainId": 10, "name": "painting"},
    {"id": 11, "trainId": 11, "name": "sofa"},
    {"id": 12, "trainId": 12, "name": "shelf"},
    {"id": 13, "trainId": 13, "name": "mirror"},
    {"id": 14, "trainId": 14, "name": "armchair"},
    {"id": 15, "trainId": 15, "name": "seat"},
    {"id": 16, "trainId": 16, "name": "fence"},
    {"id": 17, "trainId": 17, "name": "desk"},
    {"id": 18, "trainId": 18, "name": "wardrobe"},
    {"id": 19, "trainId": 19, "name": "lamp"},
    {"id": 20, "trainId": 20, "name": "bathtub"},
    {"id": 21, "trainId": 21, "name": "railing"},
    {"id": 22, "trainId": 22, "name": "cushion"},
    {"id": 23, "trainId": 23, "name": "box"},
    {"id": 24, "trainId": 24, "name": "column"},
    {"id": 25, "trainId": 25, "name": "signboard"},
    {"id": 26, "trainId": 26, "name": "chest of drawers"},
    {"id": 27, "trainId": 27, "name": "counter"},
    {"id": 28, "trainId": 28, "name": "sink"},
    {"id": 29, "trainId": 29, "name": "fireplace"},
    {"id": 30, "trainId": 30, "name": "refrigerator"},
    {"id": 31, "trainId": 31 "name": "stairs"},
    {"id": 32, "trainId": 32 "name": "case"},
    {"id": 33, "trainId": 33, "name": "pool table"},
    {"id": 34, "trainId": 34, "name": "pillow"},
    {"id": 35, "trainId": 35, "name": "screen door"},
    {"id": 36, "trainId": 36, "name": "bookcase"},
    {"id": 37, "trainId": 37, "name": "coffee table"},
    {"id": 38, "trainId": 38, "name": "toilet"},
    {"id": 39, "trainId": 39, "name": "flower"},
    {"id": 40, "trainId": 40, "name": "book"},
    {"id": 41, "trainId": 41, "name": "bench"},
    {"id": 42, "trainId": 42, "name": "countertop"},
    {"id": 43, "trainId": 43, "name": "stove"},
    {"id": 44, "trainId": 44, "name": "palm"},
    {"id": 45, "trainId": 45, "name": "kitchen island"},
    {"id": 46, "trainId": 46, "name": "computer"},
    {"id": 47, "trainId": 47, "name": "swivel chair"},
    {"id": 48, "trainId": 48, "name": "boat"},
    {"id": 49, "trainId": 49, "name": "arcade machine"},
    {"id": 50, "trainId": 50, "name": "bus"},
    {"id": 51, "trainId": 51, "name": "towel"},
    {"id": 52, "trainId": 52, "name": "light"},
    {"id": 53, "trainId": 53, "name": "truck"},
    {"id": 54, "trainId": 54, "name": "chandelier"},
    {"id": 55, "trainId": 55, "name": "awning"},
    {"id": 56, "trainId": 56, "name": "streetlight"},
    {"id": 57, "trainId": 57, "name": "booth"},
    {"id": 58, "trainId": 58, "name": "television receiver"},
    {"id": 59, "trainId": 59, "name": "airplane"},
    {"id": 60, "trainId": 60, "name": "apparel"},
    {"id": 61, "trainId": 61, "name": "pole"},
    {"id": 62, "trainId": 62, "name": "bannister"},
    {"id": 63, "trainId": 63, "name": "ottoman"},
    {"id": 64, "trainId": 64, "name": "bottle"},
    {"id": 65, "trainId": 65, "name": "van"},
    {"id": 66, "trainId": 66, "name": "ship"},
    {"id": 67, "trainId": 67, "name": "fountain"},
    {"id": 68, "trainId": 68, "name": "washer"},
    {"id": 69, "trainId": 69, "name": "plaything"},
    {"id": 70, "trainId": 70, "name": "stool"},
    {"id": 71, "trainId": 71, "name": "barrel"},
    {"id": 72, "trainId": 72, "name": "basket"},
    {"id": 73, "trainId": 73, "name": "bag"},
    {"id": 74, "trainId": 74, "name": "minibike"},
    {"id": 75, "trainId": 75, "name": "oven"},
    {"id": 76, "trainId": 76, "name": "ball"},
    {"id": 77, "trainId": 77, "name": "food"},
    {"id": 78, "trainId": 78, "name": "step"},
    {"id": 79, "trainId": 79, "name": "trade name"},
    {"id": 80, "trainId": 80, "name": "microwave"},
    {"id": 81, "trainId": 81, "name": "pot"},
    {"id": 82, "trainId": 82, "name": "animal"},
    {"id": 83, "trainId": 83, "name": "bicycle"},
    {"id": 84, "trainId": 84, "name": "dishwasher"},
    {"id": 85, "trainId": 85, "name": "screen"},
    {"id": 86, "trainId": 86, "name": "sculpture"},
    {"id": 87, "trainId": 87, "name": "hood"},
    {"id": 88, "trainId": 88, "name": "sconce"},
    {"id": 89, "trainId": 89, "name": "vase"},
    {"id": 90, "trainId": 90, "name": "traffic light"},
    {"id": 91, "trainId": 91, "name": "tray"},
    {"id": 92, "trainId": 92, "name": "ashcan"},
    {"id": 93, "trainId": 93, "name": "fan"},
    {"id": 94, "trainId": 94, "name": "plate"},
    {"id": 95, "trainId": 95, "name": "monitor"},
    {"id": 96, "trainId": 96, "name": "bulletin board"},
    {"id": 97, "trainId": 97, "name": "radiator"},
    {"id": 98, "trainId": 98, "name": "glass"},
    {"id": 99, "trainId": 99, "name": "clock"},
    {"id": 100, "trainId": 100, "name": "flag"}
]

# labels_info_train = labels_info_eval
## CityScapes -> {unify class1, unify class2, ...}
# Wall -> {Wall, fence}

class ade20k(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(ade20k, self).__init__(
                dataroot, annpath, trans_func, mode)
    
        
        self.n_cats = 100
        
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
        
    def __getitem__(self, idx):
        impth = self.img_paths[idx]
        lbpth = self.lb_paths[idx]
        
        # start = time.time()
        img = cv2.imread(impth)[:, :, ::-1]

        # img = cv2.resize(img, (1920, 1280))
        label = np.array(Image.open(lbpth).convert('RGB'))
        R = label[:,:,0]
        G = label[:,:,1]
        B = label[:,:,2]
        ObjectClassMasks = (R/10).astype(np.int32)*256+(G.astype(np.int32))
        
        # end = time.time()
        # print("idx: {}, cv2.imread time: {}".format(idx, end - start))
        # label = np.array(Image.open(lbpth).convert('RGB').resize((1920, 1280),Image.ANTIALIAS))
        
        # start = time.time()
        # label = self.convert_labels(label, impth)
        ObjectClassMasks = Image.fromarray(ObjectClassMasks)
        # end = time.time()
        # print("idx: {}, convert_labels time: {}".format(idx, end - start))

        if not self.lb_map is None:
            ObjectClassMasks = self.lb_map[ObjectClassMasks]
        im_lb = dict(im=img, lb=ObjectClassMasks)
        if not self.trans_func is None:
            # start = time.time()
            im_lb = self.trans_func(im_lb)
            # end = time.time()  
            # print("idx: {}, trans time: {}".format(idx, end - start))
        im_lb = self.to_tensor(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        if self.mode == 'ret_path':
            return impth, label.unsqueeze(0).detach()
        
        return img.detach(), label.unsqueeze(0).detach()
    
    
    def __len__(self):
        return self.len

    def convert_labels(self, label, impth):
        mask = np.full(label.shape[:2], 2, dtype=np.uint8)
        # mask = np.zeros(label.shape[:2])
        for k, v in self.color2id.items():
            mask[cv2.inRange(label, np.array(k) - 1, np.array(k) + 1) == 255] = v
            
        return mask
    
        # self.to_tensor = T.ToTensor(
        #     mean=(0.3257, 0.3690, 0.3223), # city, rgb
        #     std=(0.2112, 0.2148, 0.2115),
        # )
# sun
# wall,floor,cabinet,bed,chair,sofa,table,door,window,bookshelf,picture,counter,blinds,desk,shelves,curtain,dresser,pillow,mirror,floor_mat,clothes,ceiling,books,fridge,tv,paper,towel,shower_curtain,box,whiteboard,person,night_stand,toilet,sink,lamp,bathtub,bag

## Only return img without label
class ade20kIm(BaseDatasetIm):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(ade20kIm, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.n_cats = 100
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
