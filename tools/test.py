import sys

# from tools.train_amp_contrast import ClassRemaper
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from time import time

import lib.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file
from tools.configer import Configer
from lib.class_remap import ClassRemapOneHotLabel
from lib.get_dataloader import get_data_loader
from lib.loss_cross_datasets import CrossDatasetsLoss
from bitarray import bitarray

torch.set_grad_enabled(False)
np.random.seed(123)

# args
parse = argparse.ArgumentParser()
parse.add_argument('--weight_path', type=str, default='res/domain/model_44000.pth',)
parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2_eval.json',)
parse.add_argument('--img_path', dest='img_path', type=str, default='berlin_000543_000019_leftImg8bit.png',)
args = parse.parse_args()
# cfg = set_cfg_from_file(args.config)
configer = Configer(configs=args.config)

labels_info_eval = [
    {"name": "road", "ignoreInEval": False, "id": 7, "color": [128, 64, 128], "trainId": 0},
    {"name": "sidewalk", "ignoreInEval": False, "id": 8, "color": [244, 35, 232], "trainId": 1},
    {"name": "building", "ignoreInEval": False, "id": 11, "color": [70, 70, 70], "trainId": 2},
    {"name": "wall", "ignoreInEval": False, "id": 12, "color": [102, 102, 156], "trainId": 3},
    {"name": "fence", "ignoreInEval": False, "id": 13, "color": [190, 153, 153], "trainId": 4},
    {"name": "pole", "ignoreInEval": False, "id": 17, "color": [153, 153, 153], "trainId": 5},
    {"name": "traffic light", "ignoreInEval": False, "id": 19, "color": [250, 170, 30], "trainId": 6},
    {"name": "traffic sign", "ignoreInEval": False, "id": 20, "color": [220, 220, 0], "trainId": 7},
    {"name": "vegetation", "ignoreInEval": False, "id": 21, "color": [107, 142, 35], "trainId": 8},
    {"name": "terrain", "ignoreInEval": False, "id": 22, "color": [152, 251, 152], "trainId": 9},
    {"name": "sky", "ignoreInEval": False, "id": 23, "color": [70, 130, 180], "trainId": 10},
    {"name": "person", "ignoreInEval": False, "id": 24, "color": [220, 20, 60], "trainId": 11},
    {"name": "rider", "ignoreInEval": False, "id": 25, "color": [255, 0, 0], "trainId": 12},
    {"name": "car", "ignoreInEval": False, "id": 26, "color": [0, 0, 142], "trainId": 13},
    {"name": "truck", "ignoreInEval": False, "id": 27, "color": [0, 0, 70], "trainId": 14},
    {"name": "bus", "ignoreInEval": False, "id": 28, "color": [0, 60, 100], "trainId": 15},
    {"name": "train", "ignoreInEval": False, "id": 31, "color": [0, 80, 100], "trainId": 16},
    {"name": "motorcycle", "ignoreInEval": False, "id": 32, "color": [0, 0, 230], "trainId": 17},
    {"name": "bicycle", "ignoreInEval": False, "id": 33, "color": [119, 11, 32], "trainId": 18},
    {"name": "void", "ignoreInEval": False, "id": 34, "color": [0, 0, 0], "trainId": 19}
]

def bitarrayToInt(bitA):
    i = 0
    for bit in bitA:
        i = (i << 1) | bit
    return i

def buildPalette(labels_info):
    palette = []
    for el in labels_info:
        palette.append(el["color"])
        
    return np.array(palette)
palette = buildPalette(labels_info_eval)

ClassRemaper = ClassRemapOneHotLabel(configer)
net = model_factory[configer.get('model_name')](configer)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=True)
net.eval()
net.cuda()

dl_city, dl_cam = get_data_loader(configer, aux_mode='train', distributed=False)

mean = torch.tensor([0.3257, 0.3690, 0.3223])[:, None, None]
std = torch.tensor([0.2112, 0.2148, 0.2115])[:, None, None]

def test_showimg():
    segment_queue = net.segment_queue   

    for im, lb in dl_city:
        lb = lb.squeeze(1)
        b,h,w = lb.shape
        # print(lb.shape)
        out = net(im)
        emb = out['embed'][0]
        contrast_lable, seg_mask, hard_lb_mask = ClassRemaper.ContrastRemapping(lb, emb, segment_queue, 1)
        seg_lb = torch.zeros([b,h,w,3])
        # seg_mask = seg_mask.permute(0,2,3,1)
        for i in range(0,b):
            for j in range(0,h):
                for k in range(0,w):
                    # print(seg_mask.shape)
                    # print(seg_mask[i,j,k,:])
                    bitToInt = bitarrayToInt(bitarray(list(seg_mask[i,j,k,:])))
                    R = bitToInt & 0xFF
                    G = (bitToInt >> 8) & 0xFF
                    B = (bitToInt >> 16) & 0xFF
                    # print(R,G,B)
                    seg_lb[i,j,k] =torch.tensor([B, G, R])
        
        # cv2.imwrite('./im.jpg', im.long().squeeze().detach().cpu().numpy())
        cv2.imwrite('./lb.bmp', seg_lb.long().squeeze().detach().cpu().numpy())
        break


# label = torch.tensor([[[8, 0, 5]]])
# print(ClassRemaper.SegRemapping(label, 1))
# x = x.permute(0, 3, 1, 2)
# x = x.div_(255.)
# x = x.sub_(self.mean).div_(self.std).clone()

# print(palette[[[0,1],[2,0]]])

@torch.no_grad()
def test_loss():
    segment_queue = net.segment_queue
    net.aux_mode = 'train'
    for im, lb in dl_city:
        out = net(im.cuda())
        city_out = {}
        for k, v in out.items():
            if v is None:
                city_out[k] = None
            elif k == 'seg':
                city_logits_list = []
                for logit in v:
                    city_logits_list.append(logit[0]) 
                                    
                city_out[k] = city_logits_list
            else:
                city_out[k] = v[0]
                
        city_out['segment_queue'] = net.segment_queue
        
        contrast_losses = CrossDatasetsLoss(configer)
        lb = lb.squeeze(1)
        # print(lb.shape)
        backward_loss0, loss_seg0, loss_aux0, loss_contrast0, loss_domain0 = contrast_losses(city_out, lb.cuda(), 0, False)
        print("loss_seg:", loss_seg0)
        logit = city_out['seg'][0]
        pred = logit.argmax(dim=1)
        pred = pred[0].long().squeeze().detach().cpu().numpy()
        out_im = palette[pred]
        # cv2.imwrite('./lb.png', out_im)
        break
        

if __name__ == "__main__":
    test_loss()
