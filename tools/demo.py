
import sys
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

torch.set_grad_enabled(False) 
np.random.seed(123)

# args
parse = argparse.ArgumentParser()

parse.add_argument('--weight_path', type=str, default='res/celoss/best_model_95000.pth',)
parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2_eval.json',)
parse.add_argument('--img_path', dest='img_path', type=str, default='43.jpg',)
args = parse.parse_args()
# cfg = set_cfg_from_file(args.config)
configer = Configer(configs=args.config)

# palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
# labels_info_eval = [
#     {"name": "road", "ignoreInEval": False, "id": 7, "color": [128, 64, 128], "trainId": 0},
#     {"name": "sidewalk", "ignoreInEval": False, "id": 8, "color": [244, 35, 232], "trainId": 1},
#     {"name": "building", "ignoreInEval": False, "id": 11, "color": [70, 70, 70], "trainId": 2},
#     {"name": "wall", "ignoreInEval": False, "id": 12, "color": [102, 102, 156], "trainId": 3},
#     {"name": "fence", "ignoreInEval": False, "id": 13, "color": [190, 153, 153], "trainId": 4},
#     {"name": "pole", "ignoreInEval": False, "id": 17, "color": [153, 153, 153], "trainId": 5},
#     {"name": "traffic light", "ignoreInEval": False, "id": 19, "color": [250, 170, 30], "trainId": 6},
#     {"name": "traffic sign", "ignoreInEval": False, "id": 20, "color": [220, 220, 0], "trainId": 7},
#     {"name": "vegetation", "ignoreInEval": False, "id": 21, "color": [107, 142, 35], "trainId": 8},
#     {"name": "terrain", "ignoreInEval": False, "id": 22, "color": [152, 251, 152], "trainId": 9},
#     {"name": "sky", "ignoreInEval": False, "id": 23, "color": [70, 130, 180], "trainId": 10},
#     {"name": "person", "ignoreInEval": False, "id": 24, "color": [220, 20, 60], "trainId": 11},
#     {"name": "rider", "ignoreInEval": False, "id": 25, "color": [255, 0, 0], "trainId": 12},
#     {"name": "car", "ignoreInEval": False, "id": 26, "color": [0, 0, 142], "trainId": 13},
#     {"name": "truck", "ignoreInEval": False, "id": 27, "color": [0, 0, 70], "trainId": 14},
#     {"name": "bus", "ignoreInEval": False, "id": 28, "color": [0, 60, 100], "trainId": 15},
#     {"name": "train", "ignoreInEval": False, "id": 31, "color": [0, 80, 100], "trainId": 16},
#     {"name": "motorcycle", "ignoreInEval": False, "id": 32, "color": [0, 0, 230], "trainId": 17},
#     {"name": "bicycle", "ignoreInEval": False, "id": 33, "color": [119, 11, 32], "trainId": 18},
#     {"name": "void", "ignoreInEval": False, "id": 34, "color": [0, 0, 0], "trainId": 19}
# ]

labels_info_eval = [
    {"name": "road", "color": [128, 64, 128], "trainId": 0},
    {"name": "sidewalk", "color": [244, 35, 232], "trainId": 1},
    {"name": "building(Grid)",  "color": [170, 170, 170], "trainId": 2},
    {"name": "building",  "color": [70, 70, 70], "trainId": 3},
    {"name": "wall(Grid)",  "color": [52, 52, 106], "trainId": 4},
    {"name": "wall",  "color": [102, 102, 156], "trainId": 5},
    {"name": "fence",  "color": [190, 153, 153], "trainId": 6},
    {"name": "pole",  "color": [153, 153, 153], "trainId": 7},
    {"name": "traffic light",  "color": [250, 170, 30], "trainId": 8},
    {"name": "traffic sign",  "color": [220, 220, 0], "trainId": 9},
    {"name": "vegetation",  "color": [107, 142, 35], "trainId": 10},
    {"name": "terrain",  "color": [152, 251, 152], "trainId": 11},
    {"name": "sky",  "color": [70, 130, 180], "trainId": 12},
    {"name": "person",  "color": [220, 20, 60], "trainId": 13},
    {"name": "rider(Bicycle)",  "color": [255, 0, 0], "trainId": 14},
    {"name": "rider(Motor)",  "color": [255, 0, 50], "trainId": 15},
    {"name": "car",  "color": [0, 0, 142], "trainId": 16},
    {"name": "truck",  "color": [0, 0, 70], "trainId": 17},
    {"name": "bus",  "color": [0, 60, 100], "trainId": 18},
    {"name": "train",  "color": [0, 80, 100], "trainId": 19},
    {"name": "motorcycle",  "color": [0, 0, 230], "trainId": 20},
    {"name": "bicycle",  "color": [119, 11, 32], "trainId": 21},
    {"name": "Utility vehicle 1", "color": [255, 255, 0], "trainId": 22},
    {"name": "Sidebars", "color": [233, 100, 0], "trainId": 23},
    {"name": "Speed bumper", "color": [110, 110, 0], "trainId": 24},
    {"name": "Curbstone", "color": [128, 128, 0], "trainId": 25},
    {"name": "Solid line", "color": [255, 193, 37], "trainId": 26},
    {"name": "Irrelevant signs", "color": [64, 0, 64], "trainId": 27},
    {"name": "Road blocks", "color": [185, 122, 87], "trainId": 28},
    {"name": "Tractor", "color": [0, 0, 100], "trainId": 29},
    {"name": "Non-drivable street", "color": [139, 99, 108], "trainId": 30},
    {"name": "Zebra crossing", "color": [210, 50, 115], "trainId": 31},
    {"name": "Obstacles / trash", "color": [255, 0, 127], "trainId": 32},
    {"name": "RD restricted area", "color": [150, 0, 150], "trainId": 33},
    {"name": "Animals", "color": [204, 255, 153], "trainId": 34},
    {"name": "Signal corpus(Pole)", "color": [33, 44, 177], "trainId": 35},
    {"name": "Signal corpus(Light)", "color": [133, 144, 177], "trainId": 36},
    {"name": "Drivable cobblestone", "color": [180, 50, 180], "trainId": 37},
    {"name": "Electronic traffic", "color": [255, 70, 185], "trainId": 38},
    {"name": "Slow drive area", "color": [238, 233, 191], "trainId": 39},
    {"name": "Parking area", "color": [150, 150, 200], "trainId": 40},
    {"name": "Painted driv. instr.", "color": [200, 125, 210], "trainId": 41},
    {"name": "Traffic guide obj.(Fence)", "color": [159, 121, 238], "trainId": 42},
    {"name": "Traffic guide obj.(Cone)", "color": [109, 71, 188], "trainId": 43},
    {"name": "Dashed line", "color": [128, 0, 255], "trainId": 44},
    {"name": "Ego car", "color": [72, 209, 204], "trainId": 45},
]

def buildPalette(labels_info):
    palette = []
    for el in labels_info:
        palette.append(el["color"])
        
    return np.array(palette)
palette = buildPalette(labels_info_eval)
# print(Palette)

class E2EModel(torch.nn.Module):
        
    def __init__(self, configer, weight_path) -> None:
        super().__init__()
        
        self.mean = torch.tensor([0.3038, 0.3383, 0.3034])[:, None, None] #.cuda()
        self.std = torch.tensor([0.2071, 0.2088, 0.2090])[:, None, None] #.cuda()
        
        # self.net = model_factory[cfg.model_type](cfg.n_cats, aux_mode="pred")
        self.net = model_factory[configer.get('model_name')](configer)
        self.net.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
        self.net.eval()
        self.net.aux_mode='pred'
        # self.net.train()
        # self.net.cuda()
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x.div_(255.)
        x = x.sub_(self.mean).div_(self.std).clone()
        # out = self.net(x)[0]
        # x = torch.cat((x,x), dim=0)
        out = self.net(x, dataset=0)
        return out
    
## mean: [0.3038, 0.3383, 0.3034] std: [0.2071, 0.2088, 0.2090]    
net = E2EModel(configer, args.weight_path)
# net.load_state_dict(torch.load('res/model_50000.pth', map_location='cpu'), strict=False)

# # define model
# net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='pred')
# net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
# net.eval()
# net.cuda()

# # prepare data
# to_tensor = T.ToTensor(
#     mean=(0.3257, 0.3690, 0.3223), # city, rgb
#     std=(0.2112, 0.2148, 0.2115),
# )

print(args.img_path)
im = cv2.imread(args.img_path)[:, :, ::-1]

for i in range(1):
    t0 = time()
    # input_im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
    # input_im = cv2.resize(im, (960, 768))
    # input_im = cv2.resize(im, (1024, 512))
    input_im = im
    
    input_im = torch.tensor(input_im.astype(np.float32).copy()).unsqueeze(0) #.cuda()
    
    # test_im = torch.cat((input_im, input_im), dim=0)
    # print(input_im)
    # inference
    # out1 = net1(input_im).squeeze().detach().cpu().numpy()
    # out2 = net(input_im).long().squeeze().detach().cpu().numpy()
    # net.train()
    out2 = net(input_im)
    # print(out2.shape)
    print(out2.shape)
    out2 = out2[0].long().squeeze().detach().cpu().numpy()
    
    # # print(maxV)
    # th = 0.9
    # maxV[maxV>=th] = 12
    # maxV[maxV<th] = 19
    # maxV = maxV.long().squeeze().detach().cpu().numpy()
    # # maxV = int(maxV)
    # # print(out2.shape)
    # # print(out.shape)
    # # pred1 = palette[out1]
    # # print(out2.shape)
    pred2 = palette[out2]
    # pred1 = palette[maxV]
    print(pred2.shape)
    # print((time() - t0) * 1000)

# cv2.imwrite('./res1.jpg', pred1)
cv2.imwrite('./res.bmp', pred2)
# cv2.imwrite('./test.jpg', pred1)
