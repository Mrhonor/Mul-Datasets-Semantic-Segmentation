
import sys
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from time import time
import random

import lib.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file
from tools.configer import Configer
from lib.module.gen_graph_node_feature import gen_graph_node_feature

torch.set_grad_enabled(False) 
np.random.seed(123)

# args
parse = argparse.ArgumentParser()

parse.add_argument('--weight_path', type=str, default='res/celoss/train3_seg_model_80000.pth',)
parse.add_argument('--gnn_weight_path', type=str, default='res/celoss/train3_graph_model_80000.pth',)
parse.add_argument('--config', dest='config', type=str, default='configs/ltbgnn_3_datasets.json',)
parse.add_argument('--img_path', dest='img_path', type=str, default='img/873809_leftImg8bit.png',)
args = parse.parse_args()
# cfg = set_cfg_from_file(args.config)
configer = Configer(configs=args.config)

palette = np.random.randint(0, 256, (512, 3), dtype=np.uint8)
print(palette.shape)
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

labels_info_cam = [
    {"name": "Sky", "color": [70, 130, 180], "trainId": 0},
    {"name": "Building", "color": [70, 70, 70], "trainId": 1},
    {"name": "Column_Pole", "color": [153, 153, 153], "trainId": 2},
    {"name": "Road", "color": [128, 64, 128], "trainId": 3},
    {"name": "Sidewalk", "color": [244, 35, 232], "trainId": 4},
    {"name": "Tree", "color": [107, 142, 35], "trainId": 5},
    {"name": "SignSymbol", "color": [250, 170, 30], "trainId": 6},
    {"name": "Fence", "color": [190, 153, 153], "trainId": 7},
    {"name": "Car", "color": [0, 0, 142], "trainId": 8},
    {"name": "Pedestrian", "color": [220, 20, 60], "trainId":9},
    {"name": "Bicyclist", "color": [119, 11, 32], "trainId": 10},
    {"name": "Wall", "color": [102, 102, 156], "trainId": 11},
]

labels_info_city = [
    {"name": "road", "color": [128, 64, 128], "trainId": 0},
    {"name": "sidewalk", "color": [244, 35, 232], "trainId": 1},
    {"name": "building", "color": [70, 70, 70], "trainId": 2},
    {"name": "wall", "color": [102, 102, 156], "trainId": 3},
    {"name": "fence", "color": [190, 153, 153], "trainId": 4},
    {"name": "pole", "color": [153, 153, 153], "trainId": 5},
    {"name": "traffic light", "color": [250, 170, 30], "trainId": 6},
    {"name": "traffic sign", "color": [220, 220, 0], "trainId": 7},
    {"name": "vegetation", "color": [107, 142, 35], "trainId": 8},
    {"name": "terrain", "color": [152, 251, 152], "trainId": 9},
    {"name": "sky", "color": [70, 130, 180], "trainId": 10},
    {"name": "person", "color": [220, 20, 60], "trainId": 11},
    {"name": "rider", "color": [255, 0, 0], "trainId": 12},
    {"name": "car", "color": [0, 0, 142], "trainId": 13},
    {"name": "truck", "color": [0, 0, 70], "trainId": 14},
    {"name": "bus", "color": [0, 60, 100], "trainId": 15},
    {"name": "train", "color": [0, 80, 100], "trainId": 16},
    {"name": "motorcycle", "color": [0, 0, 230], "trainId": 17},
    {"name": "bicycle", "color": [119, 11, 32], "trainId": 18},
]

labels_info_a2d2 = [
    {"name": "Car 1", "id": 0, "color": [255, 0, 0], "trainId": 0},
    {"name": "Bicycle 1", "id": 4, "color": [182, 89, 6], "trainId": 1},
    {"name": "Pedestrian 1", "id": 8, "color": [204, 153, 255], "trainId": 2},
    {"name": "Truck 1", "id": 11, "color": [255, 128, 0], "trainId": 3},
    {"name": "Small vehicles 1", "id": 14, "color": [0, 255, 0], "trainId": 4},
    {"name": "Traffic signal 1", "id": 17, "color": [0, 128, 255], "trainId": 5},
    {"name": "Traffic sign 1", "id": 20, "color": [0, 255, 255], "trainId": 6},
    {"name": "Utility vehicle 1", "id": 23, "color": [255, 255, 0], "trainId": 7},
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

ade_labels = [
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

bdd_labels = [
{'name': 'unlabel', 'id': 255, 'color': [83, 104, 40], 'trainId': 255},
{'name': 'road', 'id': 0, 'color': [20, 109, 144], 'trainId': 0},
{'name': 'sidewalk', 'id': 1, 'color': [230, 135, 48], 'trainId': 1},
{'name': 'building', 'id': 2, 'color': [70, 167, 229], 'trainId': 2},
{'name': 'wall', 'id': 3, 'color': [220, 234, 62], 'trainId': 3},
{'name': 'fence', 'id': 4, 'color': [209, 86, 88], 'trainId': 4},
{'name': 'pole', 'id': 5, 'color': [253, 97, 48], 'trainId': 5},
{'name': 'traffic light', 'id': 6, 'color': [130, 187, 207], 'trainId': 6},
{'name': 'traffic sign', 'id': 7, 'color': [145, 78, 159], 'trainId': 7},
{'name': 'vegetation', 'id': 8, 'color': [168, 10, 202], 'trainId': 8},
{'name': 'terrain', 'id': 9, 'color': [127, 204, 255], 'trainId': 9},
{'name': 'sky', 'id': 10, 'color': [237, 183, 72], 'trainId': 10},
{'name': 'person', 'id': 11, 'color': [226, 206, 193], 'trainId': 11},
{'name': 'rider', 'id': 12, 'color': [7, 158, 237], 'trainId': 12},
{'name': 'car', 'id': 13, 'color': [172, 237, 137], 'trainId': 13},
{'name': 'truck', 'id': 14, 'color': [59, 44, 172], 'trainId': 14},
{'name': 'bus', 'id': 15, 'color': [119, 83, 141], 'trainId': 15},
{'name': 'train', 'id': 16, 'color': [175, 189, 48], 'trainId': 16},
{'name': 'motorcycle', 'id': 17, 'color': [215, 166, 104], 'trainId': 17},
{'name': 'bicycle', 'id': 18, 'color': [124, 248, 126], 'trainId': 18},
]
coco_labels = [
{"name": "person", "id": 1, "trainId": 0},
{"name": "bicycle", "id": 2, "trainId": 1},
{"name": "car", "id": 3, "trainId": 2},
{"name": "motorcycle", "id": 4, "trainId": 3},
{"name": "airplane", "id": 5, "trainId": 4},
{"name": "bus", "id": 6, "trainId": 5},
{"name": "train", "id": 7, "trainId": 6},
{"name": "truck", "id": 8, "trainId": 7},
{"name": "boat", "id": 9, "trainId": 8},
{"name": "traffic light", "id": 10, "trainId": 9},
{"name": "fire hydrant", "id": 11, "trainId": 10},
{"name": "stop sign", "id": 13, "trainId": 11},
{"name": "parking meter", "id": 14, "trainId": 12},
{"name": "bench", "id": 15, "trainId": 13},
{"name": "bird", "id": 16, "trainId": 14},
{"name": "cat", "id": 17, "trainId": 15},
{"name": "dog", "id": 18, "trainId": 16},
{"name": "horse", "id": 19, "trainId": 17},
{"name": "sheep", "id": 20, "trainId": 18},
{"name": "cow", "id": 21, "trainId": 19},
{"name": "elephant", "id": 22, "trainId": 20},
{"name": "bear", "id": 23, "trainId": 21},
{"name": "zebra", "id": 24, "trainId": 22},
{"name": "giraffe", "id": 25, "trainId": 23},
{"name": "backpack", "id": 27, "trainId": 24},
{"name": "umbrella", "id": 28, "trainId": 25},
{"name": "handbag", "id": 31, "trainId": 26},
{"name": "tie", "id": 32, "trainId": 27},
{"name": "suitcase", "id": 33, "trainId": 28},
{"name": "frisbee", "id": 34, "trainId": 29},
{"name": "skis", "id": 35, "trainId": 30},
{"name": "snowboard", "id": 36, "trainId": 31},
{"name": "sports ball", "id": 37, "trainId": 32},
{"name": "kite", "id": 38, "trainId": 33},
{"name": "baseball bat", "id": 39, "trainId": 34},
{"name": "baseball glove", "id": 40, "trainId": 35},
{"name": "skateboard", "id": 41, "trainId": 36},
{"name": "surfboard", "id": 42, "trainId": 37},
{"name": "tennis racket", "id": 43, "trainId": 38},
{"name": "bottle", "id": 44, "trainId": 39},
{"name": "wine glass", "id": 46, "trainId": 40},
{"name": "cup", "id": 47, "trainId": 41},
{"name": "fork", "id": 48, "trainId": 42},
{"name": "knife", "id": 49, "trainId": 43},
{"name": "spoon", "id": 50, "trainId": 44},
{"name": "bowl", "id": 51, "trainId": 45},
{"name": "banana", "id": 52, "trainId": 46},
{"name": "apple", "id": 53, "trainId": 47},
{"name": "sandwich", "id": 54, "trainId": 48},
{"name": "orange", "id": 55, "trainId": 49},
{"name": "broccoli", "id": 56, "trainId": 50},
{"name": "carrot", "id": 57, "trainId": 51},
{"name": "hot dog", "id": 58, "trainId": 52},
{"name": "pizza", "id": 59, "trainId": 53},
{"name": "donut", "id": 60, "trainId": 54},
{"name": "cake", "id": 61, "trainId": 55},
{"name": "chair", "id": 62, "trainId": 56},
{"name": "couch", "id": 63, "trainId": 57},
{"name": "potted plant", "id": 64, "trainId": 58},
{"name": "bed", "id": 65, "trainId": 59},
{"name": "dining table", "id": 67, "trainId": 60},
{"name": "toilet", "id": 70, "trainId": 61},
{"name": "tv", "id": 72, "trainId": 62},
{"name": "laptop", "id": 73, "trainId": 63},
{"name": "mouse", "id": 74, "trainId": 64},
{"name": "remote", "id": 75, "trainId": 65},
{"name": "keyboard", "id": 76, "trainId": 66},
{"name": "cell phone", "id": 77, "trainId": 67},
{"name": "microwave", "id": 78, "trainId": 68},
{"name": "oven", "id": 79, "trainId": 69},
{"name": "toaster", "id": 80, "trainId": 70},
{"name": "sink", "id": 81, "trainId": 71},
{"name": "refrigerator", "id": 82, "trainId": 72},
{"name": "book", "id": 84, "trainId": 73},
{"name": "clock", "id": 85, "trainId": 74},
{"name": "vase", "id": 86, "trainId": 75},
{"name": "scissors", "id": 87, "trainId": 76},
{"name": "teddy bear", "id": 88, "trainId": 77},
{"name": "hair drier", "id": 89, "trainId": 78},
{"name": "toothbrush", "id": 90, "trainId": 79},
{"name": "banner", "id": 92, "trainId": 80},
{"name": "blanket", "id": 93, "trainId": 81},
{"name": "bridge", "id": 95, "trainId": 82},
{"name": "cardboard", "id": 100, "trainId": 83},
{"name": "counter", "id": 107, "trainId": 84},
{"name": "curtain", "id": 109, "trainId": 85},
{"name": "door-stuff", "id": 112, "trainId": 86},
{"name": "floor-wood", "id": 118, "trainId": 87},
{"name": "flower", "id": 119, "trainId": 88},
{"name": "fruit", "id": 122, "trainId": 89},
{"name": "gravel", "id": 125, "trainId": 90},
{"name": "house", "id": 128, "trainId": 91},
{"name": "light", "id": 130, "trainId": 92},
{"name": "mirror-stuff", "id": 133, "trainId": 93},
{"name": "net", "id": 138, "trainId": 94},
{"name": "pillow", "id": 141, "trainId": 95},
{"name": "platform", "id": 144, "trainId": 96},
{"name": "playingfield", "id": 145, "trainId": 97},
{"name": "railroad", "id": 147, "trainId": 98},
{"name": "river", "id": 148, "trainId": 99},
{"name": "road", "id": 149, "trainId": 100},
{"name": "roof", "id": 151, "trainId": 101},
{"name": "sand", "id": 154, "trainId": 102},
{"name": "sea", "id": 155, "trainId": 103},
{"name": "shelf", "id": 156, "trainId": 104},
{"name": "snow", "id": 159, "trainId": 105},
{"name": "stairs", "id": 161, "trainId": 106},
{"name": "tent", "id": 166, "trainId": 107},
{"name": "towel", "id": 168, "trainId": 108},
{"name": "wall-brick", "id": 171, "trainId": 109},
{"name": "wall-stone", "id": 175, "trainId": 110},
{"name": "wall-tile", "id": 176, "trainId": 111},
{"name": "wall-wood", "id": 177, "trainId": 112},
{"name": "water-other", "id": 178, "trainId": 113},
{"name": "window-blind", "id": 180, "trainId": 114},
{"name": "window-other", "id": 181, "trainId": 115},
{"name": "tree-merged", "id": 184, "trainId": 116},
{"name": "fence-merged", "id": 185, "trainId": 117},
{"name": "ceiling-merged", "id": 186, "trainId": 118},
{"name": "sky-other-merged", "id": 187, "trainId": 119},
{"name": "cabinet-merged", "id": 188, "trainId": 120},
{"name": "table-merged", "id": 189, "trainId": 121},
{"name": "floor-other-merged", "id": 190, "trainId": 122},
{"name": "pavement-merged", "id": 191, "trainId": 123},
{"name": "mountain-merged", "id": 192, "trainId": 124},
{"name": "grass-merged", "id": 193, "trainId": 125},
{"name": "dirt-merged", "id": 194, "trainId": 126},
{"name": "paper-merged", "id": 195, "trainId": 127},
{"name": "food-other-merged", "id": 196, "trainId": 128},
{"name": "building-other-merged", "id": 197, "trainId": 129},
{"name": "rock-merged", "id": 198, "trainId": 130},
{"name": "wall-other-merged", "id": 199, "trainId": 131},
{"name": "rug-merged", "id": 200, "trainId": 132},
]

idd_labels = [
{'name': 'person', 'id': 0, 'color': [157, 43, 50], 'trainId': 0},
{'name': 'truck', 'id': 1, 'color': [199, 67, 146], 'trainId': 1},
{'name': 'fence', 'id': 2, 'color': [126, 41, 245], 'trainId': 2},
{'name': 'billboard', 'id': 3, 'color': [96, 226, 191], 'trainId': 3},
{'name': 'bus', 'id': 4, 'color': [222, 166, 242], 'trainId': 4},
{'name': 'out of roi', 'id': 5, 'color': [64, 225, 5], 'trainId': 5},
{'name': 'curb', 'id': 6, 'color': [218, 101, 200], 'trainId': 6},
{'name': 'obs-str-bar-fallback', 'id': 7, 'color': [136, 91, 145], 'trainId': 7},
{'name': 'tunnel', 'id': 8, 'color': [5, 191, 123], 'trainId': 8},
{'name': 'non-drivable fallback', 'id': 9, 'color': [144, 177, 10], 'trainId': 9},
{'name': 'bridge', 'id': 10, 'color': [12, 136, 61], 'trainId': 10},
{'name': 'road', 'id': 11, 'color': [36, 199, 31], 'trainId': 11},
{'name': 'wall', 'id': 12, 'color': [240, 67, 195], 'trainId': 12},
{'name': 'traffic sign', 'id': 13, 'color': [77, 2, 151], 'trainId': 13},
{'name': 'trailer', 'id': 14, 'color': [200, 210, 160], 'trainId': 14},
{'name': 'animal', 'id': 15, 'color': [197, 119, 162], 'trainId': 15},
{'name': 'building', 'id': 16, 'color': [62, 51, 2], 'trainId': 16},
{'name': 'sky', 'id': 17, 'color': [239, 31, 224], 'trainId': 17},
{'name': 'drivable fallback', 'id': 18, 'color': [1, 22, 57], 'trainId': 18},
{'name': 'guard rail', 'id': 19, 'color': [161, 185, 7], 'trainId': 19},
{'name': 'bicycle', 'id': 20, 'color': [73, 234, 230], 'trainId': 20},
{'name': 'traffic light', 'id': 21, 'color': [196, 161, 211], 'trainId': 21},
{'name': 'polegroup', 'id': 22, 'color': [232, 11, 224], 'trainId': 22},
{'name': 'motorcycle', 'id': 23, 'color': [233, 39, 115], 'trainId': 23},
{'name': 'car', 'id': 24, 'color': [177, 133, 83], 'trainId': 24},
{'name': 'parking', 'id': 25, 'color': [69, 196, 131], 'trainId': 25},
{'name': 'fallback background', 'id': 26, 'color': [202, 73, 150], 'trainId': 26},
{'name': 'license plate', 'id': 27, 'color': [193, 167, 237], 'trainId': 255},
{'name': 'rectification border', 'id': 28, 'color': [53, 35, 56], 'trainId': 27},
{'name': 'train', 'id': 29, 'color': [95, 238, 8], 'trainId': 28},
{'name': 'rider', 'id': 30, 'color': [119, 138, 146], 'trainId': 29},
{'name': 'rail track', 'id': 31, 'color': [33, 94, 35], 'trainId': 30},
{'name': 'sidewalk', 'id': 32, 'color': [244, 136, 181], 'trainId': 31},
{'name': 'caravan', 'id': 33, 'color': [34, 2, 81], 'trainId': 32},
{'name': 'pole', 'id': 34, 'color': [73, 61, 149], 'trainId': 33},
{'name': 'vegetation', 'id': 35, 'color': [138, 10, 147], 'trainId': 34},
{'name': 'autorickshaw', 'id': 36, 'color': [128, 140, 171], 'trainId': 35},
{'name': 'vehicle fallback', 'id': 37, 'color': [93, 6, 120], 'trainId': 36},
{'name': 'unlabel', 'id': 255, 'color': [86, 220, 114], 'trainId': 255},
]
sunrgb_labels = [
{'name': 'unlabeled', 'id': 0, 'trainId': 255, 'color': [242, 188, 232]},
{'name': 'wall', 'id': 1, 'trainId': 1, 'color': [233, 202, 59]},
{'name': 'floor', 'id': 2, 'trainId': 2, 'color': [213, 38, 18]},
{'name': 'cabinet', 'id': 3, 'trainId': 3, 'color': [194, 2, 224]},
{'name': 'bed', 'id': 4, 'trainId': 4, 'color': [223, 11, 214]},
{'name': 'chair', 'id': 5, 'trainId': 5, 'color': [85, 73, 155]},
{'name': 'sofa', 'id': 6, 'trainId': 6, 'color': [74, 250, 181]},
{'name': 'table', 'id': 7, 'trainId': 7, 'color': [183, 246, 222]},
{'name': 'door', 'id': 8, 'trainId': 8, 'color': [58, 98, 138]},
{'name': 'window', 'id': 9, 'trainId': 9, 'color': [85, 222, 236]},
{'name': 'bookshelf', 'id': 10, 'trainId': 10, 'color': [41, 191, 193]},
{'name': 'picture', 'id': 11, 'trainId': 11, 'color': [163, 9, 4]},
{'name': 'counter', 'id': 12, 'trainId': 12, 'color': [228, 169, 95]},
{'name': 'blinds', 'id': 13, 'trainId': 13, 'color': [122, 95, 180]},
{'name': 'desk', 'id': 14, 'trainId': 14, 'color': [223, 156, 194]},
{'name': 'shelves', 'id': 15, 'trainId': 15, 'color': [221, 123, 19]},
{'name': 'curtain', 'id': 16, 'trainId': 16, 'color': [234, 172, 3]},
{'name': 'dresser', 'id': 17, 'trainId': 17, 'color': [239, 156, 24]},
{'name': 'pillow', 'id': 18, 'trainId': 18, 'color': [89, 198, 1]},
{'name': 'mirror', 'id': 19, 'trainId': 19, 'color': [23, 233, 129]},
{'name': 'floor mat', 'id': 20, 'trainId': 20, 'color': [42, 172, 55]},
{'name': 'clothes', 'id': 21, 'trainId': 21, 'color': [160, 198, 54]},
{'name': 'ceiling', 'id': 22, 'trainId': 22, 'color': [212, 80, 88]},
{'name': 'books', 'id': 23, 'trainId': 23, 'color': [78, 137, 207]},
{'name': 'refridgerator', 'id': 24, 'trainId': 24, 'color': [176, 239, 185]},
{'name': 'television', 'id': 25, 'trainId': 25, 'color': [160, 28, 99]},
{'name': 'paper', 'id': 26, 'trainId': 26, 'color': [77, 139, 91]},
{'name': 'towel', 'id': 27, 'trainId': 27, 'color': [170, 55, 89]},
{'name': 'shower curtain', 'id': 28, 'trainId': 28, 'color': [37, 230, 191]},
{'name': 'box', 'id': 29, 'trainId': 29, 'color': [54, 74, 19]},
{'name': 'whiteboard', 'id': 30, 'trainId': 30, 'color': [180, 74, 241]},
{'name': 'person', 'id': 31, 'trainId': 31, 'color': [189, 141, 151]},
{'name': 'night stand', 'id': 32, 'trainId': 32, 'color': [155, 246, 163]},
{'name': 'toilet', 'id': 33, 'trainId': 33, 'color': [186, 133, 177]},
{'name': 'sink', 'id': 34, 'trainId': 34, 'color': [253, 11, 213]},
{'name': 'lamp', 'id': 35, 'trainId': 35, 'color': [83, 84, 29]},
{'name': 'bathtub', 'id': 36, 'trainId': 36, 'color': [159, 143, 221]},
{'name': 'bag', 'id': 37, 'trainId': 0, 'color': [117, 178, 149]},
]

def generate_random_color(existing_colors):
    while True:
        color = [random.randint(0, 255) for _ in range(3)]
        if tuple(color) not in existing_colors:
            return color

# Collect existing colors from labels_info_cam

# Assign unique random RGB values to each label in bdd_labels


# temp_labels_info = labels_info_city + labels_info_cam + labels_info_a2d2
# existing_colors = set(tuple(label['color']) for label in temp_labels_info)
# for label in bdd_labels:
#     random_color = generate_random_color(existing_colors)
#     label['color'] = random_color
#     existing_colors.add(tuple(random_color))

# for label in sunrgb_labels:
#     random_color = generate_random_color(existing_colors)
#     label['color'] = random_color
#     existing_colors.add(tuple(random_color)) 

# for label in idd_labels:
#     random_color = generate_random_color(existing_colors)
#     label['color'] = random_color
#     existing_colors.add(tuple(random_color)) 
    
# # Write the updated labels to out.txt file
# with open('bdd.txt', 'w') as outfile:
#     for label in bdd_labels:
#         outfile.write(str(label) + '\n')

# with open('sunrgb.txt', 'w') as outfile:
#     for label in sunrgb_labels:
#         outfile.write(str(label) + '\n')

# with open('idd.txt', 'w') as outfile:
#     for label in idd_labels:
#         outfile.write(str(label) + '\n')


CITY_ID = 0
CAM_ID = 1
A2D2_ID = 2
unify_ID = 3
dataset_id = unify_ID
if dataset_id is CITY_ID:
    labels_info = labels_info_city
elif dataset_id is CAM_ID:
    labels_info =  labels_info_cam
elif dataset_id is A2D2_ID:
    labels_info = labels_info_a2d2
else:
    labels_info = labels_info_eval

# labels_info = labels_info_city + labels_info_cam + sunrgb_labels + bdd_labels + idd_labels
labels_info = idd_labels
def buildPalette(label_info):
    palette = []
    for el in label_info:
        palette.append(el["color"])
        
    return np.array(palette)
# palette = buildPalette(labels_info)
# print(Palette)

class E2EModel(torch.nn.Module):
        
    def __init__(self, configer, weight_path) -> None:
        super().__init__()
        
        self.mean = torch.tensor([0.3038, 0.3383, 0.3034])[:, None, None] #.cuda()
        self.std = torch.tensor([0.2071, 0.2088, 0.2090])[:, None, None] #.cuda()
        
        
        # self.net = model_factory[cfg.model_type](cfg.n_cats, aux_mode="pred")
        self.net = model_factory[configer.get('model_name')](configer)
        state = torch.load(weight_path, map_location='cpu')
        # del state['bipartite_graphs.0']
        self.net.load_state_dict(state, strict=False)
        self.net.eval()
        self.net.aux_mode='pred'
        # self.net.aux_mode='uni'
        # self.net.train()
        self.net.cuda()
        # with open('camvid_mapping.txt', 'r') as f:
        #     lines = f.readlines()

        # bi_graphs = []
        # for i, line in enumerate(lines):
        #     bi_graph = torch.zeros((11, 224), dtype=torch.float32).cuda()
        #     ids = line.replace('\n', '').replace(' ', '').split(',')
        #     ids = [int(id) for id in ids]
        #     print(ids)
        #     for id in ids:
        #         bi_graph[i, id] = 1
        # bi_graphs.append(bi_graph)

        # graph_net = model_factory[configer.get('GNN','model_name')](configer)
        # torch.set_printoptions(profile="full")
        # graph_net.load_state_dict(torch.load(args.gnn_weight_path, map_location='cpu'), strict=False)
        # graph_net.cuda()
        # graph_net.eval()
        # graph_net.train()
        # graph_node_features = gen_graph_node_feature(configer)
        # unify_prototype, ori_bi_graphs = graph_net.get_optimal_matching(graph_node_features, init=True) 
        # if len(ori_bi_graphs) == 10:
        #     for j in range(0, len(ori_bi_graphs), 2):
        #         bi_graphs.append(ori_bi_graphs[j+1].detach())
        # else:
        #     bi_graphs = [bigh.detach() for bigh in ori_bi_graphs]
        # unify_prototype, bi_graphs, adv_out, _ = graph_net(graph_node_features)

        # print(self.net.bipartite_graphs[6])
        # print(bi_graphs[6])

        # self.net.set_unify_prototype(unify_prototype)
        # self.net.set_bipartite_graphs(bi_graphs)
                
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x.div_(255.)
        x = x.sub_(self.mean).div_(self.std).clone()
        # out = self.net(x)[0]
        # x = torch.cat((x,x), dim=0)
        out = self.net(x.cuda(), dataset=2)
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
    input_im = cv2.resize(im, (1024, 512))
    # input_im = im
    
    input_im = torch.tensor(input_im.astype(np.float32).copy()).unsqueeze(0)#.cuda()
    
    # test_im = torch.cat((input_im, input_im), dim=0)
    # print(input_im)
    # inference
    # out1 = net1(input_im).squeeze().detach().cpu().numpy()
    # out2 = net(input_im).long().squeeze().detach().cpu().numpy()
    # net.train()
    out2 = net(input_im)
    # print(out2.shape)
    print(out2.shape)
    print(torch.max(out2))
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
    # print(pred2.shape)
    # print((time() - t0) * 1000)

# cv2.imwrite('./res1.jpg', pred1)
cv2.imwrite('./res.bmp', pred2)
# cv2.imwrite('./test.jpg', pred1)
