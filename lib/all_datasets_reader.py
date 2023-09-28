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

a2d2_labels = [
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
    {"name": "unlabel", "id":255, "color":[0,0,0], "trainId": 255},
    {"name": "road", "id": 0, "color": [0, 0, 0], "trainId": 0},
    {"name": "sidewalk", "id": 1, "color": [0, 0, 0], "trainId": 1},
    {"name": "building", "id": 2, "color": [0, 0, 0], "trainId": 2},
    {"name": "wall", "id": 3, "color": [0, 0, 0], "trainId": 3},
    {"name": "fence", "id": 4, "color": [0, 0, 0], "trainId": 4},
    {"name": "pole", "id": 5, "color": [0, 0, 0], "trainId": 5},
    {"name": "traffic light", "id": 6, "color": [0, 0, 0], "trainId": 6},
    {"name": "traffic sign", "id": 7, "color": [0, 0, 0], "trainId": 7},
    {"name": "vegetation", "id": 8, "color": [0, 0, 0], "trainId": 8},
    {"name": "terrain", "id": 9, "color": [0, 0, 0], "trainId": 9},
    {"name": "sky", "id": 10, "color": [0, 0, 0], "trainId": 10},
    {"name": "person", "id": 11, "color": [0, 0, 0], "trainId": 11},
    {"name": "rider", "id": 12, "color": [0, 0, 0], "trainId": 12},
    {"name": "car", "id": 13, "color": [0, 0, 0], "trainId": 13},
    {"name": "truck", "id": 14, "color": [0, 0, 0], "trainId": 14},
    {"name": "bus", "id": 15, "color": [0, 0, 0], "trainId": 15},
    {"name": "train", "id": 16, "color": [0, 0, 0], "trainId": 16},
    {"name": "motorcycle", "id": 17, "color": [0, 0, 0], "trainId": 17},
    {"name": "bicycle", "id": 18, "color": [0, 0, 0], "trainId": 18},
]

camvid_labels = [
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

cityscapes_labels = [
    {"hasInstances": False, "category": "void", "catid": 0, "name": "unlabeled", "ignoreInEval": True, "id": 0, "color": [0, 0, 0], "trainId": -1},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "ego vehicle", "ignoreInEval": True, "id": 1, "color": [0, 0, 0], "trainId": -1},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "rectification border", "ignoreInEval": True, "id": 2, "color": [0, 0, 0], "trainId": -1},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "out of roi", "ignoreInEval": True, "id": 3, "color": [0, 0, 0], "trainId": -1},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "static", "ignoreInEval": True, "id": 4, "color": [0, 0, 0], "trainId": -1},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "dynamic", "ignoreInEval": True, "id": 5, "color": [111, 74, 0], "trainId": -1},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "ground", "ignoreInEval": True, "id": 6, "color": [81, 0, 81], "trainId": -1},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "road", "ignoreInEval": False, "id": 7, "color": [128, 64, 128], "trainId": 0},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "sidewalk", "ignoreInEval": False, "id": 8, "color": [244, 35, 232], "trainId": 1},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "parking", "ignoreInEval": True, "id": 9, "color": [250, 170, 160], "trainId": -1},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "rail track", "ignoreInEval": True, "id": 10, "color": [230, 150, 140], "trainId": -1},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "building", "ignoreInEval": False, "id": 11, "color": [70, 70, 70], "trainId": 2},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "wall", "ignoreInEval": False, "id": 12, "color": [102, 102, 156], "trainId": 3},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "fence", "ignoreInEval": False, "id": 13, "color": [190, 153, 153], "trainId": 4},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "guard rail", "ignoreInEval": True, "id": 14, "color": [180, 165, 180], "trainId": -1},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "bridge", "ignoreInEval": True, "id": 15, "color": [150, 100, 100], "trainId": -1},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "tunnel", "ignoreInEval": True, "id": 16, "color": [150, 120, 90], "trainId": -1},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "pole", "ignoreInEval": False, "id": 17, "color": [153, 153, 153], "trainId": 5},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "polegroup", "ignoreInEval": True, "id": 18, "color": [153, 153, 153], "trainId": -1},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic light", "ignoreInEval": False, "id": 19, "color": [250, 170, 30], "trainId": 6},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic sign", "ignoreInEval": False, "id": 20, "color": [220, 220, 0], "trainId": 7},
    {"hasInstances": False, "category": "nature", "catid": 4, "name": "vegetation", "ignoreInEval": False, "id": 21, "color": [107, 142, 35], "trainId": 8},
    {"hasInstances": False, "category": "nature", "catid": 4, "name": "terrain", "ignoreInEval": False, "id": 22, "color": [152, 251, 152], "trainId": 9},
    {"hasInstances": False, "category": "sky", "catid": 5, "name": "sky", "ignoreInEval": False, "id": 23, "color": [70, 130, 180], "trainId": 10},
    {"hasInstances": True, "category": "human", "catid": 6, "name": "person", "ignoreInEval": False, "id": 24, "color": [220, 20, 60], "trainId": 11},
    {"hasInstances": True, "category": "human", "catid": 6, "name": "rider", "ignoreInEval": False, "id": 25, "color": [255, 0, 0], "trainId": 12},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "car", "ignoreInEval": False, "id": 26, "color": [0, 0, 142], "trainId": 13},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "truck", "ignoreInEval": False, "id": 27, "color": [0, 0, 70], "trainId": 14},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bus", "ignoreInEval": False, "id": 28, "color": [0, 60, 100], "trainId": 15},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "caravan", "ignoreInEval": True, "id": 29, "color": [0, 0, 90], "trainId": -1},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "trailer", "ignoreInEval": True, "id": 30, "color": [0, 0, 110], "trainId": -1},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "train", "ignoreInEval": False, "id": 31, "color": [0, 80, 100], "trainId": 16},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "motorcycle", "ignoreInEval": False, "id": 32, "color": [0, 0, 230], "trainId": 17},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bicycle", "ignoreInEval": False, "id": 33, "color": [119, 11, 32], "trainId": 18},
    {"hasInstances": False, "category": "vehicle", "catid": 7, "name": "license plate", "ignoreInEval": True, "id": -1, "color": [0, 0, 142], "trainId": -1}
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
    {"name": "person", "id": 0, "color": [0, 0, 0], "trainId": 0},
    {"name": "truck", "id": 1, "color": [0, 0, 0], "trainId": 1},
    {"name": "fence", "id": 2, "color": [0, 0, 0], "trainId": 2},
    {"name": "billboard", "id": 3, "color": [0, 0, 0], "trainId": 3},
    {"name": "bus", "id": 4, "color": [0, 0, 0], "trainId": 4},
    {"name": "out of roi", "id": 5, "color": [0, 0, 0], "trainId": 5},
    {"name": "curb", "id": 6, "color": [0, 0, 0], "trainId": 6},
    {"name": "obs-str-bar-fallback", "id": 7, "color": [0, 0, 0], "trainId": 7},
    {"name": "tunnel", "id": 8, "color": [0, 0, 0], "trainId": 8},
    {"name": "non-drivable fallback", "id": 9, "color": [0, 0, 0], "trainId": 9},
    {"name": "bridge", "id": 10, "color": [0, 0, 0], "trainId": 10},
    {"name": "road", "id": 11, "color": [0, 0, 0], "trainId": 11},
    {"name": "wall", "id": 12, "color": [0, 0, 0], "trainId": 12},
    {"name": "traffic sign", "id": 13, "color": [0, 0, 0], "trainId": 13},
    {"name": "trailer", "id": 14, "color": [0, 0, 0], "trainId": 14},
    {"name": "animal", "id": 15, "color": [0, 0, 0], "trainId": 15},
    {"name": "building", "id": 16, "color": [0, 0, 0], "trainId": 16},
    {"name": "sky", "id": 17, "color": [0, 0, 0], "trainId": 17},
    {"name": "drivable fallback", "id": 18, "color": [0, 0, 0], "trainId": 18},
    {"name": "guard rail", "id": 19, "color": [0, 0, 0], "trainId": 19},
    {"name": "bicycle", "id": 20, "color": [0, 0, 0], "trainId": 20},
    {"name": "traffic light", "id": 21, "color": [0, 0, 0], "trainId": 21},
    {"name": "polegroup", "id": 22, "color": [0, 0, 0], "trainId": 22},
    {"name": "motorcycle", "id": 23, "color": [0, 0, 0], "trainId": 23},
    {"name": "car", "id": 24, "color": [0, 0, 0], "trainId": 24},
    {"name": "parking", "id": 25, "color": [0, 0, 0], "trainId": 25},
    {"name": "fallback background", "id": 26, "color": [0, 0, 0], "trainId": 26},
    {"name": "license plate", "id": 27, "color": [0, 0, 0], "trainId": 255},
    {"name": "rectification border", "id": 28, "color": [0, 0, 0], "trainId": 27},
    {"name": "train", "id": 29, "color": [0, 0, 0], "trainId": 28},
    {"name": "rider", "id": 30, "color": [0, 0, 0], "trainId": 29},
    {"name": "rail track", "id": 31, "color": [0, 0, 0], "trainId": 30},
    {"name": "sidewalk", "id": 32, "color": [0, 0, 0], "trainId": 31},
    {"name": "caravan", "id": 33, "color": [0, 0, 0], "trainId": 32},
    {"name": "pole", "id": 34, "color": [0, 0, 0], "trainId": 33},
    {"name": "vegetation", "id": 35, "color": [0, 0, 0], "trainId": 34},
    {"name": "autorickshaw", "id": 36, "color": [0, 0, 0], "trainId": 35},
    {"name": "vehicle fallback", "id": 37, "color": [0, 0, 0], "trainId": 36},
    {"name": "unlabel", "id":255, "color":[0,0,0], "trainId": 255},
]

idd_labels_eval = [
{"name": "person", "id": 0, "trainId": 0},
{"name": "truck", "id": 1, "trainId": 1},
{"name": "fence", "id": 2, "trainId": 2},
{"name": "billboard", "id": 3, "trainId": 3},
{"name": "bus", "id": 4, "trainId": 4},
{"name": "out of roi", "id": 5, "trainId": 5},
{"name": "curb", "id": 6, "trainId": 6},
{"name": "obs-str-bar-fallback", "id": 7, "trainId": 7},
{"name": "tunnel", "id": 8, "trainId": 8},
{"name": "non-drivable fallback", "id": 9, "trainId": 9},
{"name": "bridge", "id": 10, "trainId": 10},
{"name": "road", "id": 11, "trainId": 11},
{"name": "wall", "id": 12, "trainId": 12},
{"name": "traffic sign", "id": 13, "trainId": 13},
{"name": "trailer", "id": 14, "trainId": 255},
{"name": "animal", "id": 15, "trainId": 15},
{"name": "building", "id": 16, "trainId": 16},
{"name": "sky", "id": 17, "trainId": 17},
{"name": "drivable fallback", "id": 18, "trainId": 18},
{"name": "guard rail", "id": 19, "trainId": 19},
{"name": "bicycle", "id": 20, "trainId": 20},
{"name": "traffic light", "id": 21, "trainId": 21},
{"name": "polegroup", "id": 22, "trainId": 22},
{"name": "motorcycle", "id": 23, "trainId": 23},
{"name": "car", "id": 24, "trainId": 24},
{"name": "parking", "id": 25, "trainId": 25},
{"name": "fallback background", "id": 26, "trainId": 26},
{"name": "license plate", "id": 27, "trainId": 255},
{"name": "rectification border", "id": 28, "trainId": 255},
{"name": "train", "id": 29, "trainId": 255},
{"name": "rider", "id": 30, "trainId": 29},
{"name": "rail track", "id": 31, "trainId": 255},
{"name": "sidewalk", "id": 32, "trainId": 31},
{"name": "caravan", "id": 33, "trainId": 32},
{"name": "pole", "id": 34, "trainId": 33},
{"name": "vegetation", "id": 35, "trainId": 34},
{"name": "autorickshaw", "id": 36, "trainId": 35},
{"name": "vehicle fallback", "id": 37, "trainId": 36},
{"name": "unlabel", "id": 255, "trainId": 255},
]

sunrgb_labels = [
{"name": "unlabeled", "id": 0, "trainId": 255},
{"name": "wall", "id": 1, "trainId": 1},
{"name": "floor", "id": 2, "trainId": 2},
{"name": "cabinet", "id": 3, "trainId": 3},
{"name": "bed", "id": 4, "trainId": 4},
{"name": "chair", "id": 5, "trainId": 5},
{"name": "sofa", "id": 6, "trainId": 6},
{"name": "table", "id": 7, "trainId": 7},
{"name": "door", "id": 8, "trainId": 8},
{"name": "window", "id": 9, "trainId": 9},
{"name": "bookshelf", "id": 10, "trainId": 10},
{"name": "picture", "id": 11, "trainId": 11},
{"name": "counter", "id": 12, "trainId": 12},
{"name": "blinds", "id": 13, "trainId": 13},
{"name": "desk", "id": 14, "trainId": 14},
{"name": "shelves", "id": 15, "trainId": 15},
{"name": "curtain", "id": 16, "trainId": 16},
{"name": "dresser", "id": 17, "trainId": 17},
{"name": "pillow", "id": 18, "trainId": 18},
{"name": "mirror", "id": 19, "trainId": 19},
{"name": "floor mat", "id": 20, "trainId": 20},
{"name": "clothes", "id": 21, "trainId": 21},
{"name": "ceiling", "id": 22, "trainId": 22},
{"name": "books", "id": 23, "trainId": 23},
{"name": "refridgerator", "id": 24, "trainId": 24},
{"name": "television", "id": 25, "trainId": 25},
{"name": "paper", "id": 26, "trainId": 26},
{"name": "towel", "id": 27, "trainId": 27},
{"name": "shower curtain", "id": 28, "trainId": 28},
{"name": "box", "id": 29, "trainId": 29},
{"name": "whiteboard", "id": 30, "trainId": 30},
{"name": "person", "id": 31, "trainId": 31},
{"name": "night stand", "id": 32, "trainId": 32},
{"name": "toilet", "id": 33, "trainId": 33},
{"name": "sink", "id": 34, "trainId": 34},
{"name": "lamp", "id": 35, "trainId": 35},
{"name": "bathtub", "id": 36, "trainId": 36},
{"name": "bag", "id": 37, "trainId": 0},
]



label_map = {'a2d2': a2d2_labels, 'ade': ade_labels, 'bdd': bdd_labels, 'cityscapes': cityscapes_labels, 'coco': coco_labels, 'idd': idd_labels, 'camvid': camvid_labels, 'sunrgb': sunrgb_labels}

label_map_eval = {'a2d2': a2d2_labels, 'ade': ade_labels, 'bdd': bdd_labels, 'cityscapes': cityscapes_labels, 'coco': coco_labels, 'idd': idd_labels_eval, 'camvid': camvid_labels, 'sunrgb': sunrgb_labels}

class AllDatasetsReader(Dataset):
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(AllDatasetsReader, self).__init__()
        # assert mode in ('train', 'eval', 'test')
        self.mode = mode
        self.trans_func = trans_func

        
        if mode == 'eval':
            self.label_map = label_map_eval
        else:
            self.label_map = label_map

        self.lb_map = {}
        for key, item in self.label_map.items():
            this_lb_map = np.arange(256).astype(np.uint8)
            for el in item:
                this_lb_map[el['id']] = el['trainId']
                
            self.lb_map[key] = this_lb_map    
            
        self.to_tensor = T.ToTensor(
            mean=(0.3038, 0.3383, 0.3034), # city, rgb
            std=(0.2071, 0.2088, 0.2090),
        )

        with open(annpath, 'r') as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths = [], []
        self.datasets_name = []
        self.im_len = []
        for info in pairs[0].split(','):
            datasets_name, sta = info.split(':')
            self.datasets_name.append(str(datasets_name))
            self.im_len.append(int(sta))
        
        self.im_len.sort()        
        
        for pair in pairs[1:]:
            imgpth, lbpth = pair.split(',')
            self.img_paths.append(osp.join(dataroot, imgpth))
            self.lb_paths.append(osp.join(dataroot, lbpth))

        self.len = len(self.img_paths)

    def __getitem__(self, idx):
        impth, lbpth = self.img_paths[idx], self.lb_paths[idx]        
        label = self.get_label(lbpth)
        cur_datasets = None
        datasets_id = 0
        for i in range(0, len(self.im_len)):
            if idx < self.im_len[i]:
                cur_datasets = self.datasets_name[i-1]
                datasets_id = i-1
                break
        if cur_datasets is None:
            cur_datasets = self.datasets_name[len(self.im_len) - 1]
            datasets_id = len(self.im_len) - 1
            
        if not self.lb_map is None:
            label = self.lb_map[cur_datasets][label]
        if self.mode == 'ret_path':
            return impth, label, lbpth

        img = self.get_image(impth)

        im_lb = dict(im=img, lb=label)
        if not self.trans_func is None:
            im_lb = self.trans_func(im_lb)
        im_lb = self.to_tensor(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        
        return img.detach(), label.unsqueeze(0).detach(), datasets_id
        # return img.detach()

    def get_label(self, lbpth):
        return cv2.imread(lbpth, 0)

    def get_image(self, impth):
        img = cv2.imread(impth)[:, :, ::-1]
        return img

    def __len__(self):
        return self.len


