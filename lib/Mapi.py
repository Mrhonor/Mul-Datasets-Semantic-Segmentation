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
from PIL import Image

import lib.transform_cv2 as T
from lib.base_dataset import BaseDataset, BaseDatasetIm


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
labels_info_v12 = [
{'name': 'Bird', 'id': 0, 'color': [165, 42, 42], 'trainId': 0},
{'name': 'Ground Animal', 'id': 1, 'color': [0, 192, 0], 'trainId': 1},
{'name': 'Curb', 'id': 2, 'color': [196, 196, 196], 'trainId': 2},
{'name': 'Fence', 'id': 3, 'color': [190, 153, 153], 'trainId': 3},
{'name': 'Guard Rail', 'id': 4, 'color': [180, 165, 180], 'trainId': 4},
{'name': 'Barrier', 'id': 5, 'color': [90, 120, 150], 'trainId': 5},
{'name': 'Wall', 'id': 6, 'color': [102, 102, 156], 'trainId': 6},
{'name': 'Bike Lane', 'id': 7, 'color': [128, 64, 255], 'trainId': 7},
{'name': 'Crosswalk - Plain', 'id': 8, 'color': [140, 140, 200], 'trainId': 8},
{'name': 'Curb Cut', 'id': 9, 'color': [170, 170, 170], 'trainId': 9},
{'name': 'Parking', 'id': 10, 'color': [250, 170, 160], 'trainId': 10},
{'name': 'Pedestrian Area', 'id': 11, 'color': [96, 96, 96], 'trainId': 11},
{'name': 'Rail Track', 'id': 12, 'color': [230, 150, 140], 'trainId': 12},
{'name': 'Road', 'id': 13, 'color': [128, 64, 128], 'trainId': 13},
{'name': 'Service Lane', 'id': 14, 'color': [110, 110, 110], 'trainId': 14},
{'name': 'Sidewalk', 'id': 15, 'color': [244, 35, 232], 'trainId': 15},
{'name': 'Bridge', 'id': 16, 'color': [150, 100, 100], 'trainId': 16}, 
{'name': 'Building', 'id': 17, 'color': [70, 70, 70], 'trainId': 17}, 
{'name': 'Tunnel', 'id': 18, 'color': [150, 120, 90], 'trainId': 18}, 
{'name': 'Person', 'id': 19, 'color': [220, 20, 60], 'trainId': 19}, 
{'name': 'Bicyclist', 'id': 20, 'color': [255, 0, 0], 'trainId': 20}, 
{'name': 'Motorcyclist', 'id': 21, 'color': [255, 0, 100], 'trainId': 21}, 
{'name': 'Other Rider', 'id': 22, 'color': [255, 0, 200], 'trainId': 22}, 
{'name': 'Lane Marking - Crosswalk', 'id': 23, 'color': [200, 128, 128], 'trainId': 23}, 
{'name': 'Lane Marking - General', 'id': 24, 'color': [255, 255, 255], 'trainId': 24}, 
{'name': 'Mountain', 'id': 25, 'color': [64, 170, 64], 'trainId': 25}, 
{'name': 'Sand', 'id': 26, 'color': [230, 160, 50], 'trainId': 26}, 
{'name': 'Sky', 'id': 27, 'color': [70, 130, 180], 'trainId': 27},
{'name': 'Snow', 'id': 28, 'color': [190, 255, 255], 'trainId': 28}, 
{'name': 'Terrain', 'id': 29, 'color': [152, 251, 152], 'trainId': 29},
{'name': 'Vegetation', 'id': 30, 'color': [107, 142, 35], 'trainId': 30},
{'name': 'Water', 'id': 31, 'color': [0, 170, 30], 'trainId': 31}, 
{'name': 'Banner', 'id': 32, 'color': [255, 255, 128], 'trainId': 32},
{'name': 'Bench', 'id': 33, 'color': [250, 0, 30], 'trainId': 33}, 
{'name': 'Bike Rack', 'id': 34, 'color': [100, 140, 180], 'trainId': 34}, 
{'name': 'Billboard', 'id': 35, 'color': [220, 220, 220], 'trainId': 35}, 
{'name': 'Catch Basin', 'id': 36, 'color': [220, 128, 128], 'trainId': 36}, 
{'name': 'CCTV Camera', 'id': 37, 'color': [222, 40, 40], 'trainId': 37},
{'name': 'Fire Hydrant', 'id': 38, 'color': [100, 170, 30], 'trainId': 38},
{'name': 'Junction Box', 'id': 39, 'color': [40, 40, 40], 'trainId': 39}, 
{'name': 'Mailbox', 'id': 40, 'color': [33, 33, 33], 'trainId': 255},
{'name': 'Manhole', 'id': 41, 'color': [100, 128, 160], 'trainId': 40},
{'name': 'Phone Booth', 'id': 42, 'color': [142, 0, 0], 'trainId': 41},
{'name': 'Pothole', 'id': 43, 'color': [70, 100, 150], 'trainId': 42}, 
{'name': 'Street Light', 'id': 44, 'color': [210, 170, 100], 'trainId': 43},
{'name': 'Pole', 'id': 45, 'color': [153, 153, 153], 'trainId': 44}, 
{'name': 'Traffic Sign Frame', 'id': 46, 'color': [128, 128, 128], 'trainId': 45},
{'name': 'Utility Pole', 'id': 47, 'color': [0, 0, 80], 'trainId': 46}, 
{'name': 'Traffic Light', 'id': 48, 'color': [250, 170, 30], 'trainId': 47},
{'name': 'Traffic Sign (Back)', 'id': 49, 'color': [192, 192, 192], 'trainId': 48},
{'name': 'Traffic Sign (Front)', 'id': 50, 'color': [220, 220, 0], 'trainId': 49},
{'name': 'Trash Can', 'id': 51, 'color': [140, 140, 20], 'trainId': 50},
{'name': 'Bicycle', 'id': 52, 'color': [119, 11, 32], 'trainId': 51}, 
{'name': 'Boat', 'id': 53, 'color': [150, 0, 255], 'trainId': 52}, 
{'name': 'Bus', 'id': 54, 'color': [0, 60, 100], 'trainId': 53}, 
{'name': 'Car', 'id': 55, 'color': [0, 0, 142], 'trainId': 54}, 
{'name': 'Caravan', 'id': 56, 'color': [0, 0, 90], 'trainId': 55},
{'name': 'Motorcycle', 'id': 57, 'color': [0, 0, 230], 'trainId': 56}, 
{'name': 'On Rails', 'id': 58, 'color': [0, 80, 100], 'trainId': 57},
{'name': 'Other Vehicle', 'id': 59, 'color': [128, 64, 64], 'trainId': 58},
{'name': 'Trailer', 'id': 60, 'color': [0, 0, 110], 'trainId': 59},
{'name': 'Truck', 'id': 61, 'color': [0, 0, 70], 'trainId': 60},
{'name': 'Wheeled Slow', 'id': 62, 'color': [0, 0, 192], 'trainId': 61},
{'name': 'Car Mount', 'id': 63, 'color': [32, 32, 32], 'trainId': 62}, 
{'name': 'Ego Vehicle', 'id': 64, 'color': [120, 10, 10], 'trainId': 63},
{'name': 'Unlabeled', 'id': 65, 'color': [0, 0, 0], 'trainId': 255},
]
mseg_labels_info = [{'name': 'Bird', 'id': 0, 'color': [165, 42, 42], 'trainId': 0},
{'name': 'Ground Animal', 'id': 1, 'color': [0, 192, 0], 'trainId': 1},
{'name': 'Curb', 'id': 2, 'color': [196, 196, 196], 'trainId': 9},
{'name': 'Fence', 'id': 3, 'color': [190, 153, 153], 'trainId': 28},
{'name': 'Guard Rail', 'id': 4, 'color': [180, 165, 180], 'trainId': 29},
{'name': 'Barrier', 'id': 5, 'color': [90, 120, 150], 'trainId': 16},
{'name': 'Wall', 'id': 6, 'color': [102, 102, 156], 'trainId': 17},
{'name': 'Bike Lane', 'id': 7, 'color': [128, 64, 255], 'trainId': 7},
{'name': 'Crosswalk - Plain', 'id': 8, 'color': [140, 140, 200], 'trainId': 7},
{'name': 'Curb Cut', 'id': 9, 'color': [170, 170, 170], 'trainId': 9},
{'name': 'Parking', 'id': 10, 'color': [250, 170, 160], 'trainId': 7},
{'name': 'Pedestrian Area', 'id': 11, 'color': [96, 96, 96], 'trainId': 9},
{'name': 'Rail Track', 'id': 12, 'color': [230, 150, 140], 'trainId': 6},
{'name': 'Road', 'id': 13, 'color': [128, 64, 128], 'trainId': 7},
{'name': 'Service Lane', 'id': 14, 'color': [110, 110, 110], 'trainId': 7},
{'name': 'Sidewalk', 'id': 15, 'color': [244, 35, 232], 'trainId': 9},
{'name': 'Bridge', 'id': 16, 'color': [150, 100, 100], 'trainId': 3},
{'name': 'Building', 'id': 17, 'color': [70, 70, 70], 'trainId': 4},
{'name': 'Tunnel', 'id': 18, 'color': [150, 120, 90], 'trainId': 2},
{'name': 'Person', 'id': 19, 'color': [220, 20, 60], 'trainId': 11},
{'name': 'Bicyclist', 'id': 20, 'color': [255, 0, 0], 'trainId': 13},
{'name': 'Motorcyclist', 'id': 21, 'color': [255, 0, 100], 'trainId': 14},
{'name': 'Other Rider', 'id': 22, 'color': [255, 0, 200], 'trainId': 12},
{'name': 'Lane Marking - Crosswalk', 'id': 23, 'color': [200, 128, 128], 'trainId': 7},
{'name': 'Lane Marking - General', 'id': 24, 'color': [255, 255, 255], 'trainId': 7},
{'name': 'Mountain', 'id': 25, 'color': [64, 170, 64], 'trainId': 30},
{'name': 'Sand', 'id': 26, 'color': [230, 160, 50], 'trainId': 10},
{'name': 'Sky', 'id': 27, 'color': [70, 130, 180], 'trainId': 26},
{'name': 'Snow', 'id': 28, 'color': [190, 255, 255], 'trainId': 8},
{'name': 'Terrain', 'id': 29, 'color': [152, 251, 152], 'trainId': 10},
{'name': 'Vegetation', 'id': 30, 'color': [107, 142, 35], 'trainId': 32},
{'name': 'Water', 'id': 31, 'color': [0, 170, 30], 'trainId': 42},
{'name': 'Banner', 'id': 32, 'color': [255, 255, 128], 'trainId': 31},
{'name': 'Bench', 'id': 33, 'color': [250, 0, 30], 'trainId': 23},
{'name': 'Bike Rack', 'id': 34, 'color': [100, 140, 180], 'trainId': 24},
{'name': 'Billboard', 'id': 35, 'color': [220, 220, 220], 'trainId': 25},
{'name': 'Catch Basin', 'id': 36, 'color': [220, 128, 128], 'trainId': 255},
{'name': 'CCTV Camera', 'id': 37, 'color': [222, 40, 40], 'trainId': 18},
{'name': 'Fire Hydrant', 'id': 38, 'color': [100, 170, 30], 'trainId': 22},
{'name': 'Junction Box', 'id': 39, 'color': [40, 40, 40], 'trainId': 19},
{'name': 'Mailbox', 'id': 40, 'color': [33, 33, 33], 'trainId': 255},
{'name': 'Manhole', 'id': 41, 'color': [100, 128, 160], 'trainId': 255},
{'name': 'Phone Booth', 'id': 42, 'color': [142, 0, 0], 'trainId': 4},
{'name': 'Pothole', 'id': 43, 'color': [70, 100, 150], 'trainId': 7},
{'name': 'Street Light', 'id': 44, 'color': [210, 170, 100], 'trainId': 15},
{'name': 'Pole', 'id': 45, 'color': [153, 153, 153], 'trainId': 27},
{'name': 'Traffic Sign Frame', 'id': 46, 'color': [128, 128, 128], 'trainId': 20},
{'name': 'Utility Pole', 'id': 47, 'color': [0, 0, 80], 'trainId': 27},
{'name': 'Traffic Light', 'id': 48, 'color': [250, 170, 30], 'trainId': 21},
{'name': 'Traffic Sign (Back)', 'id': 49, 'color': [192, 192, 192], 'trainId': 20},
{'name': 'Traffic Sign (Front)', 'id': 50, 'color': [220, 220, 0], 'trainId': 20},
{'name': 'Trash Can', 'id': 51, 'color': [140, 140, 20], 'trainId': 5},
{'name': 'Bicycle', 'id': 52, 'color': [119, 11, 32], 'trainId': 33},
{'name': 'Boat', 'id': 53, 'color': [150, 0, 255], 'trainId': 40},
{'name': 'Bus', 'id': 54, 'color': [0, 60, 100], 'trainId': 36},
{'name': 'Car', 'id': 55, 'color': [0, 0, 142], 'trainId': 34},
{'name': 'Caravan', 'id': 56, 'color': [0, 0, 90], 'trainId': 34},
{'name': 'Motorcycle', 'id': 57, 'color': [0, 0, 230], 'trainId': 35},
{'name': 'On Rails', 'id': 58, 'color': [0, 80, 100], 'trainId': 37},
{'name': 'Other Vehicle', 'id': 59, 'color': [128, 64, 64], 'trainId': 255},
{'name': 'Trailer', 'id': 60, 'color': [0, 0, 110], 'trainId': 39},
{'name': 'Truck', 'id': 61, 'color': [0, 0, 70], 'trainId': 38},
{'name': 'Wheeled Slow', 'id': 62, 'color': [0, 0, 192], 'trainId': 41},
{'name': 'Car Mount', 'id': 63, 'color': [32, 32, 32], 'trainId': 255},
{'name': 'Ego Vehicle', 'id': 64, 'color': [120, 10, 10], 'trainId': 255},
{'name': 'Unlabeled', 'id': 65, 'color': [0, 0, 0], 'trainId': 255}]

# labels_info_train = labels_info_eval
## CityScapes -> {unify class1, unify class2, ...}
# Wall -> {Wall, fence}

# class Mapi(Dataset):
#     '''
#     '''
#     def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
#         super(Mapi, self).__init__()
                
    
#         self.mode = mode
#         self.trans_func = trans_func
#         # self.n_cats = 38
#         self.n_cats = 88
#         self.lb_map = np.arange(256).astype(np.uint8)

#         for el in labels_info:
#             self.lb_map[el['id']] = el['trainId']

#         self.ignore_lb = -1

#         with open(annpath, 'r') as fr:
#             pairs = fr.read().splitlines()

#         self.img_paths, self.lb_paths = [], []
#         for pair in pairs:
#             imgpth, lbpth = pair.split(',')
#             self.img_paths.append(osp.join(dataroot, imgpth))
#             self.lb_paths.append(osp.join(dataroot, lbpth))

#         assert len(self.img_paths) == len(self.lb_paths)
#         self.len = len(self.img_paths)

#         self.to_tensor = T.ToTensor(
#             mean=(0.3038, 0.3383, 0.3034), # city, rgb
#             std=(0.2071, 0.2088, 0.2090),
#         )
        
#         self.colors = []

#         for el in labels_info:
#             (r, g, b) = el['color']
#             self.colors.append((r, g, b))
            
#         self.color2id = dict(zip(self.colors, range(len(self.colors))))

#     def __getitem__(self, idx):
#         impth = self.img_paths[idx]
#         lbpth = self.lb_paths[idx]
#         label = np.array(Image.open(lbpth).convert('RGB'))
#         if self.mode == 'ret_path':
#             return impth, label, lbpth
#         # start = time.time()
#         img = cv2.imread(impth)[:, :, ::-1]

#         # img = cv2.resize(img, (1920, 1280))
        
#         # end = time.time()
#         # print("idx: {}, cv2.imread time: {}".format(idx, end - start))
#         # label = np.array(Image.open(lbpth).convert('RGB').resize((1920, 1280),Image.ANTIALIAS))
        
#         # start = time.time()
#         label = self.convert_labels(label, impth)
#         label = Image.fromarray(label)
#         # end = time.time()
#         # print("idx: {}, convert_labels time: {}".format(idx, end - start))

#         if not self.lb_map is None:
#             label = self.lb_map[label]
#         im_lb = dict(im=img, lb=label)
#         if not self.trans_func is None:
#             # start = time.time()
#             im_lb = self.trans_func(im_lb)
#             # end = time.time()  
#             # print("idx: {}, trans time: {}".format(idx, end - start))
#         im_lb = self.to_tensor(im_lb)
#         img, label = im_lb['im'], im_lb['lb']
        
#         return img.detach(), label.unsqueeze(0).detach()
#         # return img.detach()

#     def __len__(self):
#         return self.len

#     def convert_labels(self, label, impth):
#         mask = np.full(label.shape[:2], 2, dtype=np.uint8)
#         # mask = np.zeros(label.shape[:2])
#         for k, v in self.color2id.items():
#             mask[cv2.inRange(label, np.array(k-1), np.array(k+1)) == 255] = v
            
            
#             # if v == 30 and cv2.inRange(label, np.array(k) - 1, np.array(k) + 1).any() == True:
#             #     label[cv2.inRange(label, np.array(k) - 1, np.array(k) + 1) == 255] = [0, 0, 0]
#             #     cv2.imshow(impth, label)
#             #     cv2.waitKey(0)
#         return mask

class Mapi(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(Mapi, self).__init__(
                dataroot, annpath, trans_func, mode)
    
        
        mode = 'eval'
        self.n_cats = 88
        if mode == 'train':
            self.n_cats = 89
        
        self.lb_ignore = -1
        # self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        
        self.labels_info = labels_info
            
        for el in self.labels_info:
            if mode=='train' and el['trainId'] == 255:
                self.lb_map[el['id']] = self.n_cats - 1
            else:
                self.lb_map[el['id']] = el['trainId']
            

        self.to_tensor = T.ToTensor(
            mean=(0.3038, 0.3383, 0.3034), # city, rgb
            std=(0.2071, 0.2088, 0.2090),
        )
        

class Mapiv1(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(Mapiv1, self).__init__(
                dataroot, annpath, trans_func, mode)
        
        mode = 'eval'
        self.n_cats = 64
        if mode == 'train':
            self.n_cats = 65
        
        self.lb_ignore = -1
        # self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        
        self.labels_info = labels_info_v12
            
        for el in self.labels_info:
            if mode=='train' and el['trainId'] == 255:
                self.lb_map[el['id']] = self.n_cats - 1
            else:
                self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(0.3038, 0.3383, 0.3034), # city, rgb
            std=(0.2071, 0.2088, 0.2090),
        )

class Mapiv1_mseg(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(Mapiv1_mseg, self).__init__(
                dataroot, annpath, trans_func, mode)
        
        
        self.n_cats = 43

        
        self.lb_ignore = -1
        # self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        
        self.labels_info = mseg_labels_info
            
        for el in self.labels_info:
            self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(0.3038, 0.3383, 0.3034), # city, rgb
            std=(0.2071, 0.2088, 0.2090),
        )


## Only return img without label
class MapiIm(BaseDatasetIm):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(MapiIm, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.n_cats = 88
        self.lb_ignore = -1
        self.lb_map = np.arange(256).astype(np.uint8)
        for el in self.labels_info:
            self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(0.3038, 0.3383, 0.3034), # city, rgb
            std=(0.2071, 0.2088, 0.2090),
        )



if __name__ == "__main__":
    lb_map = np.arange(256).astype(np.uint8)
    for i, el in enumerate(mseg_labels_info):
        lb_map[labels_info_v12[i]['trainId']] = el['trainId']
    
    print(lb_map)
    np.save('mapi_relabel.npy', lb_map)
    # from tqdm import tqdm
    # from torch.utils.data import DataLoader
    # ds = CityScapes('./data/', mode='eval')
    # dl = DataLoader(ds,
    #                 batch_size = 4,
    #                 shuffle = True,
    #                 num_workers = 4,
    #                 drop_last = True)
    # for imgs, label in dl:
    #     print(len(imgs))
    #     for el in imgs:
    #         print(el.size())
    #     break
