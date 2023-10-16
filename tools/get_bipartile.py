#!/usr/bin/python
# -*- encoding: utf-8 -*-


import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import logging
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.cuda.amp as amp

from lib.models import model_factory
from lib.get_dataloader import get_data_loader, get_single_data_loader
from lib.loss.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg
from lib.loss.loss_cross_datasets import CrossDatasetsLoss, CrossDatasetsCELoss, CrossDatasetsCELoss_KMeans, CrossDatasetsCELoss_CLIP, CrossDatasetsCELoss_GNN
from lib.class_remap import ClassRemap

from evaluate import find_unuse_label

from tools.configer import Configer


from lib.module.gen_graph_node_feature import gen_graph_node_feature
import pickle

torch.set_printoptions(profile="full")

## fix all random seeds
#  torch.manual_seed(123)
#  torch.cuda.manual_seed(123)
#  np.random.seed(123)
#  random.seed(123)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')

def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def is_distributed():
    return torch.distributed.is_initialized()

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
    parse.add_argument('--port', dest='port', type=int, default=16855,)
    parse.add_argument('--finetune_from', type=str, default=None,)
    parse.add_argument('--config', dest='config', type=str, default='configs/ltbgnn_5_datasets.json',)
    return parse.parse_args()

# 使用绝对路径
args = parse_args()
configer = Configer(configs=args.config)


# cfg_city = set_cfg_from_file(configer.get('dataset1'))
# cfg_cam  = set_cfg_from_file(configer.get('dataset2'))

CITY_ID = 0
CAM_ID = 1
A2D2_ID = 2

# ClassRemaper = ClassRemap(configer=configer)

def set_model(configer):
    logger = logging.getLogger()

    net = model_factory[configer.get('model_name')](configer)

    if configer.get('train', 'finetune'):
        logger.info(f"load pretrained weights from {configer.get('train', 'finetune_from')}")
        net.load_state_dict(torch.load("res/celoss/seg_model_final.pth", map_location='cpu'), strict=False)

        
    if configer.get('use_sync_bn'): 
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.train()
    return net

def set_graph_model(configer):
    logger = logging.getLogger()

    net = model_factory[configer.get('GNN','model_name')](configer)

    # if configer.get('train', 'graph_finetune'):
    #     print("!")
    #     logger.info(f"load pretrained weights from {configer.get('train', 'graph_finetune_from')}")
    #     net.load_state_dict(torch.load(configer.get('train', 'graph_finetune_from'), map_location='cpu'), strict=True)
    # state = torch.load("res/celoss/ltbgnn_5_datasets_gnn.pth", map_location='cpu')
    state = torch.load("res/celoss/gnn_model_final.pth", map_location='cpu')
    # print(state['adj_matrix'])

    net.load_state_dict(state, strict=True) 
    if configer.get('use_sync_bn'): 
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.train()
    return net

def print_bipartite(configer, n_datasets, bi_graphs):
    
    # logger = logging.getLogger()
    # is_dist = dist.is_initialized()
    # graph_net = set_graph_model(configer=configer)
    # graph_node_features = gen_graph_node_feature(configer)
    # graph_node_features = graph_node_features.cuda()
    # net = set_model(configer)

    # net.eval()
    # graph_net.eval()
    # unify_prototype, bi_graphs,_ = graph_net(graph_node_features) 
    # unify_prototype, bi_graphs = graph_net.get_optimal_matching(graph_node_features, True)
    # print(len(bi_graphs))
    # net.set_bipartite_graphs(bi_graphs)
    # net.set_unify_prototype(unify_prototype)
    # print(unify_prototype)

    # if configer.get('GNN', 'output_max_adj') and configer.get('GNN', 'output_softmax_and_max_adj'):
    #     bi_graphs = [bi_graph for i, bi_graph in filter(lambda x : x[0] % 2 == 0, enumerate(bi_graphs))]
    # print(bi_graphs)
    for i in range(0, n_datasets):
        
        max_value, max_index = torch.max(bi_graphs[i], dim=0)
        n_cat = configer.get(f'dataset{i+1}', 'n_cats')
        
        buckets = {}
        for index, j in enumerate(max_index):
            
            if int(j) not in buckets:
                buckets[int(j)] = [index]
            else:
                buckets[int(j)].append(index)
            
        print("dataset {}:".format(i+1))    
    
        for index in range(0, n_cat):
            if index not in buckets:
                buckets[index] = []
            print("\"{}\": {}".format(index, buckets[index]))    
        
    
    return 

@torch.no_grad()
def find_unuse(configer, net):
    n_datasets = configer.get('n_datasets')
    is_dist = dist.is_initialized()
    net.eval()
    dls = get_data_loader(configer, aux_mode='train', distributed=is_dist, stage=2)

    out_buckets = {}
    for i in range(0, n_datasets): 
        print("dataset {}:".format(i+1))    
        n_cat = configer.get(f'dataset{i+1}', 'n_cats')
        buckets = find_unuse_label(configer, net, dls[i], n_cat, i)
        for index in range(0, n_cat):
            if index not in buckets:
                buckets[index] = []
            print("\"{}\": {}".format(index, buckets[index]))    
        
        out_buckets[f'dataset{i+1}'] = buckets
        
    net.train()
    return out_buckets

def find_unuse_self():
    # torch.autograd.set_detect_anomaly(True)
    n_datasets = configer.get('n_datasets')
    logger = logging.getLogger()
    is_dist = dist.is_initialized()
    graph_net = set_graph_model(configer=configer)
    graph_node_features = gen_graph_node_feature(configer)
    graph_node_features = graph_node_features.cuda()
    net = set_model(configer)

    net.eval()
    graph_net.eval()
    # unify_prototype, bi_graphs,_ = graph_net(graph_node_features) 
    unify_prototype, bi_graphs = graph_net.get_optimal_matching(graph_node_features, True)
    # print_bipartite(n_datasets, bi_graphs)
    # return
    net.set_bipartite_graphs(bi_graphs)
    net.set_unify_prototype(unify_prototype)
    # print(unify_prototype)

    # if configer.get('GNN', 'output_max_adj') and configer.get('GNN', 'output_softmax_and_max_adj'):
    #     bi_graphs = [bi_graph for i, bi_graph in filter(lambda x : x[0] % 2 == 0, enumerate(bi_graphs))]
    # # print(bi_graphs)
    # for i in range(0, n_datasets):
        
    #     max_value, max_index = torch.max(bi_graphs[i], dim=0)
    #     # print(max_value)
        
    #     buckets = {}
    #     for index, j in enumerate(max_index):
            
    #         if int(j) not in buckets:
    #             buckets[int(j)] = [index]
    #         else:
    #             buckets[int(j)].append(index)
            
    #     print("dataset {}:".format(i+1))    
    dls = get_data_loader(configer, aux_mode='eval', distributed=is_dist)


    out_buckets = {}
    for i in range(0, n_datasets): 
        print("dataset {}:".format(i+1))    
        n_cat = configer.get(f'dataset{i+1}', 'n_cats')
        buckets = find_unuse_label(configer, net, dls[i], n_cat, i)
        for index in range(0, n_cat):
            if index not in buckets:
                buckets[index] = []
            print("\"{}\": {}".format(index, buckets[index]))    
        
        out_buckets[f'dataset{i+1}'] = buckets
        
    with open('bipart.pkl', 'wb') as file:
        pickle.dump(out_buckets, file)

    return


def print_unified_label_mapping(configer, bi_graphs, adj):
    n_datasets = configer.get('n_datasets')
    total_cats = 0
    cat_buckets = []
    class_names = []
    dataset_names = []
    for i in range(0, n_datasets):
        cat_buckets.append(configer.get(f'dataset{i+1}', 'n_cats'))
        class_names.append(configer.get(f'dataset{i+1}', 'label_names'))
        dataset_names.append(configer.get(f'dataset{i+1}', 'data_reader'))
        total_cats += configer.get(f'dataset{i+1}', 'n_cats')
    
    uni_cats = int(total_cats*configer.get('GNN', 'unify_ratio'))
    uni_graph = torch.zeros((total_cats, uni_cats))
    for i in range(0, n_datasets):
        uni_graph[sum(cat_buckets[0:i]):sum(cat_buckets[0:i+1]), :] = bi_graphs[i]
        
    uni_label_mapping = {}
    for i in range(0, uni_cats):
        uni_label_mapping[i] = []
        for dataset_id in range(0, n_datasets):

            # find each col the value is 1
            for j in range(0, cat_buckets[dataset_id]):
                if bi_graphs[dataset_id][j,i] == 1:
                    
                    # print(len(adj))
                    # print(sum(cat_buckets[0:dataset_id])+i)
                    # print(adj[sum(cat_buckets[0:dataset_id])+i,uni_cats+j])
                    uni_label_mapping[i].append(f'{dataset_names[dataset_id]}:{class_names[dataset_id][j]}:adj:{adj[dataset_id][j, i]}')
                
        
    for index in range(0, uni_cats):
        print("\"{}\": {}".format(index, uni_label_mapping[index]))    
        

def main():
    configer = Configer(configs='configs/ltbgnn_7_datasets_2.json')
    net = model_factory[configer.get('model_name')](configer)
    graph_net = model_factory[configer.get('GNN','model_name')](configer)

    
        
    state = torch.load('res/celoss/seg_model_150000.pth', map_location='cpu')
    
        
        # del state['unify_prototype']
        # for i in range(0, configer.get('n_datasets')):
        #     del state[f'bipartite_graphs.{i}']
    net.load_state_dict(state, strict=False)
    
    graph_net.load_state_dict(torch.load('res/celoss/graph_model_150000.pth', map_location='cpu'), strict=False)
    node_feats = gen_graph_node_feature(configer)
    _, adj, _, _ = graph_net(node_feats)
    loaded_map = []
    loaded_map.append({
        "0": [36, 64, 68, 113, 156, 175, 197, 210, 217],
        "1": [19, 37, 49, 70, 98, 136, 141, 180],
        "2": [42, 46, 62, 112, 129, 164, 177, 187, 189],
        "3": [45, 94, 95, 133, 154, 211],
        "4": [24, 47, 66, 81, 104, 142, 146, 148, 185, 215],
        "5": [30, 60, 91, 97, 111, 163, 168, 169, 192, 199],
        "6": [75, 84, 127, 134, 145, 212, 223],
        "7": [40, 65, 76, 105, 128, 137, 165, 190],
        "8": [18, 28, 34, 38, 57, 63, 110, 153, 191, 200, 216],
        "9": [31, 96, 103, 115, 149, 178],
        "10": [8, 13, 56, 77, 92, 132, 138, 150, 159, 182],
        "11": [5, 17, 20, 48, 54, 59, 100, 155, 162, 172, 205, 208],
        "12": [35, 78, 86, 90, 114, 179, 219],
        "13": [22, 41, 51, 73, 101, 117, 124, 152, 157, 174, 203],
        "14": [3, 23, 26, 33, 43, 186, 202, 206, 214],
        "15": [2, 119, 130, 131, 140, 143, 183],
        "16": [69, 89, 121, 144, 176],
        "17": [4, 12, 53, 87, 126, 151],
        "18": [10, 14, 27, 93, 181, 222],
    })
    loaded_map.append({
        "0": [186],
"1": [23, 138],
"2": [20, 140, 193, 205, 208],
"3": [85, 109, 148, 155, 179, 216],
"4": [17, 48, 54, 187],
"5": [40, 60, 65, 165],
"6": [24, 63, 67],
"7": [96, 136, 164, 207],
"8": [84, 177, 201],
"9": [76],
"10": [30, 135],
"11": [25, 39, 56, 59, 74, 103],
"12": [31, 102, 170, 197],
"13": [49, 163, 175, 184, 209],
"14": [44, 75, 150, 166],
"15": [19, 27, 37, 98, 141, 180, 199],
"16": [89, 128, 129],
"17": [10, 46, 62, 120, 144],
"18": [14, 45, 104, 116],
"19": [121, 169, 213],
"20": [86],
"21": [196],
"22": [70],
"23": [21, 53, 82, 212],
"24": [29, 94, 113, 123, 125, 133],
"25": [28, 50, 100, 204],
"26": [111],
"27": [1, 101],
"28": [6, 51, 91, 97],
"29": [2, 87, 115, 178, 192],
"30": [18, 57, 130, 191],
"31": [95, 195, 223],
"32": [38, 114, 219],
"33": [137],
"34": [16],
"35": [5, 81, 131, 147],
"36": [132],
"37": [13, 198, 200],
"38": [171],
"39": [154, 211],
"40": [157],
"41": [9],
"42": [0],
"43": [168, 176, 221],
"44": [41, 90, 110, 119, 174],
"45": [107, 172],
"46": [47, 153, 167, 183, 222],
"47": [162],
"48": [34, 215],
"49": [126, 146, 151, 190],
"50": [161],
"51": [58],
"52": [78, 122],
"53": [22, 35, 73, 203],
"54": [12, 61, 117, 152, 182],
"55": [66],
"56": [72, 105],
"57": [188],
"58": [69, 139, 220],
"59": [33, 145, 202],
"60": [3, 11, 26, 43, 214],
"61": [142],
"62": [77, 124],
"63": [55, 68, 156, 217],
    })
    
    loaded_map.append({
        "0": [16, 78, 190],
"1": [34, 36, 45, 63, 67, 175, 200, 217],
"2": [27, 57, 59, 100, 107, 172, 204],
"3": [20, 22, 85, 110, 119, 148, 153],
"4": [2, 30, 52, 106, 135, 168, 178, 209],
"5": [17, 54, 99, 170, 195],
"6": [4, 73, 101, 117, 124, 157, 171],
"7": [40, 43, 88, 105, 122, 186],
"8": [13, 32, 94, 197],
"9": [29, 127, 134],
"10": [58, 64, 70, 96, 149],
"11": [19, 24, 112, 136, 154],
"12": [8, 28, 50, 61, 140, 203],
"13": [109, 192, 221],
"14": [39, 49, 68, 98],
"15": [76, 93, 181, 210],
"16": [38, 114, 131, 147, 183, 211, 218, 223],
"17": [14, 35, 41, 47, 90, 144, 174, 185, 222],
"18": [9, 15, 77, 91, 118, 169, 176],
"19": [0, 21, 62, 177, 189],
"20": [74],
"21": [5, 37, 95, 179, 187, 208],
"22": [18, 115, 132, 133, 198],
"23": [65, 142, 155, 205],
"24": [53, 164],
"25": [75, 82, 159, 201, 212],
"26": [44, 60, 125, 137, 163, 184],
"27": [48, 97, 162, 199],
"28": [81, 84, 113, 126, 146, 191, 215, 216],
"29": [66, 89, 111, 128, 129, 165],
"30": [12, 138, 166, 182],
"31": [10, 86, 219],
"32": [108, 145, 150, 202],
"33": [46, 120, 121, 161],
"34": [3, 69, 139, 152, 220],
"35": [92],
"36": [11, 206, 214],
    })
    
    loaded_map.append({
        "0": [15, 25, 49, 52, 56, 74, 77, 98, 103, 135, 141, 178, 209],
"1": [19, 27, 37, 97, 121, 129, 136, 169, 199],
"2": [10, 62, 112, 131, 183, 187, 189, 194, 218],
"3": [24, 29, 40, 104, 128, 137, 175],
"4": [17, 48, 54, 78, 134, 142, 148, 162, 208, 219],
"5": [13, 44, 75, 82, 92, 127, 132, 150, 166],
"6": [53, 79, 94, 96, 149, 173, 177],
"7": [76, 93, 113, 126, 146, 151, 193, 215],
"8": [18, 28, 57, 67, 110, 153, 191, 216],
"9": [2, 30, 87, 106, 130, 163, 192],
"10": [1, 31, 32, 55, 70, 73, 101, 116, 171, 196, 197],
"11": [84, 95, 99, 114, 145, 159, 207, 223],
"12": [23, 46, 86, 138],
"13": [8, 9, 12, 61, 90, 100, 124, 140, 152, 157, 182, 204],
"14": [11, 26, 43, 69, 72, 105, 108, 186, 202, 214],
"15": [4, 20, 22, 35, 41, 117, 174, 203, 222],
"16": [65, 66, 107, 144, 172, 179],
"17": [14, 64, 89, 133, 168, 184],
"18": [34, 45, 59, 115, 180, 200],
    })
    dls = get_data_loader(configer, aux_mode='train', distributed=False, stage=2)


    n_cat = configer.get('dataset5', 'n_cats')
    buckets = find_unuse_label(configer, net.cuda(), dls[4], n_cat, 4)
    for index in range(0, n_cat):
        if index not in buckets:
            buckets[index] = []
        print("\"{}\": {}".format(index, buckets[index]))    
    
    loaded_map.append(buckets)
    # loaded_map = find_unuse(configer, net.cuda())
    
    loaded_map.append({
        "0": [92],
"1": [133, 163, 191],
"2": [10, 112],
"3": [107],
"4": [57, 98],
"5": [176],
"6": [132, 200],
"7": [171],
"8": [32],
"9": [34, 67, 113, 215],
"10": [2, 207],
"11": [63],
"12": [31, 103, 141],
"13": [14],
"14": [178],
"15": [38, 154, 211],
"16": [65],
"17": [6, 199],
"18": [209],
"19": [201],
"20": [146],
"21": [69],
"22": [134],
"23": [47, 222],
"24": [1, 140],
"25": [125],
"26": [46, 161],
"27": [3],
"28": [28, 167],
"29": [74, 208],
"30": [23],
"31": [18, 153],
"32": [5, 48, 78],
"33": [95],
"34": [40, 137, 190],
"35": [91, 192],
"36": [82, 94, 115],
"37": [29],
"38": [142],
"39": [157],
"40": [101],
"41": [175, 217],
"42": [165],
"43": [44, 87],
"44": [35],
"45": [24, 129],
"46": [85],
"47": [26, 97],
"48": [100, 148, 179],
"49": [220],
"50": [96, 197],
"51": [79],
"52": [99],
"53": [55],
"54": [193],
"55": [25, 39, 156, 158],
"56": [76],
"57": [15, 118],
"58": [130],
"59": [106],
"60": [75],
"61": [169],
"62": [159, 212],
"63": [30],
"64": [109],
"65": [108],
"66": [0],
"67": [41, 203, 204],
"68": [16],
"69": [68],
"70": [136],
"71": [81, 216],
"72": [49],
"73": [123],
"74": [152],
"75": [127],
"76": [180],
"77": [58],
"78": [60],
"79": [33],
"80": [21, 149, 173],
"81": [120, 177, 189],
"82": [17, 54, 162],
"83": [13],
"84": [45],
"85": [119],
"86": [11],
"87": [62, 90],
"88": [77],
"89": [110],
"90": [73, 104],
"91": [186],
"92": [145, 150],
"93": [102, 170],
"94": [42, 182],
"95": [205],
"96": [84],
"97": [210],
"98": [138],
"99": [213],
"100": [66, 128],
"101": [50],
"102": [184],
"103": [4, 12],
"104": [116],
"105": [195],
"106": [27],
"107": [151],
"108": [86],
"109": [223],
"110": [144],
"111": [164],
"112": [117, 166],
"113": [89],
"114": [202],
"115": [198],
"116": [88, 214],
"117": [105, 206],
"118": [188],
"119": [185],
"120": [219],
"121": [172],
"122": [139],
"123": [52],
"124": [174],
"125": [122],
"126": [70, 196],
"127": [135],
"128": [53],
"129": [155],
"130": [93],
"131": [72],
"132": [9],
"133": [51],
"134": [147],
"135": [59],
"136": [114],
"137": [71],
"138": [20],
"139": [187, 194],
"140": [22],
"141": [126],
"142": [36],
"143": [61],
"144": [143],
"145": [56],
"146": [121],
"147": [124],
"148": [111],
"149": [168],
    })
    
    loaded_map.append({
        "0": [14, 72, 89, 121],
"1": [114, 220, 223],
"2": [10, 66],
"3": [41, 157, 203],
"4": [101, 204],
"5": [169],
"6": [213],
"7": [107],
"8": [35, 122],
"9": [26],
"10": [183],
"11": [108],
"12": [184],
"13": [3, 127, 139],
"14": [189],
"15": [85, 110, 216],
"16": [82],
"17": [97, 128],
"18": [109],
"19": [84, 136],
"20": [208],
"21": [75, 212],
"22": [123],
"23": [165],
"24": [74],
"25": [156],
"26": [7],
"27": [71],
"28": [18, 28, 34, 153],
"29": [12],
"30": [4],
"31": [179],
"32": [70, 166],
"33": [65],
"34": [102],
"35": [53, 80, 149],
"36": [23],
"37": [155, 172],
"38": [168],
"39": [129],
"40": [193],
"41": [196],
"42": [158],
"43": [29],
"44": [194],
"45": [69],
"46": [214],
"47": [21],
"48": [93],
"49": [51, 145],
"50": [202],
"51": [61, 152],
"52": [8, 140],
"53": [207],
"54": [99],
"55": [219],
"56": [176],
"57": [46, 161],
"58": [73, 117, 171, 222],
"59": [13, 138],
"60": [30, 209],
"61": [49, 96],
"62": [94, 201],
"63": [87],
"64": [159],
"65": [132],
"66": [92],
"67": [19],
"68": [20],
"69": [118],
"70": [50],
"71": [40],
"72": [95],
"73": [170],
"74": [90],
"75": [175],
"76": [5],
"77": [17, 48],
"78": [164],
"79": [86],
"80": [24, 104],
"81": [143],
"82": [44],
"83": [55],
"84": [98, 191],
"85": [150],
"86": [2, 106, 211],
"87": [32, 77, 182],
"88": [42],
"89": [11, 22],
"90": [147],
"91": [62],
"92": [162],
"93": [197],
"94": [116],
"95": [173, 177],
"96": [124, 188],
"97": [6, 37],
"98": [52, 56],
"99": [76],
"100": [59],
"101": [185],
"102": [43, 105, 186],
"103": [25, 141, 217],
"104": [125],
"105": [16, 126, 151],
"106": [206],
"107": [115],
"108": [120],
"109": [54, 142, 148],
"110": [205],
"111": [91, 192],
"112": [63, 221],
"113": [146],
"114": [180],
"115": [0, 187, 218],
"116": [36, 190, 215],
"117": [137],
"118": [38, 47, 67],
"119": [64, 113],
"120": [81],
"121": [130],
"122": [57, 100],
"123": [1, 15, 103],
"124": [68, 133],
"125": [163, 178, 199],
"126": [27, 60, 88, 111],
"127": [198, 200],
"128": [78, 134, 195],
"129": [119, 144, 167],
"130": [135],
"131": [45, 112, 154],
"132": [9, 174],
    })

    
    n_datasets = configer.get('n_datasets')
    total_cats = 0

    for i in range(0, n_datasets):
        total_cats += configer.get(f'dataset{i+1}', 'n_cats')
    total_cats = int(total_cats*configer.get('GNN', 'unify_ratio'))

     
    bi_graphs = []
    for dataset_id in range(1, n_datasets+1):
        n_cats = configer.get('dataset'+str(dataset_id), 'n_cats')
        this_bi_graph = torch.zeros(n_cats, total_cats)
        print(dataset_id)
        # for key, val in loaded_map['dataset'+str(dataset_id)].items():
        for key, val in loaded_map[dataset_id-1].items():
            this_bi_graph[int(key)][val] = 1
            
        bi_graphs.append(this_bi_graph.cuda())
        
    print_unified_label_mapping(configer, bi_graphs, adj)


if __name__ == "__main__":
    # dl = get_single_data_loader(configer, aux_mode='train', distributed=False)
    # dl_iter = iter(dl)
    # im_lb, i = dl_iter.next()
    # im, lb = im_lb
    # print(i)
    # # print(im_lb.shape)
    # print(im.shape)
    # print(lb.shape)
    main()
