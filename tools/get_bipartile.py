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
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg
from lib.loss.loss_cross_datasets import CrossDatasetsLoss, CrossDatasetsCELoss, CrossDatasetsCELoss_KMeans, CrossDatasetsCELoss_CLIP, CrossDatasetsCELoss_GNN
from lib.class_remap import ClassRemap

from tools.configer import Configer


from lib.module.gen_graph_node_feature import gen_graph_node_feature

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
    parse.add_argument('--config', dest='config', type=str, default='configs/ltbgnn_city_cam_a2d2.json',)
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



def set_graph_model(configer):
    logger = logging.getLogger()

    net = model_factory[configer.get('GNN','model_name')](configer)

    # if configer.get('train', 'graph_finetune'):
    #     print("!")
    #     logger.info(f"load pretrained weights from {configer.get('train', 'graph_finetune_from')}")
    #     net.load_state_dict(torch.load(configer.get('train', 'graph_finetune_from'), map_location='cpu'), strict=True)
    state = torch.load("res/celoss/gnn_model_final.pth", map_location='cpu')
    # print(state['adj_matrix'])

    net.load_state_dict(state, strict=True) 
    if configer.get('use_sync_bn'): 
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.train()
    return net

def train():
    # torch.autograd.set_detect_anomaly(True)
    n_datasets = configer.get('n_datasets')
    logger = logging.getLogger()
    is_dist = dist.is_initialized()
    graph_net = set_graph_model(configer=configer)
    graph_node_features = gen_graph_node_feature(configer)
    graph_node_features = graph_node_features.cuda()

    graph_net.eval()
    unify_prototype, bi_graphs,_ = graph_net(graph_node_features) 
    # print(bi_graphs)
    for i in range(0, n_datasets):
        
        max_value, max_index = torch.max(bi_graphs[i], dim=0)
        print(max_value)
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


def main():
    train()


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
