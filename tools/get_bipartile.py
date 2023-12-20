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
        
        # print(bi_graphs[i].shape)
        # print(max_value, max_index)
              
        buckets = {}
        for index, j in enumerate(max_index):
            if max_value[index] < 1e-4:
               continue
            
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
    print_bipartite(configer, n_datasets, net.bipartite_graphs)

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
                    uni_label_mapping[i].append(f'{dataset_names[dataset_id]}:{class_names[dataset_id][j]}:{adj[dataset_id][j, i]}')
                
        
    for index in range(0, uni_cats):
        print("\"{}\": {}".format(index, uni_label_mapping[index]))    
        

def main():
    configer = Configer(configs='configs/ltbgnn_3_datasets.json')
    net = model_factory[configer.get('model_name')](configer)
    graph_net = model_factory[configer.get('GNN','model_name')](configer)
    graph_net.load_state_dict(torch.load('res/celoss/graph_model_50000.pth', map_location='cpu'), strict=False)
    
        
    state = torch.load('res/celoss/seg_model_50000.pth', map_location='cpu')
    
        
        # del state['unify_prototype']
        # for i in range(0, configer.get('n_datasets')):
        #     del state[f'bipartite_graphs.{i}']
    net.load_state_dict(state, strict=False)
    node_feat = gen_graph_node_feature(configer)
    _, bipart_graphs = graph_net.get_optimal_matching(node_feat, True)
    
    print_bipartite(configer, 3, bipart_graphs)
    


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
