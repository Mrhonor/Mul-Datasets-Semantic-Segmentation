import sys
from time import sleep
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


from tools.configer import Configer
from evaluate import eval_model_contrast, eval_model_unseen
from lib.module.gen_graph_node_feature import gen_graph_node_feature_test, gen_graph_node_feature

import clip

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
    parse.add_argument('--port', dest='port', type=int, default=16854,)
    parse.add_argument('--finetune_from', type=str, default=None,)
    parse.add_argument('--config', dest='config', type=str, default='configs/scannet.json',)
    return parse.parse_args()

# 使用绝对路径
args = parse_args()
configer = Configer(configs=args.config)


def set_model(configer):
    logger = logging.getLogger()

    net = model_factory[configer.get('model_name')](configer)

    if configer.get('train', 'finetune'):
        logger.info(f"load pretrained weights from {configer.get('train', 'finetune_from')}")
        state = torch.load(configer.get('train', 'finetune_from'), map_location='cpu')
        
        # del state['unify_prototype']
        for i in range(0, 3):
            del state[f'bipartite_graphs.{i}']
        net.load_state_dict(state, strict=False)

        
    if configer.get('use_sync_bn'): 
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.train()
    return net

def set_graph_model(configer):
    logger = logging.getLogger()

    net = model_factory[configer.get('GNN','model_name')](configer)

    if configer.get('train', 'graph_finetune'):
        logger.info(f"load pretrained weights from {configer.get('train', 'graph_finetune_from')}")
        state = torch.load(configer.get('train', 'graph_finetune_from'), map_location='cpu')
        del state['unlable_node_features']
        del state['adj_matrix']
        
        net.load_state_dict(state, strict=False)

    
        
    if configer.get('use_sync_bn'): 
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.train()
    return net


def eval_wd():
    
    logger = logging.getLogger()
    
    ## model
    net = set_model(configer=configer)
    with open('camvid_mapping.txt', 'r') as f:
        lines = f.readlines()
    
    # bi_graphs = []
    # for i, line in enumerate(lines):
    #     bi_graph = torch.zeros((11, 224), dtype=torch.float32).cuda()
    #     ids = line.replace('\n', '').replace(' ', '').split(',')
    #     ids = [int(id) for id in ids]
    #     for id in ids:
    #         bi_graph[i, id] = 1
    # bi_graphs.append(bi_graph)
    
    
    
    graph_net = set_graph_model(configer=configer)
    graph_node_features = gen_graph_node_feature(configer).cuda()
    
    unify_prototype, bi_graphs = graph_net.get_optimal_matching(graph_node_features, True)
    # unify_prototype, bi_graphs = concat_class_lb(configer, net)

    # net.get_encode_lb_vec()
    # net.set_unify_prototype(unify_prototype)
    net.set_bipartite_graphs(bi_graphs)
    heads, mious = eval_model_contrast(configer, net)
    logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
    

if __name__ == '__main__':
    eval_wd()