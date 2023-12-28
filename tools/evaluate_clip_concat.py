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
from evaluate import eval_model_contrast, eval_model_uni, eval_model_clip
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
        # for i in range(0, configer.get('n_datasets')):
        #     del state[f'bipartite_graphs.{i}']
        net.load_state_dict(state, strict=False)

        
    if configer.get('use_sync_bn'): 
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.train()
    return net

def concat_class_lb(configer):
    n_datasets = configer.get('n_datasets')
    class_names = []
    for i in range(0, n_datasets):
        class_names.append(configer.get(f'dataset{i+1}', 'label_names'))
    
    unify_class_names = []
    for class_name in class_names:
        unify_class_names.extend(class_name)

    unify_class_names = list(set(unify_class_names))
    
    bi_graphs = []    
    for i in range(0, n_datasets):
        class_name = class_names[i]
        bi_graph = torch.zeros((1+len(class_name), len(unify_class_names)), dtype=torch.float32).cuda()
        for j in range(0, len(class_name)):
            bi_graph[j][unify_class_names.index(class_name[j])] = 1
        for j in range(0, len(unify_class_names)):
            if bi_graph[-1][j].sum() == 0:
                bi_graph[-1][j] = 1
        bi_graphs.append(bi_graph)
        
    text_feature_vecs = []
    with torch.no_grad():
        clip_model, _ = clip.load("ViT-B/32", device="cuda")
            
        lb_name = [f'a photo of {name} from dataset {i+1}.' for name in unify_class_names]
        text = clip.tokenize(lb_name).cuda()
        text_features = clip_model.encode_text(text).type(torch.float32)
        # text_feature_vecs.append(text_features)
            
    unify_prototype = nn.Parameter(text_features,
                    requires_grad=False).cuda()
    return unify_prototype, bi_graphs

def eval_clip_concat():
    
    logger = logging.getLogger()
    
    ## model
    net = set_model(configer=configer)
    
    # net.aux_mode = 'clip'
    # unify_prototype, bi_graphs = concat_class_lb(configer)

    net.get_encode_lb_vec()
    # net.set_unify_prototype(unify_prototype)
    # net.set_bipartite_graphs(bi_graphs)
    # heads, mious = eval_model_uni(configer, net)
    heads, mious = eval_model_clip(configer, net)
    print(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
    

if __name__ == '__main__':
    eval_clip_concat()