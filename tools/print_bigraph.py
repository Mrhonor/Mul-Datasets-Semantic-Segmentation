#!/usr/bin/python
# -*- encoding: utf-8 -*-


import sys
from time import sleep
sys.path.insert(0, '.')
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
from lib.loss.loss_cross_datasets import CrossDatasetsLoss, CrossDatasetsCELoss, CrossDatasetsCELoss_KMeans, CrossDatasetsCELoss_CLIP, CrossDatasetsCELoss_GNN, CrossDatasetsCELoss_AdvGNN
from lib.class_remap import ClassRemap

from tools.configer import Configer
from evaluate import eval_model_contrast, eval_model_aux, eval_model, eval_model_contrast_single, eval_model_mulbn, eval_model_dsg, eval_model_unlabel, eval_find_use_and_unuse_label

from tensorboardX import SummaryWriter

from lib.module.gen_graph_node_feature import gen_graph_node_feature
from tools.get_bipartile import print_bipartite, find_unuse, print_bipartite2
import clip

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
    parse.add_argument('--port', dest='port', type=int, default=16854,)
    parse.add_argument('--finetune_from', type=str, default=None,)
    parse.add_argument('--config', dest='config', type=str, default='configs/ltbgnn_7_datasets_snp.json',)
    return parse.parse_args()

# 使用绝对路径
args = parse_args()
configer = Configer(configs=args.config)


# cfg_city = set_cfg_from_file(configer.get('dataset1'))
# cfg_cam  = set_cfg_from_file(configer.get('dataset2'))

CITY_ID = 0
CAM_ID = 1
A2D2_ID = 2
SUN_ID = 3
ADE2016_ID = 4
BDD_ID = 5
COCO_ID = 6
IDD_ID = 7
MAPI_ID = 8

# ClassRemaper = ClassRemap(configer=configer)


def set_model(configer):
    logger = logging.getLogger()

    net = model_factory[configer.get('model_name')](configer)

    if configer.get('train', 'finetune'):
        logger.info(f"load pretrained weights from {configer.get('train', 'finetune_from')}")
        state = torch.load('res/celoss/seg_model_900000.pth', map_location='cpu')
        
        # if 'unify_prototype' in state:
        #     del state['unify_prototype']
        #     for i in range(0, configer.get('n_datasets')):
        #         del state[f'bipartite_graphs.{i}']
            
        # new_state = {}
        # for k,v in state.items():
            
        #     if 'bn' in k:
        #         if 'weight' in k:
        #             newk = k.replace('bn', 'affine_weight').replace('.weight', '')
        #             new_state[newk] = v
        #             continue
        #         elif 'bias' in k:
        #             newk = k.replace('bn', 'affine_bias').replace('.bias', '')
        #             new_state[newk] = v
        #             continue
                    
        #     if 'norm.weight' in k:
        #         newk = k.replace('norm.weight', 'affine_weight')
        #         new_state[newk] = v
        #         continue
        #     if 'norm.bias' in k:
        #         newk = k.replace('norm.bias', 'affine_bias')
        #         new_state[newk] = v
        #         continue
                
        #     if 'downsample.1.weight' in k:
        #         newk = k.replace('1.weight', 'affine_weight')
        #         new_state[newk] = v
        #         continue
        #     if 'downsample.1.bias' in k:
        #         newk = k.replace('1.bias', 'affine_bias')
        #         new_state[newk] = v
        #         continue
            
        #     new_state[k] = v
                    
            
        if 'model_state_dict' in state:
            net.load_state_dict(state['model_state_dict'], strict=True)
        else:
            net.load_state_dict(state, strict=True)
            

        
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
        if 'model_state_dict' in state:
            net.load_state_dict(state['model_state_dict'], strict=True)
        else:
            net.load_state_dict(state, strict=True)
        # net.load_state_dict(torch.load(configer.get('train', 'graph_finetune_from'), map_location='cpu'), strict=False)

        
    if configer.get('use_sync_bn'): 
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.train()
    return net

def set_ema_model(configer):
    logger = logging.getLogger()
    # net = NULL
    # if config_files is NULL:
    #     net = model_factory[config_file.model_type](config_file.n_cats)
    # 修改判定
    # if len(config_files) == 0:
    #     net = model_factory[config_file.model_type](config_file.n_cats)
    # else:
    #     n_classes = [cfg.n_cats for cfg in config_files]
    #     net = model_factory[config_file.model_type](config_file.n_cats, 'train', 2, *n_classes)

    net = model_factory[configer.get('model_name') + '_ema'](configer)

    if configer.get('train', 'finetune'):
        logger.info(f"ema load pretrained weights from {configer.get('train', 'finetune_from')}")
        net.load_state_dict(torch.load(configer.get('train', 'finetune_from'), map_location='cpu'), strict=False)

        
    if configer.get('use_sync_bn'): 
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.eval()
    return net



def set_optimizer(model, configer, lr):
    print("lr: ", lr)
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        #  wd_val = cfg.weight_decay
        wd_val = 0
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': lr},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': lr},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.requires_grad == False:
                continue
            
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    
    if configer.get('optim') == 'SGD':
        optim = torch.optim.SGD(
            params_list,
            lr=lr,
            momentum=0.9,
            weight_decay=configer.get('lr', 'weight_decay'),
        )
    elif configer.get('optim') == 'AdamW':
        optim = torch.optim.AdamW(
            params_list,
            lr=lr,
        )
        
    return optim
    
def set_optimizerD(model, configer, lr):
    if hasattr(model, 'get_discri_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_discri_params()
        #  wd_val = cfg.weight_decay
        wd_val = 0
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': lr},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': lr},
        ]
    else:
        return None
    
    if configer.get('optim') == 'SGD':
        optim = torch.optim.SGD(
            params_list,
            lr=lr,
            momentum=0.9,
            weight_decay=configer.get('lr', 'weight_decay'),
        )
    elif configer.get('optim') == 'AdamW':
        optim = torch.optim.AdamW(
            params_list,
            lr=lr,
            weight_decay=configer.get('lr', 'weight_decay'),
        )
    
    return optim

def set_model_dist(net):
    local_rank = dist.get_rank()
    net = nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank, ],
        find_unused_parameters=True,
        output_device=local_rank
        )
    return net

def set_contrast_loss(configer):
    loss_factory = {
        'GNN': CrossDatasetsCELoss_GNN,
        'Adv_GNN': CrossDatasetsCELoss_AdvGNN
    }
    # return CrossDatasetsCELoss_KMeans(configer)
    return loss_factory[configer.get('loss', 'type')](configer)
    # return CrossDatasetsLoss(configer)

def set_meters(configer):
    time_meter = TimeMeter(configer.get('lr', 'max_iter'))
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(configer.get('loss', 'aux_num'))]
    loss_contrast_meter = AvgMeter('loss_contrast')
    loss_domain_meter = AvgMeter('loss_domian')
    kl_loss_meter = AvgMeter('Kl_loss')
            
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters, loss_contrast_meter, loss_domain_meter, kl_loss_meter


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp


def train():
    # torch.autograd.set_detect_anomaly(True)
    n_datasets = configer.get('n_datasets')
    logger = logging.getLogger()
    is_dist = dist.is_initialized()
    # writer = SummaryWriter(configer.get('res_save_pth'))
    use_ema = configer.get('use_ema')
    use_contrast = configer.get('contrast', 'use_contrast')
    joint_train = configer.get('train', 'joint_train')
    mse_or_adv = configer.get('GNN', 'mse_or_adv')
    ## dataset

    
    ## model
    net = set_model(configer=configer)
    total_cats = 0
    for dataset_id in range(1, n_datasets+1):
        total_cats += configer.get('dataset'+str(dataset_id), 'n_cats')
    total_cats = int(total_cats*configer.get('GNN', 'unify_ratio'))

    if is_distributed():
        loaded_map = find_unuse(configer, net.module)
    else:
        loaded_map = find_unuse(configer, net)
    bi_graphs = []
    for dataset_id in range(1, n_datasets+1):
        n_cats = configer.get('dataset'+str(dataset_id), 'n_cats')
        this_bi_graph = torch.zeros(n_cats, total_cats)
        for key, val in loaded_map['dataset'+str(dataset_id)].items():
            for target in val:
                this_bi_graph[key][target[0]] = target[1]
            
        bi_graphs.append(this_bi_graph.cuda())

    # if is_distributed():
    #     net.module.set_bipartite_graphs(bi_graphs)
    # else:
    #     net.set_bipartite_graphs(bi_graphs) 
    print_bipartite2(configer, n_datasets, bi_graphs)


    return


def main():
    if configer.get('use_sync_bn'):
        local_rank = int(os.environ["LOCAL_RANK"])
        # torch.cuda.set_device(args.local_rank)
        # dist.init_process_group(
        #     backend='nccl',
        #     init_method='tcp://127.0.0.1:{}'.format(args.port),
        #     world_size=torch.cuda.device_count(),
        #     rank=args.local_rank
        # )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:{}'.format(args.port),
            world_size=torch.cuda.device_count(),
            rank=local_rank
        )
    
    if not osp.exists(configer.get('res_save_pth')): os.makedirs(configer.get('res_save_pth'))

    setup_logger(f"{configer.get('model_name')}-train", configer.get('res_save_pth'))
    train()

def temp():
    net = set_model(configer=configer)
    

if __name__ == "__main__":
    # dl = get_single_data_loader(configer, aux_mode='train', distributed=False)
    # dl_iter = iter(dl)
    # im_lb, i = dl_iter.next()
    # im, lb = im_lb
    # print(i)
    # # print(im_lb.shape)
    # print(im.shape)
    # print(lb.shape)
    # dls = get_data_loader(configer, aux_mode='train', distributed=False)
    # dl_iters = [iter(dl) for dl in dls]
    
    # for j in range(0,len(dl_iters)):
    #     print("!!!!!!!!!")
    #     print(j)
    #     im, lb = next(dl_iters[j])
    #     sleep(10)
                
    
    main()
