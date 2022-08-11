#!/usr/bin/python
# -*- encoding: utf-8 -*-

# from asyncio.windows_events import NULL # 有bug，去掉改为别的判定
import sys

import os
import os.path as osp
import random
import logging
import time
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.get_dataloader import get_data_loader
# from tools.evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg

# import fvcore.nn as fvnn
from lib.precise_bn import update_bn_stats

from lib.models.bisenetv2 import SegmentBranch, DetailBranch, BGALayer, SegmentHead
from lib.get_dataloader import get_a2d2_city_loader

weight_pth = 'res/same_affine_layer.pth'

## fix all random seeds
#  torch.manual_seed(123)
#  torch.cuda.manual_seed(123)
#  np.random.seed(123)
#  random.seed(123)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')

class BGAandHead(nn.Module):

    def __init__(self, n_classes, aux_mode='train', n_bn=1, *other_n_classes):
        super(BGAandHead, self).__init__()
        self.aux_mode = aux_mode

        self.bga = BGALayer(n_bn=n_bn)

        ## TODO: what is the number of mid chan ?
        self.head = nn.ModuleList([SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)])

        for n in other_n_classes:
            self.head.append(SegmentHead(128, 1024, n, up_factor=8, aux=False))



    def forward(self, x, dataset=0, *other_x):
        # x = x.cuda()
        ## other_x 其他数据集的输入
        # size = x.size()[2:]


        feat_head = self.bga(x[0], x[1], dataset)

        # logits = self.head[0](feat_head[0])
        ## 修改为多数据集模式，返回list
        
        # logits = [logit(feat_head[i]) for i, logit in enumerate(self.head)]
        ## 修改后支持单张图片输入
        if (len(other_x) == 0):
            # logits = [self.head[dataset](feat_head[0])]
            logits = [self.head[1](feat_head[0])]
        else:
            logits = [self.head[i](feat_head[i]) for i in range(0, len(other_x) + 1)]
        ## TODO 修改下面的多数据集模式
        if (len(other_x) == 0):
            if self.aux_mode == 'train':

                return logits
            elif self.aux_mode == 'eval':
                return logits,
            elif self.aux_mode == 'pred':
                # pred = logits.argmax(dim=1)
                pred = [logit.argmax(dim=1) for logit in logits]
                return pred
            else:
                raise NotImplementedError
        else:
            if self.aux_mode == 'train':

                return logits
            elif self.aux_mode == 'eval':
                return logits,
            elif self.aux_mode == 'pred':
                # pred = logits.argmax(dim=1)
                pred = [logit.argmax(dim=1) for logit in logits]
                return pred
            else:
                raise NotImplementedError


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
    parse.add_argument('--port', dest='port', type=int, default=12345,)
    parse.add_argument('--config', dest='config', type=str,
            default='/configs/bisenetv2.py',)
    parse.add_argument('--finetune_from', type=str, default=None,)
    return parse.parse_args()

# 使用绝对路径
args = parse_args()
cfg = set_cfg_from_file(args.config)
cfg_a2d2 = set_cfg_from_file('configs/bisenetv2_a2d2.py')
cfg_city = set_cfg_from_file('configs/bisenetv2_city.py')
cfg_cam = set_cfg_from_file('configs/bisenetv2_cam.py')


def load_pretrained(net):
    state = torch.load(args.finetune_from, map_location='cpu')

    detail_state = {}
    detail_state['S1_1.conv.weight'] = state['detail']['S1.0.conv.weight']
    detail_state['S1_2.conv.weight'] = state['detail']['S1.1.conv.weight']
    detail_state['S2_1.conv.weight'] = state['detail']['S2.0.conv.weight']
    detail_state['S2_2.conv.weight'] = state['detail']['S2.1.conv.weight']
    detail_state['S2_3.conv.weight'] = state['detail']['S2.2.conv.weight']
    detail_state['S3_1.conv.weight'] = state['detail']['S3.0.conv.weight']
    detail_state['S3_2.conv.weight'] = state['detail']['S3.1.conv.weight']
    detail_state['S3_3.conv.weight'] = state['detail']['S3.2.conv.weight']

    segment_state = {}
    segment_state['S1S2.conv.conv.weight'] = state['segment']['S1S2.conv.conv.weight']
    segment_state['S1S2.left_1.conv.weight'] = state['segment']['S1S2.left.0.conv.weight']
    segment_state['S1S2.left_2.conv.weight'] = state['segment']['S1S2.left.1.conv.weight']
    segment_state['S1S2.fuse.conv.weight'] = state['segment']['S1S2.fuse.conv.weight']

    def loadGELayerS2(srcDict, src_name, targerdict, target_name):
        targerdict[target_name + '.conv1.conv.weight'] = srcDict[src_name + '.conv1.conv.weight']
        targerdict[target_name + '.dwconv1.conv.weight'] = srcDict[src_name + '.dwconv1.0.weight']
        targerdict[target_name + '.dwconv2.conv.weight'] = srcDict[src_name + '.dwconv2.0.weight']
        targerdict[target_name + '.conv2.conv.weight'] = srcDict[src_name + '.conv2.0.weight']
        targerdict[target_name + '.shortcut_1.conv.weight'] = srcDict[src_name + '.shortcut.0.weight']
        targerdict[target_name + '.shortcut_2.conv.weight'] = srcDict[src_name + '.shortcut.2.weight']

    def loadGELayerS1(srcDict, src_name, targerdict, target_name):
        targerdict[target_name + '.conv1.conv.weight'] = srcDict[src_name + '.conv1.conv.weight']
        targerdict[target_name + '.dwconv.conv.weight'] = srcDict[src_name + '.dwconv.0.weight']
        targerdict[target_name + '.conv2.conv.weight'] = srcDict[src_name + '.conv2.0.weight']

    loadGELayerS2(state['segment'], 'S3.0', segment_state, 'S3_1')
    loadGELayerS1(state['segment'], 'S3.1', segment_state, 'S3_2')
    loadGELayerS2(state['segment'], 'S4.0', segment_state, 'S4_1')
    loadGELayerS1(state['segment'], 'S4.1', segment_state, 'S4_2')
    loadGELayerS2(state['segment'], 'S5_4.0', segment_state, 'S5_4_1')
    loadGELayerS1(state['segment'], 'S5_4.1', segment_state, 'S5_4_2')
    loadGELayerS1(state['segment'], 'S5_4.2', segment_state, 'S5_4_3')
    loadGELayerS1(state['segment'], 'S5_4.3', segment_state, 'S5_4_4')
    segment_state['S5_5.conv_gap.conv.weight'] = state['segment']['S5_5.conv_gap.conv.weight']
    segment_state['S5_5.conv_last.conv.weight'] = state['segment']['S5_5.conv_last.conv.weight']

    bga_state = {}
    bga_state['left1_convbn.conv.weight'] = state['bga']['left1.0.weight']
    bga_state['left1_conv.weight'] = state['bga']['left1.2.weight']
    bga_state['left2_convbn.conv.weight'] = state['bga']['left2.0.weight']
    bga_state['right1.conv.weight'] = state['bga']['right1.0.weight']
    bga_state['right2_convbn.conv.weight'] = state['bga']['right2.0.weight']
    bga_state['right2_conv.weight'] = state['bga']['right2.2.weight']
    bga_state['conv.conv.weight'] = state['bga']['conv.0.weight']

    net.detail.load_state_dict(detail_state, strict=True)
    net.segment.load_state_dict(segment_state, strict=True)
    net.bga.load_state_dict(bga_state, strict=True)


def set_model(config_file, *config_files):
    logger = logging.getLogger()
    # net = NULL
    # if config_files is NULL:
    #     net = model_factory[config_file.model_type](config_file.n_cats)
    # 修改判定
    if len(config_files) == 0:
        net = model_factory[config_file.model_type](config_file.n_cats)
    else:
        n_classes = [cfg.n_cats for cfg in config_files]
        # net = model_factory[config_file.model_type](config_file.n_cats, 'train', 2, *n_classes)
        net = model_factory[config_file.model_type](config_file.n_cats, 'eval', 2, *n_classes)

    if not args.finetune_from is None:
        logger.info(f'load pretrained weights from {args.finetune_from}')
        net.load_state_dict(torch.load(args.finetune_from, map_location='cpu'))
    # if config_file.use_sync_bn: net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    # net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7)
    criteria_aux = [OhemCELoss(0.7) for _ in range(config_file.num_aux_heads)]
    return net, criteria_pre, criteria_aux


def set_optimizer(model, config_file):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        #  wd_val = cfg.weight_decay
        wd_val = 0
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': config_file.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': config_file.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=config_file.lr_start,
        momentum=0.9,
        weight_decay=config_file.weight_decay,
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


def set_meters(config_file):
    time_meter = TimeMeter(config_file.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(config_file.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters

def get_segment_state(state):
    segment_state = {}
    for key in state.keys():
        if key.find('segment') != -1 and key.find('bn.1.') == -1:
    #         print(key[8:-1])
            segment_state[key[8:]] = state[key]
    return(segment_state)
    
def get_detail_state(state):
    detail_state = {}
    for key in state.keys():
        if key.find('detail') != -1 and key.find('bn.1.') == -1:
    #         print(key[8:-1])
            detail_state[key[7:]] = state[key]
    return(detail_state)

def train_bn():
    n_dataset = 2
    logger = logging.getLogger()
    # is_dist = dist.is_initialized()
    ## dataset
    dl = get_data_loader(cfg, mode='val', distributed=False)
    # dl_a2d2 = get_data_loader(cfg_a2d2, mode='train', distributed=False)
    # dl = get_a2d2_city_loader(cfg_a2d2, cfg_city, mode='train', distributed=False)
    
    ## model
    net, criteria_pre, criteria_aux = set_model(cfg_a2d2, cfg_city)
    # net = DetailBranch(n_bn=1)
    # detail_state = get_detail_state(torch.load(weight_pth, map_location='cuda:0'))
    # net = SegmentBranch(n_bn=1)
    # segment_state = get_segment_state(torch.load(weight_pth, map_location='cuda:0'))
    # net.eval()
    # net = BGAandHead(38, 'eval', 2, 19)
    # net.cuda()
    net.load_state_dict(torch.load(weight_pth, map_location='cpu'), strict=False)
    # net.load_state_dict(segment_state, strict=True)

    # # ## ddp training
    # # net = set_model_dist(net)

    
    # # net.load_state_dict(torch.load(weight_pth), strict=False)
    
    net.aux_mode = 'eval'
    update_bn_stats(net, dl, len(dl), "tqdm")
    
    save_pth = osp.join(cfg.respth, 'cam_vid_cityhead.pth')
    logger.info('\nsave models to {}'.format(save_pth))
    state = net.state_dict()
    torch.save(state, save_pth)


def main():
    torch.cuda.set_device(args.local_rank)
    # dist.init_process_group(
    #     backend='nccl',
    #     init_method='tcp://127.0.0.1:{}'.format(args.port),
    #     world_size=torch.cuda.device_count(),
    #     rank=args.local_rank
    # )
    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)

    setup_logger(f'{cfg.model_type}-{cfg.dataset.lower()}-train', cfg.respth)
    train_bn()

def test():
    # with torch.no_grad():
    #     test_fd = torch.randn(39, 128, 96, 120).cuda()
    #     test_fs = torch.randn(39, 128, 24, 30).cuda()
    #     net = BGAandHead(38, 'eval', 2, 19)
    #     net.cuda()
    #     net.load_state_dict(torch.load(weight_pth, map_location='cuda:0'), strict=False)
    #     net([[test_fd], [test_fs]])
    dl = get_data_loader(cfg_a2d2, mode='val', distributed=False)
    # dl = get_data_loader(cfg_cam, mode='val', distributed=False)
    for im, lb in dl:
        continue
    
    

if __name__ == "__main__":
    main()
    # test()
