#!/usr/bin/python
# -*- encoding: utf-8 -*-

from asyncio.windows_events import NULL
import sys
sys.path.insert(0, '.')
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
from evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg


## fix all random seeds
#  torch.manual_seed(123)
#  torch.cuda.manual_seed(123)
#  np.random.seed(123)
#  random.seed(123)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')




def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
    parse.add_argument('--port', dest='port', type=int, default=44554,)
    parse.add_argument('--config', dest='config', type=str,
            default='configs/bisenetv2.py',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    return parse.parse_args()

args = parse_args()
cfg = set_cfg_from_file(args.config)
cfg_a2d2 = set_cfg_from_file('configs/bisenetv2_a2d2.py')
cfg_city = set_cfg_from_file('configs/bisenetv2_city.py')


def set_model(config_file, *config_files):
    logger = logging.getLogger()
    net = NULL
    if config_files is NULL:
        net = model_factory[config_file.model_type](config_file.n_cats)
    else:
        n_classes = [cfg.n_cats for cfg in config_files]
        net = model_factory[config_file.model_type](config_file.n_cats, 'train', 2, *n_classes)

    if not args.finetune_from is None:
        logger.info(f'load pretrained weights from {args.finetune_from}')
        net.load_state_dict(torch.load(args.finetune_from, map_location='cpu'))
    if config_file.use_sync_bn: net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
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
        #  find_unused_parameters=True,
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



def train():
    n_dataset = 2
    logger = logging.getLogger()
    is_dist = dist.is_initialized()

    ## dataset
    # dl = get_data_loader(cfg, mode='train', distributed=is_dist)
    dl_a2d2 = get_data_loader(cfg_a2d2, mode='train', distributed=is_dist)
    dl_city = get_data_loader(cfg_city, mode='train', distributed=is_dist)

    ## model
    net, criteria_pre, criteria_aux = set_model(cfg_a2d2, cfg_city)

    ## optimizer
    optim = set_optimizer(net, cfg)

    ## mixed precision training
    scaler = amp.GradScaler()

    ## ddp training
    net = set_model_dist(net)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters(cfg)

    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    # 两个数据集分别处理
    a2d2_list = enumerate(dl_a2d2)
    a2d2_len = len(a2d2_list)
    city_list = enumerate(dl_city)
    city_len = len(city_list)

    ## train loop
    # for it, (im, lb) in enumerate(dl):
    for i in range(0, cfg.max_iter):
        _, (im_a2d2, lb_a2d2) = a2d2_list[i % a2d2_len]
        _, (im_city, lb_city) = city_list[i % city_len]

        im_a2d2 = im_a2d2.cuda()
        lb_a2d2 = lb_a2d2.cuda()

        im_city = im_city.cuda()
        lb_city = lb_city.cuda()

        lb_a2d2 = torch.squeeze(lb_a2d2, 1)
        lb_city = torch.squeeze(lb_city, 1)

        im = [im_a2d2, im_city]
        lb = [lb_a2d2, lb_city]

        optim.zero_grad()
        with amp.autocast(enabled=cfg.use_fp16):
            ## 修改为多数据集模式
            logits, *logits_aux = net(*im)
            loss_pre = [criteria_pre(logits[i], lb[i]) for i in range(0, n_dataset)]
            loss_aux = [[crit(lgt[i], lb[i]) for crit, lgt in zip(criteria_aux, logits_aux)] for i in range(0, n_dataset)]
            loss_all = [loss_pre[i] + sum(loss_aux[i]) for i in range(0, n_dataset)]
            # 不同数据集loss的权重
            w1 = 1
            w2 = 1
            loss = w1 * loss_all[0] + w2 * loss_all[1]

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        torch.cuda.synchronize()

        time_meter.update()
        loss_meter.update(loss.item())
        # loss_pre_meter.update(loss_pre.item())
        _ = [loss_pre_meter.update(loss_pre[i].item()) for i in range(0, n_dataset)]
        _ = [[mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux[i])] for i in range(0, n_dataset)]

        ## print training log message
        if (i + 1) % 100 == 0:
            lr = lr_schdr.get_lr()
            lr = sum(lr) / len(lr)
            print_log_msg(
                i, cfg.max_iter, lr, time_meter, loss_meter,
                loss_pre_meter, loss_aux_meters)
        lr_schdr.step()

    ## dump the final model and evaluate the result
    save_pth = osp.join(cfg.respth, 'model_final.pth')
    logger.info('\nsave models to {}'.format(save_pth))
    state = net.module.state_dict()
    if dist.get_rank() == 0: torch.save(state, save_pth)

    logger.info('\nevaluating the final model')
    torch.cuda.empty_cache()
    heads, mious = eval_model(cfg, net.module)
    logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))

    return


def main():
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:{}'.format(args.port),
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )
    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger(f'{cfg.model_type}-{cfg.dataset.lower()}-train', cfg.respth)
    train()


if __name__ == "__main__":
    main()
