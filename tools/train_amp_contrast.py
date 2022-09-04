#!/usr/bin/python
# -*- encoding: utf-8 -*-

# from asyncio.windows_events import NULL # 有bug，去掉改为别的判定
import imp
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
from lib.loss_cross_datasets import CrossDatasetsLoss
from tools.configer import Configer
from lib.class_remap import ClassRemap


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
    parse.add_argument('--port', dest='port', type=int, default=12345,)
    parse.add_argument('--config', dest='config', type=str,
            default='/root/autodl-tmp/project/BiSeNet/configs/bisenetv2.py',)
    parse.add_argument('--finetune_from', type=str, default=None,)
    return parse.parse_args()

# 使用绝对路径
args = parse_args()
cfg = set_cfg_from_file(args.config)
cfg_a2d2 = set_cfg_from_file('/root/autodl-tmp/project/BiSeNet/configs/bisenetv2_a2d2.py')
cfg_city = set_cfg_from_file('/root/autodl-tmp/project/BiSeNet/configs/bisenetv2_city.py')

configer = Configer(args_parser=args)
ClassRemaper = ClassRemap(configer=configer)


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
        find_unused_parameters=True,
        output_device=local_rank
        )
    return net

def set_contrast_loss(configer):
    return CrossDatasetsLoss(configer=configer)

def set_meters(config_file):
    time_meter = TimeMeter(config_file.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(config_file.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters

def dequeue_and_enqueue(configer, keys, labels,
                            segment_queue, segment_queue_ptr,
                            pixel_queue=None, pixel_queue_ptr=None):
    ## 更新memory bank
    
    batch_size = keys.shape[0]
    feat_dim = keys.shape[1]
    network_stride = configer.get('network', 'stride')
    memory_size = configer.get('contrast', 'memory_size')
    # pixel_update_freq = configer.get('contrast', 'pixel_update_freq')

    labels = labels[:, ::network_stride, ::network_stride]

    for bs in range(batch_size):
        this_feat = keys[bs].contiguous().view(feat_dim, -1)
        this_label = labels[bs].contiguous().view(-1)
        this_label_ids = torch.unique(this_label)
        this_label_ids = [x for x in this_label_ids if x > 0]

        for lb in this_label_ids:
            idxs = (this_label == lb).nonzero()

            # segment enqueue and dequeue
            feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
            ptr = int(segment_queue_ptr[lb])
            segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
            segment_queue_ptr[lb] = (segment_queue_ptr[lb] + 1) % memory_size

            # # pixel enqueue and dequeue
            # num_pixel = idxs.shape[0]
            # perm = torch.randperm(num_pixel)
            # K = min(num_pixel, pixel_update_freq)
            # feat = this_feat[:, perm[:K]]
            # feat = torch.transpose(feat, 0, 1)
            # ptr = int(pixel_queue_ptr[lb])

            # if ptr + K >= memory_size:
            #     pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
            #     pixel_queue_ptr[lb] = 0
            # else:
            #     pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
            #     pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + 1) % memory_size

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
    n_dataset = 2
    logger = logging.getLogger()
    is_dist = dist.is_initialized()
    ## dataset
    # dl = get_data_loader(cfg, mode='train', distributed=is_dist)
    dl_a2d2 = get_data_loader(cfg_a2d2, mode='train', distributed=is_dist)
    dl_city = get_data_loader(cfg_city, mode='train', distributed=is_dist)
    ## model
    net, criteria_pre, criteria_aux = set_model(cfg_a2d2, cfg_city)
    pixel_loss = set_contrast_loss(configer)

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
    # 使用迭代器读取数据
    city_iter = iter(dl_city)
    a2d2_iter = iter(dl_a2d2)
    ## train loop
    # for it, (im, lb) in enumerate(dl):
    starti = 20000

    contrast_warmup_iters = configer.get("contrast", "warmup_iters")
    with_memory = configer.exists('contrast', 'with_memory')

    for i in range(starti, cfg.max_iter + starti):
        try:
            im_a2d2, lb_a2d2 = next(a2d2_iter)
            if not im_a2d2.size()[0] == cfg_a2d2.ims_per_gpu:
                raise StopIteration
        except StopIteration:
            a2d2_iter = iter(dl_a2d2)
            im_a2d2, lb_a2d2 = next(a2d2_iter)
        a2d2_epoch = i * cfg_a2d2.ims_per_gpu / 37150
        try:
            im_city, lb_city = next(city_iter)
            if not im_city.size()[0] == cfg_city.ims_per_gpu:
                raise StopIteration
        except StopIteration:
            city_iter = iter(dl_city)
            im_city, lb_city = next(city_iter)
        city_epoch = i * cfg_city.ims_per_gpu / 2976

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
            out = net(*im)
            logits, *logits_aux = out['seg']
            emb = out['embed']
            # logits, *logits_aux = net(*im)
            # loss_pre = [criteria_pre(logits[i], lb[i]) for i in range(0, n_dataset)]  # logits[i]仍是一个len为1的元组
            loss_pre = [criteria_pre(logits[i], lb[i]) for i in range(0, n_dataset)]
            loss_aux = [[crit(lgt[i], lb[i]) for crit, lgt in zip(criteria_aux, logits_aux)] for i in range(0, n_dataset)]
            loss_all = [loss_pre[i] + sum(loss_aux[i]) for i in range(0, n_dataset)]
            # 不同数据集loss的权重
            w1 = 1
            w2 = 1
            loss = w1 * loss_all[0] + w2 * loss_all[1]
            
            with_embed = True if i >= contrast_warmup_iters else False
            if is_distributed():
                # out['pixel_queue'] = net.pixel_queue
                # out['pixel_queue_ptr'] = net.pixel_queue_ptr
                out['segment_queue'] = net.segment_queue
                out['segment_queue_ptr'] = net.segment_queue_ptr
            else:
                # out['pixel_queue'] = net.module.pixel_queue
                # out['pixel_queue_ptr'] = net.module.pixel_queue_ptr
                out['segment_queue'] = net.module.segment_queue
                out['segment_queue_ptr'] = net.module.segment_queue_ptr
            
            backward_loss = pixel_loss(out, lb, with_embed=with_embed)
            display_loss = reduce_tensor(backward_loss) / get_world_size()

        if with_memory and 'key' in out and 'lb_key' in out:
            # dequeue_and_enqueue(configer, out['key'], out['lb_key'],
            #                     segment_queue=net.module.segment_queue,
            #                     segment_queue_ptr=net.module.segment_queue_ptr,
            #                     pixel_queue=net.module.pixel_queue,
            #                     pixel_queue_ptr=net.module.pixel_queue_ptr)
            dequeue_and_enqueue(configer, out['key'], out['lb_key'],
                                segment_queue=out['segment_queue'],
                                segment_queue_ptr=out['segment_queue_ptr'],
                                )


        scaler.scale(backward_loss).backward()

        # self.configer.plus_one('iters')

        # scaler.scale(loss).backward()
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
                i, a2d2_epoch, city_epoch, cfg.max_iter+starti, lr, time_meter, loss_meter,
                loss_pre_meter, loss_aux_meters)

        if (i + 1) % 1000 == 0:
            save_pth = osp.join(cfg.respth, 'model_{}.pth'.format(i+1))
            logger.info('\nsave models to {}'.format(save_pth))
            state = net.module.state_dict()
            if dist.get_rank() == 0: torch.save(state, save_pth)

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
