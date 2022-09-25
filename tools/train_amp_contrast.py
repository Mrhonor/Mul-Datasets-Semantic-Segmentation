#!/usr/bin/python
# -*- encoding: utf-8 -*-

# from asyncio.windows_events import NULL # 有bug，去掉改为别的判定

from cgi import print_directory
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
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg
from lib.loss_cross_datasets import CrossDatasetsLoss
from tools.configer import Configer
from lib.class_remap import ClassRemap
from evaluate import eval_model_contrast

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
    parse.add_argument('--port', dest='port', type=int, default=16746,)
    parse.add_argument('--finetune_from', type=str, default=None,)
    parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2_city_cam.json',)
    return parse.parse_args()

# 使用绝对路径
args = parse_args()
configer = Configer(configs=args.config)


# cfg_city = set_cfg_from_file(configer.get('dataset1'))
# cfg_cam  = set_cfg_from_file(configer.get('dataset2'))

CITY_ID = 0
CAM_ID = 1


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

def set_model(configer):
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

    net = model_factory[configer.get('model_name')](configer)

    if configer.get('train', 'finetune'):
        logger.info(f"load pretrained weights from {configer.get('train', 'finetune_from')}")
        net.load_state_dict(torch.load(configer.get('train', 'finetune_from'), map_location='cpu'))
    if configer.get('use_sync_bn'): 
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.train()
    return net

def set_optimizer(model, configer):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        #  wd_val = cfg.weight_decay
        wd_val = 0
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': configer.get('lr', 'lr_start')},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': configer.get('lr', 'lr_start')},
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
        lr=configer.get('lr', 'lr_start'),
        momentum=0.9,
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
    return [CrossDatasetsLoss(configer=configer) for _ in range(0,configer.get('n_datasets'))]

def set_meters(configer):
    time_meter = TimeMeter(configer.get('lr', 'max_iter'))
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(configer.get('loss', 'aux_num'))]
    loss_contrast_meter = AvgMeter('loss_contrast')
            
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters, loss_contrast_meter

def dequeue_and_enqueue(configer, keys, labels,
                            segment_queue, iter, dataset_id):
    ## 更新memory bank
    
    batch_size = keys.shape[0]
    feat_dim = keys.shape[1]
    network_stride = configer.get('network', 'stride')
    coefficient = configer.get('contrast', 'coefficient')
    warmup_iter = configer.get('lr', 'warmup_iters')
    ignore_index = configer.get('loss', 'ignore_index')
    if iter < warmup_iter:
        coefficient = coefficient * iter / warmup_iter
    
    classRemapper = ClassRemap(configer=configer)
    
    labels = labels[:, ::network_stride, ::network_stride]

    emb = keys.permute(1, 0, 2, 3)
    # lbs = labels.permute(1, 0, 2, 3)
    this_feat = emb.contiguous().view(feat_dim, -1)
    this_label = labels.contiguous().view(-1)
    this_label_ids = torch.unique(this_label)
    this_label_ids = [x for x in this_label_ids if x >= 0 and x != ignore_index]

    for lb_id in this_label_ids:
        lb = lb_id.item()

        idxs = (this_label == lb).nonzero()
        # segment enqueue and dequeue
        feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)

        if len(classRemapper.getAnyClassRemap(lb, dataset_id)) > 1:
            remap_lb = lb
        else:
            remap_lb = classRemapper.getAnyClassRemap(lb, dataset_id)[0]

        # print(nn.functional.normalize(feat.view(-1), p=2, dim=0))
        # print(segment_queue[lb])
            
        segment_queue[remap_lb] = coefficient * segment_queue[remap_lb] + (1 - coefficient) * nn.functional.normalize(feat, p=2, dim=0)

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
    n_dataset = configer.get('n_datasets')
    logger = logging.getLogger()
    is_dist = dist.is_initialized()
    ## dataset
    # dl = get_data_loader(cfg, mode='train', distributed=is_dist)
    
    dl_city, dl_cam = get_data_loader(configer, aux_mode='train', distributed=is_dist)
    # dl_cam = get_data_loader(configer, distributed=is_dist)
    ## model
    net = set_model(configer=configer)
    contrast_losses = set_contrast_loss(configer)

    ## optimizer
    optim = set_optimizer(net, configer)

    ## mixed precision training
    scaler = amp.GradScaler()

    ## ddp training
    if is_distributed():
        net = set_model_dist(net)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters, loss_contrast_meter = set_meters(configer)
    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=configer.get('lr','max_iter'), warmup_iter=configer.get('lr','warmup_iters'),
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)
    # 两个数据集分别处理
    # 使用迭代器读取数据
    city_iter = iter(dl_city)
    cam_iter = iter(dl_cam)
    ## train loop
    # for it, (im, lb) in enumerate(dl):
    starti = 0

    contrast_warmup_iters = configer.get("lr", "warmup_iters")
    with_memory = configer.exists('contrast', 'with_memory')
    with_aux = configer.get('loss', 'with_aux')

    for i in range(starti, configer.get('lr','max_iter') + starti):
        try:
            im_cam, lb_cam = next(cam_iter)
            if not im_cam.size()[0] == configer.get('dataset2', 'ims_per_gpu'):
                raise StopIteration
        except StopIteration:
            cam_iter = iter(dl_cam)
            im_cam, lb_cam = next(cam_iter)
        cam_epoch = i * configer.get('dataset2', 'ims_per_gpu') / 469
        try:
            im_city, lb_city = next(city_iter)
            if not im_city.size()[0] == configer.get('dataset1', 'ims_per_gpu'):
                raise StopIteration
        except StopIteration:
            city_iter = iter(dl_city)
            im_city, lb_city = next(city_iter)
        city_epoch = i * configer.get('dataset2', 'ims_per_gpu') / 2976

        im_city = im_city.cuda()
        lb_city = lb_city.cuda()

        im_cam = im_cam.cuda()
        lb_cam = lb_cam.cuda()

        lb_city = torch.squeeze(lb_city, 1)
        
        lb_cam = torch.squeeze(lb_cam, 1)

        
        im = torch.cat((im_city, im_cam), 0)
        lb = torch.cat((lb_city, lb_cam), 0)
        perm_index = torch.randperm(im.shape[0])
        im = im[perm_index]
        lb = lb[perm_index]

        optim.zero_grad()
        with amp.autocast(enabled=configer.get('use_fp16')):
            ## 修改为多数据集模式
            out = net(im)
            # logits, *logits_aux = out['seg']
            # emb = out['embed']
            
            with_embed = True if i >= contrast_warmup_iters else False

            cam_size = im_cam.shape[0]
            city_size = im_city.shape[0]
            
            city_out = {}
            cam_out = {}
            
            for k, v in out.items():
                if k == 'seg':
                    city_logits_list = []
                    cam_logits_list = []
                    for logit in v:
                        city_logits_list.append(logit[0][perm_index.sort()[1]][:city_size]) 
                        cam_logits_list.append(logit[0][perm_index.sort()[1]][city_size:city_size+cam_size])
                        
                    city_out[k] = city_logits_list
                    cam_out[k] = cam_logits_list
                else:
                    emb = v[0]
                    city_out[k] = emb[perm_index.sort()[1]][:city_size] 
                    cam_out[k] = emb[perm_index.sort()[1]][city_size:city_size+cam_size]
            
            if is_distributed():
                city_out['segment_queue'] = net.module.segment_queue
                cam_out['segment_queue'] = net.module.segment_queue
            else:
                city_out['segment_queue'] = net.segment_queue
                cam_out['segment_queue'] = net.segment_queue
            
            if i < configer.get('lr', 'warmup_iters') or not configer.get('contrast', 'use_contrast'):
                backward_loss0, loss_seg0, loss_aux0 = contrast_losses[CITY_ID](city_out, lb_city, CITY_ID, True)
                backward_loss1, loss_seg1, loss_aux1 = contrast_losses[CAM_ID](cam_out, lb_cam, CAM_ID, True)
                
                
                backward_loss =  backward_loss0 + backward_loss1
                loss_seg =  loss_seg0 + loss_seg1
                if with_aux:
                    loss_aux = [aux0 + aux1 for aux0, aux1 in zip(loss_aux0, loss_aux1)]
                # print(backward_loss0)
                # print(backward_loss1)
                # print(loss_seg0)
                # print(loss_seg1)
                # print(loss_aux0)
                # print(loss_aux1)
                # print("***************************")
            else:
                backward_loss0, loss_seg0, loss_aux0, loss_contrast0 = contrast_losses[CITY_ID](city_out, lb_city, CITY_ID, False)
                backward_loss1, loss_seg1, loss_aux1, loss_contrast1 = contrast_losses[CAM_ID](cam_out, lb_cam, CAM_ID, False)
                
                
                backward_loss =  backward_loss0 + backward_loss1
                loss_seg =  loss_seg0 + loss_seg1
                if with_aux:
                    loss_aux = [aux0 + aux1 for aux0, aux1 in zip(loss_aux0, loss_aux1)]
                    
                loss_contrast = loss_contrast0 + loss_contrast1
                
            
            # display_loss = reduce_tensor(backward_loss) / get_world_size()

        if with_memory and 'key' in out:
            # dequeue_and_enqueue(configer, out['key'], out['lb_key'],
            #                     segment_queue=net.module.segment_queue,
            #                     segment_queue_ptr=net.module.segment_queue_ptr,
            #                     pixel_queue=net.module.pixel_queue,
            #                     pixel_queue_ptr=net.module.pixel_queue_ptr)

            dequeue_and_enqueue(configer, city_out['key'], lb_city.detach(),
                                city_out['segment_queue'], i, CITY_ID)

            dequeue_and_enqueue(configer, cam_out['key'], lb_cam.detach(),
                                cam_out['segment_queue'], i, CAM_ID)


        scaler.scale(backward_loss).backward()

        # self.configer.plus_one('iters')

        # scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        torch.cuda.synchronize()

        time_meter.update()
        loss_meter.update(backward_loss.item())
        # loss_pre_meter.update(loss_pre.item())
        loss_pre_meter.update(loss_seg.item()) 
        if with_aux:
            _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]
            
        if i >= configer.get('lr', 'warmup_iters') and configer.get('contrast', 'use_contrast'):
            loss_contrast_meter.update(loss_contrast.item())

        ## print training log message
        if (i + 1) % 100 == 0:
            lr = lr_schdr.get_lr()
            lr = sum(lr) / len(lr)
            print_log_msg(
                i, cam_epoch, city_epoch, configer.get('lr', 'max_iter')+starti, lr, time_meter, loss_meter,
                loss_pre_meter, loss_aux_meters, loss_contrast_meter)

        if (i + 1) % 1000 == 0:
            save_pth = osp.join(configer.get('res_save_pth'), 'nll_model_{}.pth'.format(i+1))
            logger.info('\nsave models to {}'.format(save_pth))

            if is_distributed():
                heads, mious = eval_model_contrast(configer, net.module)
            else:
                heads, mious = eval_model_contrast(configer, net)
            logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
            net.train()
            
            if is_distributed():
                state = net.module.state_dict()
            else:
                state = net.state_dict()
            torch.save(state, save_pth)

        lr_schdr.step()

    ## dump the final model and evaluate the result
    save_pth = osp.join(configer.get('res_save_pth'), 'upsample_model_final.pth')
    logger.info('\nsave models to {}'.format(save_pth))
    
    if is_distributed():
        state = net.module.state_dict()
    else:
        state = net.state_dict()

    if dist.get_rank() == 0: torch.save(state, save_pth)

    logger.info('\nevaluating the final model')
    torch.cuda.empty_cache()
    if is_distributed():
        heads, mious = eval_model_contrast(configer, net.module)
    else:
        heads, mious = eval_model_contrast(configer, net)
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
    if not osp.exists(configer.get('res_save_pth')): os.makedirs(configer.get('res_save_pth'))

    setup_logger(f"{configer.get('model_name')}-train", configer.get('res_save_pth'))
    train()


if __name__ == "__main__":
    main()
