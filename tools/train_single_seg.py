#!/usr/bin/python
# -*- encoding: utf-8 -*-


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
from lib.get_dataloader import get_data_loader, get_single_data_loader
from lib.loss.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg
from lib.loss.loss_cross_datasets import CrossDatasetsLoss, CrossDatasetsCELoss, CrossDatasetsCELoss_KMeans, CrossDatasetsCELoss_CLIP, CrossDatasetsCELoss_GNN, CrossDatasetsCELoss_AdvGNN
from lib.class_remap import ClassRemap

from tools.configer import Configer
from evaluate import eval_model_contrast, eval_model_aux, eval_model, eval_model_contrast_single

from tensorboardX import SummaryWriter

from lib.module.gen_graph_node_feature import gen_graph_node_feature
from tools.get_bipartile import print_bipartite, find_unuse



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
    parse.add_argument('--port', dest='port', type=int, default=16853,)
    parse.add_argument('--finetune_from', type=str, default=None,)
    parse.add_argument('--config', dest='config', type=str, default='configs/ltbgnn_7_datasets_segonly.json',)
    return parse.parse_args()

# 使用绝对路径
args = parse_args()
configer = Configer(configs=args.config)


# cfg_city = set_cfg_from_file(configer.get('dataset1'))
# cfg_cam  = set_cfg_from_file(configer.get('dataset2'))

# ClassRemaper = ClassRemap(configer=configer)


def set_model(configer):
    logger = logging.getLogger()

    net = model_factory[configer.get('model_name')](configer)

    if configer.get('train', 'finetune'):
        logger.info(f"load pretrained weights from {configer.get('train', 'finetune_from')}")
        net.load_state_dict(torch.load(configer.get('train', 'finetune_from'), map_location='cpu'), strict=False)

        
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
            lr=configer.get('lr', 'lr_start'),
            momentum=0.9,
            weight_decay=configer.get('lr', 'weight_decay'),
        )
    elif configer.get('optim') == 'AdamW':
        optim = torch.optim.AdamW(
            params_list,
            lr=configer.get('lr', 'lr_start'),
        )
        
    return optim
    

def set_model_dist(net):
    local_rank = dist.get_rank()
    net = nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank, ],
        find_unused_parameters=False,
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
    writer = SummaryWriter(configer.get('res_save_pth'))
    use_ema = configer.get('use_ema')
    use_contrast = configer.get('contrast', 'use_contrast')
    joint_train = configer.get('train', 'joint_train')
    ## dataset

    # dl = get_single_data_loader(configer, aux_mode='train', distributed=is_dist)
    
    # dl_city, dl_cam = get_data_loader(configer, aux_mode='train', distributed=is_dist)
    
    ## model
    net = set_model(configer=configer)
    
    contrast_losses = set_contrast_loss(configer)

    ## optimizer


    ## mixed precision training
    scaler = amp.GradScaler()

    optim = set_optimizer(net, configer)
    # gnn_optim = set_optimizer(graph_net, configer)
    # gnn_optimD = set_optimizerD(graph_net, configer)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters, loss_contrast_meter, loss_domain_meter, kl_loss_meter = set_meters(configer)
    ## lr scheduler
    # gnn_lr_schdr = WarmupPolyLrScheduler(gnn_optim, power=0.9,
    #     max_iter=configer.get('train','gnn_iters'), warmup_iter=configer.get('lr','warmup_iters'),
    #     warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    # gnn_lr_schdrD = WarmupPolyLrScheduler(gnn_optimD, power=0.9,
    #     max_iter=configer.get('train','gnn_iters'), warmup_iter=configer.get('lr','warmup_iters'),
    #     warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    # lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
    #     max_iter=configer.get('lr','max_iter'), warmup_iter=configer.get('lr','warmup_iters'),
    #     warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    # 两个数据集分别处理
    # 使用迭代器读取数据
    
    # dl_iters = [iter(dl) for dl in dls]
    
    ## train loop
    # for it, (im, lb) in enumerate(dl):
    starti = 0

    use_dataset_aux_head = configer.get('dataset_aux_head', 'use_dataset_aux_head')
    train_aux = use_dataset_aux_head
    aux_iter = 0

    if use_dataset_aux_head:
        aux_iter = configer.get('dataset_aux_head', 'aux_iter')
        net.set_train_dataset_aux(starti < aux_iter)
        criteria_pre = OhemCELoss(0.7)
        criteria_aux = [OhemCELoss(0.7) for _ in range(configer.get('loss', 'aux_num'))]
    
    
    finetune = configer.get('train', 'finetune')
    fix_param_iters = 0
    # if finetune:
    #     fix_param_iters = configer.get('lr', 'fix_param_iters')
    #     net.switch_require_grad_state(False)
    
    ## ddp training
    if is_distributed():
        net = set_model_dist(net)
        # graph_net = set_model_dist(graph_net)
    

    with_aux = configer.get('loss', 'with_aux')

    init_gnn_stage = False


    dls = get_data_loader(configer, aux_mode='train', distributed=is_dist)
                    
    dl_iters = [iter(dl) for dl in dls]
    
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=configer.get('lr','max_iter'), warmup_iter=configer.get('lr','warmup_iters'),
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,) 

    configer.update(['iter'], 0)
    for i in range(starti, configer.get('lr','max_iter') + starti):
        configer.plus_one('iter')



        # try:
        #     im_lb, dataset_lbs = next(dl_iter)
        #     im, lb = im_lb
        #     if not im.size()[0] == (configer.get('dataset1', 'ims_per_gpu') + configer.get('dataset2', 'ims_per_gpu')):
        #         raise StopIteration
        # except StopIteration:
        #     city_iter = iter(dl_iter)
        #     im_lb, dataset_lbs = next(dl_iter)
        #     im, lb = im_lb
        # epoch = i * (configer.get('dataset1', 'ims_per_gpu') + configer.get('dataset2', 'ims_per_gpu')) / 2976


        ims = []
        lbs = []    
        for j in range(0,len(dl_iters)):
            try:
                im, lb = next(dl_iters[j])
                while torch.min(lb) == 255:
                    im, lb = next(dl_iters[j])

                if not im.size()[0] == configer.get('dataset'+str(j+1), 'ims_per_gpu'):
                    raise StopIteration
            except StopIteration:
                dl_iters[j] = iter(dls[j])
                im, lb = next(dl_iters[j])
                while torch.min(lb) == 255:
                    im, lb = next(dl_iters[j])
            
            ims.append(im)
            lbs.append(lb)
                

        im = torch.cat(ims, dim=0)
        lb = torch.cat(lbs, dim=0)

        im = im.cuda()
        lb = lb.cuda()
        

        dataset_lbs = torch.cat([i*torch.ones(this_lb.shape[0], dtype=torch.int) for i,this_lb in enumerate(lbs)], dim=0)
        dataset_lbs = dataset_lbs.cuda()
        # print(dataset_lbs)

        lb = torch.squeeze(lb, 1)


        # net.eval()
        # with torch.no_grad():
        #     seg_out = net(im)

        optim.zero_grad()
        with amp.autocast(enabled=configer.get('use_fp16')):

            ## 修改为多数据集模式
            
            if train_aux:
                train_aux = False
                if is_distributed():
                    net.module.set_train_dataset_aux(False)
                else:
                    net.set_train_dataset_aux(False)    
            
                
                        
            seg_out = {}

            is_adv = False
            
            net.train()
            
    

        
            seg_out = net(im)
            seg_out['unify_prototype'] = None
            
            

            seg_out['bi_graphs'] = [torch.eye(configer.get('dataset1', 'n_cats')).cuda()]
            seg_out['adv_out'] = None
                
            backward_loss, adv_loss = contrast_losses(seg_out, lb, dataset_lbs, is_adv, False)
            # print(backward_loss)
            kl_loss = None
            loss_seg = backward_loss
            loss_aux = None

            

        scaler.scale(backward_loss).backward()
        
        scaler.step(optim)

        
        scaler.update()
        torch.cuda.synchronize()


            

        # print('synchronize')
        time_meter.update()
        loss_meter.update(backward_loss.item())
        if kl_loss:
            kl_loss_meter.update(kl_loss.item())
        
        if is_adv:
            loss_domain_meter.update(adv_loss.item())
        # loss_pre_meter.update(loss_pre.item())
        
        if not use_dataset_aux_head or i > aux_iter:
            loss_pre_meter.update(loss_seg.item()) 
            # if with_domain_adversarial:
            #     loss_domain_meter.update(loss_domain)
            if with_aux:
                _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]
            
        # if i >= configer.get('lr', 'warmup_iters') and use_contrast:
        #     loss_contrast_meter.update(loss_contrast.item())

    
        ## print training log message
        if (i + 1) % 100 == 0:
            writer.add_scalars("loss",{"seg":loss_pre_meter.getWoErase(),"contrast":loss_contrast_meter.getWoErase(), "domain":loss_domain_meter.getWoErase()},configer.get("iter")+1)
            lr = lr_schdr.get_lr()

            lr = sum(lr) / len(lr)
            print_log_msg(
                i, 0, 0, configer.get('lr', 'max_iter')+starti, lr, time_meter, loss_meter,
                loss_pre_meter, loss_aux_meters, loss_contrast_meter, loss_domain_meter, kl_loss_meter)
            

        if (i + 1) % 10000 == 0:
            seg_save_pth = osp.join(configer.get('res_save_pth'), 'seg_{}_model{}_.pth'.format(configer.get('dataset1', 'data_reader'), i))
            logger.info('\nsave seg_models to {}'.format(seg_save_pth))
            if is_distributed():
                seg_state = net.module.state_dict()
                if dist.get_rank() == 0: 
                    torch.save(seg_state, seg_save_pth)
            else:
                seg_state = net.state_dict()
                torch.save(seg_state, seg_save_pth)

            # if fix_graph == False:
            # with torch.no_grad():
            #     # input_feats = torch.cat([graph_node_features, graph_net.unify_node_features], dim=0)
            #     if is_distributed():
            #         unify_prototype, bi_graphs = graph_net.module.get_optimal_matching(graph_node_features)
            #     else:
            #         unify_prototype, bi_graphs = graph_net.get_optimal_matching(graph_node_features) 
                
            eval_model_func = eval_model_contrast
            optim.zero_grad()
            if is_distributed():

                torch.cuda.empty_cache()
                heads, mious = eval_model_func(configer, net.module)
            else:
                torch.cuda.empty_cache()
                heads, mious = eval_model_func(configer, net)
                
            
            # writer.export_scalars_to_json(osp.join(configer.get('res_save_pth'), str(time())+'_writer.json'))
            logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
            net.train()
        
                
        lr_schdr.step()

        

    ## dump the final model and evaluate the result
    seg_save_pth = osp.join(configer.get('res_save_pth'), 'seg_model_final.pth')
    gnn_save_pth = osp.join(configer.get('res_save_pth'), 'gnn_model_final.pth')
    logger.info('\nsave seg_models to {}, gnn_models to {}'.format(seg_save_pth, gnn_save_pth))
    
    writer.close()
    if is_distributed():
        # gnn_state = graph_net.module.state_dict()
        seg_state = net.module.state_dict()
        if dist.get_rank() == 0: 
            # torch.save(gnn_state, gnn_save_pth)
            torch.save(seg_state, seg_save_pth)
    else:
        # gnn_state = graph_net.state_dict()
        seg_state = net.state_dict()
        # torch.save(gnn_state, gnn_save_pth)
        torch.save(seg_state, seg_save_pth)

    # logger.info('\nevaluating the final model')
    torch.cuda.empty_cache()
    # if is_distributed():
    #     heads, mious = eval_model_func(configer, net.module)
    # else:
    #     heads, mious = eval_model_func(configer, net)
    # logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))

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
