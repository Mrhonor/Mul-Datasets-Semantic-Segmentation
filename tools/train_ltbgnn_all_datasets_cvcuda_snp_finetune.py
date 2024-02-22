#!/usr/bin/python
# -*- encoding: utf-8 -*-

import pycuda.driver as cuda
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
import cvcuda
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.cuda.amp as amp

from lib.models import model_factory
from lib.get_dataloader import get_data_loader, get_single_data_loader
from lib.cvCudaDataLoader import getDataLoaderCVCUDA
from lib.loss.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg
from lib.loss.loss_cross_datasets import CrossDatasetsLoss, CrossDatasetsCELoss, CrossDatasetsCELoss_KMeans, CrossDatasetsCELoss_CLIP, CrossDatasetsCELoss_GNN, CrossDatasetsCELoss_AdvGNN, CrossDatasetsCELoss_AdvGNN_ce 
from lib.class_remap import ClassRemap

from tools.configer import Configer
from evaluate import eval_model_contrast, eval_model_aux, eval_model, eval_model_contrast_single, eval_model_mulbn, eval_model_dsg, eval_model_unlabel, eval_find_use_and_unuse_label, eval_model_cvcuda, eval_find_use_and_unuse_label_CVCUDA

from tensorboardX import SummaryWriter

from lib.module.gen_graph_node_feature import gen_graph_node_feature
from tools.get_bipartile import print_bipartite, find_unuse
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
    parse.add_argument('--config', dest='config', type=str, default='configs/ltbgnn_7_datasets_snp_train_tg.json',)
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
        state = torch.load(configer.get('train', 'finetune_from'), map_location='cpu')
        
        if 'unify_prototype' in state:
           del state['unify_prototype']
           for i in range(0, configer.get('n_datasets')):
               del state[f'bipartite_graphs.{i}']
           for k in ['logits.norm.weight', 'logits.norm.bias', 'logits.norm.running_mean', 'logits.norm.running_var', 'logits.norm.num_batches_tracked', 'logits.conv.weight', 'logits.conv.bias']:
               del state[k] 
            
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
            net.load_state_dict(state['model_state_dict'], strict=False)
        else:
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
    # elif hasattr(model, 'get_optim_params'):
    #     params_list = model.get_optim_params()
    #     print(len(params_list))
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.requires_grad == False:
                continue
            
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
            else:
                non_wd_params.append(param)
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
    
def set_optimizer_two(model, seg_model, configer, lr, seg_lr):
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
            {'params': wd_params, 'lr': seg_lr},
            {'params': non_wd_params, 'weight_decay': 0, 'lr': seg_lr},
        ]
    
    # params_list2 = seg_model.get_optim_params()
    wd_params, non_wd_params = [], []
    for name, param in seg_model.named_parameters():
        if param.requires_grad == False:
            continue
        
        if param.dim() == 1:
            non_wd_params.append(param)
        elif param.dim() == 2 or param.dim() == 4:
            wd_params.append(param)
    params_list2 = [
        {'params': wd_params, },
        {'params': non_wd_params, 'weight_decay': 0},
    ]

    params_list.extend(params_list2)
    
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
        'Adv_GNN': CrossDatasetsCELoss_AdvGNN,
        'CE_GNN': CrossDatasetsCELoss_AdvGNN_ce
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
    dls = None
    # torch.autograd.set_detect_anomaly(True)
    device_id = torch.cuda.current_device()
    cuda_device = cuda.Device(device_id)
    cuda_ctx = cuda_device.retain_primary_context()
    # cuda_ctx = cuda_device.make_context()

    # cuda_ctx.pop()
    cvcuda_stream = cvcuda.Stream()
    torch_stream = torch.cuda.ExternalStream(cvcuda_stream.handle)
    
    n_datasets = configer.get('n_datasets')
    logger = logging.getLogger()
    
    # writer = SummaryWriter(configer.get('res_save_pth'))
    use_ema = configer.get('use_ema')
    use_contrast = configer.get('contrast', 'use_contrast')
    joint_train = configer.get('train', 'joint_train')
    mse_or_adv = configer.get('GNN', 'mse_or_adv')
    ## dataset

    # dl = get_single_data_loader(configer, aux_mode='train', distributed=is_dist)
    # dls = get_data_loader(configer, aux_mode='train', distributed=is_dist, stage=1)
    # dl_city, dl_cam = get_data_loader(configer, aux_mode='train', distributed=is_dist)
    cuda_ctx.push()
    with cvcuda_stream, torch.cuda.stream(torch_stream):
        is_dist = dist.is_initialized()
        ## model
        net = set_model(configer=configer)
        graph_net = set_graph_model(configer=configer)
        
        if use_ema:
            ema_net = set_ema_model(configer=configer)
            
        contrast_losses = set_contrast_loss(configer)
        net.unify_prototype.requires_grad = True
        
        ## optimizer
        optim = set_optimizer(net, configer, configer.get('lr', 'seg_lr_start'))


        ## mixed precision training
        scaler = amp.GradScaler()

        ori_graph_node_features = gen_graph_node_feature(configer)
        graph_node_features = ori_graph_node_features.cuda()

        ## meters
        time_meter, loss_meter, loss_pre_meter, loss_aux_meters, loss_contrast_meter, loss_domain_meter, kl_loss_meter = set_meters(configer)
        ## lr scheduler
        # gnn_lr_schdr = WarmupPolyLrScheduler(gnn_optim, power=1.2,
        #     max_iter=configer.get('train','gnn_iters'), warmup_iter=configer.get('lr','warmup_iters'),
        #     warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

        # gnn_lr_schdrD = WarmupPolyLrScheduler(gnn_optimD, power=1.2,
        #     max_iter=configer.get('train','gnn_iters'), warmup_iter=configer.get('lr','warmup_iters'),
        #     warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

        lr_schdr = WarmupPolyLrScheduler(optim, power=1.2,
            max_iter=configer.get('lr','init_iter'), warmup_iter=configer.get('lr','warmup_iters'),
            warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

        if configer.get('train', 'finetune'):
            state = torch.load(configer.get('train', 'finetune_from'))
            # net.load_state_dict(torch.load(configer.get('train', 'finetune_from'), map_location='cpu'), strict=False)
            if 'optimizer_state_dict' in state.keys():
                optim.load_state_dict(state['optimizer_state_dict'])
            if 'scheduler_state_dict' in state.keys():
                lr_schdr.load_state_dict(state['scheduler_state_dict'])

        # 两个数据集分别处理
        # 使用迭代器读取数据

        unify_prototype = None
        
        ## train loop
        # for it, (im, lb) in enumerate(dl):
        starti = 0

        use_dataset_aux_head = configer.get('dataset_aux_head', 'use_dataset_aux_head')
        train_aux = use_dataset_aux_head
        aux_iter = 0

        ## ddp training
        if is_distributed():
            net = set_model_dist(net)
            graph_net = set_model_dist(graph_net)

        bi_graphs = [torch.eye(19).cuda()]
        # bi_graphs = None
        net.set_bipartite_graphs(bi_graphs)
    
        ft_start_i = 0
        if configer.get('lr','init_iter') > 0:
            dls = getDataLoaderCVCUDA(configer, device_id, cuda_ctx)
        for i in range(ft_start_i, configer.get('lr','init_iter')):
            configer.plus_one('iter')
            im, lb, ids = dls()
            im = torch.as_tensor(
                im.cuda(), device="cuda:%d" % device_id
            )
            # lb = torch.as_tensor(
            #     lb.cuda(), device="cuda:%d" % device_id
            # ).squeeze(3).to(torch.long)
            lb = lb.squeeze(3).to(torch.long)
            
                
            # ids = []
            # for num in range(n_datasets):
            #     for _ in range(configer.get('dataset'+str(num+1), 'ims_per_gpu')):
            #         ids.append(num)
                
            dataset_lbs = torch.tensor(ids)
            dataset_lbs = dataset_lbs.cuda()
            # print(dataset_lbs)

            lb = torch.squeeze(lb, 1)

            optim.zero_grad()
            with amp.autocast(enabled=configer.get('use_fp16')):
                ## 修改为多数据集模式
                # net.aux_mode = 'train_head'
                seg_out = {}
                is_adv = False
                adv_out = None
                net.train()

                seg_out = net(im, dataset_lbs)
                seg_out['unify_prototype'] = None
                
                seg_out['bi_graphs'] = net.bipartite_graphs
                seg_out['adv_out'] = adv_out
                    
                backward_loss, adv_loss, _, _ = contrast_losses(seg_out, lb, dataset_lbs, is_adv, False)
                # print(backward_loss)
                kl_loss = None
                loss_seg = backward_loss
                loss_aux = None

            scaler.scale(backward_loss).backward()
            # print(backward_loss)
            scaler.step(optim)
            
            scaler.update()
            torch.cuda.synchronize()       


            # print('synchronize')
            time_meter.update()
            loss_meter.update(backward_loss.item())
            lr_schdr.step()
                
            ## print training log message
            if (i + 1) % 100 == 0:
                # writer.add_scalars("loss",{"seg":loss_pre_meter.getWoErase(),"contrast":loss_contrast_meter.getWoErase(), "domain":loss_domain_meter.getWoErase()},configer.get("iter")+1)
                lr = lr_schdr.get_lr()
                lr = sum(lr) / len(lr)
                print_log_msg(
                    i, 0, 0, configer.get('lr', 'max_iter')+starti, lr, time_meter, loss_meter,
                    loss_pre_meter, loss_aux_meters, loss_contrast_meter, loss_domain_meter, kl_loss_meter)
                

            if (i + 1) % 10000 == 0:
                seg_save_pth = osp.join(configer.get('res_save_pth'), 'clip_model_{}.pth'.format(i+1))
                logger.info('\nsave seg_models to {}'.format(seg_save_pth))

                if is_distributed():
                    seg_state = net.module.state_dict()
                    torch.save(seg_state, seg_save_pth)

                else:
                    seg_state = net.state_dict()
                    torch.save(seg_state, seg_save_pth)
                    
                torch.save({
                    'model_state_dict': seg_state,
                    'optimizer_state_dict': optim.state_dict(),
                    'scheduler_state_dict': lr_schdr.state_dict(),
                }, seg_save_pth)

                if use_dataset_aux_head and i < aux_iter:
                    eval_model_func = eval_model_aux
                else:
                    # eval_model = eval_model_contrast
                    eval_model_func = eval_model_contrast
                    # eval_model_func = eval_model_mulbn

                optim.zero_grad()
                if is_distributed():

                    # net.module.set_bipartite_graphs(bi_graphs)
                    torch.cuda.empty_cache()
                    heads, mious = eval_model_func(configer, net.module)
                else:

                    # net.set_bipartite_graphs(bi_graphs)
                    torch.cuda.empty_cache()
                    heads, mious = eval_model_func(configer, net)
                    
                # writer.add_scalars("mious",{"Cityscapes":mious[CITY_ID],"Camvid":mious[CAM_ID]},configer.get("iter")+1)
                # writer.export_scalars_to_json(osp.join(configer.get('res_save_pth'), str(time())+'_writer.json'))
                logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))

                # if is_distributed():

                #     # net.module.set_bipartite_graphs(bi_graphs)
                #     torch.cuda.empty_cache()
                #     heads, mious = eval_model_func(configer, net.module, mode='train')
                # else:

                #     # net.set_bipartite_graphs(bi_graphs)
                #     torch.cuda.empty_cache()
                #     heads, mious = eval_model_func(configer, net, mode='train')
                    
                # # writer.add_scalars("mious",{"Cityscapes":mious[CITY_ID],"Camvid":mious[CAM_ID]},configer.get("iter")+1)
                # # writer.export_scalars_to_json(osp.join(configer.get('res_save_pth'), str(time())+'_writer.json'))
                # logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
                net.train()
                
                if is_distributed():
                    net.module.aux_mode = 'train'
                else:
                    net.aux_mode = 'train'
        
        print("Alter Train")
        contrast_warmup_iters = configer.get("lr", "warmup_iters")
        with_aux = configer.get('loss', 'with_aux')
        with_domain_adversarial = configer.get('network', 'with_domain_adversarial')
        alter_iter = 0
        SEG = 0
        GNN = 1
        init_gnn_stage = configer.get('GNN', 'init_by_eval')
        fix_graph = False
        train_seg_or_gnn = GNN
        first_stage_gnn_iters = int(configer.get('train','gnn_iters')/2)
        sec_stage_gnn_iters = configer.get('train','gnn_iters') - first_stage_gnn_iters
        isSecondStage = False
        adv_out = None
        unify_prototype = None
        target_bi_graph = None
        # if alter_iter != 0:
        #     if is_distributed():
        #         _, _, target_bi_graph = eval_find_use_and_unuse_label_CVCUDA(configer, net.module, device_id, cuda_ctx)
        #     else:
        #         _, _, target_bi_graph = eval_find_use_and_unuse_label_CVCUDA(configer, net, device_id, cuda_ctx)
        if init_gnn_stage:
            init_adj = torch.load(configer.get('GNN', 'init_adj_path')).cuda()
            if is_distributed():
                graph_net.module.set_adj_matrix(init_adj, grad=False)
                # graph_net.module.output_max_adj = False
            else:
                graph_net.set_adj_matrix(init_adj, grad=False)
                # graph_net.output_max_adj = False
            
        GNN_INIT = configer.get('train', 'graph_finetune')
        gnn_optim = set_optimizer_two(graph_net, net, configer, configer.get('lr', 'gnn_lr_start'), configer.get('lr', 'seg_lr_start'))
        if mse_or_adv == 'adv':
            gnn_optimD = set_optimizerD(graph_net, configer, configer.get('lr', 'gnn_lr_start'))
        # GNN_INIT = True
        if init_gnn_stage:
            gnn_lr_schdr = WarmupPolyLrScheduler(gnn_optim, power=1.2,
                        max_iter=configer.get('train','gnn_iters')+1, warmup_iter=configer.get('lr','warmup_iters'),
                        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)
            # gnn_lr_schdr = WarmupPolyLrScheduler(gnn_optim, power=1.2,
            #             max_iter=10000, warmup_iter=configer.get('lr','warmup_iters'),
            #             warmup_ratio=0.1, warmup='exp', last_epoch=-1,)
        else:
            if is_distributed():
                graph_net.module.adj_matrix.requires_grad=True
                # graph_net.module.output_max_adj = False
            else:
                graph_net.adj_matrix.requires_grad=True
            gnn_optim = set_optimizer(graph_net, configer, configer.get('lr', 'gnn_lr_start'))
            gnn_lr_schdr = WarmupPolyLrScheduler(gnn_optim, power=1.2,
                        max_iter=first_stage_gnn_iters, warmup_iter=configer.get('lr','warmup_iters'),
                        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

        if mse_or_adv == 'adv':
            gnn_lr_schdrD = WarmupPolyLrScheduler(gnn_optimD, power=1.2,
                max_iter=configer.get('train','gnn_iters'), warmup_iter=configer.get('lr','warmup_iters'),
                warmup_ratio=0.1, warmup='exp', last_epoch=-1,)
        
        
        

        if train_seg_or_gnn == SEG:   
            if is_distributed():
                with torch.no_grad():

                    unify_prototype, ori_bi_graphs = graph_net.module.get_optimal_matching(graph_node_features, True)

                    # print(torch.norm(unify_prototype[0][0], p=2))
                    unify_prototype = unify_prototype.detach()
                    new_bi_graphs = []
                    if len(ori_bi_graphs) == 2*n_datasets:
                        for j in range(0, len(ori_bi_graphs), 2):
                            new_bi_graphs.append(ori_bi_graphs[j+1].detach())
                    else:
                        new_bi_graphs = [bigh.detach() for bigh in ori_bi_graphs]
                    fix_graph = True
                    adv_out = None
                    # _, ori_bi_graphs, _, _ = graph_net(graph_node_features)
                    # bi_graphs = []
                    # if len(ori_bi_graphs) == 2*n_datasets:
                    #     for j in range(0, len(ori_bi_graphs), 2):
                    #         bi_graphs.append(ori_bi_graphs[j+1].detach())
                    # else:
                    #     bi_graphs = [bigh.detach() for bigh in ori_bi_graphs]


                    # init_gnn_stage = False
                
                    # graph_net.module.init_stage = False
                

                    net.module.set_unify_prototype(unify_prototype, False)
                    net.module.set_bipartite_graphs(new_bi_graphs)
                    print_bipartite(configer, n_datasets, net.module.bipartite_graphs)

                    # net.module.unify_prototype.requires_grad = True
                        
            else:
                with torch.no_grad():

                    unify_prototype, ori_bi_graphs = graph_net.get_optimal_matching(graph_node_features, True)

                    # print(torch.norm(unify_prototype[0][0], p=2))
                    unify_prototype = unify_prototype.detach()
                    new_bi_graphs = []
                    if len(ori_bi_graphs) == 2*n_datasets:
                        for j in range(0, len(ori_bi_graphs), 2):
                            new_bi_graphs.append(ori_bi_graphs[j+1].detach())
                    else:
                        new_bi_graphs = [bigh.detach() for bigh in ori_bi_graphs]
                    fix_graph = True
                    adv_out = None
                    # _, ori_bi_graphs, _, _ = graph_net(graph_node_features)
                    # bi_graphs = []
                    # if len(ori_bi_graphs) == 2*n_datasets:
                    #     for j in range(0, len(ori_bi_graphs), 2):
                    #         bi_graphs.append(ori_bi_graphs[j+1].detach())
                    # else:
                    #     bi_graphs = [bigh.detach() for bigh in ori_bi_graphs]


                    # init_gnn_stage = False
                    
                    # graph_net.init_stage = False

                    net.set_unify_prototype(unify_prototype, False)
                    net.set_bipartite_graphs(new_bi_graphs)
                    print_bipartite(configer, n_datasets, net.bipartite_graphs)
                # net.unify_prototype.requires_grad = True
            
        if is_distributed():
            optim = set_optimizer(net.module, configer, configer.get('lr', 'seg_lr_start'))
        else:
            optim = set_optimizer(net, configer, configer.get('lr', 'seg_lr_start'))
        
        lr_schdr = WarmupPolyLrScheduler(optim, power=1.2,
            max_iter=configer.get('train','seg_iters'), warmup_iter=configer.get('lr','warmup_iters'),
            warmup_ratio=0.1, warmup='exp', last_epoch=-1,)
        
        # if train_seg_or_gnn==SEG and configer.get('train', 'finetune'):
        #     state = torch.load(configer.get('train', 'finetune_from'))
        #     # net.load_state_dict(torch.load(configer.get('train', 'finetune_from'), map_location='cpu'), strict=False)
        #     if 'optimizer_state_dict' in state.keys():
        #         optim.load_state_dict(state['optimizer_state_dict'])
        #     if 'scheduler_state_dict' in state.keys():
        #         lr_schdr.load_state_dict(state['scheduler_state_dict'])
        
        # if train_seg_or_gnn==GNN and configer.get('train', 'graph_finetune'):
        #     state = torch.load(configer.get('train', 'graph_finetune_from'))
        #     # net.load_state_dict(torch.load(configer.get('train', 'finetune_from'), map_location='cpu'), strict=False)
        #     if 'optimizer_state_dict' in state.keys():
        #         gnn_optim.load_state_dict(state['optimizer_state_dict'])
        #     if 'scheduler_state_dict' in state.keys():
        #         gnn_lr_schdr.load_state_dict(state['scheduler_state_dict'])
        
        total_max_iter = configer.get('lr', 'max_iter')
        configer.update(['iter'], 0)
        starti = 0
        if configer.get('lr','max_iter') > 0:
            dls = getDataLoaderCVCUDA(configer, device_id, cuda_ctx, stage=1)
        for i in range(starti, configer.get('lr','max_iter')):
            configer.plus_one('iter')
            ratio = 1 - float(i) / total_max_iter
            alter_iter += 1

            if init_gnn_stage == False and train_seg_or_gnn == GNN and alter_iter > first_stage_gnn_iters and isSecondStage == False:
                if is_distributed():
                    graph_net.module.frozenAdj(True)
                    gnn_optim = set_optimizer_two(graph_net.module, net.module, configer, ratio*configer.get('lr', 'gnn_lr_start'), ratio*configer.get('lr', 'seg_lr_start'))
                    gnn_lr_schdr = WarmupPolyLrScheduler(gnn_optim, power=1.2,
                        max_iter=sec_stage_gnn_iters, warmup_iter=configer.get('lr','warmup_iters'),
                        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)
                else:
                    graph_net.frozenAdj(True)
                    gnn_optim = set_optimizer_two(graph_net, net, configer, ratio*configer.get('lr', 'gnn_lr_start'), ratio*configer.get('lr', 'seg_lr_start'))
                    gnn_lr_schdr = WarmupPolyLrScheduler(gnn_optim, power=1.2,
                        max_iter=sec_stage_gnn_iters, warmup_iter=configer.get('lr','warmup_iters'),
                        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)
                isSecondStage = True
            # for j in range(0,len(dl_iters)):

                # try:
                #     im, lb = next(dl_iters[j])
                #     if not im.size()[0] == configer.get('dataset'+str(j+1), 'ims_per_gpu'):
                #         raise StopIteration
                #     while torch.min(lb) == 255:
                #         im, lb = next(dl_iters[j])
                #         if not im.size()[0] == configer.get('dataset'+str(j+1), 'ims_per_gpu'):
                #             raise StopIteration

                    
                # except StopIteration:
                #     dl_iters[j] = iter(dls[j])
                #     im, lb = next(dl_iters[j])
                #     while torch.min(lb) == 255:
                #         im, lb = next(dl_iters[j])
            im, lb, ids = dls()
            im = torch.as_tensor(
                im.cuda(), device="cuda:%d" % device_id
            )
            lb = torch.as_tensor(
                lb.cuda(), device="cuda:%d" % device_id
            ).squeeze(3).to(torch.long)
            
                
            # ids = []
            # for num in range(n_datasets):
            #     for _ in range(configer.get('dataset'+str(num+1), 'ims_per_gpu')):
            #         ids.append(num)
                
            dataset_lbs = torch.tensor(ids)
            dataset_lbs = dataset_lbs.cuda()
            # print(dataset_lbs)

            lb = torch.squeeze(lb, 1)
            # print(dataset_lbs)


            if train_seg_or_gnn == GNN and alter_iter > configer.get('train', 'gnn_iters'):

                # if is_distributed():
                #     _, ori_ema_bi_graphs = graph_net.module.get_optimal_matching(graph_node_features, GNN_INIT)
                #     # unify_prototype, bi_graphs, adv_out = graph_net(graph_node_features)    
                # else:
                #     _, ori_ema_bi_graphs = graph_net.get_optimal_matching(graph_node_features, GNN_INIT)     
                #     # unify_prototype, bi_graphs, adv_out = graph_net(graph_node_features)     
                # # print(bi_graphs)

                # # print(torch.norm(unify_prototype[0][0], p=2))
                # ema_bi_graphs = []
                # if len(ori_ema_bi_graphs) == 2*n_datasets:
                #     for j in range(0, len(ori_ema_bi_graphs), 2):
                #         ema_bi_graphs.append(ori_ema_bi_graphs[j+1].detach())
                # else:
                #     ema_bi_graphs = [bigh.detach() for bigh in ori_ema_bi_graphs]
                if is_distributed():
                    graph_net.module.init_stage = False
                else:
                    graph_net.init_stage = False
                isSecondStage = False
                init_gnn_stage = False
                if is_distributed():
                    graph_net.module.frozenAdj(False)
                    gnn_optim = set_optimizer(graph_net.module, configer, ratio * configer.get('lr', 'gnn_lr_start'))
                    if mse_or_adv == 'adv':
                        gnn_optimD = set_optimizerD(graph_net.module, configer, ratio * configer.get('lr', 'gnn_lr_start'))
                else:
                    graph_net.frozenAdj(False)
                    gnn_optim = set_optimizer(graph_net, configer, ratio * configer.get('lr', 'gnn_lr_start'))
                    if mse_or_adv == 'adv':
                        gnn_optimD = set_optimizerD(graph_net, configer, ratio * configer.get('lr', 'gnn_lr_start'))
                    

                alter_iter = 0
                train_seg_or_gnn = GNN
                gnn_lr_schdr = WarmupPolyLrScheduler(gnn_optim, power=1.2,
                    max_iter=first_stage_gnn_iters+1, warmup_iter=configer.get('lr','warmup_iters'),
                    warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

                if mse_or_adv == 'adv':
                    gnn_lr_schdrD = WarmupPolyLrScheduler(gnn_optimD, power=1.2,
                        max_iter=configer.get('train','gnn_iters')+1, warmup_iter=configer.get('lr','warmup_iters'),
                        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)


            # net.eval()
            # with torch.no_grad():
            #     seg_out = net(im)

            if train_seg_or_gnn == SEG:
                optim.zero_grad()
            else:
                gnn_optim.zero_grad()
                if mse_or_adv == 'adv':
                    gnn_optimD.zero_grad()
            with amp.autocast(enabled=configer.get('use_fp16')):
                
                ## 修改为多数据集模式
                
                seg_out = {}
                if train_seg_or_gnn == GNN:
                    is_adv = True
                    fix_graph = False
                    GNN_INIT = True
                    graph_net.train()
                    if init_gnn_stage or isSecondStage:
                        if is_distributed():
                            net.module.aux_mode = 'train_tg'
                        else:
                            net.aux_mode = 'train_tg'
                        net.train()
                        seg_out = net(im,dataset_lbs)
                    else:
                        net.eval()
                        with torch.no_grad():
                            seg_out = net(im,dataset_lbs)

                    unify_prototype, bi_graphs, adv_out, adj_feat = graph_net(graph_node_features)
                    
                    seg_out['unify_prototype'] = unify_prototype
                    seg_out['adj_feat'] = adj_feat
                    seg_out['gnn_stage'] = isSecondStage
                else:
                    if is_distributed():
                        net.module.aux_mode = 'train'
                    else:
                        net.aux_mode = 'train'
                    is_adv = False
                    graph_net.eval()
                    net.train()
                    unify_prototype = None
                    adv_out = None
                    if configer.get("GNN", "ema_graph") == True:
                        this_bi_graphs = [configer.get("GNN", "ema_graph_rate") * bg + (1 - configer.get("GNN", "ema_graph_rate")) * nbg for bg, nbg  in zip(bi_graphs, new_bi_graphs)] 
                        if is_distributed():
                            net.module.set_bipartite_graphs(this_bi_graphs)
                        else:
                            net.set_bipartite_graphs(this_bi_graphs)
                            
                    
                    seg_out = net(im,dataset_lbs)
                    if is_distributed():
                        bi_graphs = net.module.bipartite_graphs
                    else:
                        bi_graphs = net.bipartite_graphs
                    
                seg_out['unify_prototype'] = unify_prototype
                seg_out['bi_graphs'] = bi_graphs
                seg_out['adv_out'] = adv_out
                if target_bi_graph != None:
                    seg_out['target_bi_graph'] = target_bi_graph
                    
                adj_loss = None
                backward_loss, adv_loss, aux_loss, adj_loss = contrast_losses(seg_out, lb, dataset_lbs, is_adv, False)
                # print(backward_loss)
                # aux_loss = None
                loss_seg = backward_loss
                loss_aux = None
                loss_contrast = None


            if is_adv and mse_or_adv == 'adv':
                backward_loss += adv_loss
                # scaler.scale(adv_loss).backward()
                # scaler.step(gnn_optimD)
                # scaler.update()
                # torch.cuda.synchronize()
                # gnn_optimD.zero_grad()
            

            scaler.scale(backward_loss).backward()
            if train_seg_or_gnn == SEG: 
                scaler.step(optim)
            else:
                scaler.step(gnn_optim)
                if mse_or_adv == 'adv':
                    scaler.step(gnn_optimD)
            
            scaler.update()
            torch.cuda.synchronize()       

            if use_ema:
                ema_net.EMAUpdate(net.module)
            # print('synchronize')
            time_meter.update()
            loss_meter.update(backward_loss.item())
            if aux_loss:
                kl_loss_meter.update(aux_loss.item())
            
            if is_adv and adv_loss != None:
                loss_domain_meter.update(adv_loss.item())
            # loss_pre_meter.update(loss_pre.item())
            
            if not use_dataset_aux_head or i > aux_iter:
                loss_pre_meter.update(loss_seg.item()) 
                # if with_domain_adversarial:
                #     loss_domain_meter.update(loss_domain)
                if with_aux:
                    _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]
                
            if adj_loss != None:
                loss_contrast_meter.update(adj_loss.item())

        
            ## print training log message
            if (i + 1) % 100 == 0:
                # writer.add_scalars("loss",{"seg":loss_pre_meter.getWoErase(),"contrast":loss_contrast_meter.getWoErase(), "domain":loss_domain_meter.getWoErase()},configer.get("iter")+1)
                if train_seg_or_gnn == SEG:
                    lr = lr_schdr.get_lr()
                else:
                    lr = gnn_lr_schdr.get_lr()
                lr = sum(lr) / len(lr)
                print_log_msg(
                    i, 0, 0, configer.get('lr', 'max_iter'), lr, time_meter, loss_meter,
                    loss_pre_meter, loss_aux_meters, loss_contrast_meter, loss_domain_meter, kl_loss_meter)

            if (i + 1) % 10000 == 0:
                seg_save_pth = osp.join(configer.get('res_save_pth'), 'tg_seg_model_{}.pth'.format(i+1))
                gnn_save_pth = osp.join(configer.get('res_save_pth'), 'tg_graph_model_{}.pth'.format(i+1))
                logger.info('\nsave seg_models to {}, gnn_models to {}'.format(seg_save_pth, gnn_save_pth))
                print("before get")

                if is_distributed():
                    if unify_prototype != None:
                        net.module.set_unify_prototype(unify_prototype)
                        
                    net.module.set_bipartite_graphs(bi_graphs)
                    gnn_state = graph_net.module.state_dict()
                    seg_state = net.module.state_dict()
                    if dist.get_rank() == 0: 
                        torch.save({
                            'model_state_dict': gnn_state,
                            'optimizer_state_dict': gnn_optim.state_dict(),
                            'scheduler_state_dict': gnn_lr_schdr.state_dict(),
                        }, gnn_save_pth)
                        torch.save({
                            'model_state_dict': seg_state,
                            'optimizer_state_dict': optim.state_dict(),
                            'scheduler_state_dict': lr_schdr.state_dict(),
                        }, seg_save_pth)
                else:
                    if unify_prototype != None:
                        net.set_unify_prototype(unify_prototype)
                    net.set_bipartite_graphs(bi_graphs)
                    gnn_state = graph_net.state_dict()
                    seg_state = net.state_dict()
                    torch.save({
                        'model_state_dict': gnn_state,
                        'optimizer_state_dict': gnn_optim.state_dict(),
                        'scheduler_state_dict': gnn_lr_schdr.state_dict(),
                    }, gnn_save_pth)
                    torch.save({
                        'model_state_dict': seg_state,
                        'optimizer_state_dict': optim.state_dict(),
                        'scheduler_state_dict': lr_schdr.state_dict(),
                    }, seg_save_pth)

                # if fix_graph == False:

                # if (i + 1) % 50000 == 0:            
                if use_dataset_aux_head and i < aux_iter:
                    eval_model_func = eval_model_aux
                else:
                    
                    # eval_model_func = eval_model_unlabel
                    # eval_model_func = eval_find_use_and_unuse_label
                    # eval_model_func = eval_model_cvcuda
                    eval_model_func = eval_model_contrast

                if train_seg_or_gnn == SEG:
                    optim.zero_grad()
                else:
                    gnn_optim.zero_grad()
                    if mse_or_adv == 'adv':
                        gnn_optimD.zero_grad()
                
                torch.cuda.empty_cache()
                if is_distributed():
                    # if eval_model_func == eval_find_use_and_unuse_label:
                    if train_seg_or_gnn == GNN and not isSecondStage and not init_gnn_stage:
                        heads, mious, target_bi_graph = eval_find_use_and_unuse_label(configer, net.module)
                    # else:
                    # heads, mious = eval_model_func(configer, net.module, device_id, cuda_ctx)
                    heads, mious = eval_model_func(configer, net.module)
                else:
                    # if eval_model_func == eval_find_use_and_unuse_label:
                    if train_seg_or_gnn == GNN and not isSecondStage and not init_gnn_stage:
                        heads, mious, target_bi_graph = eval_find_use_and_unuse_label(configer, net)
                    # else:
                    # heads, mious = eval_model_func(configer, net, device_id, cuda_ctx)
                    heads, mious = eval_model_func(configer, net)
                    
                torch.cuda.empty_cache()
                # writer.add_scalars("mious",{"Cityscapes":mious[CITY_ID],"Camvid":mious[CAM_ID]},configer.get("iter")+1)
                # writer.export_scalars_to_json(osp.join(configer.get('res_save_pth'), str(time())+'_writer.json'))
                logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
                net.train()
                
                if is_distributed():
                    net.module.aux_mode = 'train'
                else:
                    net.aux_mode = 'train'
            
                    
            if train_seg_or_gnn == SEG:
                lr_schdr.step()
            else:
                gnn_lr_schdr.step()
                if mse_or_adv == 'adv':
                    gnn_lr_schdrD.step()    
            
        with torch.no_grad():
            # input_feats = torch.cat([graph_node_features, graph_net.unify_node_features], dim=0)
            if is_distributed():
                unify_prototype, bi_graphs = graph_net.module.get_optimal_matching(graph_node_features, True)
            else:
                unify_prototype, bi_graphs = graph_net.get_optimal_matching(graph_node_features, True) 
            
            new_bi_graphs = []
            if len(bi_graphs) == 2*n_datasets:
                for j in range(0, len(bi_graphs), 2):
                    new_bi_graphs.append(bi_graphs[j+1].detach())
            else:
                new_bi_graphs = [bigh.detach() for bigh in bi_graphs]

        if is_distributed():
            net.module.set_unify_prototype(unify_prototype, True)
            net.module.set_bipartite_graphs(new_bi_graphs)
        else:
            net.set_unify_prototype(unify_prototype, True)
            net.set_bipartite_graphs(new_bi_graphs)

        # ## dump the final model and evaluate the result
        # seg_save_pth = osp.join(configer.get('res_save_pth'), 'seg_model_joint_stage.pth')
        # gnn_save_pth = osp.join(configer.get('res_save_pth'), 'gnn_model_joint_stage.pth')
        # logger.info('\nsave seg_models to {}, gnn_models to {}'.format(seg_save_pth, gnn_save_pth))
        
        # # writer.close()
        # if is_distributed():
        #     gnn_state = graph_net.module.state_dict()
        #     seg_state = net.module.state_dict()
        #     if dist.get_rank() == 0: 
        #         torch.save(gnn_state, gnn_save_pth)
        #         torch.save(seg_state, seg_save_pth)
        # else:
        #     gnn_state = graph_net.state_dict()
        #     seg_state = net.state_dict()
        #     torch.save(gnn_state, gnn_save_pth)
        #     torch.save(seg_state, seg_save_pth)

        del graph_net
        
        # logger.info('\nevaluating the final model')
        torch.cuda.empty_cache()
        
            

        total_cats = 0
        for i in range(1, n_datasets+1):
            total_cats += configer.get('dataset'+str(i), 'n_cats')

        total_cats = int(total_cats * configer.get('GNN', 'unify_ratio'))
        
        for stage in ['stage1', 'stage2']:    
            print(stage)
            if dls is not None:
                del dls
            if stage == 'stage2':
                dls = getDataLoaderCVCUDA(configer, device_id, cuda_ctx)
                # if is_distributed():
                #     loaded_map = find_unuse(configer, net.module)
                # else:
                #     loaded_map = find_unuse(configer, net)
                # bi_graphs = []
                # for dataset_id in range(1, n_datasets+1):
                #     n_cats = configer.get('dataset'+str(dataset_id), 'n_cats')
                #     this_bi_graph = torch.zeros(n_cats, total_cats)
                #     for key, val in loaded_map['dataset'+str(dataset_id)].items():
                #         this_bi_graph[int(key)][val] = 1
                        
                #     bi_graphs.append(this_bi_graph.cuda())

                # if is_distributed():
                #     net.module.set_bipartite_graphs(bi_graphs)
                # else:
                #     net.set_bipartite_graphs(bi_graphs) 
                            
                optim = set_optimizer(net, configer, configer.get('lr', 'seg_lr_start'))
            else:
                dls = getDataLoaderCVCUDA(configer, device_id, cuda_ctx, stage=1)

            # finetune stage1

            if is_distributed():
                net.module.aux_mode = 'train'
                net.module.with_datasets_aux = False
                net.module.set_aux_grad(False)
                contrast_losses.with_datasets_aux = False
            else:
                net.aux_mode = 'train'
                net.with_datasets_aux = False
                net.set_aux_grad(False)
                contrast_losses.with_datasets_aux = False
                
            if is_distributed():
                net.module.unify_prototype.requires_grad = True
                optim = set_optimizer(net.module, configer, configer.get('lr', 'seg_lr_start'))
            else:
                net.unify_prototype.requires_grad = True
                optim = set_optimizer(net, configer, configer.get('lr', 'seg_lr_start'))
                
            lr_schdr = WarmupPolyLrScheduler(optim, power=1.2,
                max_iter=configer.get('train',f'finetune_{stage}_iters'), warmup_iter=configer.get('lr','warmup_iters'),
                warmup_ratio=0.1, warmup='exp', last_epoch=-1,) 

        
            configer.update(['iter'], 0)
            for i in range(starti, configer.get('train',f'finetune_{stage}_iters') + starti):
                configer.plus_one('iter')
                alter_iter += 1
                # if i > 10000:
                #     init_gnn_stage = False


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

                im, lb, ids = dls()
                im = torch.as_tensor(
                    im.cuda(), device="cuda:%d" % device_id
                )
                lb = torch.as_tensor(
                    lb.cuda(), device="cuda:%d" % device_id
                ).squeeze(3).to(torch.long)
                
                    
                # ids = []
                # for num in range(n_datasets):
                #     for _ in range(configer.get('dataset'+str(num+1), 'ims_per_gpu')):
                #         ids.append(num)
                    
                dataset_lbs = torch.tensor(ids)
                dataset_lbs = dataset_lbs.cuda()
                # print(dataset_lbs)

                lb = torch.squeeze(lb, 1)


                # net.eval()
                # with torch.no_grad():
                #     seg_out = net(im)

                optim.zero_grad()
                with amp.autocast(enabled=configer.get('use_fp16')):
                    # if finetune and i >= fix_param_iters + aux_iter:
                    #     finetune = False
                    #     if is_distributed():
                    #         net.module.switch_require_grad_state(True)
                    #     else:
                    #         net.switch_require_grad_state(True)
                        
                    
                    ## 修改为多数据集模式
                    
                    # if train_aux:
                    #     train_aux = False
                    #     if is_distributed():
                    #         net.module.set_train_dataset_aux(False)
                    #     else:
                    #         net.set_train_dataset_aux(False)    
                    
                        
                                
                    seg_out = {}

                    is_adv = False
                    
                    net.train()

                    seg_out = net(im,dataset_lbs)
                    seg_out['unify_prototype'] = None
                    

                        
                    # seg_out['seg'] = seg_out['seg']
                    
                    if is_distributed():
                        seg_out['bi_graphs'] = net.module.bipartite_graphs
                    else:
                        seg_out['bi_graphs'] = net.bipartite_graphs
                    seg_out['adv_out'] = None
                        
                    backward_loss, adv_loss, aux_loss, adj_loss = contrast_losses(seg_out, lb, dataset_lbs, is_adv, False)
                    # print(backward_loss)
                    kl_loss = None
                    loss_seg = backward_loss
                    loss_aux = None
                    loss_contrast = None

                    
                # if with_memory and 'key' in out:
                #     dequeue_and_enqueue(configer, city_out['seg'], city_out['key'], lb_city.detach(),
                #                         city_out['segment_queue'], i, CITY_ID)
                #     dequeue_and_enqueue(configer, cam_out['seg'], cam_out['key'], lb_cam.detach(),
                #                         cam_out['segment_queue'], i, CAM_ID)


                # set_trace()
                # with torch.autograd.detect_anomaly():
                # print(backward_loss)

                if is_adv and mse_or_adv == 'adv':
                    backward_loss += adv_loss
                    # scaler.scale(adv_loss).backward()
                    # scaler.step(gnn_optimD)
                    # scaler.update()
                    # torch.cuda.synchronize()
                    # gnn_optimD.zero_grad()
                

                scaler.scale(backward_loss).backward()
                
                scaler.step(optim)
                
                scaler.update()
                torch.cuda.synchronize()

                if use_ema:
                    ema_net.EMAUpdate(net.module)
                # print('synchronize')
                time_meter.update()
                loss_meter.update(backward_loss.item())
                if aux_loss:
                    kl_loss_meter.update(aux_loss.item())
                
                if is_adv and adv_loss != None:
                    loss_domain_meter.update(adv_loss.item())
                # loss_pre_meter.update(loss_pre.item())
                
                if not use_dataset_aux_head or i > aux_iter:
                    loss_pre_meter.update(loss_seg.item()) 
                    # if with_domain_adversarial:
                    #     loss_domain_meter.update(loss_domain)
                    if with_aux:
                        _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]
                    
                if adj_loss != None:
                    loss_contrast_meter.update(adj_loss.item())
                    # if i >= configer.get('lr', 'warmup_iters') and use_contrast:
                    #     loss_contrast_meter.update(loss_contrast.item())

                
                ## print training log message
                if (i + 1) % 100 == 0:
                    # writer.add_scalars("loss",{"seg":loss_pre_meter.getWoErase(),"contrast":loss_contrast_meter.getWoErase(), "domain":loss_domain_meter.getWoErase()},configer.get("iter")+1)

                    lr = lr_schdr.get_lr()
                    if type(lr) != float:
                        lr = sum(lr) / len(lr)
                    
                    print_log_msg(
                        i, 0, 0, configer.get('train', 'finetune_stage1_iters')+configer.get('train', 'finetune_stage2_iters'), lr, time_meter, loss_meter,
                        loss_pre_meter, loss_aux_meters, loss_contrast_meter, loss_domain_meter, kl_loss_meter)
                    

                if (i + 1) % 10000 == 0:
                    stage_iter = 0 if stage == 'stage1' else configer.get('train', 'finetune_stage1_iters')
                    seg_save_pth = osp.join(configer.get('res_save_pth'), 'test_seg_model_{}_{}.pth'.format(stage, i+1+configer.get('lr','max_iter')))
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
                    torch.cuda.empty_cache()
                    if is_distributed():
                        heads, mious = eval_model_func(configer, net.module)
                    else:
                        heads, mious = eval_model_func(configer, net)
                        
                    # writer.add_scalars("mious",{"Cityscapes":mious[CITY_ID],"Camvid":mious[CAM_ID]},configer.get("iter")+1)
                    # writer.export_scalars_to_json(osp.join(configer.get('res_save_pth'), str(time())+'_writer.json'))
                    logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
                    net.train()
                
                            
                lr_schdr.step()

        cuda_ctx.pop()
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

