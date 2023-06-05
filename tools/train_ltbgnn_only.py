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
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg
from lib.loss.loss_cross_datasets import CrossDatasetsLoss, CrossDatasetsCELoss, CrossDatasetsCELoss_KMeans, CrossDatasetsCELoss_CLIP, CrossDatasetsCELoss_GNN, CrossDatasetsCELoss_AdvGNN
from lib.class_remap import ClassRemap

from tools.configer import Configer
from evaluate import eval_model_contrast, eval_model_aux, eval_model, eval_model_contrast_single

from tensorboardX import SummaryWriter

from lib.module.gen_graph_node_feature import gen_graph_node_feature



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
    parse.add_argument('--config', dest='config', type=str, default='configs/ltbgnn_city_cam_a2d2.json',)
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
        net.load_state_dict(torch.load(configer.get('train', 'finetune_from'), map_location='cpu'), strict=False)

        
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
        net.load_state_dict(torch.load(configer.get('train', 'graph_finetune_from'), map_location='cpu'), strict=False)

        
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
    
def set_optimizerD(model, configer):
    if hasattr(model, 'get_discri_params'):
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
        return None
    
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
            weight_decay=configer.get('lr', 'weight_decay'),
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

def dequeue_and_enqueue(configer, seg_out, keys, labels,
                            segment_queue, iter, dataset_id):
    ## 更新memory bank
    out = seg_out[0]
    probs = F.softmax(out, dim=1)
    
    batch_size = keys.shape[0]
    feat_dim = keys.shape[1]
    network_stride = configer.get('network', 'stride')
    coefficient = configer.get('contrast', 'coefficient')
    warmup_iter = configer.get('lr', 'warmup_iters')
    ignore_index = configer.get('loss', 'ignore_index')
    update_confidence_thresh = configer.get('contrast', 'update_confidence_thresh')
    num_unify_classes = configer.get('num_unify_classes')
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
    
    this_probs = probs.permute(1, 0, 2, 3)
    this_probs = probs.contiguous().view(num_unify_classes, -1)

    for lb_id in this_label_ids:
        lb = lb_id.item()
        
        if len(classRemapper.getAnyClassRemap(lb, dataset_id)) > 1:
            ## 当一个标签对应多个的情况下，不应该进行更新
            # remap_lb = lb
            continue
        else:
            remap_lb = classRemapper.getAnyClassRemap(lb, dataset_id)[0]

        idxs = (this_label == lb).nonzero().squeeze()
        feat_idxs = this_feat[:, idxs]
        
        
        # 计算对应置信度是否大于阈值
        update_id = (this_probs[remap_lb, idxs] > update_confidence_thresh).nonzero()
        
        if update_id.shape[0] == 0 or update_id.shape[1] == 0:
            continue
        else:
            update_id = update_id.squeeze(1)
        
        # print("remap_lb :", remap_lb)
        # print("idx:", idxs.shape)
        # print("feat_idxs shape: ", feat_idxs.shape)
        # print("update_id shape:", update_id.shape)
        # print("this_probs:", this_probs.shape)
        # print("this_probs[remap_lb, idxs]:", this_probs[remap_lb, idxs].shape)
        
        # segment enqueue and dequeue
        # feat_idxs = feat_idxs.squeeze(2)

        # update_id = update_id[:, 0]

        feat = torch.mean(feat_idxs[:, update_id], dim=1)
        # print('lb_id: ', lb_id)
        # print("feat_idxs: ", torch.isnan(feat_idxs).any())
        # print("feat: ", torch.isnan(feat).any())
        
        if torch.isnan(feat).any():
            raise Exception("feat nan")

        # # segment enqueue and dequeue
        # feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)

        # print(nn.functional.normalize(feat.view(-1), p=2, dim=0))
        # print(segment_queue[lb])
            
        segment_queue[remap_lb] =  nn.functional.normalize((coefficient * segment_queue[remap_lb] + (1 - coefficient) * nn.functional.normalize(feat, p=2, dim=0)), p=2, dim=0)

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
    ## dataset

    # dl = get_single_data_loader(configer, aux_mode='train', distributed=is_dist)
    dls = get_data_loader(configer, aux_mode='train', distributed=is_dist)
    # dl_city, dl_cam = get_data_loader(configer, aux_mode='train', distributed=is_dist)
    
    ## model
    net = set_model(configer=configer)
    graph_net = set_graph_model(configer=configer)
    
    if use_ema:
        ema_net = set_ema_model(configer=configer)
        
    contrast_losses = set_contrast_loss(configer)

    ## optimizer
    optim = set_optimizer(net, configer)
    gnn_optim = set_optimizer(graph_net, configer)
    gnn_optimD = set_optimizerD(graph_net, configer)

    ## mixed precision training
    scaler = amp.GradScaler()

    graph_node_features = gen_graph_node_feature(configer)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters, loss_contrast_meter, loss_domain_meter, kl_loss_meter = set_meters(configer)
    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=configer.get('lr','max_iter'), warmup_iter=configer.get('lr','warmup_iters'),
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    gnn_lr_schdr = WarmupPolyLrScheduler(gnn_optim, power=0.9,
        max_iter=configer.get('lr','max_iter'), warmup_iter=configer.get('lr','warmup_iters'),
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    gnn_lr_schdrD = WarmupPolyLrScheduler(gnn_optimD, power=0.9,
        max_iter=configer.get('lr','max_iter'), warmup_iter=configer.get('lr','warmup_iters'),
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)
    # 两个数据集分别处理
    # 使用迭代器读取数据
    
    dl_iters = [iter(dl) for dl in dls]
    
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
        graph_net = set_model_dist(graph_net)

    contrast_warmup_iters = configer.get("lr", "warmup_iters")
    with_aux = configer.get('loss', 'with_aux')
    with_domain_adversarial = configer.get('network', 'with_domain_adversarial')

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
                if not im.size()[0] == configer.get('dataset'+str(j+1), 'ims_per_gpu'):
                    raise StopIteration
            except StopIteration:
                dl_iters[j] = iter(dls[j])
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


        SEG = 0
        GNN = 1
        # if configer.get('iter') // configer.get('train', 'seg_gnn_alter_iters') % 2 == 0:
        #     train_seg_or_gnn = SEG
        # else:
        train_seg_or_gnn = GNN
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
            
            if train_aux:
                train_aux = False
                if is_distributed():
                    net.module.set_train_dataset_aux(False)
                else:
                    net.set_train_dataset_aux(False)    
                
            fix_graph = False            
            if train_seg_or_gnn == GNN:
                is_adv = True
                fix_graph = False
                graph_net.train()
                net.eval()
                with torch.no_grad():
                    seg_out = net(im)
                
                unify_prototype, bi_graphs, adv_out = graph_net(graph_node_features)
            else:
                is_adv = False
                graph_net.eval()
                net.train()
                seg_out = net(im)
                
                if fix_graph == False:
                    with torch.no_grad():
                        unify_prototype, bi_graphs, adv_out = graph_net(graph_node_features)
                        unify_prototype = unify_prototype.detach()
                        bi_graphs = [bigh.detach() for bigh in bi_graphs]
                        fix_graph = True
                
            
            seg_out['seg'] = seg_out['seg']
            seg_out['unify_prototype'] = unify_prototype
            seg_out['bi_graphs'] = bi_graphs
            seg_out['adv_out'] = adv_out

                
            backward_loss, adv_loss = contrast_losses(seg_out, lb, dataset_lbs, is_adv)
            kl_loss = None
            loss_seg = backward_loss
            loss_aux = None
            loss_contrast = None

            
        # if with_memory and 'key' in out:
        #     dequeue_and_enqueue(configer, city_out['seg'], city_out['key'], lb_city.detach(),
        #                         city_out['segment_queue'], i, CITY_ID)
        #     dequeue_and_enqueue(configer, cam_out['seg'], cam_out['key'], lb_cam.detach(),
        #                         cam_out['segment_queue'], i, CAM_ID)


        # print('before backward')
        # set_trace()
        # with torch.autograd.detect_anomaly():
        
        scaler.scale(backward_loss).backward()
        scaler.scale(adv_loss).backward()
        # print(backward_loss.item())
            
        # print('after backward')

        # configer.plus_one('iters')
        # self.configer.plus_one('iters')

        # for name, param in graph_net.named_parameters():
        #     if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
        #         print("Graph NaN or Inf value found in gradients")

        # for param in net.parameters():
        #     if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
        #         print("seg NaN or Inf value found in gradients")
        
        if train_seg_or_gnn == SEG: 
            scaler.step(optim)
        else:
            scaler.step(gnn_optim)
            scaler.step(gnn_optimD)
            
        scaler.update()
        torch.cuda.synchronize()
        if use_ema:
            ema_net.EMAUpdate(net.module)
        # print('synchronize')
        time_meter.update()
        loss_meter.update(backward_loss.item())
        if kl_loss:
            kl_loss_meter.update(kl_loss.item())
        
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
            lr = gnn_lr_schdr.get_lr()
            lr = sum(lr) / len(lr)
            print_log_msg(
                i, 0, 0, configer.get('lr', 'max_iter')+starti, lr, time_meter, loss_meter,
                loss_pre_meter, loss_aux_meters, loss_contrast_meter, loss_domain_meter, kl_loss_meter)
            

        if (i + 1) % 5000 == 0:
            seg_save_pth = osp.join(configer.get('res_save_pth'), 'seg_model_{}.pth'.format(i+1))
            gnn_save_pth = osp.join(configer.get('res_save_pth'), 'graph_model_{}.pth'.format(i+1))
            logger.info('\nsave seg_models to {}, gnn_models to {}'.format(seg_save_pth, gnn_save_pth))

            # if fix_graph == False:
            with torch.no_grad():
                # input_feats = torch.cat([graph_node_features, graph_net.unify_node_features], dim=0)
                unify_prototype, bi_graphs, _ = graph_net(graph_node_features)
                if configer.get('GNN', 'output_max_adj') and configer.get('GNN', 'output_softmax_and_max_adj'):
                    bi_graphs = [bi_graph for i, bi_graph in filter(lambda x : x[0] % 2 == 0, enumerate(bi_graphs))]

            if use_dataset_aux_head and i < aux_iter:
                eval_model_func = eval_model_aux
            else:
                # eval_model = eval_model_contrast
                eval_model_func = eval_model_contrast

            optim.zero_grad()
            if is_distributed():
                net.module.set_unify_prototype(unify_prototype)
                net.module.set_bipartite_graphs(bi_graphs)
                heads, mious = eval_model_func(configer, net.module)
            else:
                net.set_unify_prototype(unify_prototype)
                net.set_bipartite_graphs(bi_graphs)
                heads, mious = eval_model_func(configer, net)
                
            writer.add_scalars("mious",{"Cityscapes":mious[CITY_ID],"Camvid":mious[CAM_ID]},configer.get("iter")+1)
            # writer.export_scalars_to_json(osp.join(configer.get('res_save_pth'), str(time())+'_writer.json'))
            logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
            net.train()
            
            if is_distributed():
                gnn_state = graph_net.module.state_dict()
                seg_state = net.module.state_dict()
                if dist.get_rank() == 0: 
                    torch.save(gnn_state, gnn_save_pth)
                    torch.save(seg_state, seg_save_pth)
            else:
                gnn_state = graph_net.state_dict()
                seg_state = net.state_dict()
                torch.save(gnn_state, gnn_save_pth)
                torch.save(seg_state, seg_save_pth)
                
        if train_seg_or_gnn == SEG:
            lr_schdr.step()
        else:
            gnn_lr_schdr.step()
            gnn_lr_schdrD.step()
        

    ## dump the final model and evaluate the result
    seg_save_pth = osp.join(configer.get('res_save_pth'), 'seg_model_final.pth')
    gnn_save_pth = osp.join(configer.get('res_save_pth'), 'gnn_model_final.pth')
    logger.info('\nsave seg_models to {}, gnn_models to {}'.format(seg_save_pth, gnn_save_pth))
    
    writer.close()
    if is_distributed():
        gnn_state = graph_net.module.state_dict()
        seg_state = net.module.state_dict()
        if dist.get_rank() == 0: 
            torch.save(gnn_state, gnn_save_pth)
            torch.save(seg_state, seg_save_pth)
    else:
        gnn_state = graph_net.state_dict()
        seg_state = net.state_dict()
        torch.save(gnn_state, gnn_save_pth)
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
    if False:
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
    main()
