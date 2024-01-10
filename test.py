import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from time import sleep
sys.path.insert(0, '.')
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import os.path as osp
import argparse
from lib.models import model_factory
from lib.loss.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg
from lib.loss.loss_cross_datasets import CrossDatasetsLoss, CrossDatasetsCELoss, CrossDatasetsCELoss_KMeans, CrossDatasetsCELoss_CLIP, CrossDatasetsCELoss_GNN, CrossDatasetsCELoss_AdvGNN
from lib.class_remap import ClassRemap
import math

import cv2
from tools.configer import Configer
from torch.utils.data import Dataset, DataLoader

from lib.cityscapes_cv2 import CityScapes, CityScapesIm
from lib.a2d2_lb_cv2 import A2D2Data
from lib.a2d2_cv2 import A2D2Data_L
from lib.ADE20K import ade20k
from lib.ade2016_data import ade2016, ade2016_mseg
from lib.bdd100k_data import Bdd100k
from lib.idd_cv2 import Idd
from lib.Mapi import Mapi, Mapiv1, Mapiv1_mseg
from lib.sunrgbd import Sunrgbd
from lib.coco_data import Coco_data, Coco_data_mseg
from lib.a2d2_city_dataset import A2D2CityScapes
from lib.CamVid_lb import CamVid
from lib.WD2 import wd2
from lib.scannet import scannet
from lib.MultiSetReader import MultiSetReader
from lib.all_datasets_reader import AllDatasetsReader

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

class Compose(object):

    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb

class RandomResizedCrop(object):
    '''
    size should be a tuple of (H, W)
    '''
    def __init__(self, scales=(0.5, 1.), size=(384, 384)):
        self.scales = scales
        self.size = size

    def __call__(self, im_lb):
        if self.size is None:
            return im_lb

        im, lb = im_lb['im'], im_lb['lb']
        H, W = im.shape[:2]        

        assert im.shape[:2] == lb.shape[:2]
        
        crop_h, crop_w = self.size
        
        scale = min(self.scales)
        if np.min([H, W]) < 1080:
            scale = scale * (1080 / np.min([H, W]))
            
        
        im_h, im_w = [math.ceil(el * scale) for el in im.shape[:2]]
        im = cv2.resize(im, (im_w, im_h))
        lb = cv2.resize(lb, (im_w, im_h), interpolation=cv2.INTER_NEAREST)

        if (im_h, im_w) == (crop_h, crop_w): return dict(im=im, lb=lb)
        pad_h, pad_w = 0, 0
        if im_h < crop_h:
            pad_h = (crop_h - im_h) // 2 + 1
        if im_w < crop_w:
            pad_w = (crop_w - im_w) // 2 + 1
        if pad_h > 0 or pad_w > 0:
            im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
            lb = np.pad(lb, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=255)

        im_h, im_w, _ = im.shape
        sh, sw = 1, 1
        sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))
        return dict(
            im=im[sh:sh+crop_h, sw:sw+crop_w, :].copy(),
            lb=lb[sh:sh+crop_h, sw:sw+crop_w].copy()
        )
        
class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im_lb):
        # if np.random.random() < self.p:
        #     return im_lb
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]
        return dict(
            im=im[:, ::-1, :],
            lb=lb[:, ::-1],
        )

class RandomResizedCrop_Flip_ColorJitterGPU(object):
    def __init__(self, scales=(0.5, 1.), size=(384, 384), p=0.5, brightness=None, contrast=None, saturation=None):
        self.scales = scales
        self.size = size
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, im_lb):
        if self.size is None:
            return im_lb

        im, lb = im_lb['im'], im_lb['lb']
        H, W = im.shape[:2]

        assert im.shape[:2] == lb.shape[:2]

        crop_h, crop_w = self.size

        scale = np.random.uniform(min(self.scales), max(self.scales))
        if np.min([H, W]) < 1080:
            scale = scale * (1080 / np.min([H, W]))

        im_h, im_w = [math.ceil(el * scale) for el in im.shape[:2]]
        im_gpu = cv2.cuda_GpuMat()
        lb_gpu = cv2.cuda_GpuMat()
        im_gpu.upload(im)
        lb_gpu.upload(lb)
        im_gpu = cv2.cuda.resize(im_gpu, (im_w, im_h))
        lb_gpu = cv2.cuda.resize(lb_gpu, (im_w, im_h), interpolation=cv2.INTER_NEAREST)

        if (im_h, im_w) == (crop_h, crop_w): return dict(im=im_gpu.download(), lb=lb_gpu.download())
               # 计算需要填充的大小
        pad_h = max(0, (crop_h - im_h) // 2 + 1)
        pad_w = max(0, (crop_w - im_w) // 2 + 1)
        
        # 使用copyMakeBorder进行填充
        im_gpu = cv2.cuda.copyMakeBorder(im_gpu, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
        lb_gpu = cv2.cuda.copyMakeBorder(lb_gpu, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=255)
 
        # 随机裁剪的起始位置
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))
        
        # 使用ROI进行随机裁剪
        im_gpu = im_gpu(cv2.cuda_GpuMat(), roi=(sw, sh, crop_w, crop_h))
        lb_gpu = lb_gpu(cv2.cuda_GpuMat(), roi=(sw, sh, crop_w, crop_h))

        im_gpu = cv2.cuda.flip(im_gpu, flipCode=1)
        lb_gpu = cv2.cuda.flip(lb_gpu, flipCode=1)
        
        rate = 1.25
        if self.brightness is not None:
            self.adj_brightness_gpu(im_gpu, rate)
        if self.contrast is not None:
            self.adj_contrast_gpu(im_gpu, rate)
        if self.saturation is not None:

            self.adj_saturation_gpu(im_gpu, rate)
        
        return dict(im=im_gpu.download(), lb=lb_gpu.download())
    
    def adj_saturation_gpu(self, im_gpu, rate):
        M = np.float32([
            [1 + 2 * rate, 1 - rate, 1 - rate],
            [1 - rate, 1 + 2 * rate, 1 - rate],
            [1 - rate, 1 - rate, 1 + 2 * rate]
        ])
        im_gpu_color = cv2.cuda.cvtColor(im_gpu, cv2.COLOR_BGR2BGRA)
        im_gpu_color_channels = cv2.cuda.split(im_gpu_color)
        for i in range(3):
            cv2.cuda.LUT(im_gpu_color_channels[i], M[i], im_gpu_color_channels[i])
        cv2.cuda.merge(im_gpu_color_channels, im_gpu_color)
        cv2.cuda.cvtColor(im_gpu_color, cv2.COLOR_BGRA2BGR, dst=im_gpu)

    def adj_brightness_gpu(self, im_gpu, rate):
        table = np.array([i * rate for i in range(256)]).clip(0, 255).astype(np.uint8)
        cv2.cuda.LUT(im_gpu, table, im_gpu)

    def adj_contrast_gpu(self, im_gpu, rate):
        table = np.array([74 + (i - 74) * rate for i in range(256)]).clip(0, 255).astype(np.uint8)
        cv2.cuda.LUT(im_gpu, table, im_gpu)


class ColorJitter(object):

    def __init__(self, brightness=None, contrast=None, saturation=None):
        if not brightness is None and brightness >= 0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast >= 0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation >= 0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]
        if not self.brightness is None:
            rate = 1.25
            im = self.adj_brightness(im, rate)
        if not self.contrast is None:
            rate = 1.25
            im = self.adj_contrast(im, rate)
        if not self.saturation is None:
            rate = 1.25
            im = self.adj_saturation(im, rate)
        return dict(im=im, lb=lb,)

    def adj_saturation(self, im, rate):
        M = np.float32([
            [1+2*rate, 1-rate, 1-rate],
            [1-rate, 1+2*rate, 1-rate],
            [1-rate, 1-rate, 1+2*rate]
        ])
        shape = im.shape
        im = np.matmul(im.reshape(-1, 3), M).reshape(shape)/3
        im = np.clip(im, 0, 255).astype(np.uint8)
        return im

    def adj_brightness(self, im, rate):
        table = np.array([
            i * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[im]

    def adj_contrast(self, im, rate):
        table = np.array([
            74 + (i - 74) * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[im]

class TransformationTrain(object):

    def __init__(self, scales, cropsize):
        self.trans_func = Compose([
            RandomResizedCrop_Flip_ColorJitterGPU(scales, cropsize, p=0.5, brightness=0.4, contrast=0.4, saturation=0.4),
            # RandomHorizontalFlip(),
            # ColorJitter(
            #     brightness=0.4,
            #     contrast=0.4,
            #     saturation=0.4
            # ),
        ])
        # self.trans_func = T.Compose([
            # T.TensorRandomResizedCrop(scales, cropsize),
            # T.RandomHorizontalFlip(),
            # T.TensorColorJitter(
            #     brightness=0.4,
            #     contrast=0.4,
            #     saturation=0.4
            # ),
        # ])

    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)
        return im_lb


class TransformationVal(object):

    def __init__(self, scales, cropsize):
        self.trans_func = Compose([
            RandomResizedCrop(scales, cropsize),
            RandomHorizontalFlip(),
            ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
        ])
        # self.trans_func = T.Compose([
            # T.TensorRandomResizedCrop(scales, cropsize),
            # T.RandomHorizontalFlip(),
            # T.TensorColorJitter(
            #     brightness=0.4,
            #     contrast=0.4,
            #     saturation=0.4
            # ),
        # ])

    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)
        return im_lb




def get_data_loader(configer, aux_mode='eval', distributed=True, stage=None):
    mode = aux_mode
    n_datasets = configer.get('n_datasets')
    max_iter = configer.get('lr', 'max_iter')
    scales = configer.get('train', 'scales')
    cropsize = configer.get('train', 'cropsize')
    if mode == 'train':
        trans_func = TransformationTrain(scales, cropsize)
    else:
        trans_func = TransformationVal(scales, cropsize)

    
    if stage != None:
        annpath = [configer.get('dataset'+str(i), 'train_im_anns').replace('.txt', f'_{stage}.txt') for i in range(1, n_datasets+1)]
        print(annpath)
        batchsize = [configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, n_datasets+1)]
    else:
        annpath = [configer.get('dataset'+str(i), 'train_im_anns') for i in range(1, n_datasets+1)]
        batchsize = [configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, n_datasets+1)]
    imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
    data_reader = [configer.get('dataset'+str(i), 'data_reader') for i in range(1, n_datasets+1)]
    
    shuffle = True
    drop_last = True

        

    ds = [eval(reader)(root, path, trans_func=trans_func, mode=mode)
          for reader, root, path in zip(data_reader, imroot, annpath)]
    # ds = [eval(reader)(root, path, trans_func=trans_func, mode=mode)
    #       for reader, root, path in zip(data_reader, imroot, annpath)]


        # n_train_imgs = cfg.ims_per_gpu * cfg.max_iter
        # sampler = RepeatedDistSampler(ds, n_train_imgs, shuffle=shuffle, num_replicas=1, rank=0)
        # batchsampler = torch.utils.data.sampler.BatchSampler(
        #     sampler, batchsize, drop_last=drop_last
        # )
        # dl = DataLoader(
        #     ds,
        #     batch_sampler=batchsampler,
        #     num_workers=4,
        #     pin_memory=True,
        # )
    dl = [DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        drop_last=drop_last,
        num_workers=1,
        pin_memory=False,
    ) for dataset, bs in zip(ds, batchsize)]
    return dl

dls = get_data_loader(configer, aux_mode='eval', distributed=False, stage=1)

dl_iters = [iter(dl) for dl in dls]
np_im, np_lb = next(dl_iters[0])


mean=[0.3038, 0.3383, 0.3034]
std=[0.2071, 0.2088, 0.2090]
mean = torch.as_tensor(mean)[None, :, None, None].cuda()
std = torch.as_tensor(std)[None, :, None, None].cuda()

class TensorRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im, lb):
        # if np.random.random() < self.p:
        #     return im, lb

        assert im.shape[:2] == lb.shape[:2]
        return im.flip(1), lb.flip(0)

        
class TensorColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None):
        if not brightness is None and brightness >= 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        if not contrast is None and contrast >= 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        if not saturation is None and saturation >= 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, im, lb):
        assert im.shape[:2] == lb.shape[:2]

        if not self.brightness is None:
            rate = torch.tensor([1.25]).cuda()
            im = self.adj_brightness(im, rate)

        if not self.contrast is None:
            rate = torch.tensor([1.25]).cuda()
            im = self.adj_contrast(im, rate)

        if not self.saturation is None:
            rate = torch.tensor([1.25]).cuda()
            im = self.adj_saturation(im, rate)

        return im, lb

    def adj_saturation(self, im, rate):
        M = torch.tensor([
            [1 + 2 * rate, 1 - rate, 1 - rate],
            [1 - rate, 1 + 2 * rate, 1 - rate],
            [1 - rate, 1 - rate, 1 + 2 * rate]
        ]).float().cuda()

        shape = im.shape
        im_flat = im.reshape(-1, 3)
        im_flat_adjusted = torch.matmul(im_flat.float(), M)
        im_adjusted = im_flat_adjusted.reshape(shape) / 3

        # 剪裁和数据类型转换
        im_adjusted = torch.clamp(im_adjusted, 0, 255)#.to(torch.uint8)

        return im_adjusted

    def adj_brightness(self, im, rate):
        # table = torch.tensor([i * rate for i in range(256)]).clip(0, 255).byte()
        im = (im.float() * rate).clamp(0, 255).to(torch.uint8)
        return im

    def adj_contrast(self, im, rate):
        im = (74 + (im.float()-74) * rate).clamp(0, 255).to(torch.uint8)
        # table = torch.tensor([74 + (i - 74) * rate for i in range(256)]).clip(0, 255).byte()
        return im
    

randomHorizontalFlip = TensorRandomHorizontalFlip()
colorJitter = TensorColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
dls = get_data_loader(configer, aux_mode='train', distributed=False, stage=1)
dl_iters = [iter(dl) for dl in dls]
ims = []
lbs = []  
ids = []  
dl_iters = [iter(dl) for dl in dls]
im, lb = next(dl_iters[0])
# for j in range(0,len(dl_iters)):

#     try:
#         im, lb = next(dl_iters[j])
#         if not im.size()[0] == configer.get('dataset'+str(j+1), 'ims_per_gpu'):
#             raise StopIteration
#         while torch.min(lb) == 255:
#             im, lb = next(dl_iters[j])
#             if not im.size()[0] == configer.get('dataset'+str(j+1), 'ims_per_gpu'):
#                 raise StopIteration

        
#     except StopIteration:
#         dl_iters[j] = iter(dls[j])
#         im, lb = next(dl_iters[j])
#         while torch.min(lb) == 255:
#             im, lb = next(dl_iters[j])
    

#     # for idx in range(im.shape[0]):
#     #     this_im = im[idx].cuda()
#     #     this_lb = lb[idx].squeeze().cuda()
#     #     this_im, this_lb = randomHorizontalFlip(this_im, this_lb)
#     #     this_im, this_lb = colorJitter(this_im, this_lb)
    
#     #     ims.append(this_im[None].float())
#     #     lbs.append(this_lb[None][None].float())
#     ims.append(im)
#     lbs.append(lb)
#     ids.append(j*torch.ones(lb.shape[0], dtype=torch.int))
        

# im = torch.cat(ims, dim=0)
# lb = torch.cat(lbs, dim=0)
# print(im.shape)
# print(lb.shape)
# im = im.permute(0, 3, 1, 2)
# im = im.float().cuda()
# lb = lb.cuda()
# im = im.div_(255)

# im = im.sub_(mean).div_(std)
print(np_im.shape, np_lb.shape)
print(im.shape, lb.shape)
print(np_im[0])
print(im[0])
print(torch.sum(np_im[0] != im[0]))
print(torch.max(np_im[0] - im[0]), torch.min(np_im[0] - im[0]))
print(torch.sum(np_im[1] != im[1]))
print(torch.max(np_im[1] - im[1]), torch.min(np_im[0] - im[0]))
print(torch.sum(np_lb[0] != lb[0]))
print(torch.max(np_lb[0] - lb[0]))
