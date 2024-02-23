
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

import lib.transform_cv2 as T
from lib.sampler import RepeatedDistSampler
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
import types
import collections
import numpy as np
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from lib.base_dataset import ExternalInputIterator, ExternalInputIteratorMul
from nvidia.dali import pipeline_def
import lib.cityscapes_cv2 as cityscapes_cv2
import lib.ADE20K as ADE20K
import lib.ade2016_data as ade2016_data
import lib.bdd100k_data as bdd100k_data
import lib.idd_cv2 as idd_cv2
import lib.Mapi as Mapi 
import lib.sunrgbd as sunrgbd
import lib.coco_data as coco_data
from nvidia.dali.plugin.pytorch import DALIClassificationIterator as PyTorchIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy

class TransformationTrain(object):

    def __init__(self, scales, cropsize):
        self.trans_func = T.Compose([
            T.RandomResizedCrop(scales, cropsize),
            T.RandomHorizontalFlip(),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
        ])

    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)
        return im_lb

# class TransformationTrain(object):

#     def __init__(self, scales, cropsize):
#         self.trans_func = T.Compose([
#             T.RandomResizedCrop_Flip_ColorJitterGPU(scales, cropsize, 
#                 p=0.5,                
#                 brightness=0.4,
#                 contrast=0.4,
#                 saturation=0.4),
#         ])

#     def __call__(self, im_lb):
#         im_lb = self.trans_func(im_lb)
#         return im_lb

class TransformationVal(object):

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        return dict(im=im, lb=lb)


def get_data_loader(configer, aux_mode='eval', distributed=True, stage=None):
    mode = aux_mode
    n_datasets = configer.get('n_datasets')
    max_iter = configer.get('lr', 'max_iter')
    
    if mode == 'train':
        scales = configer.get('train', 'scales')
        cropsize = configer.get('train', 'cropsize')
        trans_func = TransformationTrain(scales, cropsize)
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
    elif mode == 'eval':
        trans_func = TransformationVal()
        batchsize = [configer.get('dataset'+str(i), 'eval_ims_per_gpu') for i in range(1, n_datasets+1)]
        annpath = [configer.get('dataset'+str(i), 'val_im_anns') for i in range(1, n_datasets+1)]
        imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
        data_reader = [configer.get('dataset'+str(i), 'data_reader') for i in range(1, n_datasets+1)]
        
        shuffle = False
        drop_last = False
    elif mode == 'ret_path':
        trans_func = TransformationVal()
        batchsize = [1 for i in range(1, n_datasets+1)]
        if stage != None:
            annpath = [configer.get('dataset'+str(i), 'train_im_anns').replace('.txt', f'_{stage}.txt') for i in range(1, n_datasets+1)]
            print(annpath)
        else:
            annpath = [configer.get('dataset'+str(i), 'train_im_anns') for i in range(1, n_datasets+1)]
            
        imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
        data_reader = [configer.get('dataset'+str(i), 'data_reader') for i in range(1, n_datasets+1)]
        
        shuffle = False
        drop_last = False
        

    ds = [eval(reader)(root, path, trans_func=trans_func, mode=mode)
          for reader, root, path in zip(data_reader, imroot, annpath)]
    # ds = [eval(reader)(root, path, trans_func=trans_func, mode=mode)
    #       for reader, root, path in zip(data_reader, imroot, annpath)]

    if distributed:
        assert dist.is_available(), "dist should be initialzed"
        if mode == 'train' and stage != 2:
            assert not max_iter is None
            n_train_imgs = [ims_per_gpu * dist.get_world_size() * max_iter for ims_per_gpu in batchsize]
            sampler = [RepeatedDistSampler(dataset, n_train_img, shuffle=shuffle) for n_train_img, dataset in zip(n_train_imgs, ds)] 
        else:
            sampler = [torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=shuffle) for dataset in ds] 
            
        batchsampler = [torch.utils.data.sampler.BatchSampler(
            samp, bs, drop_last=drop_last
        ) for samp, bs in zip(sampler, batchsize)]
        dl = [DataLoader(
            dataset,
            batch_sampler=batchsamp,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4
        ) for dataset, batchsamp in zip(ds, batchsampler)]
    else:
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
        dl = []
        for idx in range(len(ds)):
            dataset = ds[idx]
            bs = batchsize[idx]
             
            if idx == 1:
                dl.append(DataLoader(
                    dataset,
                    batch_size=bs,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    num_workers=3,
                    pin_memory=True,
                    ))
            else:
                dl.append(DataLoader(
                    dataset,
                    batch_size=bs,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    num_workers=3,
                    pin_memory=True,
                    # prefetch_factor=4
                    ))
    return dl

def get_data_loader_all_in_one(configer, aux_mode='eval', distributed=True):
    mode = aux_mode
    n_datasets = configer.get('n_datasets')
    max_iter = configer.get('lr', 'max_iter')
    
    if mode == 'train':
        scales = configer.get('train', 'scales')
        cropsize = configer.get('train', 'cropsize')
        trans_func = TransformationTrain(scales, cropsize)
        batchsize = 0
        for i in range(1, n_datasets+1):
            batchsize += configer.get('dataset'+str(i), 'ims_per_gpu')
        annpath = "datasets/all/train.txt"
        imroot = "/home1/marong/datasets/"
        data_reader = "AllDatasetsReader"
        
        shuffle = True
        drop_last = True
    else:
        trans_func = TransformationVal()
        batchsize = 0
        for i in range(1, n_datasets+1):
            batchsize += configer.get('dataset'+str(i), 'eval_ims_per_gpu')
        annpath = "datasets/all/val.txt"
        imroot = "/home1/marong/datasets/"
        data_reader = "AllDatasetsReader"
        
        shuffle = False
        drop_last = False

    ds = eval(data_reader)(imroot, annpath, trans_func=trans_func, mode=mode)
          
    # ds = [eval(reader)(root, path, trans_func=trans_func, mode=mode)
    #       for reader, root, path in zip(data_reader, imroot, annpath)]

    if distributed:
        assert dist.is_available(), "dist should be initialzed"
        if mode == 'train':
            assert not max_iter is None
            n_train_imgs = batchsize * dist.get_world_size() * max_iter
            sampler = RepeatedDistSampler(ds, n_train_imgs, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, shuffle=shuffle)
            
        batchsampler = torch.utils.data.sampler.BatchSampler(
            sampler, batchsize, drop_last=drop_last
        )
        dl = DataLoader(
            ds,
            batch_sampler=batchsampler,
            num_workers=2,
            pin_memory=False,
        )
    else:
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
        dl = DataLoader(
            ds,
            batch_size=batchsize,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=2,
            pin_memory=False,
        )
    return dl


def get_single_data_loader(configer, aux_mode='eval', distributed=True):
    mode = aux_mode
    n_datasets = configer.get('n_datasets')
    max_iter = configer.get('lr', 'max_iter')
    
    if mode == 'train':
        scales = configer.get('train', 'scales')
        cropsize = configer.get('train', 'cropsize')
        trans_func = TransformationTrain(scales, cropsize)
        batchsize = [configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, n_datasets+1)]
        annpath = [configer.get('dataset'+str(i), 'train_im_anns') for i in range(1, n_datasets+1)]
        imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
        data_reader = [configer.get('dataset'+str(i), 'data_reader') for i in range(1, n_datasets+1)]
        
        shuffle = True
        drop_last = True
    elif mode == 'eval':
        trans_func = TransformationVal()
        batchsize = [configer.get('dataset'+str(i), 'eval_ims_per_gpu') for i in range(1, n_datasets+1)]
        annpath = [configer.get('dataset'+str(i), 'val_im_anns') for i in range(1, n_datasets+1)]
        imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
        data_reader = [configer.get('dataset'+str(i), 'data_reader') for i in range(1, n_datasets+1)]
        
        shuffle = False
        drop_last = False
        

    ds = [eval(reader)(root, path, trans_func=trans_func, mode=mode)
          for reader, root, path in zip(data_reader, imroot, annpath)]

    Mds = MultiSetReader(ds)

    total_batchsize = 0
    for ims_per_gpu in batchsize:
        total_batchsize += ims_per_gpu

    if distributed:
        assert dist.is_available(), "dist should be initialzed"

            
        if mode == 'train':
            assert not max_iter is None
            
            n_train_imgs = total_batchsize * dist.get_world_size() * max_iter

            sampler = RepeatedDistSampler(Mds, n_train_imgs, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(Mds, shuffle=shuffle)
            
        batchsampler = torch.utils.data.sampler.BatchSampler(
            sampler, total_batchsize, drop_last=drop_last)
        dl = DataLoader(
            Mds,
            batch_sampler=batchsampler,
            num_workers=1,
            pin_memory=False)
    else:

        dl = DataLoader(
            Mds,
            batch_size=total_batchsize,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=1,
            pin_memory=False,
        )
    return dl


def get_city_loader(configer, aux_mode='eval', distributed=True):
    mode = aux_mode
    n_datasets = configer.get('n_datasets')
    max_iter = configer.get('lr', 'max_iter')
    
    if mode == 'train':
        scales = configer.get('train', 'scales')
        cropsize = configer.get('train', 'cropsize')
        trans_func = TransformationTrain(scales, cropsize)
        batchsize = [configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, n_datasets+1)]
        annpath = [configer.get('dataset'+str(i), 'train_im_anns') for i in range(1, n_datasets+1)]
        imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
        data_reader = [configer.get('dataset'+str(i), 'data_reader') for i in range(1, n_datasets+1)]
        
        shuffle = True
        drop_last = True
    elif mode == 'eval':
        trans_func = TransformationVal()
        batchsize = [configer.get('dataset'+str(i), 'eval_ims_per_gpu') for i in range(1, n_datasets+1)]
        annpath = [configer.get('dataset'+str(i), 'val_im_anns') for i in range(1, n_datasets+1)]
        imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
        data_reader = [configer.get('dataset'+str(i), 'data_reader') for i in range(1, n_datasets+1)]
        
        shuffle = False
        drop_last = False
        

    ds = [eval(reader)(root, path, trans_func=trans_func, mode='train')
          for reader, root, path in zip(data_reader, imroot, annpath)]

    if distributed:
        assert dist.is_available(), "dist should be initialzed"
        if mode == 'train':
            assert not max_iter is None
            n_train_imgs = [ims_per_gpu * dist.get_world_size() * max_iter for ims_per_gpu in batchsize]
            sampler = [RepeatedDistSampler(dataset, n_train_img, shuffle=shuffle) for n_train_img, dataset in zip(n_train_imgs, ds)] 
        else:
            sampler = [torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=shuffle) for dataset in ds] 
            
        batchsampler = [torch.utils.data.sampler.BatchSampler(
            samp, bs, drop_last=drop_last
        ) for samp, bs in zip(sampler, batchsize)]
        dl = [DataLoader(
            dataset,
            batch_sampler=batchsamp,
            num_workers=8,
            pin_memory=False,
        ) for dataset, batchsamp in zip(ds, batchsampler)]
    else:
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
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=8,
            pin_memory=False,
        ) for dataset, bs in zip(ds, batchsize)]
    return dl[0]

# def ExternalSourcePipeline(batch_size, num_threads, device_id, external_data, lb_map=None, scales=(0.5, 1.), size=(768, 768), p=0.5, brightness=0.4, contrast=0.4, saturation=0.4):
    
#     pipe = Pipeline(batch_size, num_threads, device_id)
#     crop_h, crop_w = size
#     if not brightness is None and brightness >= 0:
#         brightness = [max(1-brightness, 0), 1+brightness]
#     if not contrast is None and contrast >= 0:
#         contrast = [max(1-contrast, 0), 1+contrast]
#     if not saturation is None and saturation >= 0:
#         saturation = [max(1-saturation, 0), 1+saturation]

#     mean=[0.3257, 0.3690, 0.3223] # city, rgb
#     std=[0.2112, 0.2148, 0.2115]
    
#     with pipe:
#         jpegs, labels = fn.external_source(source=external_data, num_outputs=2, dtype=types.UINT8)
#         images = fn.decoders.image(jpegs, device="cpu")
#         labels = fn.decoders.image(labels, device="cpu", output_type=types.GRAY)
#         # images = fn.random_resized_crop()
#         # for i in range(len(images)):
#         # print(fn.peek_image_shape(labels))
#         # shape = fn.peek_image_shape(labels)
#         shape = fn.shapes(images)
#         H = shape[0]
#         W = shape[1]
#         # print(shape)
#         scale = fn.random.uniform(range=(min(scales), max(scales)))
        
#         # if np.min([H, W]) < 1080:
#         # min_scale = 1080 / fn.reductions.min(shape[0:2])
#         # scale = fn.reductions.min([min_scale, scale])
#         new_size = shape[0:2]*scale
#         # print(scale)
#         sh = fn.random.uniform(range=(0,1))
#         sw = fn.random.uniform(range=(0,1))
#         # h = H*scale
#         # h2 = h-crop_h
#         # sh = sh * (H*scale - crop_h)
#         # sw = sw * (W*scale - crop_w)

#         if np.random.random() < p:
#             mirror = 0
#         else:
#             mirror = 1
#         images, labels = images.gpu(), labels.gpu()
#         # images = fn.resize(images, size=new_size, interp_type=types.INTERP_LINEAR)
#         # labels = fn.resize(labels, size=new_size, interp_type=types.INTERP_NN)
#         images = fn.resize_crop_mirror(images, crop=size, crop_pos_x=sw, crop_pos_y=sh, mirror=mirror, size=new_size, interp_type=types.INTERP_LINEAR)
#         labels = fn.resize_crop_mirror(labels, antialias=False, crop=size, crop_pos_x=sw, crop_pos_y=sh, mirror=mirror, size=new_size, interp_type=types.INTERP_NN, mag_filter=types.INTERP_NN, min_filter=types.INTERP_NN)


#         # do_flip = fn.random.coin_flip(probability=0.5, dtype=types.BOOL)
#         # images_true_branch, images_false_branch = fn._conditional.split(images, predicate=do_flip)
#         # labels_true_branch, labels_false_branch = fn._conditional.split(labels, predicate=do_flip)

#         # images_true_branch = fn.resize_crop_mirror(images_true_branch, crop=size, crop_pos_x=sw, crop_pos_y=sh, mirror=1, resize_x=scale, resize_y=scale)
#         # labels_true_branch = fn.resize_crop_mirror(labels_true_branch, crop=size, crop_pos_x=sw, crop_pos_y=sh, mirror=1, resize_x=scale, resize_y=scale, mag_filter=types.INTERP_NN, min_filter=types.INTERP_NN)

#         # images_false_branch = fn.resize_crop_mirror(images_false_branch, crop=size, crop_pos_x=sw, crop_pos_y=sh, mirror=0, resize_x=scale, resize_y=scale)
#         # labels_false_branch = fn.resize_crop_mirror(labels_false_branch, crop=size, crop_pos_x=sw, crop_pos_y=sh, mirror=0, resize_x=scale, resize_y=scale, mag_filter=types.INTERP_NN, min_filter=types.INTERP_NN)

#         # images = fn._conditional.merge(images_true_branch, images_false_branch, predicate=do_flip)
#         # labels = fn._conditional.merge(labels_true_branch, labels_false_branch, predicate=do_flip)

#         brightness_rate = fn.random.uniform(range=(min(brightness), max(brightness)))
#         contrast_rate = fn.random.uniform(range=(min(contrast), max(contrast)))
#         saturation_rate = fn.random.uniform(range=(min(saturation), max(saturation)))
#         images = fn.brightness_contrast(images, brightness=brightness_rate, contrast_center=74, contrast=contrast_rate)
#         images = fn.saturation(images, saturation=saturation_rate)
#         images = fn.resize(images, resize_x=240, resize_y=240)
#         images = fn.cast(images, dtype=types.UINT8)
#         # images = fn.normalize(images, scale=1/255)
#         # images = fn.normalize(images, mean=mean, stddev=std)
        
#         # if lb_map is not None: 
#         # print(lb_map)
#         labels = fn.lookup_table(labels, keys=list(range(len(lb_map))), values=list(lb_map), default_value=255)
#         labels = fn.cast(labels, dtype=types.UINT8)
#         pipe.set_outputs(images, labels)
#     return pipe

def ExternalSourcePipeline(batch_size, num_threads, device_id, external_data, lb_map=None, mode='train', scales=(0.5, 1.), size=(768, 768), p=0.5, brightness=0.4, contrast=0.4, saturation=0.4):
    
    pipe = Pipeline(batch_size, num_threads, device_id, prefetch_queue_depth=4)
    crop_h, crop_w = size
    if not brightness is None and brightness >= 0:
        brightness = [max(1-brightness, 0), 1+brightness]
    if not contrast is None and contrast >= 0:
        contrast = [max(1-contrast, 0), 1+contrast]
    if not saturation is None and saturation >= 0:
        saturation = [max(1-saturation, 0), 1+saturation]

    # mean=[0.3257, 0.3690, 0.3223] # city, rgb
    # std=[0.2112, 0.2148, 0.2115]
    MEAN = np.asarray([0.3257, 0.3690, 0.3223])[None, None, :]
    STD = np.asarray([0.2112, 0.2148, 0.2115])[None, None, :]
    SCALE = 1 / 255.
    with pipe:
        jpegs, labels = fn.external_source(source=external_data, num_outputs=2, dtype=types.UINT8)
        images = fn.decoders.image(jpegs, device="mixed")
        labels = fn.decoders.image(labels, device="mixed", output_type=types.GRAY)
        # images = fn.random_resized_crop()
        # for i in range(len(images)):
        # print(fn.peek_image_shape(labels))
        # shape = fn.peek_image_shape(labels)
        # images = images.gpu()
        # labels = labels.gpu()
        if mode == 'train':
            images = fn.random_resized_crop(images, interp_type=types.INTERP_LINEAR, size=size, seed=1234)
            labels = fn.random_resized_crop(labels, antialias=False, interp_type=types.INTERP_NN, size=size, seed=1234)

            brightness_rate = fn.random.uniform(range=(min(brightness), max(brightness)))
            contrast_rate = fn.random.uniform(range=(min(contrast), max(contrast)))
            saturation_rate = fn.random.uniform(range=(min(saturation), max(saturation)))
            images = fn.brightness_contrast(images, brightness=brightness_rate, contrast_center=74, contrast=contrast_rate)
            images = fn.saturation(images, saturation=saturation_rate)

        # images = fn.cast(images, dtype=types.FLOAT)
        # images = fn.normalize(images, scale=1/255)
        # images = fn.normalize(images, axes=[0,1], mean=mean, stddev=std)
        images = fn.normalize(
            images,
            mean=MEAN / SCALE,
            stddev=STD,
            scale=SCALE,
            dtype=types.FLOAT,
        )
        
        # if lb_map is not None: 
        # print(lb_map)
        labels = fn.lookup_table(labels, keys=list(range(len(lb_map))), values=list(lb_map), default_value=255)
        labels = fn.cast(labels, dtype=types.UINT8)
        pipe.set_outputs(images, labels)
    return pipe

def get_DALI_data_loader(configer, aux_mode='eval', stage=None):
    mode = aux_mode
    n_datasets = configer.get('n_datasets')
    max_iter = configer.get('lr', 'max_iter')
    
    if mode == 'train':
        scales = configer.get('train', 'scales')
        cropsize = configer.get('train', 'cropsize')
        
        if stage != None:
            annpath = [configer.get('dataset'+str(i), 'train_im_anns').replace('.txt', f'_{stage}.txt') for i in range(1, n_datasets+1)]
            print(annpath)
            batchsize = [configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, n_datasets+1)]
        else:
            annpath = [configer.get('dataset'+str(i), 'train_im_anns') for i in range(1, n_datasets+1)]
            batchsize = [configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, n_datasets+1)]
        imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
        data_reader = [configer.get('dataset'+str(i), 'dataset_name') for i in range(1, n_datasets+1)]
        
        shuffle = True
        drop_last = True
    elif mode == 'eval':
        
        batchsize = [configer.get('dataset'+str(i), 'eval_ims_per_gpu') for i in range(1, n_datasets+1)]
        annpath = [configer.get('dataset'+str(i), 'val_im_anns') for i in range(1, n_datasets+1)]
        imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
        data_reader = [configer.get('dataset'+str(i), 'dataset_name') for i in range(1, n_datasets+1)]
        
        shuffle = False
        drop_last = False
    elif mode == 'ret_path':
        
        batchsize = [1 for i in range(1, n_datasets+1)]
        if stage != None:
            annpath = [configer.get('dataset'+str(i), 'train_im_anns').replace('.txt', f'_{stage}.txt') for i in range(1, n_datasets+1)]
            print(annpath)
        else:
            annpath = [configer.get('dataset'+str(i), 'train_im_anns') for i in range(1, n_datasets+1)]
            
        imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
        data_reader = [configer.get('dataset'+str(i), 'dataset_name') for i in range(1, n_datasets+1)]
        
        shuffle = False
        drop_last = False
        

    ds = [ExternalInputIterator(bs, root, path, mode=mode)
          for bs, root, path in zip(batchsize, imroot, annpath)]
    
    pipes = []
    for i, data_name in enumerate(data_reader):
        label_info = eval(data_name).labels_info
        lb_map = np.arange(256).astype(np.uint8)
       
        for el in label_info:
            lb_map[el['id']] = el['trainId']
        pipe = ExternalSourcePipeline(batch_size=batchsize[i], num_threads=8, device_id=0, external_data=ds[i], lb_map=lb_map, mode=mode)
        pipes.append(pipe)

    if mode == 'train':
        dl = [PyTorchIterator(pipe, last_batch_padded=True, last_batch_policy=LastBatchPolicy.DROP) for pipe in pipes]
    else:
        dl = [PyTorchIterator(pipe, last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL) for pipe in pipes]

    return dl

def ExternalSourcePipelineMul(batch_size, num_threads, device_id, external_data, mode='train', scales=(0.5, 1.), size=(768, 768), p=0.5, brightness=0.4, contrast=0.4, saturation=0.4):
    
    pipe = Pipeline(batch_size, num_threads, device_id, prefetch_queue_depth=4)
    crop_h, crop_w = size
    if not brightness is None and brightness >= 0:
        brightness = [max(1-brightness, 0), 1+brightness]
    if not contrast is None and contrast >= 0:
        contrast = [max(1-contrast, 0), 1+contrast]
    if not saturation is None and saturation >= 0:
        saturation = [max(1-saturation, 0), 1+saturation]

    # mean=[0.3257, 0.3690, 0.3223] # city, rgb
    # std=[0.2112, 0.2148, 0.2115]
    MEAN = np.asarray([0.3257, 0.3690, 0.3223])[None, None, :]
    STD = np.asarray([0.2112, 0.2148, 0.2115])[None, None, :]
    SCALE = 1 / 255.
    with pipe:
        jpegs, labels = fn.external_source(source=external_data, num_outputs=2, dtype=types.UINT8)
        images = fn.decoders.image(jpegs, device="mixed")
        labels = fn.decoders.image(labels, device="mixed", output_type=types.GRAY)
        # images = fn.random_resized_crop()
        # for i in range(len(images)):
        # print(fn.peek_image_shape(labels))
        # shape = fn.peek_image_shape(labels)
        # images = images.gpu()
        # labels = labels.gpu()
        if mode == 'train':
            images = fn.random_resized_crop(images, interp_type=types.INTERP_LINEAR, size=size, seed=1234)
            labels = fn.random_resized_crop(labels, antialias=False, interp_type=types.INTERP_NN, size=size, seed=1234)

            brightness_rate = fn.random.uniform(range=(min(brightness), max(brightness)))
            contrast_rate = fn.random.uniform(range=(min(contrast), max(contrast)))
            saturation_rate = fn.random.uniform(range=(min(saturation), max(saturation)))
            images = fn.brightness_contrast(images, brightness=brightness_rate, contrast_center=74, contrast=contrast_rate)
            images = fn.saturation(images, saturation=saturation_rate)

        # images = fn.cast(images, dtype=types.FLOAT)
        # images = fn.normalize(images, scale=1/255)
        # images = fn.normalize(images, axes=[0,1], mean=mean, stddev=std)
        images = fn.normalize(
            images,
            mean=MEAN / SCALE,
            stddev=STD,
            scale=SCALE,
            dtype=types.FLOAT,
        )
        
        # if lb_map is not None: 
        # print(lb_map)
        # labels = fn.lookup_table(labels, keys=list(range(len(lb_map))), values=list(lb_map), default_value=255)
        labels = fn.cast(labels, dtype=types.UINT8)
        pipe.set_outputs(images, labels)
    return pipe

def get_DALI_data_loaderMul(configer, aux_mode='eval', stage=None):
    mode = aux_mode
    n_datasets = configer.get('n_datasets')
    max_iter = configer.get('lr', 'max_iter')
    
    if mode == 'train':
        scales = configer.get('train', 'scales')
        cropsize = configer.get('train', 'cropsize')
        
        if stage != None:
            annpath = [configer.get('dataset'+str(i), 'train_im_anns').replace('.txt', f'_{stage}.txt') for i in range(1, n_datasets+1)]
            print(annpath)
            batchsize = [configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, n_datasets+1)]
        else:
            annpath = [configer.get('dataset'+str(i), 'train_im_anns') for i in range(1, n_datasets+1)]
            batchsize = [configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, n_datasets+1)]
        imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
        data_reader = [configer.get('dataset'+str(i), 'dataset_name') for i in range(1, n_datasets+1)]
        
        shuffle = True
        drop_last = True
    elif mode == 'eval':
        
        batchsize = [configer.get('dataset'+str(i), 'eval_ims_per_gpu') for i in range(1, n_datasets+1)]
        annpath = [configer.get('dataset'+str(i), 'val_im_anns') for i in range(1, n_datasets+1)]
        imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
        data_reader = [configer.get('dataset'+str(i), 'dataset_name') for i in range(1, n_datasets+1)]
        
        shuffle = False
        drop_last = False
    elif mode == 'ret_path':
        
        batchsize = [1 for i in range(1, n_datasets+1)]
        if stage != None:
            annpath = [configer.get('dataset'+str(i), 'train_im_anns').replace('.txt', f'_{stage}.txt') for i in range(1, n_datasets+1)]
            print(annpath)
        else:
            annpath = [configer.get('dataset'+str(i), 'train_im_anns') for i in range(1, n_datasets+1)]
            
        imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
        data_reader = [configer.get('dataset'+str(i), 'dataset_name') for i in range(1, n_datasets+1)]
        
        shuffle = False
        drop_last = False
        

    ds = ExternalInputIteratorMul(batchsize, imroot, annpath, mode=mode)
         
    total_bs = 0
    for bs in batchsize:
        total_bs += bs
    pipe = ExternalSourcePipelineMul(batch_size=total_bs, num_threads=64, device_id=0, external_data=ds, mode=mode)
    lb_maps = []
    for i, data_name in enumerate(data_reader):
        label_info = eval(data_name).labels_info
        lb_map = np.arange(256).astype(np.uint8)
       
        for el in label_info:
            lb_map[el['id']] = el['trainId']
        
        lb_maps.append(lb_map)
        
        

    if mode == 'train':
        dl = PyTorchIterator(pipe, last_batch_padded=True, last_batch_policy=LastBatchPolicy.DROP) 
    else:
        dl = PyTorchIterator(pipe, last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL)

    return dl, lb_maps


if __name__ == "__main__":

    batch_size = 8
    dataroot = "/cpfs01/projects-HDD/pujianxiangmuzu_HDD/public/mr"
    ann_path = "datasets/Cityscapes/val.txt"
    labels_info = [
        {"hasInstances": False, "category": "void", "catid": 0, "name": "unlabeled", "ignoreInEval": True, "id": 0, "color": [0, 0, 0], "trainId": 255},
        {"hasInstances": False, "category": "void", "catid": 0, "name": "ego vehicle", "ignoreInEval": True, "id": 1, "color": [0, 0, 0], "trainId": 255},
        {"hasInstances": False, "category": "void", "catid": 0, "name": "rectification border", "ignoreInEval": True, "id": 2, "color": [0, 0, 0], "trainId": 255},
        {"hasInstances": False, "category": "void", "catid": 0, "name": "out of roi", "ignoreInEval": True, "id": 3, "color": [0, 0, 0], "trainId": 255},
        {"hasInstances": False, "category": "void", "catid": 0, "name": "static", "ignoreInEval": True, "id": 4, "color": [0, 0, 0], "trainId": 255},
        {"hasInstances": False, "category": "void", "catid": 0, "name": "dynamic", "ignoreInEval": True, "id": 5, "color": [111, 74, 0], "trainId": 255},
        {"hasInstances": False, "category": "void", "catid": 0, "name": "ground", "ignoreInEval": True, "id": 6, "color": [81, 0, 81], "trainId": 255},
        {"hasInstances": False, "category": "flat", "catid": 1, "name": "road", "ignoreInEval": False, "id": 7, "color": [128, 64, 128], "trainId": 0},
        {"hasInstances": False, "category": "flat", "catid": 1, "name": "sidewalk", "ignoreInEval": False, "id": 8, "color": [244, 35, 232], "trainId": 1},
        {"hasInstances": False, "category": "flat", "catid": 1, "name": "parking", "ignoreInEval": True, "id": 9, "color": [250, 170, 160], "trainId": 255},
        {"hasInstances": False, "category": "flat", "catid": 1, "name": "rail track", "ignoreInEval": True, "id": 10, "color": [230, 150, 140], "trainId": 255},
        {"hasInstances": False, "category": "construction", "catid": 2, "name": "building", "ignoreInEval": False, "id": 11, "color": [70, 70, 70], "trainId": 2},
        {"hasInstances": False, "category": "construction", "catid": 2, "name": "wall", "ignoreInEval": False, "id": 12, "color": [102, 102, 156], "trainId": 3},
        {"hasInstances": False, "category": "construction", "catid": 2, "name": "fence", "ignoreInEval": False, "id": 13, "color": [190, 153, 153], "trainId": 4},
        {"hasInstances": False, "category": "construction", "catid": 2, "name": "guard rail", "ignoreInEval": True, "id": 14, "color": [180, 165, 180], "trainId": 255},
        {"hasInstances": False, "category": "construction", "catid": 2, "name": "bridge", "ignoreInEval": True, "id": 15, "color": [150, 100, 100], "trainId": 255},
        {"hasInstances": False, "category": "construction", "catid": 2, "name": "tunnel", "ignoreInEval": True, "id": 16, "color": [150, 120, 90], "trainId": 255},
        {"hasInstances": False, "category": "object", "catid": 3, "name": "pole", "ignoreInEval": False, "id": 17, "color": [153, 153, 153], "trainId": 5},
        {"hasInstances": False, "category": "object", "catid": 3, "name": "polegroup", "ignoreInEval": True, "id": 18, "color": [153, 153, 153], "trainId": 255},
        {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic light", "ignoreInEval": False, "id": 19, "color": [250, 170, 30], "trainId": 6},
        {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic sign", "ignoreInEval": False, "id": 20, "color": [220, 220, 0], "trainId": 7},
        {"hasInstances": False, "category": "nature", "catid": 4, "name": "vegetation", "ignoreInEval": False, "id": 21, "color": [107, 142, 35], "trainId": 8},
        {"hasInstances": False, "category": "nature", "catid": 4, "name": "terrain", "ignoreInEval": False, "id": 22, "color": [152, 251, 152], "trainId": 9},
        {"hasInstances": False, "category": "sky", "catid": 5, "name": "sky", "ignoreInEval": False, "id": 23, "color": [70, 130, 180], "trainId": 10},
        {"hasInstances": True, "category": "human", "catid": 6, "name": "person", "ignoreInEval": False, "id": 24, "color": [220, 20, 60], "trainId": 11},
        {"hasInstances": True, "category": "human", "catid": 6, "name": "rider", "ignoreInEval": False, "id": 25, "color": [255, 0, 0], "trainId": 12},
        {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "car", "ignoreInEval": False, "id": 26, "color": [0, 0, 142], "trainId": 13},
        {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "truck", "ignoreInEval": False, "id": 27, "color": [0, 0, 70], "trainId": 14},
        {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bus", "ignoreInEval": False, "id": 28, "color": [0, 60, 100], "trainId": 15},
        {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "caravan", "ignoreInEval": True, "id": 29, "color": [0, 0, 90], "trainId": 255},
        {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "trailer", "ignoreInEval": True, "id": 30, "color": [0, 0, 110], "trainId": 255},
        {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "train", "ignoreInEval": False, "id": 31, "color": [0, 80, 100], "trainId": 16},
        {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "motorcycle", "ignoreInEval": False, "id": 32, "color": [0, 0, 230], "trainId": 17},
        {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bicycle", "ignoreInEval": False, "id": 33, "color": [119, 11, 32], "trainId": 18},
        {"hasInstances": False, "category": "vehicle", "catid": 7, "name": "license plate", "ignoreInEval": True, "id": -1, "color": [0, 0, 142], "trainId": 255}
    ]
    lb_map = np.arange(256).astype(np.uint8)

        
    for el in labels_info:
        # if mode=='train' and el['trainId'] == 255:
        #     self.lb_map[el['id']] = 19
        # else:
        lb_map[el['id']] = el['trainId']

    eii = ExternalInputIterator(batch_size, dataroot, ann_path, mode='train')
    pipe = ExternalSourcePipeline(batch_size=batch_size, num_threads=2, device_id=0, external_data=eii, lb_map=lb_map)
    pii = PyTorchIterator(pipe, last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL)
    dl_iter = iter(pii)
    for i in [1]:
        data = next(dl_iter)
        print("epoch: {}, iter {}, real batch size: {}".format(0, i, len(data[0]["data"])))
        import matplotlib.pyplot as plt
        from nvidia.dali import tensors


        def display(output):
            data_idx = 0
            fig, axes = plt.subplots(len(output['data']), 2, figsize=(15, 15))
            if len(output) == 1:
                axes = [axes]
            for i, out in enumerate(output['data']):
                img = out.cpu().numpy() #if isinstance(out, tensors.TensorCPU) else out.as_cpu()
                axes[i, 0].imshow(img)
                axes[i, 0].axis("off")
                lb = output['label'][i].cpu().numpy() #if isinstance(output['label'][i], tensors.TensorCPU) else output['label'][i].as_cpu()
                axes[i, 1].imshow(lb)
                axes[i, 1].axis("off")
            plt.savefig('res.png')
        # print(data)
        display(data[0]) 
        exit(0)
    pii.reset()