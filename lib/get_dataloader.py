
from distutils.command.config import config
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

import lib.transform_cv2 as T
from lib.sampler import RepeatedDistSampler
from lib.cityscapes_cv2 import CityScapes, CityScapesIm
from lib.a2d2_lb_cv2 import A2D2Data
from lib.a2d2_cv2 import A2D2Data_L
from lib.ADE20K import ade20k
from lib.ade2016_data import ade2016
from lib.bdd100k_data import Bdd100k
from lib.idd_cv2 import Idd
from lib.Mapi import Mapi
from lib.sunrgbd import Sunrgbd
from lib.coco_data import Coco_data
from lib.a2d2_city_dataset import A2D2CityScapes
from lib.CamVid_lb import CamVid
from lib.MultiSetReader import MultiSetReader
from lib.all_datasets_reader import AllDatasetsReader



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


class TransformationVal(object):

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        return dict(im=im, lb=lb)


def get_data_loader(configer, aux_mode='eval', distributed=True):
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
    elif mode == 'ret_path':
        trans_func = TransformationVal()
        batchsize = [1 for i in range(1, n_datasets+1)]
        annpath = []
        for i in range(1, n_datasets+1):
            annpath.append(configer.get('dataset'+str(i), 'train_im_anns'))
            
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
            num_workers=1,
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
            num_workers=1,
            pin_memory=False,
        ) for dataset, bs in zip(ds, batchsize)]
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

