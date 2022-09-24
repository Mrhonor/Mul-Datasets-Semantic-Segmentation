
from distutils.command.config import config
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

import lib.transform_cv2 as T
from lib.sampler import RepeatedDistSampler
from lib.cityscapes_cv2 import CityScapes, CityScapesIm
from lib.a2d2_lb_cv2 import A2D2Data
from lib.coco import CocoStuff
from lib.a2d2_city_dataset import A2D2CityScapes
from lib.CamVid_lb import CamVid



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


def get_data_loader(configer, distributed=True):
    mode = configer.get('aux_mode')
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
            num_workers=4,
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
            num_workers=4,
            pin_memory=False,
        ) for dataset, bs in zip(ds, batchsize)]
    return dl

def get_a2d2_city_loader(cfg_a2d2, cfg_city, mode='train', distributed=True):
    if mode == 'train':
        trans_func = TransformationTrain(cfg_city.scales, cfg_city.cropsize)
        batchsize = cfg_city.ims_per_gpu
        annpath_a2d2 = cfg_a2d2.train_im_anns
        annpath = cfg_city.train_im_anns
        
        shuffle = True
        drop_last = True
    elif mode == 'val':
        trans_func = TransformationVal()
        batchsize = cfg_city.eval_ims_per_gpu
        annpath_a2d2 = cfg_a2d2.val_im_anns
        annpath = cfg_city.val_im_anns
        shuffle = False
        drop_last = False
        
        # ## For precise bn
        # shuffle = True
        # drop_last = True

    ds_a2d2 = eval(cfg_a2d2.dataset)(cfg_a2d2.im_root, annpath_a2d2, trans_func=trans_func, mode=mode)
    ds_city = eval(cfg_city.dataset)(cfg_city.im_root, annpath, trans_func=trans_func, mode=mode)

    ds = A2D2CityScapes(ds_a2d2, ds_city)

    dl = DataLoader(
        ds,
        batch_size=batchsize,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=4,
        pin_memory=False,
    )
    return dl

