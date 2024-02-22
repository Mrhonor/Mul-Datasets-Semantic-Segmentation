#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

from lib.sampler import RepeatedDistSampler
import lib.transform_cv2 as T
import types
from random import shuffle

import torch
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator

class BaseDataset(Dataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(BaseDataset, self).__init__()
        # assert mode in ('train', 'eval', 'test')
        self.mode = mode
        self.trans_func = trans_func

        self.lb_map = None

        with open(annpath, 'r') as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths = [], []
        for pair in pairs:
            imgpth, lbpth = pair.split(',')
            self.img_paths.append(osp.join(dataroot, imgpth))
            self.lb_paths.append(osp.join(dataroot, lbpth))

        assert len(self.img_paths) == len(self.lb_paths)
        self.len = len(self.img_paths)

    def __getitem__(self, idx):
        impth, lbpth = self.img_paths[idx], self.lb_paths[idx]        
        label = self.get_label(lbpth)
        if not self.lb_map is None:
            label = self.lb_map[label]
        if self.mode == 'ret_path':
            return impth, label, lbpth

        img = self.get_image(impth)

        im_lb = dict(im=img, lb=label)
        if not self.trans_func is None:
            im_lb = self.trans_func(im_lb)
            
        im_lb = self.to_tensor(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        # self.to_tensor = T.ToTensorCUDA(
        #     mean=(0.3038, 0.3383, 0.3034), # city, rgb
        #     std=(0.2071, 0.2088, 0.2090),
        # )
        # img, label = self.to_tensor(img, label)
        # return img.copy(), label[None].copy()
        
        return img.detach(), label.unsqueeze(0).detach()
        # return img.detach()

    def get_label(self, lbpth):
        return cv2.imread(lbpth, 0)

    def get_image(self, impth):
        img = cv2.imread(impth)[:, :, ::-1]
        return img

    def __len__(self):
        return self.len

## read img without label
class BaseDatasetIm(Dataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(BaseDatasetIm, self).__init__()
        assert mode in ('train', 'eval', 'test')
        self.mode = mode
        self.trans_func = trans_func

        self.lb_map = None

        with open(annpath, 'r') as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths = [], []
        for pair in pairs:
            imgpth, lbpth = pair.split(',')
            self.img_paths.append(osp.join(dataroot, imgpth))
            self.lb_paths.append(osp.join(dataroot, lbpth))

        assert len(self.img_paths) == len(self.lb_paths)
        self.len = len(self.img_paths)

    def __getitem__(self, idx):
        impth, lbpth = self.img_paths[idx], self.lb_paths[idx]
        img, label = self.get_image(impth, lbpth)
        if not self.lb_map is None:
            label = self.lb_map[label]
        im_lb = dict(im=img, lb=label)
        if not self.trans_func is None:
            im_lb = self.trans_func(im_lb)
        im_lb = self.to_tensor(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        # return img.detach(), label.unsqueeze(0).detach()
        return img.detach()

    def get_image(self, impth, lbpth):
        img, label = cv2.imread(impth)[:, :, ::-1], cv2.imread(lbpth, 0)
        return img, label

    def __len__(self):
        return self.len

class ExternalInputIterator(object):
    def __init__(self, batch_size, dataroot, annpath, mode='train'):
        # 这一块其实与 dateset 都比较像
        self.batch_size = batch_size
        # self.num_instances = num_instances
        self.shuffled = False
        if mode == 'train':
            self.shuffled = True

        # self.img_seq_length = num_instances

        self.lb_map = None

        with open(annpath, 'r') as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths = [], []
        for pair in pairs:
            imgpth, lbpth = pair.split(',')
            self.img_paths.append(osp.join(dataroot, imgpth))
            self.lb_paths.append(osp.join(dataroot, lbpth))

        assert len(self.img_paths) == len(self.lb_paths)
        self.len = len(self.img_paths)

        # self.list_of_pids = list(images_dict.keys())
        self._num_classes = len(self.img_paths) #len(self.list_of_pids)
        self.all_indexs = list(range(self._num_classes))
        self.n = self.__len__()


    def __iter__(self):
        self.i = 0
        if self.shuffled:
            shuffle(self.all_indexs)
        return self

    def __len__(self):
        return len(self.all_indexs)

    @staticmethod
    def image_open(path):
        return np.fromfile(path, dtype=np.uint8)

    def __next__(self):
        # 如果溢出了，就终止
        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        batch_images = []
        batch_labels = []

        leave_num = self.n - self.i
        current_batch_size = min(self.batch_size, leave_num) # 保证最后一个 batch 不溢出
        for _ in range(current_batch_size):
            tmp_index = self.all_indexs[self.i]
            # p_id = self.list_of_pids[tmp_index]
            imp = self.img_paths[tmp_index]
            lbp = self.lb_paths[tmp_index]

            # # images = images_dict["images"] # 取 n 个
            # images = list(map(self.image_open, imp)) # 分别读取为 numpy，也可以是 batch
            # # 这一块都比较像，但是不作 transform 处理
            # label = list(map(self.image_open, lbp)) #images_dict["label"]

            batch_images.append(np.fromfile(imp, dtype=np.uint8))
            batch_labels.append(np.fromfile(lbp, dtype=np.uint8))

            self.i += 1

        # batch_data = []
        # for ins_i in range(self.num_instances):
        #     elem = []
        #     for batch_idx in range(current_batch_size):
        #         elem.append(batch_images[batch_idx][ins_i])
        #     batch_data.append(elem)
        # 其实这块也可以通过 tensor 的 permute 实现？我之前没有注意，大家有兴趣可以试试

        return batch_images, batch_labels

    # next = __next__
    # len = __len__
class ExternalInputIteratorMul(object):
    def __init__(self, batch_size, dataroot, annpath, mode='train'):
        # 这一块其实与 dateset 都比较像
        self.batch_size = batch_size
        # self.num_instances = num_instances
        self.shuffled = False
        if mode == 'train':
            self.shuffled = True

        # self.img_seq_length = num_instances

        self.lb_map = None
        self.img_paths, self.lb_paths = [], []
        for root, anp in zip(dataroot, annpath):
            with open(anp, 'r') as fr:
                pairs = fr.read().splitlines()
            self.img_path, self.lb_path = [], []
            for pair in pairs:
                imgpth, lbpth = pair.split(',')
                self.img_path.append(osp.join(root, imgpth))
                self.lb_path.append(osp.join(root, lbpth))

            assert len(self.img_path) == len(self.lb_path)
            self.img_paths.append(self.img_path)
            self.lb_paths.append(self.lb_path)
            
        self.len = len(self.img_paths)

        # self.list_of_pids = list(images_dict.keys())
        self._num_classes = len(self.img_paths) #len(self.list_of_pids)
        self.all_indexs = list(range(self._num_classes))
        self.n = self.__len__()


    def __iter__(self):
        self.i = 0
        if self.shuffled:
            shuffle(self.all_indexs)
        return self

    def __len__(self):
        return len(self.all_indexs)

    @staticmethod
    def image_open(path):
        return np.fromfile(path, dtype=np.uint8)

    def __next__(self):
        # 如果溢出了，就终止
        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        batch_images = []
        batch_labels = []

        leave_num = self.n - self.i
        current_batch_size = min(self.batch_size, leave_num) # 保证最后一个 batch 不溢出
        for _ in range(current_batch_size):
            tmp_index = self.all_indexs[self.i]
            # p_id = self.list_of_pids[tmp_index]
            imp = self.img_paths[tmp_index]
            lbp = self.lb_paths[tmp_index]

            # # images = images_dict["images"] # 取 n 个
            # images = list(map(self.image_open, imp)) # 分别读取为 numpy，也可以是 batch
            # # 这一块都比较像，但是不作 transform 处理
            # label = list(map(self.image_open, lbp)) #images_dict["label"]

            batch_images.append(np.fromfile(imp, dtype=np.uint8))
            batch_labels.append(np.fromfile(lbp, dtype=np.uint8))

            self.i += 1

        # batch_data = []
        # for ins_i in range(self.num_instances):
        #     elem = []
        #     for batch_idx in range(current_batch_size):
        #         elem.append(batch_images[batch_idx][ins_i])
        #     batch_data.append(elem)
        # 其实这块也可以通过 tensor 的 permute 实现？我之前没有注意，大家有兴趣可以试试

        return batch_images, batch_labels


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    ds = CityScapes('./data/', mode='eval')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
