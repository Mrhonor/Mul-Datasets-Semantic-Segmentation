import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

import lib.transform_cv2 as T
from lib.base_dataset import BaseDataset, BaseDatasetIm

class MultiSetReader(Dataset):
    def __init__(self, readers):
        super(MultiSetReader, self).__init__()
        self.readers = readers
        self.n_datasets = len(self.readers)
        self.single_readers_len = [len(reader) for reader in self.readers]
        
        self.total_len = 0
        for l in self.single_readers_len:
            self.total_len += l
        
        
    def __getitem__(self, idx):
        index = idx
        for i in range(self.n_datasets):
            if index < self.single_readers_len[i]:
                return self.readers[i][index], i
            else:
                index -= self.single_readers_len[i]
        
        raise Exception("MultiSetReader idx invalid")

            
    
    def __len__(self):
        return self.total_len
