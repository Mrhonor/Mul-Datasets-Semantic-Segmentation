# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
import sys
import glob
import logging
import torch
import torchnvjpeg
import torchvision.transforms.functional as F
import random
import cv2
import threading
import numpy as np
from multiprocessing.pool import ThreadPool
import torch.distributed as dist
from pathlib import Path
import math
import lib.transform_cv2 as T

# Bring module folders from the samples directory into our path so that
# we can import modules from it.
samples_dir = Path(os.path.abspath(__file__)).parents[2]  # samples/
sys.path.insert(0, os.path.join(samples_dir, ""))

from lib.batch import Batch  # noqa: E402

# docs_tag: begin_init_imagebatchdecoder_pytorch

class LabelReaderThread(threading.Thread):
    def __init__(self, lb_path, lb_map=None, isLabel=True):
        super(LabelReaderThread, self).__init__()
        self.lb_path = lb_path
        self.lb_image = None
        self.lb_map=lb_map
        self.isLabel = isLabel
        

    def run(self):
        # 在这里执行图像读取操作
        if self.isLabel:
            self.lb_image = cv2.imread(self.lb_path, 0)
            if not self.lb_map is None:
                self.lb_image = self.lb_map[self.lb_image]
        else:
            self.lb_image = cv2.imread(self.lb_path)[:, :, ::-1]

            
class IMLbReaderThread(threading.Thread):
    def __init__(self, im_lb_path, lb_map=None, trans_func=None):
        super(IMLbReaderThread, self).__init__()
        self.im_lb_path = im_lb_path
        self.im_lb = []
        self.lb_map=lb_map
        self.trans_func = trans_func
        

    def run(self):
        # 在这里执行图像读取操作
        for im_lb_p in self.im_lb_path:
            imp, lbp = im_lb_p
            lb = cv2.imread(lbp, 0)
            if not self.lb_map is None:
                lb = self.lb_map[lb]
            
            im = cv2.imread(imp)[:, :, ::-1]
            im_lb = dict(im=im, lb=lb)
            if not self.trans_func is None:
                im_lb = self.trans_func(im_lb)
                im = im_lb['im']
                lb = im_lb['lb']
            self.im_lb.append([im, lb])


class ImageBatchDecoderPyTorch:
    def __init__(
        self,
        dataroot, 
        annpath, 
        batch_size,
        device_id,
        cuda_ctx,
        mode='train',
    ):
        # self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.device_id = device_id
        # self.total_decoded = 0
        self.batch_idx = 0
        self.cuda_ctx = cuda_ctx
        self.mode = mode
        if mode == 'train':
            self.shuffle = True
        else:
            self.shuffle = False


        self.lb_map = None
        self.imStack = True

        with open(annpath, 'r') as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths = [], []
        for pair in pairs:
            imgpth, lbpth = pair.split(',')
            self.img_paths.append(osp.join(dataroot, imgpth))
            self.lb_paths.append(osp.join(dataroot, lbpth))

        assert len(self.img_paths) == len(self.lb_paths)
        self.len = len(self.img_paths)
        # docs_tag: end_parse_imagebatchdecoder_pytorch

        self.file_idx = np.arange(0, self.len)
        if self.shuffle:
            random.shuffle(self.file_idx)
            
        
        # docs_tag: begin_batch_imagebatchdecoder_pytorch
        # self.file_name_batches = [
        #     self.file_names[i : i + self.batch_size]  # noqa: E203
        #     for i in range(0, len(self.file_names), self.batch_size)
        # ]

        self.max_image_size =  8192 * 4096 * 3  # Maximum possible image size.
        self.decoder = torchnvjpeg.Decoder(
                device_padding=0,
                host_padding=0,
                gpu_huffman=True,
                device_id=self.device_id,
                bath_size=batch_size,
                max_cpu_threads=3,  # this is max_cpu_threads parameter. Not used internally.
                max_image_size=self.max_image_size,
                stream=None,
            )

        # self.logger.info("Using torchnvjpeg as decoder.")

        # docs_tag: end_init_imagebatchdecoder_pytorch

    def __call__(self):
        if self.batch_idx >= self.len:
            if self.mode != 'train':
                return None
            
            self.decoder = torchnvjpeg.Decoder(
                device_padding=0,
                host_padding=0,
                gpu_huffman=True,
                device_id=self.device_id,
                bath_size=self.batch_size,
                max_cpu_threads=3,  # this is max_cpu_threads parameter. Not used internally.
                max_image_size=self.max_image_size,
                stream=None,
            )
            self.batch_idx = 0
            if self.shuffle:
                random.shuffle(self.file_idx)
                

        # docs_tag: begin_call_imagebatchdecoder_pytorch
        file_name_batch = self.img_paths[self.batch_idx:self.batch_idx+self.batch_size]
        label_name_batch = self.lb_paths[self.batch_idx:self.batch_idx+self.batch_size]
        # print(file_name_batch, label_name_batch)
        effective_batch_size = len(file_name_batch)
        data_batch = [open(path, "rb").read() for path in file_name_batch]
        # label_batch = [open(path, "rb").read() for path in label_name_batch]

        # docs_tag: end_read_imagebatchdecoder_pytorch

        # docs_tag: begin_decode_imagebatchdecoder_pytorch
        if self.decoder is None or effective_batch_size != self.batch_size:
            self.decoder = torchnvjpeg.Decoder(
                device_padding=0,
                host_padding=0,
                gpu_huffman=True,
                device_id=self.device_id,
                bath_size=effective_batch_size,
                max_cpu_threads=3,  # this is max_cpu_threads parameter. Not used internally.
                max_image_size=self.max_image_size,
                stream=None,
            )

        image_tensor_list = self.decoder.batch_decode(data_batch)
        # image_tensor_list = [torch.ones(3), torch.ones(3)]
        label_tensor_list = self.get_label(label_name_batch)
        

        # Convert the list of tensors to a tensor itself.
        if self.imStack:
            image_tensors_nhwc = torch.stack(image_tensor_list)
            lb_tensors_nhwc = torch.stack(label_tensor_list).cuda(self.device_id)
        else:
            # image_tensor_list = [image_tensor.unsqueeze(0) for image_tensor in image_tensor_list]
            # label_tensor_list = [label_tensor.unsqueeze(0) for label_tensor in label_tensor_list]
            image_tensors_nhwc = image_tensor_list
            lb_tensors_nhwc = [lb.cuda(self.device_id) for lb in label_tensor_list] 
        
        
        # self.total_decoded += len(image_tensor_list)
        # docs_tag: end_decode_imagebatchdecoder_pytorch

        # docs_tag: begin_return_imagebatchdecoder_pytorch
        # batch = Batch(
        #     batch_idx=self.batch_idx, data=image_tensors_nhwc, fileinfo=file_name_batch, lb=lb_tensors_nhwc
        # )
        self.batch_idx += self.batch_size

        return image_tensors_nhwc, lb_tensors_nhwc
        # docs_tag: end_return_imagebatchdecoder_pytorch

    def start(self):
        pass

    def join(self):
        pass

    def __len__(self):
        return self.len
    
    def get_label(self, lbpth):
        
        threads = [LabelReaderThread(lb_path, self.lb_map) for lb_path in lbpth]

        # 启动线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        lbs = [torch.from_numpy((thread.lb_image).astype(np.int32)).to(torch.uint8).unsqueeze(2) for thread in threads]
        return lbs


class ImageBatchPNGDecoderPyTorch:
    def __init__(
        self,
        dataroot, 
        annpath, 
        batch_size,
        device_id,
        cuda_ctx,
        mode='train',
    ):
        # self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.device_id = device_id
        # self.total_decoded = 0
        self.batch_idx = 0
        self.cuda_ctx = cuda_ctx
        self.mode = mode
        if mode == 'train':
            self.shuffle = True
        else:
            self.shuffle = False


        self.lb_map = None
        self.imStack = False

        with open(annpath, 'r') as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths = [], []
        for pair in pairs:
            imgpth, lbpth = pair.split(',')
            self.img_paths.append(osp.join(dataroot, imgpth))
            self.lb_paths.append(osp.join(dataroot, lbpth))

        assert len(self.img_paths) == len(self.lb_paths)
        self.len = len(self.img_paths)
        # docs_tag: end_parse_imagebatchdecoder_pytorch

        self.file_idx = np.arange(0, self.len)
        if self.shuffle:
            random.shuffle(self.file_idx)
            
        
        # docs_tag: begin_batch_imagebatchdecoder_pytorch
        # self.file_name_batches = [
        #     self.file_names[i : i + self.batch_size]  # noqa: E203
        #     for i in range(0, len(self.file_names), self.batch_size)
        # ]

        self.max_image_size = 8192 * 4096 * 3  # Maximum possible image size.
        

        # self.logger.info("Using torchnvjpeg as decoder.")

        # docs_tag: end_init_imagebatchdecoder_pytorch

    def __call__(self):
        if self.batch_idx >= self.len:
            if self.mode != 'train':
                return None
            self.batch_idx = 0
            if self.shuffle:
                random.shuffle(self.file_idx)
                

        # docs_tag: begin_call_imagebatchdecoder_pytorch
        file_name_batch = self.img_paths[self.batch_idx:self.batch_idx+self.batch_size]
        label_name_batch = self.lb_paths[self.batch_idx:self.batch_idx+self.batch_size]
        # print(file_name_batch, label_name_batch)
        effective_batch_size = len(file_name_batch)
        # data_batch = [open(path, "rb").read() for path in file_name_batch]
        # # label_batch = [open(path, "rb").read() for path in label_name_batch]

        # # docs_tag: end_read_imagebatchdecoder_pytorch

        # # docs_tag: begin_decode_imagebatchdecoder_pytorch
        # if self.decoder is None or effective_batch_size != self.batch_size:
        #     self.decoder = torchnvjpeg.Decoder(
        #         device_padding=0,
        #         host_padding=0,
        #         gpu_huffman=True,
        #         device_id=self.device_id,
        #         bath_size=effective_batch_size,
        #         max_cpu_threads=8,  # this is max_cpu_threads parameter. Not used internally.
        #         max_image_size=self.max_image_size,
        #         stream=None,
        #     )

        # image_tensor_list = self.decoder.batch_decode(data_batch)
        # # image_tensor_list = [torch.ones(3), torch.ones(3)]
        # label_tensor_list = self.get_label(label_name_batch)
        

        # Convert the list of tensors to a tensor itself.
        image_tensor_list, label_tensor_list = self.get_image_label(file_name_batch, label_name_batch)
        if self.imStack:
            image_tensors_nhwc = torch.stack(image_tensor_list).cuda(self.device_id)
            lb_tensors_nhwc = torch.stack(label_tensor_list).cuda(self.device_id)
        else:
            # image_tensor_list = [image_tensor.unsqueeze(0) for image_tensor in image_tensor_list]
            # label_tensor_list = [label_tensor.unsqueeze(0) for label_tensor in label_tensor_list]
            image_tensors_nhwc = [im.cuda(self.device_id) for im in image_tensor_list] 
            lb_tensors_nhwc = [lb.cuda(self.device_id) for lb in label_tensor_list] 
        # image_tensors_nhwc = torch.stack(image_tensor_list).cuda(self.device_id)
        # lb_tensors_nhwc = torch.stack(label_tensor_list).cuda(self.device_id)
        
        # self.total_decoded += len(image_tensor_list)
        # docs_tag: end_decode_imagebatchdecoder_pytorch

        # docs_tag: begin_return_imagebatchdecoder_pytorch
        # batch = Batch(
        #     batch_idx=self.batch_idx, data=image_tensors_nhwc, fileinfo=file_name_batch, lb=lb_tensors_nhwc
        # )
        self.batch_idx += self.batch_size

        return image_tensors_nhwc, lb_tensors_nhwc
        # docs_tag: end_return_imagebatchdecoder_pytorch

    def start(self):
        pass

    def join(self):
        pass

    def __len__(self):
        return self.len
    
    def get_image_label(self, impth, lbpth):
        threads = []
        for i in range(0, len(impth), 2):
            if i+1 < len(impth):
                im_lb_path = [[impth[i], lbpth[i]], [impth[i+1], lbpth[i+1]]]
            else:
                im_lb_path = [[impth[i], lbpth[i]]]
            threads.append(IMLbReaderThread(im_lb_path, self.lb_map))

        # 启动线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        ims = []
        lbs = []
        for thread in threads:
            im_lbs = thread.im_lb
            for im_lb in im_lbs:
                im, lb = im_lb
                ims.append(torch.from_numpy(im.copy()))
                lbs.append(torch.from_numpy((lb).astype(np.int32)).to(torch.uint8).unsqueeze(2))

        return ims, lbs

        
    
    def get_label(self, lbpth):
        
        threads = [LabelReaderThread(lb_path, self.lb_map) for lb_path in lbpth]

        # 启动线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        lbs = [torch.from_numpy((thread.lb_image).astype(np.int32)).to(torch.uint8).unsqueeze(2) for thread in threads]
        return lbs

class ImageBatchPNGDecoderPyTorchDist:
    def __init__(
        self,
        dataroot, 
        annpath, 
        batch_size,
        device_id,
        cuda_ctx,
        mode='train',
    ):
        
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            print(f'cur rank:{self.rank}, num_replicas:{self.num_replicas}')
            if self.rank >= self.num_replicas or self.rank < 0:
                raise ValueError(
                    f"Invalid rank {self.rank}, rank should be in the interval [0, {self.num_replicas - 1}]")
        # self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.device_id = device_id
        # self.total_decoded = 0
        self.batch_idx = 0
        self.cuda_ctx = cuda_ctx
        self.mode = mode

        self.lb_map = None
        self.imStack = True

        with open(annpath, 'r') as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths = [], []
        for pair in pairs:
            imgpth, lbpth = pair.split(',')
            self.img_paths.append(osp.join(dataroot, imgpth))
            self.lb_paths.append(osp.join(dataroot, lbpth))

        self.img_paths = np.array(self.img_paths)
        self.lb_paths = np.array(self.lb_paths)

        assert len(self.img_paths) == len(self.lb_paths)
        self.total_size = len(self.img_paths)
        self.file_idx = np.arange(0, self.total_size)
        # docs_tag: end_parse_imagebatchdecoder_pytorch
        if self.mode == 'train':
            self.shuffle = True
            self.trans_func = T.Compose([
                T.RandomResizedCrop((0,5,1.5), (768, 768)),
                T.RandomHorizontalFlip(),
                # T.ColorJitter(
                #     brightness=0.4,
                #     contrast=0.4,
                #     saturation=0.4
                # ),
            ])
            if dist.is_initialized():
                self.num_samples = int(math.ceil(self.total_size / self.num_replicas))
                self.total_size = self.num_samples * self.num_replicas
                self.file_idx += self.file_idx[:(self.total_size - len(self.file_idx))]
                self.file_idx = self.file_idx[self.rank:self.total_size:self.num_replicas]
                random.shuffle(self.file_idx)
                self.total_size = len(self.file_idx)
        else:
            self.shuffle = False
            self.trans_func = None
        
        # docs_tag: begin_batch_imagebatchdecoder_pytorch
        # self.file_name_batches = [
        #     self.file_names[i : i + self.batch_size]  # noqa: E203
        #     for i in range(0, len(self.file_names), self.batch_size)
        # ]

        self.max_image_size = 8192 * 4096 * 3  # Maximum possible image size.
        

        # self.logger.info("Using torchnvjpeg as decoder.")

        # docs_tag: end_init_imagebatchdecoder_pytorch

    def __call__(self):
        if self.batch_idx >= self.total_size:
            if self.mode != 'train':
                return None
            self.batch_idx = 0
            if self.shuffle:
                random.shuffle(self.file_idx)
                

        # docs_tag: begin_call_imagebatchdecoder_pytorch
        file_name_batch = self.img_paths[self.file_idx[self.batch_idx:self.batch_idx+self.batch_size]]
        label_name_batch = self.lb_paths[self.file_idx[self.batch_idx:self.batch_idx+self.batch_size]]
        # print(file_name_batch, label_name_batch)
        effective_batch_size = len(file_name_batch)
        # data_batch = [open(path, "rb").read() for path in file_name_batch]
        # # label_batch = [open(path, "rb").read() for path in label_name_batch]

        # # docs_tag: end_read_imagebatchdecoder_pytorch

        # # docs_tag: begin_decode_imagebatchdecoder_pytorch
        # if self.decoder is None or effective_batch_size != self.batch_size:
        #     self.decoder = torchnvjpeg.Decoder(
        #         device_padding=0,
        #         host_padding=0,
        #         gpu_huffman=True,
        #         device_id=self.device_id,
        #         bath_size=effective_batch_size,
        #         max_cpu_threads=8,  # this is max_cpu_threads parameter. Not used internally.
        #         max_image_size=self.max_image_size,
        #         stream=None,
        #     )

        # image_tensor_list = self.decoder.batch_decode(data_batch)
        # # image_tensor_list = [torch.ones(3), torch.ones(3)]
        # label_tensor_list = self.get_label(label_name_batch)
        

        # Convert the list of tensors to a tensor itself.
        image_tensor_list, label_tensor_list = self.get_image_label(file_name_batch, label_name_batch)
        # if self.imStack:
        image_tensors_nhwc = torch.stack(image_tensor_list).cuda(self.device_id)
        lb_tensors_nhwc = torch.stack(label_tensor_list).cuda(self.device_id)
        # else:
        #     # image_tensor_list = [image_tensor.unsqueeze(0) for image_tensor in image_tensor_list]
        #     # label_tensor_list = [label_tensor.unsqueeze(0) for label_tensor in label_tensor_list]
        #     image_tensors_nhwc = [im.cuda(self.device_id) for im in image_tensor_list] 
        #     lb_tensors_nhwc = [lb.cuda(self.device_id) for lb in label_tensor_list] 
        # image_tensors_nhwc = torch.stack(image_tensor_list).cuda(self.device_id)
        # lb_tensors_nhwc = torch.stack(label_tensor_list).cuda(self.device_id)
        
        # self.total_decoded += len(image_tensor_list)
        # docs_tag: end_decode_imagebatchdecoder_pytorch

        # docs_tag: begin_return_imagebatchdecoder_pytorch
        # batch = Batch(
        #     batch_idx=self.batch_idx, data=image_tensors_nhwc, fileinfo=file_name_batch, lb=lb_tensors_nhwc
        # )
        self.batch_idx += self.batch_size

        return image_tensors_nhwc, lb_tensors_nhwc
        # docs_tag: end_return_imagebatchdecoder_pytorch

    def start(self):
        pass

    def join(self):
        pass

    def __len__(self):
        return self.total_size
    
    def get_image_label(self, impth, lbpth):
        threads = []
        for i in range(0, len(impth), 2):
            if i+1 < len(impth):
                im_lb_path = [[impth[i], lbpth[i]], [impth[i+1], lbpth[i+1]]]
            else:
                im_lb_path = [[impth[i], lbpth[i]]]
            threads.append(IMLbReaderThread(im_lb_path, self.lb_map, trans_func=self.trans_func))

        # 启动线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        ims = []
        lbs = []
        for thread in threads:
            im_lbs = thread.im_lb
            for im_lb in im_lbs:
                im, lb = im_lb
                ims.append(torch.from_numpy(im.copy()))
                lbs.append(torch.from_numpy((lb).astype(np.int32)).to(torch.uint8).unsqueeze(2))

        return ims, lbs

        
    
    def get_label(self, lbpth):
        
        threads = [LabelReaderThread(lb_path, self.lb_map) for lb_path in lbpth]

        # 启动线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        lbs = [torch.from_numpy((thread.lb_image).astype(np.int32)).to(torch.uint8).unsqueeze(2) for thread in threads]
        return lbs

class ImageParallelDecoderPyTorch:
    def __init__(
        self,
        dataroot, 
        annpath, 
        batch_size,
        device_id,
        cuda_ctx,
        mode='train',
    ):
        # self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.device_id = device_id
        # self.total_decoded = 0
        self.batch_idx = 0
        self.cuda_ctx = cuda_ctx
        self.mode = mode
        if mode == 'train':
            self.shuffle = True
        else:
            self.shuffle = False

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
        # docs_tag: end_parse_imagebatchdecoder_pytorch

        self.file_idx = np.arange(0, self.len)
        if self.shuffle:
            random.shuffle(self.file_idx)
            
        
        # docs_tag: begin_batch_imagebatchdecoder_pytorch
        # self.file_name_batches = [
        #     self.file_names[i : i + self.batch_size]  # noqa: E203
        #     for i in range(0, len(self.file_names), self.batch_size)
        # ]

        self.decoder_list = [torchnvjpeg.Decoder() for _ in range(batch_size)]
        self.cpu_threads = int((self.batch_size+1) / 2)
        self.pool = ThreadPool(cpu_threads)

        # self.logger.info("Using torchnvjpeg as decoder.")

        # docs_tag: end_init_imagebatchdecoder_pytorch

    def __call__(self):
        if self.batch_idx >= self.len:
            if self.mode != 'train':
                return None
            self.batch_idx = 0
            if self.shuffle:
                random.shuffle(self.file_idx)
                

        # docs_tag: begin_call_imagebatchdecoder_pytorch
        file_name_batch = self.img_paths[self.batch_idx:self.batch_idx+self.batch_size]
        label_name_batch = self.lb_paths[self.batch_idx:self.batch_idx+self.batch_size]
        print(file_name_batch, label_name_batch)
        effective_batch_size = len(file_name_batch)
        data_batch = [open(path, "rb").read() for path in file_name_batch]
        # label_batch = [open(path, "rb").read() for path in label_name_batch]

        # docs_tag: end_read_imagebatchdecoder_pytorch

        # docs_tag: begin_decode_imagebatchdecoder_pytorch
        def run(args):
            decoder, data = args
            return decoder.decode(data)

        image_tensor_list = pool.map(run, zip(self.decoder_list, data_batch))

        # image_tensor_list = self.decoder.batch_decode(data_batch)
        # image_tensor_list = [torch.ones(3), torch.ones(3)]
        label_tensor_list = self.get_label(label_name_batch)
        

        # Convert the list of tensors to a tensor itself.
        image_tensors_nhwc = torch.stack(image_tensor_list)
        lb_tensors_nhwc = torch.stack(label_tensor_list).cuda(self.device_id)
        
        # self.total_decoded += len(image_tensor_list)
        # docs_tag: end_decode_imagebatchdecoder_pytorch

        # docs_tag: begin_return_imagebatchdecoder_pytorch
        # batch = Batch(
        #     batch_idx=self.batch_idx, data=image_tensors_nhwc, fileinfo=file_name_batch, lb=lb_tensors_nhwc
        # )
        self.batch_idx += self.batch_size

        return image_tensors_nhwc, lb_tensors_nhwc
        # docs_tag: end_return_imagebatchdecoder_pytorch

    def start(self):
        pass

    def join(self):
        pass

    def __len__(self):
        return self.len
    
    def get_label(self, lbpth):
        
        threads = [IMLbReaderThread(lb_path, self.lb_map) for lb_path in lbpth]

        # 启动线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        lbs = [torch.from_numpy((thread.lb_image).astype(np.int32)).to(torch.uint8).unsqueeze(2) for thread in threads]
        return lbs


# docs_tag: begin_init_imagebatchencoder_pytorch
class ImageBatchEncoderPyTorch:
    def __init__(
        self,
        output_path,
        fps,
        device_id,
        cuda_ctx,
        cvcuda_perf,
    ):
        self.logger = logging.getLogger(__name__)
        self._encoder = None
        self.input_layout = "NCHW"
        self.gpu_input = True
        self.output_path = output_path
        self.device_id = device_id
        self.cvcuda_perf = cvcuda_perf

        self.logger.info("Using PyTorch/PIL as encoder.")
        # docs_tag: end_init_imagebatchencoder_pytorch

    # docs_tag: begin_call_imagebatchencoder_pytorch
    def __call__(self, batch):
        self.cvcuda_perf.push_range("encoder.torch")

        image_tensors_nchw = batch.data

        # Bring the image_tensors_nchw to CPU and convert it to a PIL
        # image and save those.
        for img_idx in range(image_tensors_nchw.shape[0]):
            img_name = os.path.splitext(os.path.basename(batch.fileinfo[img_idx]))[0]
            results_path = os.path.join(self.output_path, "out_%s.jpg" % img_name)
            self.logger.info("Saving the overlay result to: %s" % results_path)
            overlay_cpu = image_tensors_nchw[img_idx].detach().cpu()
            overlay_pil = F.to_pil_image(overlay_cpu)
            overlay_pil.save(results_path)

        self.cvcuda_perf.pop_range()

        # docs_tag: end_call_imagebatchencoder_pytorch

    def start(self):
        pass

    def join(self):
        pass
