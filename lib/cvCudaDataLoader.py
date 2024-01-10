#!/usr/bin/python
# -*- encoding: utf-8 -*-
import pycuda.driver as cuda
import sys
from typing import Any
sys.path.insert(0, '.')
import cvcuda
import torch

from lib.cityscapes_cv2 import CityScapesCVCUDA
from lib.ade2016_data import ade2016CVCUDA
from lib.bdd100k_data import Bdd100kCVCUDA
from lib.idd_cv2 import IddCVCUDA
from lib.Mapi import Mapiv1CVCUDA
from lib.sunrgbd import SunrgbdCVCUDA
from lib.coco_data import Coco_dataCVCUDA

from lib.cvCudaPreprocess import PreprocessorCvcuda
import multiprocessing
import time
from multiprocessing import Queue
import threading
import queue
# multiprocessing.set_start_method('spawn')

class MyDataLoader:
    def __init__(self, n_datasets, ds_readers, preprocessor, ctx_handle, max_pool_size=10, mode='train'):
        self.ctx_handle = ctx_handle
        self.n_datasets = n_datasets
        self.max_pool_size = max_pool_size
        self.ds_readers = ds_readers
        self.mode = mode
        self.data_pools = [Queue(maxsize=max_pool_size) for _ in range(n_datasets)]
        self.out_data_pool = Queue(maxsize=max_pool_size)
        self.preprocessor = preprocessor
        self.datasets_end_flag = [0 for _ in range(self.n_datasets)]
        

    def reader(self, dataset_index, ctx_handle):
        # cuda.Context.attach(self.ctx)
        ctx = cuda.Context(ctx_handle)
        while True:
            # Simulate data reading, replace this with your actual data loading logic
            data = self.ds_readers[dataset_index]()
            if data is None:
                # self.datasets_end_flag[dataset_index] = 1
                self.data_pools[dataset_index].put(None)
                break

            # Put data into the data pool
            self.data_pools[dataset_index].put(data)

            # If the data pool is full, sleep
            while self.data_pools[dataset_index].full():
                time.sleep(1)

    def start_readers(self):
        # Create and start a process for each dataset
        self.processes = []
        for i in range(self.n_datasets):
            process = multiprocessing.Process(target=self.reader, args=(i, self.ctx_handle))
            process.daemon = True
            process.start()
            self.processes.append(process)


    def main_process(self,ctx_handle):
        ctx = cuda.Context(ctx_handle)
        while True:
            im_list = []
            lb_list = []
            for i in range(self.n_datasets):
                if self.datasets_end_flag[i] != 0:
                    continue
                # Get data from the data pool
                # if self.data_pools[i].empty() and :
                data = self.data_pools[i].get()
                if data is None:
                    self.datasets_end_flag[i] = 1
                    continue

                # Process the data 
                im, lb = data
                im_list.append(im)
                lb_list.append(lb)
                
            if len(im_list) == 0:
                break
            
            processed_data = self.preprocessor(im_list, lb_list)
            self.out_data_pool.put(processed_data)
            
            
    def run(self):
        # Start reader processes
        self.start_readers()

        # Start the main process
        main_process = multiprocessing.Process(target=self.main_process, args=(self.ctx_handle, ))
        main_process.daemon = True
        main_process.start()

        self.processes.append(main_process)
                
    def __call__(self):
        # for i in range(self.dataset_count):
                # Get data from the data pool
        if min(self.datasets_end_flag) != 0 and self.out_data_pool.empty():
            return None, None
        
        data = self.out_data_pool.get()
        im, lb = data
        return im, lb
        
    
    def stop_worker(self):
        for process in self.processes:
            if process and process.is_alive():
                process.terminate()
                process.join()  # Wait for the child process to finish

    def __del__(self):
        self.stop_worker()

class MyDataLoaderThread:
    def __init__(self, n_datasets, ds_readers, preprocessor, max_pool_size=10, mode='train'):
        self.n_datasets = n_datasets
        self.max_pool_size = max_pool_size
        self.ds_readers = ds_readers
        self.mode = mode
        self.data_pools = [queue.Queue(maxsize=max_pool_size) for _ in range(n_datasets)]
        self.out_data_pool = queue.Queue(maxsize=max_pool_size)
        self.preprocessor = preprocessor
        self.datasets_end_flag = [0 for _ in range(self.n_datasets)]
        self.stop_event = threading.Event()
        

    def reader(self, dataset_index):
        # cuda.Context.attach(self.ctx)
        while not self.stop_event.is_set():
            # Simulate data reading, replace this with your actual data loading logic
            data = self.ds_readers[dataset_index]()
            if data is None:
                # self.datasets_end_flag[dataset_index] = 1
                self.data_pools[dataset_index].put(None)
                break

            # Put data into the data pool
            self.data_pools[dataset_index].put(data)

            # If the data pool is full, sleep
            while self.data_pools[dataset_index].full() and not self.stop_event.is_set():
                time.sleep(1)

    def start_readers(self):
        # Create and start a process for each dataset
        self.processes = []
        for i in range(self.n_datasets):
            process = threading.Thread(target=self.reader, args=(i,), daemon=True)
            process.start()
            self.processes.append(process)


    def main_process(self):
        while not self.stop_event.is_set():
            im_list = []
            lb_list = []
            ids = []
            for i in range(self.n_datasets):
                if self.datasets_end_flag[i] != 0:
                    continue
                # Get data from the data pool
                # if self.data_pools[i].empty() and :
                data = self.data_pools[i].get()
                if data is None:
                    self.datasets_end_flag[i] = 1
                    continue

                # Process the data 
                im, lb = data
                im_list.append(im)
                lb_list.append(lb)
                for _ in range(len(im)):
                    ids.append(i)
                
            if len(im_list) == 0:
                break
            
            processed_data = self.preprocessor(im_list, lb_list)
            self.out_data_pool.put([processed_data, ids])
            
            
    def run(self):
        # Start reader processes
        self.start_readers()

        # Start the main process
        main_process = threading.Thread(target=self.main_process, daemon=True)
        main_process.start()

        self.processes.append(main_process)
                
    def __call__(self):
        # for i in range(self.dataset_count):
                # Get data from the data pool
        if min(self.datasets_end_flag) != 0 and self.out_data_pool.empty():
            return None, None
        
        data = self.out_data_pool.get()
        im_lb, ids = data
        im, lb = im_lb
        return im, lb, ids
        
    
    def stop_worker(self):
        self.stop_event.set()
        if self.out_data_pool.full():
            self.out_data_pool.get()
        
        for dp in self.data_pools:
            if dp.full():
                dp.get()
        
        for process in self.processes:
            if process and process.is_alive():
                process.join()  # Wait for the child process to finish

    def __del__(self):
        self.stop_worker()

class MyDataLoaderWOMainThread:
    def __init__(self, n_datasets, ds_readers, preprocessor, max_pool_size=10, mode='train'):
        self.n_datasets = n_datasets
        self.max_pool_size = max_pool_size
        self.ds_readers = ds_readers
        self.mode = mode
        self.data_pools = [queue.Queue(maxsize=max_pool_size) for _ in range(n_datasets)]
        self.preprocessor = preprocessor
        self.datasets_end_flag = [0 for _ in range(self.n_datasets)]
        self.stop_event = threading.Event()
        

    def reader(self, dataset_index):
        # cuda.Context.attach(self.ctx)
        while not self.stop_event.is_set():
            # Simulate data reading, replace this with your actual data loading logic
            data = self.ds_readers[dataset_index]()
            if data is None:
                # self.datasets_end_flag[dataset_index] = 1
                self.data_pools[dataset_index].put(None)
                break

            # Put data into the data pool
            self.data_pools[dataset_index].put(data)

            # If the data pool is full, sleep
            while self.data_pools[dataset_index].full() and not self.stop_event.is_set():
                time.sleep(1)

    def start_readers(self):
        # Create and start a process for each dataset
        self.processes = []
        for i in range(self.n_datasets):
            process = threading.Thread(target=self.reader, args=(i,), daemon=True)
            process.start()
            self.processes.append(process)


            
    def run(self):
        # Start reader processes
        self.start_readers()

        # Start the main process

                
    def __call__(self):
        # for i in range(self.dataset_count):
                # Get data from the data pool

        im_list = []
        lb_list = []
        ids = []
        for i in range(self.n_datasets):
                    
            if self.datasets_end_flag[i] != 0 and self.data_pools[i].empty():
                continue
            # Get data from the data pool
            # if self.data_pools[i].empty() and :
            data = self.data_pools[i].get()
            if data is None:
                self.datasets_end_flag[i] = 1
                continue

            # Process the data 
            im, lb = data
            im_list.append(im)
            lb_list.append(lb)
            for _ in range(len(im)):
                ids.append(i)
            
        if len(im_list) == 0:
            return None, None, None
        
        im, lb = self.preprocessor(im_list, lb_list)
        return im, lb, ids
        
    
    def stop_worker(self):
        self.stop_event.set()
        
        for dp in self.data_pools:
            if dp.full():
                dp.get()
        
        for process in self.processes:
            if process and process.is_alive():
                process.join()  # Wait for the child process to finish

    def __del__(self):
        self.stop_worker()

class MyDataLoaderWOThread:
    def __init__(self, n_datasets, ds_readers, preprocessor, max_pool_size=10, mode='train'):
        self.n_datasets = n_datasets
        self.max_pool_size = max_pool_size
        self.ds_readers = ds_readers
        self.mode = mode
        self.preprocessor = preprocessor

            
    def run(self):
        # Start reader processes
        pass
                
    def __call__(self):
        im_list = []
        lb_list = []
        ids = []
        for dataset_index in range(self.n_datasets):
            im, lb = self.ds_readers[dataset_index]()
            if im is None:
                continue
            im_list.append(im)
            lb_list.append(lb)
            for _ in range(len(im)):
                ids.append(dataset_index)
        
        if len(im_list) == 0:
            return None, None, None
        
        im, lb = self.preprocessor(im_list, lb_list)

        return im, lb, ids


    def __del__(self):
        # self.stop_worker()
        pass

class getDataLoaderCVCUDA:
    def __init__(self, configer, device_id, cuda_ctx, mode='train', stage=None):
        self.configer = configer
        self.mode = mode
        self.n_datasets = self.configer.get('n_datasets')
        self.max_iter = configer.get('lr', 'max_iter')
        
        if mode == 'train':
            self.scales = configer.get('train', 'scales')
            self.cropsize = configer.get('train', 'cropsize')
            # trans_func = TransformationTrain(scales, cropsize)
            if stage != None:
                self.annpaths = [configer.get('dataset'+str(i), 'train_im_anns').replace('.txt', f'_{stage}.txt') for i in range(1, self.n_datasets+1)]
                print(self.annpaths)
                self.batchsizes = [configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, self.n_datasets+1)]
            else:
                self.annpaths = [configer.get('dataset'+str(i), 'train_im_anns') for i in range(1, self.n_datasets+1)]
                self.batchsizes = [configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, self.n_datasets+1)]
            self.imroots = [configer.get('dataset'+str(i), 'im_root') for i in range(1, self.n_datasets+1)]
            self.data_readers = [configer.get('dataset'+str(i), 'data_reader')+'CVCUDA' for i in range(1, self.n_datasets+1)]
            
        elif mode == 'eval':
            self.batchsizes = [configer.get('dataset'+str(i), 'eval_ims_per_gpu') for i in range(1, self.n_datasets+1)]
            self.annpaths = [configer.get('dataset'+str(i), 'val_im_anns') for i in range(1, self.n_datasets+1)]
            self.imroots = [configer.get('dataset'+str(i), 'im_root') for i in range(1, self.n_datasets+1)]
            self.data_readers = [configer.get('dataset'+str(i), 'data_reader')+'CVCUDA' for i in range(1, self.n_datasets+1)]
        elif mode == 'eval_link':
            self.scales = configer.get('train', 'scales')
            self.cropsize = configer.get('train', 'cropsize')
            # trans_func = TransformationTrain(scales, cropsize)
            if stage != None:
                self.annpaths = [configer.get('dataset'+str(i), 'train_im_anns').replace('.txt', f'_{stage}.txt') for i in range(1, self.n_datasets+1)]
                print(self.annpaths)
                self.batchsizes = [configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, self.n_datasets+1)]
            else:
                self.annpaths = [configer.get('dataset'+str(i), 'train_im_anns') for i in range(1, self.n_datasets+1)]
                self.batchsizes = [configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, self.n_datasets+1)]
            self.imroots = [configer.get('dataset'+str(i), 'im_root') for i in range(1, self.n_datasets+1)]
            self.data_readers = [configer.get('dataset'+str(i), 'data_reader')+'CVCUDA' for i in range(1, self.n_datasets+1)]            
           
        # self.ds = [eval(reader)(root, path, 1, device_id, cuda_ctx, mode=mode)
        #   for reader, root, path, bs in zip(self.data_readers, self.imroots, self.annpaths, self.batchsizes)]
        scales = configer.get('train', 'scales')
        cropsize = configer.get('train', 'cropsize')
        if mode == 'eval_link':
            self.ds = [eval(reader)(root, path, bs, device_id, cuda_ctx, mode='train')
                for reader, root, path, bs in zip(self.data_readers, self.imroots, self.annpaths, self.batchsizes)]
            self.preprocessor = PreprocessorCvcuda(scales, cropsize, device_id, p=0.5, brightness=0.4, contrast=0.4, mode='train')
        else:
            self.ds = [eval(reader)(root, path, bs, device_id, cuda_ctx, mode=self.mode)
                for reader, root, path, bs in zip(self.data_readers, self.imroots, self.annpaths, self.batchsizes)]
            self.preprocessor = PreprocessorCvcuda(scales, cropsize, device_id, p=0.5, brightness=0.4, contrast=0.4, mode=self.mode)
        
        # self.dataset_lens = [len(ds) for ds in self.ds]
        # self.cur_lens = [0 for _ in range(self.n_datasets)]

        # shared_ctx = multiprocessing.Manager().Value('i', cuda_ctx.handle)
        
        if self.mode == 'train':
            self.dataLoader = MyDataLoaderWOMainThread(self.n_datasets, self.ds, self.preprocessor, mode=self.mode)
            self.dataLoader.run()
        elif self.mode == 'eval':
            self.dataLoader = [MyDataLoaderWOMainThread(1, [ds], self.preprocessor, mode=self.mode) for ds in self.ds]
        elif self.mode == 'eval_link':
            self.dataLoader = [MyDataLoaderWOMainThread(1, [ds], self.preprocessor, mode='train') for ds in self.ds]
            
            
    def __call__(self):
        if self.mode == 'train':
            im, lb, ids = self.dataLoader()
            return im, lb, ids
        else:
            return self.dataLoader
            
    def __del__(self):
        if isinstance(self.dataLoader, list):
            for dl in self.dataLoader:
                dl.stop_worker()
        else:
            self.dataLoader.stop_worker()


if __name__ == "__main__":
    from tools.configer import Configer
    # cuda.init()
    
    configer = Configer(configs="configs/ltbgnn_7_datasets_snp.json")
    device_id = 0
    cuda_device = cuda.Device(device_id)
    cuda_ctx = cuda_device.retain_primary_context()
    # cuda_ctx = cuda_device.make_context()


    
    # cuda_ctx.pop()
    cvcuda_stream = cvcuda.Stream()
    torch_stream = torch.cuda.ExternalStream(cvcuda_stream.handle)
    annpaths = configer.get('dataset7', 'train_im_anns')
    batchsizes = configer.get('dataset7', 'ims_per_gpu')
    imroots = configer.get('dataset7', 'im_root')
    
    # # 创建共享的设备数组，这里使用了Manager中的Array
    # shared_array = multiprocessing.Manager().Array('f', 1000)
            
    cuda_ctx.push()
    with cvcuda_stream, torch.cuda.stream(torch_stream):
        # reader = Coco_dataCVCUDA(imroots, annpaths, 2, device_id, cuda_ctx)
        # for i in range(100):
        #     im, lb = reader()
        #     print(i)
            # print("im shape: ", im.shape)
            # print("lb shape: ", lb.shape)
        dls = getDataLoaderCVCUDA(configer, device_id, cuda_ctx, stage=None)
        for i in range(100):
            im, lb = dls()
            print("im shape: ", im.shape)
            print("lb shape: ", lb.shape)
        
    cuda_ctx.pop()
    # cuda_ctx.detach()
    