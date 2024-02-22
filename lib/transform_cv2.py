#!/usr/bin/python
# -*- encoding: utf-8 -*-


import random
import math

import numpy as np
import cv2
import torch



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
        
        scale = np.random.uniform(min(self.scales), max(self.scales))
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
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))
        return dict(
            im=im[sh:sh+crop_h, sw:sw+crop_w, :].copy(),
            lb=lb[sh:sh+crop_h, sw:sw+crop_w].copy()
        )



class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im_lb):
        if np.random.random() < self.p:
            return im_lb
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]
        return dict(
            im=im[:, ::-1, :],
            lb=lb[:, ::-1],
        )



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
            rate = np.random.uniform(*self.brightness)
            im = self.adj_brightness(im, rate)
        if not self.contrast is None:
            rate = np.random.uniform(*self.contrast)
            im = self.adj_contrast(im, rate)
        if not self.saturation is None:
            rate = np.random.uniform(*self.saturation)
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

class RandomResizedCropGPU(object):
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

        
        return dict(im=im_gpu.download(), lb=lb_gpu.download())
    


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

        if (im_h, im_w) == (crop_h, crop_w): 
            return dict(im=im_gpu.download(), lb=lb_gpu.download())
            # return dict(im=im_gpu, lb=lb_gpu)
               # 计算需要填充的大小
        pad_h = max(0, (crop_h - im_h) // 2 + 1)
        pad_w = max(0, (crop_w - im_w) // 2 + 1)
        
        # 使用copyMakeBorder进行填充
        im_gpu = cv2.cuda.copyMakeBorder(im_gpu, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
        lb_gpu = cv2.cuda.copyMakeBorder(lb_gpu, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=255)
 
        # 随机裁剪的起始位置

        
        
        # 使用ROI进行随机裁剪
        # im_gpu = im_gpu(cv2.cuda_GpuMat(), roi=(sw, sh, crop_w, crop_h))
        # lb_gpu = lb_gpu(cv2.cuda_GpuMat(), roi=(sw, sh, crop_w, crop_h))
        # cropped = cv2.UMat(im_gpu, [minX, maxX], [minY, maxY])
        # im_gpu = im_gpu.adjustROI(sw, sw+crop_w, sh, sh+crop_h)
        # lb_gpu = lb_gpu.adjustROI(sw, sw+crop_w, sh, sh+crop_h)

        if np.random.random() >= self.p:
            im_gpu = cv2.cuda.flip(im_gpu, flipCode=1)
            lb_gpu = cv2.cuda.flip(lb_gpu, flipCode=1)
        
        # if self.brightness is not None:
        #     rate = np.random.uniform(max(1 - self.brightness, 0), 1 + self.brightness)
        #     self.adj_brightness_gpu(im_gpu, rate)
        # if self.contrast is not None:
        #     rate = np.random.uniform(max(1 - self.contrast, 0), 1 + self.contrast)
        #     self.adj_contrast_gpu(im_gpu, rate)
        # if self.saturation is not None:
        #     rate = np.random.uniform(max(1 - self.saturation, 0), 1 + self.saturation)
        #     self.adj_saturation_gpu(im_gpu, rate)
        
        im = im_gpu.download()
        lb = lb_gpu.download()
        im_h, im_w, _ = im.shape
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))
        # print(im.shape)
        # print(lb.shape)
        # print(sh, sw, crop_h, crop_w)
        # return dict(im=im_gpu.download()[sh:sh+crop_h, sw:sw+crop_w, :].copy(), lb=lb_gpu.download()[sh:sh+crop_h, sw:sw+crop_w].copy())
        # return dict(im=im_gpu, lb=lb_gpu)
        return dict(
            im=im[sh:sh+crop_h, sw:sw+crop_w, :].copy(),
            lb=lb[sh:sh+crop_h, sw:sw+crop_w].copy()
        )
    
    def adj_saturation_gpu(self, im_gpu, rate):
        M = np.float32([
            [1 + 2 * rate, 1 - rate, 1 - rate],
            [1 - rate, 1 + 2 * rate, 1 - rate],
            [1 - rate, 1 - rate, 1 + 2 * rate]
        ])
        im_gpu_color = cv2.cuda.cvtColor(im_gpu, cv2.COLOR_BGR2BGRA)
        im_gpu_color_channels = cv2.cuda.split(im_gpu_color)
        for i in range(3):
            cv2.cuda.gemm(im_gpu_color_channels[i], M[i], 1, None, 0, im_gpu_color_channels[i])
        cv2.cuda.merge(im_gpu_color_channels, im_gpu_color)
        cv2.cuda.cvtColor(im_gpu_color, cv2.COLOR_BGRA2BGR, dst=im_gpu)

    def adj_brightness_gpu(self, im_gpu, rate):
        table = np.array([i * rate for i in range(256)]).clip(0, 255).astype(np.uint8)
        cv2.cuda.LookUpTable(im_gpu, table, im_gpu)

    def adj_contrast_gpu(self, im_gpu, rate):
        table = np.array([74 + (i - 74) * rate for i in range(256)]).clip(0, 255).astype(np.uint8)
        cv2.cuda.LookUpTable(im_gpu, table, im_gpu)

class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()
        if not lb is None:
            lb = torch.from_numpy(lb.astype(np.int64))#.clone()
        return dict(im=im, lb=lb)

class ToTensorCUDA(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im, lb):
        # im, lb = im_lb['im'], im_lb['lb']
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).cuda().div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[None, :, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[None, :, None, None]
        im = im.sub_(mean).div_(std)#.clone()
        if not lb is None:
            lb = torch.from_numpy(lb.astype(np.int64)).cuda()#.clone()
        return im, lb

class TensorToIMG(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, in_tensor):
        
        dtype, device = in_tensor.dtype, in_tensor.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        
        im = in_tensor.mul_(std).add_(mean).mul_(255).cpu().numpy()
        
        im = im.transpose(1, 2, 0).astype(np.float32)
        

        return im

class GaussianNoise(object):
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, image):
        image=image/255
        noise=np.random.normal(self.mean,self.sigma,image.shape)
        gaussian_out=image+noise
        gaussian_out=np.clip(gaussian_out,0,1)
        gaussian_out=np.uint8(gaussian_out*255)
        noise=np.uint8(noise*255)
        return gaussian_out

class ColorJitter_im(object):

    def __init__(self, brightness=None, contrast=None, saturation=None):
        if not brightness is None and brightness >= 0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast >= 0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation >= 0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im):
        if not self.brightness is None:
            rate = np.random.uniform(*self.brightness)
            im = self.adj_brightness(im, rate)
        if not self.contrast is None:
            rate = np.random.uniform(*self.contrast)
            im = self.adj_contrast(im, rate)
        if not self.saturation is None:
            rate = np.random.uniform(*self.saturation)
            im = self.adj_saturation(im, rate)
        return im

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

class ToTensor_im(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im):
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()

        return im

class Compose(object):

    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb





if __name__ == '__main__':
    pass

