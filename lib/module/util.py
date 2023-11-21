import random
# from torch.utils.data import Dataset
import torch
import numpy as np
# import pickle
# from collections import defaultdict
from PIL import Image as pimg
import torch.nn.functional as F
import warnings

from torch import nn as nn

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)
batchnorm_momentum = 0.01 / 2


def get_n_params(parameters):
    pp = 0
    for p in parameters:
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1,
                 drop_rate=.0, separable=False):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        conv_class = SeparableConv2d if separable else nn.Conv2d
        warnings.warn(f'Using conv type {k}x{k}: {conv_class}')
        self.add_module('conv', conv_class(num_maps_in, num_maps_out, kernel_size=k, padding=padding, bias=bias,
                                           dilation=dilation))
        if drop_rate > 0:
            warnings.warn(f'Using dropout with p: {drop_rate}')
            self.add_module('dropout', nn.Dropout2d(drop_rate, inplace=True))

class BNReLUConv(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, n_bn=1):
        ## n_bn bn层数量，对应混合的数据集数量
        super(BNReLUConv, self).__init__()
        self.bn = nn.ModuleList([nn.BatchNorm2d(in_chan, affine=False) for i in range(0, n_bn)])
        ## 采用共享的affine parameter
        self.affine_weight = nn.Parameter(torch.empty(in_chan))
        self.affine_bias = nn.Parameter(torch.empty(in_chan))

        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x, dataset_id):
        # feat = self.conv(x)
        feat = x
        feat_list = []
        cur_pos = 0
        for i in range(0, len(dataset_id)):
            if dataset_id[i] != dataset_id[cur_pos]:
                feat_ = self.bn[dataset_id[cur_pos]](feat[cur_pos:i])
                feat_list.append(feat_)
                cur_pos = i
        feat_ = self.bn[dataset_id[cur_pos]](feat[cur_pos:])
        feat_list.append(feat_)
        feat = torch.cat(feat_list, dim=0)
        feat = feat * self.affine_weight.reshape(1,-1,1,1) + self.affine_bias.reshape(1,-1,1,1) 
        feat = self.relu(feat)
        feat = self.conv(feat)
        return feat

class _Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3, use_skip=True, only_skip=False,
                 detach_skip=False, fixed_size=None, separable=False, bneck_starts_with_bn=True):
        super(_Upsample, self).__init__()
        print(f'Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn and bneck_starts_with_bn)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn, separable=separable)
        self.use_skip = use_skip
        self.only_skip = only_skip
        self.detach_skip = detach_skip
        warnings.warn(f'\tUsing skips: {self.use_skip} (only skips: {self.only_skip})', UserWarning)
        self.upsampling_method = upsample
        if fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)
            warnings.warn(f'Fixed upsample size', UserWarning)

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        if self.detach_skip:
            skip = skip.detach()
        skip_size = skip.size()[2:4]
        x = self.upsampling_method(x, skip_size)
        if self.use_skip:
            x = x + skip
        x = self.blend_conv.forward(x)
        return x


class _UpsampleBlend(nn.Module):
    def __init__(self, num_features, use_bn=True, use_skip=True, detach_skip=False, fixed_size=None, k=3,
                 separable=False):
        super(_UpsampleBlend, self).__init__()
        self.blend_conv = _BNReluConv(num_features, num_features, k=k, batch_norm=use_bn, separable=separable)
        self.use_skip = use_skip
        self.detach_skip = detach_skip
        warnings.warn(f'Using skip connections: {self.use_skip}', UserWarning)
        self.upsampling_method = upsample
        if fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)
            warnings.warn(f'Fixed upsample size', UserWarning)

    def forward(self, x, skip):
        if self.detach_skip:
            warnings.warn(f'Detaching skip connection {skip.shape[2:4]}', UserWarning)
            skip = skip.detach()
        skip_size = skip.size()[-2:]
        x = self.upsampling_method(x, skip_size)
        if self.use_skip:
            x = x + skip
        x = self.blend_conv.forward(x)
        return x

class _UpsampleBlend_mulbn(nn.Module):
    def __init__(self, num_features, use_bn=True, use_skip=True, detach_skip=False, fixed_size=None, k=3,
                 separable=False, n_bn=1):
        super(_UpsampleBlend_mulbn, self).__init__()
        self.blend_conv = BNReLUConv(num_features, num_features, ks=k, n_bn=n_bn)
        self.use_skip = use_skip
        self.detach_skip = detach_skip
        warnings.warn(f'Using skip connections: {self.use_skip}', UserWarning)
        self.upsampling_method = upsample
        if fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)
            warnings.warn(f'Fixed upsample size', UserWarning)

    def forward(self, x, skip, dataset_id):
        if self.detach_skip:
            warnings.warn(f'Detaching skip connection {skip.shape[2:4]}', UserWarning)
            skip = skip.detach()
        skip_size = skip.size()[-2:]
        x = self.upsampling_method(x, skip_size)
        if self.use_skip:
            x = x + skip
        x = self.blend_conv.forward(x, dataset_id)
        return x

class SpatialPyramidPooling(nn.Module):
    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True, drop_rate=.0,
                 fixed_size=None, starts_with_bn=True):
        super(SpatialPyramidPooling, self).__init__()
        self.fixed_size = fixed_size
        self.grids = grids
        if self.fixed_size:
            ref = min(self.fixed_size)
            self.grids = list(filter(lambda x: x <= ref, self.grids))
        self.square_grid = square_grid
        self.upsampling_method = upsample
        if self.fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)
            warnings.warn(f'Fixed upsample size', UserWarning)
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn', _BNReluConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum,
                                                  batch_norm=use_bn and starts_with_bn))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i),
                                _BNReluConv(num_features, level_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn,
                                            drop_rate=drop_rate))
        self.spp.add_module('spp_fuse',
                            _BNReluConv(final_size, out_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))

    def forward(self, x):
        levels = []
        target_size = self.fixed_size if self.fixed_size is not None else x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)

            level = self.upsampling_method(level, target_size)
            levels.append(level)

        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def disparity_distribution_uniform(max_disp, num_bins):
    return np.linspace(0, max_disp, num_bins - 1)


def disparity_distribution_log(num_bins):
    return np.power(np.sqrt(2), np.arange(num_bins - 1))


def downsample_distribution(labels, factor, num_classes):
    h, w = labels.shape
    assert h % factor == 0 and w % factor == 0
    new_h = h // factor
    new_w = w // factor
    labels_4d = np.ascontiguousarray(labels.reshape(new_h, factor, new_w, factor), labels.dtype)
    labels_oh = np.eye(num_classes, dtype=np.float32)[labels_4d]
    target_dist = labels_oh.sum((1, 3)) / factor ** 2
    return target_dist


def downsample_distribution_th(labels, factor, num_classes, ignore_id=None):
    n, h, w = labels.shape
    assert h % factor == 0 and w % factor == 0
    new_h = h // factor
    new_w = w // factor
    labels_4d = labels.view(n, new_h, factor, new_w, factor)
    labels_oh = torch.eye(num_classes).to(labels_4d.device)[labels_4d]
    target_dist = labels_oh.sum(2).sum(3) / factor ** 2
    return target_dist


def downsample_labels_th(labels, factor, num_classes):
    '''
    :param labels: Tensor(N, H, W)
    :param factor: int
    :param num_classes:  int
    :return: FloatTensor(-1, num_classes), ByteTensor(-1, 1)
    '''
    n, h, w = labels.shape
    assert h % factor == 0 and w % factor == 0
    new_h = h // factor
    new_w = w // factor
    labels_4d = labels.view(n, new_h, factor, new_w, factor)
    # +1 class here because ignore id = num_classes
    labels_oh = torch.eye(num_classes + 1).to(labels_4d.device)[labels_4d]
    target_dist = labels_oh.sum(2).sum(3) / factor ** 2
    C = target_dist.shape[-1]
    target_dist = target_dist.view(-1, C)
    # keep only boxes which have p(ignore) < 0.5
    valid_mask = target_dist[:, -1] < 0.5
    target_dist = target_dist[:, :-1].contiguous()
    dist_sum = target_dist.sum(1, keepdim=True)
    # avoid division by zero
    dist_sum[dist_sum == 0] = 1
    # renormalize distribution after removing p(ignore)
    target_dist /= dist_sum
    return target_dist, valid_mask


def equalize_hist_disparity_distribution(d, L):
    cd = np.cumsum(d / d.sum())
    Y = np.round((L - 1) * cd).astype(np.uint8)
    return np.array([np.argmax(Y == i) for i in range(L - 1)])


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def one_hot_encoding(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).to(labels.device).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    return target


def crop_and_scale_img(img: pimg, crop_box, target_size, pad_size, resample, blank_value):
    target = pimg.new(img.mode, pad_size, color=blank_value)
    target.paste(img)
    res = target.crop(crop_box).resize(target_size, resample=resample)
    return res
