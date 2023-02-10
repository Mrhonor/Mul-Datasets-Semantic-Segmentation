import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm
from collections import OrderedDict

class MySequential(nn.Module):

    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, dataset, x, *other_x):
        feat = [x, *other_x]
        for module in self._modules.values():
            feat = module(dataset, *feat)
        return feat

class MySequential_only_x(nn.Module):

    def __init__(self, *args):
        super(MySequential_only_x, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, dataset, x):
        for module in self._modules.values():
            x = module(dataset, x)
        return x



class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, n_bn=1, inplace=True):
        ## n_bn bn层数量，对应混合的数据集数量
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        
        self.n_bn = n_bn
        self.bn = nn.ModuleList([nn.BatchNorm2d(out_chan, affine=False) for i in range(0, self.n_bn)])
        # 采用共享的affine parameter
        self.affine_weight = nn.Parameter(torch.empty(out_chan))
        self.affine_bias = nn.Parameter(torch.empty(out_chan))
        
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, dataset, x, *other_x):
        ## 处理多数据集情况
        if self.n_bn != 1 and len(other_x) != 0 and self.training:
            batch_size = [x.shape[0]]
            for i in range(0, len(other_x)):
                batch_size.append(other_x[i].shape[0])
                x = torch.cat((x, other_x[i]), 0)
            feat = self.conv(x)
            feats = []
            begin_index = 0
            for i in range(0, len(other_x) + 1):
                end_index = begin_index + batch_size[i]
                if i == dataset:
                    feat_ = self.bn[i](feat[begin_index: end_index])
                else:
                    self.bn[i].eval()
                    feat_ = self.bn[i](feat[begin_index: end_index])
                    self.bn[i].train()

                feat_ = feat_ * self.affine_weight.reshape(1,-1,1,1) + self.affine_bias.reshape(1,-1,1,1)
                feat_ = self.relu(feat_)
                feats.append(feat_)

            return feats
        
        if len(other_x) != 0:
            batch_size = [x.shape[0]]
            for i in range(0, len(other_x)):
                batch_size.append(other_x[i].shape[0])
                x = torch.cat((x, other_x[i]), 0)
            feat = self.conv(x)
            feats = []
            begin_index = 0
            for i in range(0, len(other_x) + 1):
                end_index = begin_index + batch_size[i]
                feat_ = self.bn[i](feat[begin_index: end_index])
                
                ## affine param
                feat_ = feat_ * self.affine_weight.reshape(1,-1,1,1) + self.affine_bias.reshape(1,-1,1,1)
                
                feat_ = self.relu(feat_)
                feats.append(feat_)
                begin_index = end_index
        else:
            if dataset >= self.n_bn or dataset < 0:
                dataset = 0
 
            feat = self.conv(x)
            feat = self.bn[dataset](feat)
            
            ## affine param
            feat = feat * self.affine_weight.reshape(1,-1,1,1) + self.affine_bias.reshape(1,-1,1,1)
            
            feat = self.relu(feat)
            feats = [feat]
        # for i, xs in enumerate(other_x):
        #     feat = self.conv(xs)
        #     feat = self.bn[i+1](feat)
        #     feat = self.relu(feat)
        #     feats.append(feat)

        return feats

class ConvBN(nn.Module):
    ## ConvBNReLU类去掉ReLu层
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, n_bn=1):
        ## n_bn bn层数量，对应混合的数据集数量
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.n_bn = n_bn
        self.bn = nn.ModuleList([nn.BatchNorm2d(out_chan, affine=False) for i in range(0, self.n_bn)])
        
        # 采用共享的affine parameter
        self.affine_weight = nn.Parameter(torch.empty(out_chan))
        self.affine_bias = nn.Parameter(torch.empty(out_chan))
        

    def forward(self, dataset, x, *other_x):
        ## 处理多数据集情况
        if self.n_bn != 1 and len(other_x) != 0 and self.training:
            batch_size = [x.shape[0]]
            for i in range(0, len(other_x)):
                batch_size.append(other_x[i].shape[0])
                x = torch.cat((x, other_x[i]), 0)
            feat = self.conv(x)
            feats = []
            begin_index = 0
            for i in range(0, len(other_x) + 1):
                end_index = begin_index + batch_size[i]
                self.bn[i].eval()
                feat_ = self.bn[i](feat[begin_index: end_index])
                self.bn[i].train()

                feat_ = feat_ * self.affine_weight.reshape(1,-1,1,1) + self.affine_bias.reshape(1,-1,1,1)

                feats.append(feat_)

            return feats
        
        if len(other_x) != 0:
            batch_size = [x.shape[0]]
            for i in range(0, len(other_x)):
                batch_size.append(other_x[i].shape[0])
                x = torch.cat((x, other_x[i]), 0)
            feat = self.conv(x)
            feats = []
            begin_index = 0
            for i in range(0, len(other_x) + 1):
                end_index = begin_index + batch_size[i]
                feat_ = self.bn[i](feat[begin_index: end_index])
                
                ## affine param
                feat_ = feat_ * self.affine_weight.reshape(1,-1,1,1) + self.affine_bias.reshape(1,-1,1,1)
                
                feats.append(feat_)
                begin_index = end_index
        else:
            if dataset >= self.n_bn or dataset < 0:
                dataset = 0
            
            feat = self.conv(x)
            feat = self.bn[dataset](feat)
            
            ## affine param
            feat = feat * self.affine_weight.reshape(1,-1,1,1) + self.affine_bias.reshape(1,-1,1,1)
            
            feats = [feat]
        return feats

    def SetLastBNAttr(self, attr):
        self.affine_weight.last_bn = attr
        self.affine_bias.last_bn = attr
        # for bn in self.bn:
        #     bn.last_bn = attr

class ConvWNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        ## n_bn bn层数量，对应混合的数据集数量
        super(ConvWNReLU, self).__init__()
        self.conv = weight_norm(nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias), dim=None)
        
        ## 采用共享的affine parameter
        # self.affine_weight = nn.Parameter(torch.empty(out_chan))
        # self.affine_bias = nn.Parameter(torch.empty(out_chan))
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.relu(feat)
        
        return feat

class ConvWN(nn.Module):
    ## ConvBNReLU类去掉ReLu层
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        ## n_bn bn层数量，对应混合的数据集数量
        super(ConvWN, self).__init__()
        self.conv = weight_norm(nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias), dim=None)
        
        
        ## 采用共享的affine parameter
        # self.affine_weight = nn.Parameter(torch.empty(out_chan))
        # self.affine_bias = nn.Parameter(torch.empty(out_chan))
        

    def forward(self, x):
        
        return self.conv(x)

