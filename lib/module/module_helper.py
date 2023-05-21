from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import pdb
import math
from copy import deepcopy
from typing import List, Tuple

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

from lib.logger import Logger as Log
from torch.nn.functional import interpolate


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




class ModuleHelper(object):

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        if bn_type == 'torchbn':
            return nn.Sequential(
                nn.BatchNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'torchsyncbn':
            return nn.Sequential(
                nn.SyncBatchNorm(num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'syncbn':
            from lib.extensions.syncbn.module import BatchNorm2d
            return nn.Sequential(
                BatchNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'sn':
            from lib.extensions.switchablenorms.switchable_norm import SwitchNorm2d
            return nn.Sequential(
                SwitchNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'gn':
            return nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'fn':
            Log.error('Not support Filter-Response-Normalization: {}.'.format(bn_type))
            exit(1)
        elif bn_type == 'inplace_abn':
            torch_ver = torch.__version__[:3]
            # Log.info('Pytorch Version: {}'.format(torch_ver))
            if torch_ver == '0.4':
                from lib.extensions.inplace_abn.bn import InPlaceABNSync
                return InPlaceABNSync(num_features, **kwargs)
            elif torch_ver in ('1.0', '1.1'):
                from lib.extensions.inplace_abn_1.bn import InPlaceABNSync
                return InPlaceABNSync(num_features, **kwargs)
            elif torch_ver == '1.2':
                from inplace_abn import InPlaceABNSync
                return InPlaceABNSync(num_features, **kwargs)

        else:
            Log.error('Not support BN type: {}.'.format(bn_type))
            exit(1)

    @staticmethod
    def BatchNorm2d(bn_type='torch', ret_cls=False):
        if bn_type == 'torchbn':
            return nn.BatchNorm2d

        elif bn_type == 'torchsyncbn':
            return nn.SyncBatchNorm

        elif bn_type == 'syncbn':
            from lib.extensions.syncbn.module import BatchNorm2d
            return BatchNorm2d

        elif bn_type == 'sn':
            from lib.extensions.switchablenorms.switchable_norm import SwitchNorm2d
            return SwitchNorm2d

        elif bn_type == 'gn':
            return functools.partial(nn.GroupNorm, num_groups=32)

        elif bn_type == 'inplace_abn':
            torch_ver = torch.__version__[:3]
            if torch_ver == '0.4':
                from lib.extensions.inplace_abn.bn import InPlaceABNSync
                if ret_cls:
                    return InPlaceABNSync

                return functools.partial(InPlaceABNSync, activation='none')

            elif torch_ver in ('1.0', '1.1'):
                from lib.extensions.inplace_abn_1.bn import InPlaceABNSync
                if ret_cls:
                    return InPlaceABNSync

                return functools.partial(InPlaceABNSync, activation='none')

            elif torch_ver == '1.2':
                from inplace_abn import InPlaceABNSync
                if ret_cls:
                    return InPlaceABNSync

                return functools.partial(InPlaceABNSync, activation='identity')

        else:
            Log.error('Not support BN type: {}.'.format(bn_type))
            exit(1)

    @staticmethod
    def load_model(model, pretrained=None, all_match=True, network='resnet101'):
        if pretrained is None:
            return model

        if all_match:
            Log.info('Loading pretrained model:{}'.format(pretrained))
            pretrained_dict = torch.load(pretrained, map_location=lambda storage, loc: storage)
            model_dict = model.state_dict()
            load_dict = dict()
            for k, v in pretrained_dict.items():
                if 'resinit.{}'.format(k) in model_dict:
                    load_dict['resinit.{}'.format(k)] = v
                else:
                    load_dict[k] = v
            model.load_state_dict(load_dict)

        else:
            Log.info('Loading pretrained model:{}'.format(pretrained))
            pretrained_dict = torch.load(pretrained, map_location=lambda storage, loc: storage)

            # settings for "wide_resnet38"  or network == "resnet152"
            if network == "wide_resnet":
                pretrained_dict = pretrained_dict['state_dict']

            model_dict = model.state_dict()

            if network == "hrnet_plus":
                # pretrained_dict['conv1_full_res.weight'] = pretrained_dict['conv1.weight']
                # pretrained_dict['conv2_full_res.weight'] = pretrained_dict['conv2.weight']
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}

            elif network == 'pvt':
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                   k in model_dict.keys()}
                pretrained_dict['pos_embed1'] = \
                    interpolate(pretrained_dict['pos_embed1'].unsqueeze(dim=0), size=[16384, 64])[0]
                pretrained_dict['pos_embed2'] = \
                    interpolate(pretrained_dict['pos_embed2'].unsqueeze(dim=0), size=[4096, 128])[0]
                pretrained_dict['pos_embed3'] = \
                    interpolate(pretrained_dict['pos_embed3'].unsqueeze(dim=0), size=[1024, 320])[0]
                pretrained_dict['pos_embed4'] = \
                    interpolate(pretrained_dict['pos_embed4'].unsqueeze(dim=0), size=[256, 512])[0]
                pretrained_dict['pos_embed7'] = \
                    interpolate(pretrained_dict['pos_embed1'].unsqueeze(dim=0), size=[16384, 64])[0]
                pretrained_dict['pos_embed6'] = \
                    interpolate(pretrained_dict['pos_embed2'].unsqueeze(dim=0), size=[4096, 128])[0]
                pretrained_dict['pos_embed5'] = \
                    interpolate(pretrained_dict['pos_embed3'].unsqueeze(dim=0), size=[1024, 320])[0]
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}

            elif network == 'pcpvt' or network == 'svt':
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
                Log.info('Missing keys: {}'.format(list(set(model_dict) - set(load_dict))))

            elif network == 'transunet_swin':
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                   k in model_dict.keys()}
                for item in list(pretrained_dict.keys()):
                    if item.startswith('layers.0') and not item.startswith('layers.0.downsample'):
                        pretrained_dict['dec_layers.2' + item[15:]] = pretrained_dict[item]
                    if item.startswith('layers.1') and not item.startswith('layers.1.downsample'):
                        pretrained_dict['dec_layers.1' + item[15:]] = pretrained_dict[item]
                    if item.startswith('layers.2') and not item.startswith('layers.2.downsample'):
                        pretrained_dict['dec_layers.0' + item[15:]] = pretrained_dict[item]

                for item in list(pretrained_dict.keys()):
                    if 'relative_position_index' in item:
                        pretrained_dict[item] = \
                            interpolate(pretrained_dict[item].unsqueeze(dim=0).unsqueeze(dim=0).float(),
                                        size=[256, 256])[0][0]
                    if 'relative_position_bias_table' in item:
                        pretrained_dict[item] = \
                            interpolate(pretrained_dict[item].unsqueeze(dim=0).unsqueeze(dim=0).float(),
                                        size=[961, pretrained_dict[item].size(1)])[0][0]
                    if 'attn_mask' in item:
                        pretrained_dict[item] = \
                            interpolate(pretrained_dict[item].unsqueeze(dim=0).unsqueeze(dim=0).float(),
                                        size=[pretrained_dict[item].size(0), 256, 256])[0][0]

            elif network == "hrnet" or network == "xception" or network == 'resnest':
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
                Log.info('Missing keys: {}'.format(list(set(model_dict) - set(load_dict))))

            elif network == "dcnet" or network == "resnext":
                load_dict = dict()
                for k, v in pretrained_dict.items():
                    if 'resinit.{}'.format(k) in model_dict:
                        load_dict['resinit.{}'.format(k)] = v
                    else:
                        if k in model_dict:
                            load_dict[k] = v
                        else:
                            pass

            elif network == "wide_resnet":
                load_dict = {'.'.join(k.split('.')[1:]): v \
                             for k, v in pretrained_dict.items() \
                             if '.'.join(k.split('.')[1:]) in model_dict}
            else:
                load_dict = {'.'.join(k.split('.')[1:]): v \
                             for k, v in pretrained_dict.items() \
                             if '.'.join(k.split('.')[1:]) in model_dict}

            # used to debug
            if int(os.environ.get("debug_load_model", 0)):
                Log.info('Matched Keys List:')
                for key in load_dict.keys():
                    Log.info('{}'.format(key))
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)

        return model

    @staticmethod
    def load_url(url, map_location=None):
        model_dir = os.path.join('~', '.PyTorchCV', 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        filename = url.split('/')[-1]
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            Log.info('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)

        Log.info('Loading pretrained model:{}'.format(cached_file))
        return torch.load(cached_file, map_location=map_location)

    @staticmethod
    def constant_init(module, val, bias=0):
        nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def xavier_init(module, gain=1, bias=0, distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def uniform_init(module, a=0, b=1, bias=0):
        nn.init.uniform_(module.weight, a, b)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def kaiming_init(module,
                     mode='fan_in',
                     nonlinearity='leaky_relu',
                     bias=0,
                     distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)


def MLP(channels: List[int]) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Linear(channels[i - 1], channels[i], bias=True))

    return nn.Sequential(*layers)

def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

def graph_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, adj_matrix) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[0]
    # print(query.shape, key.shape, value.shape)
    scores = torch.einsum("ab, cb -> ac", query, key) / dim**.5
    adj_scores = torch.mul(scores, adj_matrix)
    adj_scores[abs(adj_scores) < 1e-5] = -1e9
    prob = torch.nn.functional.softmax(adj_scores, dim=-1)
    return torch.einsum("ab, bc -> ac", prob, value), prob

class GraphMultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Linear(d_model, d_model)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, adj_matrix) -> torch.Tensor:
        # batch_dim = query.size(0)
        query, key, value = [l(x)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = graph_attention(query, key, value, adj_matrix)
        return self.merge(x)

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = GraphMultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor, adj_matrix) -> torch.Tensor:
        message = self.attn(x, source, source, adj_matrix)
        return self.mlp(torch.cat([x, message], dim=1))


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
            
class Discriminator(nn.Module):
    def __init__(self, infeat, hidfeat, outfeat, dropout):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(infeat, hidfeat),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidfeat, outfeat),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output

    def weights_init(self):
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.apply(_weights_init)