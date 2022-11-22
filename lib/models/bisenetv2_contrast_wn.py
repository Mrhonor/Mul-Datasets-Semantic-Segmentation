
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.projection import ProjectionHead
from lib.domain_classifier_head import DomainClassifierHead
from lib.functions import ReverseLayerF
from lib.module.ConvNorm import ConvNorm

import numpy as np
from timm.models.layers import trunc_normal_

from lib.sinkhorn import distributed_sinkhorn
from lib.momentum_update import momentum_update
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
from lib.module.module_helper import ConvWNReLU, ConvWN

import torch.distributed as dist


# backbone_url = 'https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth'
# backbone_url = '/root/autodl-tmp/project/BiSeNet/pth/backbone_v2.pth'
backbone_url = './res/backbone_v2.pth'
# backbone_url = './res/model_3000.pth'





class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)

        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)



class DetailBranch(nn.Module):

    def __init__(self):
        ## n_bn bn层数量，对应混合的数据集数量

        super(DetailBranch, self).__init__()
        self.S1_1 = ConvWNReLU(3, 64, 3, stride=2)
        self.S1_2 = ConvWNReLU(64, 64, 3, stride=1)

        self.S2_1 = ConvWNReLU(64, 64, 3, stride=2)
        self.S2_2 = ConvWNReLU(64, 64, 3, stride=1)
        self.S2_3 = ConvWNReLU(64, 64, 3, stride=1)
        
        self.S3_1 = ConvWNReLU(64, 128, 3, stride=2)
        self.S3_2 = ConvWNReLU(128, 128, 3, stride=1)
        self.S3_3 = ConvWNReLU(128, 128, 3, stride=1)
        

    def forward(self, x):
        ## other_x 其他数据集的输入
        ## 拆分列表传参
        # x = x.cuda()
        feats = self.S1_1(x)
        feats = self.S1_2(feats)

        feats = self.S2_1(feats)
        feats = self.S2_2(feats)
        feats = self.S2_3(feats)

        feats = self.S3_1(feats)
        feats = self.S3_2(feats)
        feats = self.S3_3(feats)

        return feats


class StemBlock(nn.Module):

    def __init__(self):
        ## n_bn bn层数量，对应混合的数据集数量

        super(StemBlock, self).__init__()
        self.conv = ConvWNReLU(3, 16, 3, stride=2)

        # self.left = nn.Sequential(
        #     ConvWNReLU(16, 8, 1, stride=1, padding=0),
        #     ConvWNReLU(8, 16, 3, stride=2),
        # )

        self.left_1 = ConvWNReLU(16, 8, 1, stride=1, padding=0)
        self.left_2 = ConvWNReLU(8, 16, 3, stride=2)
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvWNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left_1(feat)
        feat_left = self.left_2(feat_left)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)

        return feat


class CEBlock(nn.Module):

    def __init__(self):
        ## n_bn bn层数量，对应混合的数据集数量

        super(CEBlock, self).__init__()
        
        # self.bn = nn.ModuleList([nn.BatchNorm2d(128) for i in range(0, self.n_bn)])
        
        self.conv_gap = ConvWNReLU(128, 128, 1, stride=1, padding=0)
        #TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvWNReLU(128, 128, 3, stride=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        # feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        
        return feat


class GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        ## n_bn bn层数量，对应混合的数据集数量
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvWNReLU(in_chan, in_chan, 3, stride=1)
        
        self.dwconv = ConvWNReLU(in_chan, mid_chan, 3, groups=in_chan)
        
        self.conv2 = ConvWN(mid_chan, out_chan, ks=1, stride=1, padding=0)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat

class GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvWNReLU(in_chan, in_chan, 3, stride=1)

        self.dwconv1 = ConvWN(in_chan, mid_chan, ks=3, stride=2, 
                            padding=1, groups=in_chan, bias=False)
        self.dwconv2 = ConvWN(mid_chan, mid_chan, ks=3, stride=1, 
                            padding=1, groups=mid_chan, bias=False)
        self.conv2 = ConvWN(mid_chan, out_chan, ks=1, stride=1,
                            padding=0)
        
        self.shortcut_1 = ConvWN(in_chan, in_chan, ks=3, stride=2, padding=1, groups=in_chan, bias=False)
        self.shortcut_2 = ConvWN(in_chan, out_chan, ks=1, stride=1, padding=0, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut_1(x)
        shortcut = self.shortcut_2(shortcut)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class SegmentBranch(nn.Module):

    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3_1 = GELayerS2(16, 32)
        self.S3_2 = GELayerS1(32, 32)
        
        self.S4_1 = GELayerS2(32, 64)
        self.S4_2 = GELayerS1(64, 64)
        
        self.S5_4_1 = GELayerS2(64, 128)
        self.S5_4_2 = GELayerS1(128, 128)
        self.S5_4_3 = GELayerS1(128, 128)
        self.S5_4_4 = GELayerS1(128, 128)
        
        self.S5_5 = CEBlock()

    def forward(self, x):
        # x = x.cuda()
        feat2 = self.S1S2(x)

        feat3 = self.S3_1(feat2)
        feat3 = self.S3_2(feat3)

        feat4 = self.S4_1(feat3)
        feat4 = self.S4_2(feat4)

        feat5_4 = self.S5_4_1(feat4)
        feat5_4 = self.S5_4_2(feat5_4)
        feat5_4 = self.S5_4_3(feat5_4)
        feat5_4 = self.S5_4_4(feat5_4)

        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(nn.Module):

    def __init__(self):
        super(BGALayer, self).__init__()

        self.left1_ConvWN = ConvWN(128, 128, ks=3, groups=128)
        self.left1_conv = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.left2_ConvWN = ConvWN(128, 128, ks=3, stride=2)
        self.left2_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        
        self.right1 = ConvWN(128, 128, ks=3)
        
        self.right2_ConvWN = ConvWN(128, 128, ks=3, groups=128)
        self.right2_conv = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)

        ##TODO: does this really has no relu?
        self.conv = ConvWNReLU(128, 128, ks=3)
        

    def forward(self, x_d, x_s):
        ## x_d, x_s都是多数据集的list
        ## TODO 实现不定长参数版本
        # dsize = x_d.size()[2:]
        left1 = self.left1_ConvWN(x_d)
        left1 = self.left1_conv(left1) 

        left2 = self.left2_ConvWN(x_d)
        left2 = self.left2_pool(left2)

        right1 = self.right1(x_s)

        right2 = self.right2_ConvWN(x_s)
        right2 = self.right2_conv(right2)

        right1 = self.up1(right1)

        left = left1 * F.sigmoid(right1) 
        right = left2 * F.sigmoid(right2)
        right = self.up2(right)

        feats = left + right
        out = self.conv(feats)
        return out



class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, proj='ConvNorm', aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvWNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor

        # self.conv_out = nn.Sequential(
        #     nn.Sequential(
        #         nn.Upsample(scale_factor=2),
        #         ConvWNReLU(mid_chan, mid_chan2, 3, stride=1)
        #         ) if aux else nn.Identity(),
        #     nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
        #     nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        # )

        self.aux = aux
        self.up_sample1 = nn.Upsample(scale_factor=2)
        self.conv1 = ConvWNReLU(mid_chan, mid_chan2, 3, stride=1)

        if proj == 'convmlp':
            self.proj = nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True)
        elif proj == 'ConvNorm': 
            self.proj = ConvNorm(mid_chan2, out_chan)
        
        
        if self.up_factor > 1:
            self.up_sample2 = nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=True)
        

    def forward(self, x):
        # print(x.size())
        
        # 采用多分割头，所以不应该返回list
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = feat

        if self.aux is True:
            feat = self.up_sample1(feat)
            feat = self.conv1(feat)
            feat = self.proj(feat)
        else:
            feat = self.proj(feat)
            
        if self.up_factor > 1:
            feat = self.up_sample2(feat)

        return feat


class BiSeNetV2_Contrast_WN(nn.Module):

    def __init__(self, configer):
        super(BiSeNetV2_Contrast_WN, self).__init__()
        self.configer = configer
        self.aux_mode = self.configer.get('aux_mode')
        self.num_unify_classes = self.configer.get("num_unify_classes")
        self.n_datasets = self.configer.get('n_datasets')
        
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()
        self.network_stride = self.configer.get('network', 'stride')
        ## unify proj head
        self.proj_dim = self.configer.get('contrast', 'proj_dim')
        self.upsample = self.configer.get('contrast', 'upsample') 
        self.downsample = self.configer.get('contrast', 'downsample')
        self.num_prototype = self.configer.get('contrast', 'num_prototype')
        self.coefficient = self.configer.get('contrast', 'coefficient')
        
        # 用于分数据集训练
        self.batch_sizes = [self.configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, self.n_datasets+1)]
        
        # 分数据集训练阶段
        self.train_dataset_aux = False
        
        self.use_dataset_aux_head = self.configer.get('dataset_aux_head', 'use_dataset_aux_head')
        if self.use_dataset_aux_head:
            self.dataset_aux_head =  nn.ModuleList([SegmentHead(128, 1024, self.configer.get('dataset'+str(i), 'n_cats'), up_factor=8, aux=False)
                                                                                    for i in range(1, self.n_datasets+1)])
            
            self.train_dataset_aux = True
            
            if self.aux_mode == 'train':
                self.dataset_aux2 = nn.ModuleList([SegmentHead(16, 128, self.configer.get('dataset'+str(i), 'n_cats'), 
                                                               up_factor=4, aux=True) for i in range(1, self.n_datasets+1)])
                self.dataset_aux3 = nn.ModuleList([SegmentHead(32, 128, self.configer.get('dataset'+str(i), 'n_cats'), 
                                                               up_factor=8, aux=True) for i in range(1, self.n_datasets+1)])
                self.dataset_aux4 = nn.ModuleList([SegmentHead(64, 128, self.configer.get('dataset'+str(i), 'n_cats'), 
                                                               up_factor=16, aux=True) for i in range(1, self.n_datasets+1)])
                self.dataset_aux5_4 = nn.ModuleList([SegmentHead(128, 128, self.configer.get('dataset'+str(i), 'n_cats'), 
                                                               up_factor=32, aux=True) for i in range(1, self.n_datasets+1)])
                
            
        self.use_contrast = self.configer.get('contrast', 'use_contrast')

        if self.use_contrast:
            
            if configer.get('use_sync_bn'):
                self.projHead = ProjectionHead(dim_in=128, proj_dim=self.proj_dim, up_factor=self.network_stride, bn_type='torchsyncbn', up_sample=self.upsample, down_sample=self.downsample)
            else:
                self.projHead = ProjectionHead(dim_in=128, proj_dim=self.proj_dim, up_factor=self.network_stride, bn_type='torchbn', up_sample=self.upsample, down_sample=self.downsample)
            
            
        self.with_domain_adversarial = self.configer.get('network', 'with_domain_adversarial')
        if self.with_domain_adversarial:
            if configer.get('use_sync_bn'):
                self.DomainClassifierHead1 = DomainClassifierHead(dim_in=128, n_domain=self.n_datasets, classifier='convmlp_small', bn_type='torchsyncbn')
                self.DomainClassifierHead2 = DomainClassifierHead(dim_in=128, n_domain=self.n_datasets, classifier='convmlp', bn_type='torchsyncbn')
            else:
                self.DomainClassifierHead1 = DomainClassifierHead(dim_in=128, n_domain=self.n_datasets, classifier='convmlp_small', bn_type='torchbn')
                self.DomainClassifierHead2 = DomainClassifierHead(dim_in=128, n_domain=self.n_datasets, classifier='convmlp', bn_type='torchsyncbn')

        # ## TODO: what is the number of mid chan ?
        # self.head = nn.ModuleList([])
        # if self.aux_mode == 'train':
        #     self.aux2 = nn.ModuleList([])
        #     self.aux3 = nn.ModuleList([])
        #     self.aux4 = nn.ModuleList([])
        #     self.aux5_4 = nn.ModuleList([])

        ## 多数据集的头
        # self.n_head = len(other_n_classes) + 1
        # if self.n_head > 1:
        # for n in other_n_classes:
        #     self.head.append(SegmentHead(128, 1024, n, up_factor=8, aux=False))
        #     if self.aux_mode == 'train':
        #         self.aux2.append(SegmentHead(16, 128, n, up_factor=4))
        #         self.aux3.append(SegmentHead(32, 128, n, up_factor=8))
        #         self.aux4.append(SegmentHead(64, 128, n, up_factor=16))
        #         self.aux5_4.append(SegmentHead(128, 128, n, up_factor=32))

        self.head = SegmentHead(128, 1024, self.num_unify_classes, up_factor=8, aux=False)

            
        if self.aux_mode == 'train':
            self.aux2 = SegmentHead(16, 128, self.num_unify_classes, up_factor=4, aux=True)
            self.aux3 = SegmentHead(32, 128, self.num_unify_classes, up_factor=8, aux=True)
            self.aux4 = SegmentHead(64, 128, self.num_unify_classes, up_factor=16, aux=True)
            self.aux5_4 = SegmentHead(128, 128, self.num_unify_classes, up_factor=32, aux=True)

            
        self.prototypes = nn.Parameter(torch.zeros(self.num_unify_classes, self.num_prototype, self.proj_dim),
                                       requires_grad=False)

        trunc_normal_(self.prototypes, std=0.02)
        self.init_weights()

    def forward(self, x, perm_index=None, dataset=0):
        # perm_index用于恢复batch size中数据集来源信息，临时使用
        
        # x = x.cuda()
        ## other_x 其他数据集的输入
        # size = x.size()[2:]

        
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)

        # logits = self.head[0](feat_head[0])
        ## 修改为多数据集模式，返回list
        
        # logits = [logit(feat_head[i]) for i, logit in enumerate(self.head)]
        ## 修改后支持单张图片输入

        

        if self.aux_mode == 'train' and self.train_dataset_aux == False:
            logits = self.head(feat_head)
            logits_aux2 = self.aux2(feat2)
            logits_aux3 = self.aux3(feat3)
            logits_aux4 = self.aux4(feat4)
            logits_aux5_4 = self.aux5_4(feat5_4)
            
            domain_pred1 = None
            domain_pred2 = None
            if self.with_domain_adversarial:
                p = float(self.configer.get('iter')) / self.configer.get('lr', 'max_iter')
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                reverse_feat1 = ReverseLayerF.apply(feat_s, alpha)
                domain_pred1 = self.DomainClassifierHead1(reverse_feat1)
                
                reverse_feat2 = ReverseLayerF.apply(feat_head, alpha)
                domain_pred2 = self.DomainClassifierHead2(reverse_feat2)
                
            
            if not self.use_contrast:
                return {'seg': [logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4], 'domain': [domain_pred1, domain_pred2]}
   
            emb = self.projHead(feat_s)
            
            return {'seg': [logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4], 'embed': emb, 'domain': [domain_pred1, domain_pred2]}
                
            # return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        elif self.aux_mode == 'train' and self.train_dataset_aux == True and perm_index != None:
            acc = 0
            sort_index = perm_index.sort()[1]
            recover_index = []
            
            for i in range(self.n_datasets):
                recover_index.append(sort_index[acc:acc+self.batch_sizes[i]])
                acc += self.batch_sizes[i] 
            
            
            ## 多数据集模式
            ## 修改后支持单张图片输入
            logits = [self.dataset_aux_head[i](feat_head[recover_index[i]]) for i in range(0, self.n_datasets)]
            
            logits_aux2 = [self.dataset_aux2[i](feat2[recover_index[i]]) for i in range(0, self.n_datasets)]
            logits_aux3 = [self.dataset_aux3[i](feat3[recover_index[i]]) for i in range(0, self.n_datasets)]
            logits_aux4 = [self.dataset_aux4[i](feat4[recover_index[i]]) for i in range(0, self.n_datasets)]
            logits_aux5_4 = [self.dataset_aux5_4[i](feat5_4[recover_index[i]]) for i in range(0, self.n_datasets)]
            
            return {'seg': [logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4]}
   
 
            # return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        elif self.aux_mode == 'eval':
            if self.train_dataset_aux:
                logits = self.dataset_aux_head[dataset](feat_head)
            else:    
                logits = self.head(feat_head)
                # logits = [self.head(feat_head[i]) for i in range(0, len(other_x) + 1)]

                
            return logits
        elif self.aux_mode == 'pred':
            # logits = [self.head(feat_head[i]) for i in range(0, len(other_x) + 1)]
            logits = self.head(feat_head)
            # pred = logits.argmax(dim=1)
            # if self.upsample is False:
            #     logits = [self.up_sample(logit) for logit in logits] 
            # print(logits[0].argmax(dim=1).shape)
            pred = logits.argmax(dim=1)
            
            
            logit = F.softmax(logits, dim=1)
            maxV = torch.max(logit, dim=1)[0]
            
            return pred, maxV
        elif self.aux_mode == 'pred_by_emb':
            emb = self.projHead(feat_head)
            
            simScore = torch.einsum('bchw,nc->bnhw', emb, self.prototypes)
            # if self.upsample is False:
            #     simScore = [self.up_sample(score) for score in simScore] 
            # print(simScore[0].argmax(dim=1).shape)
            
            MaxSimIndex = simScore.argmax(dim=1)
            return MaxSimIndex
            
        else:
            raise NotImplementedError

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for name, param in self.named_parameters():
            if name.find('affine_weight') != -1:
                if hasattr(param, 'last_bn') and param.last_bn:
                    nn.init.zeros_(param)
                else:
                    nn.init.ones_(param)
            elif name.find('affine_bias') != -1:
                nn.init.zeros_(param)

        self.load_pretrain()


    def load_pretrain(self):
        # state = modelzoo.load_url(backbone_url)
        state = torch.load(backbone_url)
        detail_state = {}
        def loadConvWNReLU(srcDict, src_name, targetdict, target_name):
            targetdict[target_name+'.conv.weight'] = srcDict[src_name+'.conv.weight']
            targetdict[target_name+'.affine_weight'] = srcDict[src_name+'.bn.weight']
            targetdict[target_name+'.affine_bias'] = srcDict[src_name+'.bn.bias']

        loadConvWNReLU(state['detail'], 'S1.0', detail_state, 'S1_1')
        loadConvWNReLU(state['detail'], 'S1.1', detail_state, 'S1_2')
        loadConvWNReLU(state['detail'], 'S2.0', detail_state, 'S2_1')
        loadConvWNReLU(state['detail'], 'S2.1', detail_state, 'S2_2')
        loadConvWNReLU(state['detail'], 'S2.2', detail_state, 'S2_3')
        loadConvWNReLU(state['detail'], 'S3.0', detail_state, 'S3_1')
        loadConvWNReLU(state['detail'], 'S3.1', detail_state, 'S3_2')
        loadConvWNReLU(state['detail'], 'S3.2', detail_state, 'S3_3')

        segment_state = {}
        loadConvWNReLU(state['segment'], 'S1S2.conv', segment_state, 'S1S2.conv')
        loadConvWNReLU(state['segment'], 'S1S2.left.0', segment_state, 'S1S2.left_1')
        loadConvWNReLU(state['segment'], 'S1S2.left.1', segment_state, 'S1S2.left_2')
        loadConvWNReLU(state['segment'], 'S1S2.fuse', segment_state, 'S1S2.fuse')

        def loadGELayerS2(srcDict, src_name, targetdict, target_name):
            loadConvWNReLU(srcDict, src_name+'.conv1', targetdict, target_name+'.conv1')
            
            targetdict[target_name+'.dwconv1.conv.weight'] = srcDict[src_name+'.dwconv1.0.weight']
            targetdict[target_name+'.dwconv1.affine_weight'] = srcDict[src_name+'.dwconv1.1.weight']
            targetdict[target_name+'.dwconv1.affine_bias'] = srcDict[src_name+'.dwconv1.1.bias']
            
            targetdict[target_name+'.dwconv2.conv.weight'] = srcDict[src_name+'.dwconv2.0.weight']
            targetdict[target_name+'.dwconv2.affine_weight'] = srcDict[src_name+'.dwconv2.1.weight']
            targetdict[target_name+'.dwconv2.affine_bias'] = srcDict[src_name+'.dwconv2.1.bias']
            
            targetdict[target_name+'.conv2.conv.weight'] = srcDict[src_name+'.conv2.0.weight']
            targetdict[target_name+'.conv2.affine_weight'] = srcDict[src_name+'.conv2.1.weight']
            targetdict[target_name+'.conv2.affine_bias'] = srcDict[src_name+'.conv2.1.bias']
            
            targetdict[target_name+'.shortcut_1.conv.weight'] = srcDict[src_name+'.shortcut.0.weight']
            targetdict[target_name+'.shortcut_1.affine_weight'] = srcDict[src_name+'.shortcut.1.weight']
            targetdict[target_name+'.shortcut_1.affine_bias'] = srcDict[src_name+'.shortcut.1.bias']
            
            targetdict[target_name+'.shortcut_2.conv.weight'] = srcDict[src_name+'.shortcut.2.weight']
            targetdict[target_name+'.shortcut_2.affine_weight'] = srcDict[src_name+'.shortcut.3.weight']
            targetdict[target_name+'.shortcut_2.affine_bias'] = srcDict[src_name+'.shortcut.3.bias']

        def loadGELayerS1(srcDict, src_name, targetdict, target_name):
            loadConvWNReLU(srcDict, src_name+'.conv1', targetdict, target_name+'.conv1')
            
            targetdict[target_name+'.dwconv.conv.weight'] = srcDict[src_name+'.dwconv.0.weight']
            targetdict[target_name+'.dwconv.affine_weight'] = srcDict[src_name+'.dwconv.1.weight']
            targetdict[target_name+'.dwconv.affine_bias'] = srcDict[src_name+'.dwconv.1.bias']
            
            targetdict[target_name+'.conv2.conv.weight'] = srcDict[src_name+'.conv2.0.weight']
            targetdict[target_name+'.conv2.affine_weight'] = srcDict[src_name+'.conv2.1.weight']
            targetdict[target_name+'.conv2.affine_bias'] = srcDict[src_name+'.conv2.1.bias']
            
        loadGELayerS2(state['segment'], 'S3.0', segment_state, 'S3_1')
        loadGELayerS1(state['segment'], 'S3.1', segment_state, 'S3_2')
        loadGELayerS2(state['segment'], 'S4.0', segment_state, 'S4_1')
        loadGELayerS1(state['segment'], 'S4.1', segment_state, 'S4_2')
        loadGELayerS2(state['segment'], 'S5_4.0', segment_state, 'S5_4_1')
        loadGELayerS1(state['segment'], 'S5_4.1', segment_state, 'S5_4_2')
        loadGELayerS1(state['segment'], 'S5_4.2', segment_state, 'S5_4_3')
        loadGELayerS1(state['segment'], 'S5_4.3', segment_state, 'S5_4_4')
        loadConvWNReLU(state['segment'], 'S5_5.conv_gap', segment_state, 'S5_5.conv_gap')
        loadConvWNReLU(state['segment'], 'S5_5.conv_last', segment_state, 'S5_5.conv_last')
        # segment_state['S5_5.conv_gap.conv.weight'] = state['segment']['S5_5.conv_gap.conv.weight']
        # segment_state['S5_5.conv_last.conv.weight'] = state['segment']['S5_5.conv_last.conv.weight']

        bga_state = {}
        bga_state['left1_ConvWN.conv.weight'] = state['bga']['left1.0.weight']
        bga_state['left1_ConvWN.affine_weight'] = state['bga']['left1.1.weight']
        bga_state['left1_ConvWN.affine_bias'] = state['bga']['left1.1.bias']

        bga_state['left1_conv.weight'] = state['bga']['left1.2.weight']

        bga_state['left2_ConvWN.conv.weight'] = state['bga']['left2.0.weight']
        bga_state['left2_ConvWN.affine_weight'] = state['bga']['left2.1.weight']
        bga_state['left2_ConvWN.affine_bias'] = state['bga']['left2.1.bias']

        bga_state['right1.conv.weight'] = state['bga']['right1.0.weight']
        bga_state['right1.affine_weight'] = state['bga']['right1.1.weight']
        bga_state['right1.affine_bias'] = state['bga']['right1.1.bias']

        bga_state['right2_ConvWN.conv.weight'] = state['bga']['right2.0.weight']
        bga_state['right2_ConvWN.affine_weight'] = state['bga']['right2.1.weight']
        bga_state['right2_ConvWN.affine_bias'] = state['bga']['right2.1.bias']

        bga_state['right2_conv.weight'] = state['bga']['right2.2.weight']

        bga_state['conv.conv.weight'] = state['bga']['conv.0.weight']
        bga_state['conv.affine_weight'] = state['bga']['conv.1.weight']
        bga_state['conv.affine_bias'] = state['bga']['conv.1.bias']

        
        self.detail.load_state_dict(detail_state, strict=False)
        self.segment.load_state_dict(segment_state, strict=False)
        self.bga.load_state_dict(bga_state, strict=False)

    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.requires_grad == False:
                    continue
                
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4:
                    wd_params.append(param)
                else:
                    print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
    
    def set_train_dataset_aux(self, new_val=False):
        self.train_dataset_aux = new_val

    def switch_require_grad_state(self, require_grad_state=True):
        for p in self.detail.parameters():
            p.requires_grad = require_grad_state
            
        for p in self.segment.parameters():
            p.requires_grad = require_grad_state
            
        for p in self.bga.parameters():
            p.requires_grad = require_grad_state
        
    def PrototypesUpdate(self, new_proto):
        self.prototypes = nn.Parameter(F.normalize(new_proto, p=2, dim=-1),
                                        requires_grad=False)

        # if dist.is_available() and dist.is_initialized():
        #     protos = self.prototypes.data.clone()
        #     dist.all_reduce(protos.div_(dist.get_world_size()))
        #     self.prototypes = nn.Parameter(protos, requires_grad=False)
            

if __name__ == "__main__":

    # x1 = torch.randn(1, 3, 512, 1024).cuda()
    # x2 = torch.randn(1, 3, 512, 1024).cuda()
    # model = BiSeNetV2(38, 'eval', 2, 19)
    
    # model.cuda()
    # model.eval()
    
    # out = model(x1, 0, x2)
    # print(out)
    
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # repetitions = 100
    # timings = np.zeros((repetitions, 1))
    # end = time.time()
    # # for i in range(0, repetitions):
    # #     starter.record()
    # #     start = end
    # #     outs = model(x1, x2)
    # #     ender.record()
    # #     end = time.time()
    # #     torch.cuda.synchronize()
    # #     curr_time = starter.elapsed_time(ender)
    # #     curr_time1 = end - start
    # #     print('inference time:', 1000 * curr_time1)
    # #     timings[i] = curr_time1
    # #     print(curr_time)
    
    # for i in range(0, repetitions):
    #     start = end
    #     outs = model(x1, 0)
    #     end = time.time()
    #     print(outs[0][0].shape)
    #     curr_time1 = end - start
    #     print('inference time:', 1000 * curr_time1)
    #     timings[i] = curr_time1
    # timings = timings[10:]
    # mean_time = timings.mean().item()
    # print("Inference time: {:.6f}, FPS: {} ".format(mean_time*1000, 1/mean_time))
    

    # # for name, param in model.named_parameters():
    # #     if len(param.size()) == 1:
    # #         print(name)
    # total = sum([param.nelement() for param in model.parameters()])
    # print(total / 1e6)
    # d_total = sum([param.nelement() for param in model.bga.parameters()])
    # print(d_total / 1e6)
    # d_total = sum([param.nelement() for param in model.detail.parameters()])
    # print(d_total / 1e6)
    # d_total = sum([param.nelement() for param in model.segment.parameters()])
    # print(d_total / 1e6)
    # d_total = sum([param.nelement() for param in model.head.parameters()])
    # print(d_total / 1e6)
    
    weight = nn.Parameter(torch.tensor([1.0,-1.0]))
    bias = nn.Parameter(torch.tensor([1.0,-1.0]))
    print(weight)
    print(bias)
    a = torch.randn(2,2,2,2)
    print(a)

    a = a * weight.reshape(1,-1,1,1) + bias.reshape(1,-1,1,1)

    print(a)
    
    init_test = nn.Parameter(torch.empty(10))
    print(init_test)
    nn.init.ones_(init_test)
    print(init_test)