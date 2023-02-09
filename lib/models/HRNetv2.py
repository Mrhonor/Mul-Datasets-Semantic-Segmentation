import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.HRNet_backbone import HRNetBackbone
from lib.models.tools.module_helper import ConvBNReLU
from lib.module.projection import ProjectionHead
from lib.module.domain_classifier_head import DomainClassifierHead

class SegmentHead(nn.Module):

    def __init__(self, in_chan, n_classes, up_factor=4, n_bn=1):
        super(SegmentHead, self).__init__()
         
        self.up_factor = int(up_factor)
        self.n_bn = n_bn

        self.conv1 = ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=1, padding=1, n_bn=self.n_bn)
        self.dropout = nn.Dropout2d(0.10),
        self.conv2 = nn.Conv2d(in_channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        
        
    def forward(self, dataset, x, *other_x):
        _, _, h, w = x.shape
        
        # 采用多分割头，所以不应该返回list
        feats = self.conv1(dataset, x, *other_x)
        feats = [self.dropout(feat) for feat in feats]
        feats = [self.conv2(feat) for feat in feats]
            
        if self.up_factor > 1:
            feats = [F.interpolate(input=feat, size=(h*self.up_factor, w*self.up_factor), mode='bilinear', align_corners=True) for feat in feats]

        return feats

# class DomainClassifierHead(nn.Module):
#     def __init__(self, dim_in, n_domain, n_bn=1):
#         super(DomainClassifierHead, self).__init__()

#         self.n_bn = n_bn

#         self.drop = nn.Dropout(0.1)
#         self.conv1 = ConvWNReLU(dim_in, 128, ks=3, stride=2, padding=1)
#         self.conv2 = ConvWNReLU(128, 192, ks=3, stride=2, padding=1)
#         self.conv3 = ConvWNReLU(192, 256, ks=3, stride=2, padding=1)
#         self.conv_last = nn.Conv2d(256, n_domain, kernel_size=1) # out: 8 x 16
            


#     def forward(self, x, *other_x):
#         if len(other_x) !=0:
#             batch_size = [x.shape[0]]
#             for i in range(0, len(other_x)):
#                 batch_size.append(other_x[i].shape[0])
#                 x = torch.cat((x, other_x[i]), 0)

#         feat = self.drop(x)
                
#         feat = self.conv1(feat)
#         feat = self.conv2(feat)
#         feat = self.conv3(feat)
        
#         feat = self.conv_last(feat)
            
#         # feat = self.classifierHead(x)

#         if self.classifier == 'convmlp' or self.classifier == 'convmlp_small':
#             feat = torch.mean(feat, dim=[2,3])
        
#         feats = []
#         if len(other_x) != 0:
#             begin_index = 0
#             for i in range(0, len(other_x) + 1):
#                 end_index = begin_index + batch_size[i]
#                 feats.append(feat[begin_index: end_index])
#         else:
#             feats.append(feat)
            
#         # feat = self.proj(feat)
#         return feats

class HRNet_W48_CONTRAST(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, configer):
        super(HRNet_W48_CONTRAST, self).__init__()
        self.configer = configer
        self.aux_mode = self.configer.get('aux_mode')
        self.n_bn = self.configer.get('n_bn')
        self.num_classes = self.configer.get('data', 'num_classes')
        self.n_datasets = self.configer.get('n_datasets')
        self.backbone = HRNetBackbone(configer)
        self.proj_dim = self.configer.get('contrast', 'proj_dim')
        self.full_res_stem = self.configer.get('hrnet', 'full_res_stem')
        
        if self.full_res_stem:
            up_fac = 1
        else:
            up_fac = 4

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384
        self.cls_head = SegmentHead(in_channels, self.num_classes, up_factor=up_fac, n_bn=self.n_bn)

        self.use_contrast = self.configer.get('contrast', 'use_contrast')
        if self.use_contrast:
            self.proj_head = ProjectionHead(dim_in=in_channels, proj_dim=self.proj_dim)
            
        # self.with_domain_adversarial = self.configer.get('network', 'with_domain_adversarial')
        # if self.with_domain_adversarial:
        #     self.DomainCls_head = DomainClassifierHead(in_channels, n_domain=self.n_datasets, )
        

    def forward(self, x_, *other_x, dataset=0):
        x = self.backbone(x_, *other_x, dataset=0)
        _, _, h, w = x[0][0].size()

        feat1 = x[0]
        feat2 = [F.interpolate(x_data, size=(h, w), mode="bilinear", align_corners=True) for x_data in x[1]]
        feat3 = [F.interpolate(x_data, size=(h, w), mode="bilinear", align_corners=True) for x_data in x[2]]
        feat4 = [F.interpolate(x_data, size=(h, w), mode="bilinear", align_corners=True) for x_data in x[3]]

        feats = [torch.cat([feat1_data, feat2_data, feat3_data, feat4_data], 1) 
                for feat1_data, feat2_data, feat3_data, feat4_data in zip(feat1, feat2, feat3, feat4)]
        
        out = self.cls_head(dataset, *feats)

        if self.aux_mode == 'train':
            emb = None
            if self.use_contrast:
                emb = self.proj_head(dataset, *feats)
            return {'seg': out, 'embed': emb}
        elif self.aux_mode == 'eval':
            return out[0]
        elif self.aux_mode == 'pred':
            pred = out[0].argmax(dim=1)
        else:
            raise NotImplementedError