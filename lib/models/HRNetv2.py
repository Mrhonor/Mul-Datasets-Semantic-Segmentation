import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.HRNet_backbone import HRNetBackbone
from lib.module.module_helper import ConvBNReLU
from lib.module.projection import ProjectionHead
from lib.module.domain_classifier_head import DomainClassifierHead
from timm.models.layers import trunc_normal_

backbone_url = './res/hrnetv2_w48_imagenet_pretrained.pth'

class SegmentHead(nn.Module):

    def __init__(self, in_chan, n_classes, up_factor=4, n_bn=1):
        super(SegmentHead, self).__init__()
         
        self.up_factor = int(up_factor)
        self.n_bn = n_bn

        self.conv1 = ConvBNReLU(in_chan, in_chan, ks=3, stride=1, padding=1, n_bn=self.n_bn)
        self.dropout = nn.Dropout2d(0.10)
        self.conv2 = nn.Conv2d(in_chan, n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        
        
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
        self.num_unify_classes = self.configer.get('num_unify_classes')
        self.n_datasets = self.configer.get('n_datasets')
        self.backbone = HRNetBackbone(configer)
        self.proj_dim = self.configer.get('contrast', 'proj_dim')
        self.full_res_stem = self.configer.get('hrnet', 'full_res_stem')
        self.num_prototype = self.configer.get('contrast', 'num_prototype')
        
        if self.full_res_stem:
            up_fac = 1
        else:
            up_fac = 4

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384
        self.cls_head = SegmentHead(in_channels, self.num_unify_classes, up_factor=up_fac, n_bn=self.n_bn)

        self.use_contrast = self.configer.get('contrast', 'use_contrast')
        if self.use_contrast:
            self.proj_head = ProjectionHead(dim_in=in_channels, proj_dim=self.proj_dim)
            
        self.prototypes = nn.Parameter(torch.zeros(self.num_unify_classes, self.num_prototype, self.proj_dim),
                                       requires_grad=False)

        trunc_normal_(self.prototypes, std=0.02)
        self.init_weights()    
        
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
                    nowd_params.append(param)
                    # print(param.dim())
                    # print(param)
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
        
    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            # elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            #     if hasattr(module, 'last_bn') and module.last_bn:
            #         nn.init.zeros_(module.weight)
            #     else:
            #         nn.init.ones_(module.weight)
            #     nn.init.zeros_(module.bias)
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
        pass
        