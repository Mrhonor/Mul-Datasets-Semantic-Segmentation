import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.module_helper import ModuleHelper
from lib.module.module_helper import ConvBNReLU
# from lib.utils.tools.logger import Logger as Log


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, up_factor=8, proj='convmlp', bn_type='torchsyncbn', up_sample=False, down_sample=False, n_bn=1):
        super(ProjectionHead, self).__init__()

        # Log.info('proj_dim: {}'.format(proj_dim))
        self.n_bn = n_bn
        
        self.up_sample = up_sample
        if self.up_sample:
            self.Upsample = nn.Upsample(scale_factor=up_factor, mode='nearest')
            

        if proj == 'linear':
            raise Exception("Not Imp error")
            # if down_sample:
            #     raise Exception("Not Imp error")
                
            # self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            if down_sample:
                self.conv1 = ConvBNReLU(dim_in, dim_in*2, ks=3, stride=1, padding=1, n_bn=self.n_bn)
                self.conv_last = nn.Conv2d(dim_in*2, proj_dim, kernel_size=1, bias=True)
                
                # self.proj = nn.Sequential(
                #     nn.Conv2d(dim_in, dim_in*2, kernel_size=3, stride=1, padding=1),
                #     ModuleHelper.BNReLU(dim_in*2, bn_type=bn_type),
                #     nn.Conv2d(dim_in*2, proj_dim, kernel_size=1, bias=True)
                # )
            else:
                self.conv1 = ConvBNReLU(dim_in, dim_in*2, ks=3, stride=1, padding=1, n_bn=self.n_bn)
                self.conv_last = nn.Conv2d(dim_in*2, proj_dim, kernel_size=1, bias=True)
                
                
                # self.proj = nn.Sequential(
                #     nn.Conv2d(dim_in, dim_in, kernel_size=1),
                #     ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                #     nn.Conv2d(dim_in, proj_dim, kernel_size=1)
                # )

    def forward(self, dataset, x, *other_x):
        feats = self.conv1(dataset, x, *other_x)
        feats = [self.conv_last(feat) for feat in feats]

        feats = [F.normalize(feat, p=2, dim=1) for feat in feats]
        if self.up_sample:
            feats = [self.Upsample(feat) for feat in feats]

            
        # feat = self.proj(feat)

        return feats