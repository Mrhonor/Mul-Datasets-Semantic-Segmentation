import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.module_helper import ModuleHelper
# from lib.utils.tools.logger import Logger as Log


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, up_factor=8, proj='convmlp', bn_type='torchsyncbn', up_sample=False, down_sample=False):
        super(ProjectionHead, self).__init__()

        # Log.info('proj_dim: {}'.format(proj_dim))
        self.up_sample = up_sample
        if self.up_sample:
            self.Upsample = nn.Upsample(scale_factor=up_factor, mode='nearest')
            
        

        if proj == 'linear':
            if down_sample:
                raise Exception("Not Imp error")
                
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            if down_sample:
                self.proj = nn.Sequential(
                    nn.Conv2d(dim_in, dim_in*2, kernel_size=3, stride=2, padding=1),
                    ModuleHelper.BNReLU(dim_in*2, bn_type=bn_type),
                    nn.Conv2d(dim_in*2, proj_dim, kernel_size=1)
                )
            else:
                self.proj = nn.Sequential(
                    nn.Conv2d(dim_in, dim_in, kernel_size=1),
                    ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                    nn.Conv2d(dim_in, proj_dim, kernel_size=1)
                )

    def forward(self, x):
        feat = x
        feat = self.proj(feat)
        feat = F.normalize(feat, p=2, dim=1)
        if self.up_sample:
            feat = self.Upsample(feat)

            
        # feat = self.proj(feat)

        return feat