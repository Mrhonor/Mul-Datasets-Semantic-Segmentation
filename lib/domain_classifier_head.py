import torch
import torch.nn as nn
from tools.module_helper import ModuleHelper

class DomainClassifierHead(nn.Module):
    def __init__(self, dim_in, n_domain, classifier='convmlp', bn_type='torchsyncbn'):
        super(DomainClassifierHead, self).__init__()



        # self.drop = nn.Dropout(0.1)
        self.classifier = classifier
        if self.classifier == 'linear':
            self.classifierHead = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(dim_in, 1024),
                ModuleHelper.BNReLU(1024, bn_type='torchbn1d'),
                nn.Linear(1024, n_domain)
            )
        elif self.classifier == 'convmlp':
            self.classifierHead = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(dim_in, 128, kernel_size=3, stride=2, padding=1), # out: 32 X 64
                ModuleHelper.BNReLU(128, bn_type=bn_type),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # out: 16 x 32
                ModuleHelper.BNReLU(256, bn_type=bn_type),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # out: 8 x 16
                ModuleHelper.BNReLU(512, bn_type=bn_type),
                nn.Conv2d(512, n_domain, kernel_size=3, stride=2, padding=1), # out: 4 x 8
            )

    def forward(self, x):
        feat = self.classifierHead(x)

        if self.classifier == 'convmlp':
            feat = torch.mean(feat, dim=[2,3])

            
        # feat = self.proj(feat)

        return feat