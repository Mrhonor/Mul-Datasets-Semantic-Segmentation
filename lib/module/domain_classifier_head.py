import torch
import torch.nn as nn
from lib.module.module_helper import ConvBN, ConvBNReLU, ConvWNReLU, ModuleHelper

class DomainClassifierHead(nn.Module):
    def __init__(self, dim_in, n_domain, classifier='convmlp', bn_type='torchsyncbn', n_bn=1):
        super(DomainClassifierHead, self).__init__()

        self.n_bn = n_bn

        # self.drop = nn.Dropout(0.1)
        self.classifier = classifier
        if self.classifier == 'linear':
            # self.classifierHead = nn.Sequential(
            #     nn.Dropout(0.1),
            #     nn.Linear(dim_in, 1024),
            #     ModuleHelper.BNReLU(1024, bn_type='torchbn1d'),
            #     nn.Linear(1024, n_domain)
            # )
            raise Exception("domain classifier linear version no imp yet!")
        elif self.classifier == 'convmlp':
            # self.classifierHead = nn.Sequential(
            #     nn.Dropout(0.1),
            #     nn.Conv2d(dim_in, 128, kernel_size=3, stride=2, padding=1), # out: 32 X 64
            #     ModuleHelper.BNReLU(128, bn_type=bn_type),
            #     nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1), # out: 16 x 32
            #     ModuleHelper.BNReLU(192, bn_type=bn_type),
            #     nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1), # out: 8 x 16
            #     ModuleHelper.BNReLU(256, bn_type=bn_type),
            #     nn.Conv2d(256, n_domain, kernel_size=1), # out: 8 x 16
            # )
            
            self.drop = nn.Dropout(0.1)
            self.conv1 = ConvWNReLU(dim_in, 128, ks=3, stride=2, padding=1)
            self.conv2 = ConvWNReLU(128, 192, ks=3, stride=2, padding=1)
            self.conv3 = ConvWNReLU(192, 256, ks=3, stride=2, padding=1)
            self.conv_last = nn.Conv2d(256, n_domain, kernel_size=1) # out: 8 x 16
            
        elif self.classifier == 'convmlp_small':
            self.drop = nn.Dropout(0.1)
            self.conv1 = ConvWNReLU(dim_in, 256, ks=3, stride=2, padding=1)
            self.conv2 = ConvWNReLU(256, 512, ks=3, stride=2, padding=1)
            self.conv3 = ConvWNReLU(512, 1024, ks=3, stride=2, padding=1)
            self.conv_last = nn.Conv2d(1024, n_domain, kernel_size=1) # out: 8 x 16
            
            # self.classifierHead = nn.Sequential(
            #     nn.Dropout(0.1),
            #     nn.Conv2d(dim_in, 256, kernel_size=3, stride=2, padding=1), # out: 8 X 16
            #     ModuleHelper.BNReLU(256, bn_type=bn_type),
            #     nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # out: 4 x 8
            #     ModuleHelper.BNReLU(512, bn_type=bn_type),
            #     nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), # out: 2 x 4
            #     ModuleHelper.BNReLU(1024, bn_type=bn_type),
            #     nn.Conv2d(1024, n_domain, kernel_size=1), # out: 2 x 4
            # )
            

    def forward(self, x, *other_x):
        if len(other_x) !=0:
            batch_size = [x.shape[0]]
            for i in range(0, len(other_x)):
                batch_size.append(other_x[i].shape[0])
                x = torch.cat((x, other_x[i]), 0)

        feat = self.drop(x)
                
        feat = self.conv1(feat)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        
        feat = self.conv_last(feat)
            
        # feat = self.classifierHead(x)

        if self.classifier == 'convmlp' or self.classifier == 'convmlp_small':
            feat = torch.mean(feat, dim=[2,3])
        
        feats = []
        if len(other_x) != 0:
            begin_index = 0
            for i in range(0, len(other_x) + 1):
                end_index = begin_index + batch_size[i]
                feats.append(feat[begin_index: end_index])
        else:
            feats.append(feat)
            
        # feat = self.proj(feat)
        return feats
    
