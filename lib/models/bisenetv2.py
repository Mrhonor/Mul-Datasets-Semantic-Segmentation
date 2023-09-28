
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

# backbone_url = 'https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth'
# backbone_url = '/root/autodl-tmp/project/BiSeNet/pth/backbone_v2.pth'
backbone_url = './res/backbone_v2.pth'
# backbone_url = './res/model_3000.pth'

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, n_bn=1):
        ## n_bn bn层数量，对应混合的数据集数量
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.ModuleList([nn.BatchNorm2d(out_chan, affine=False) for i in range(0, n_bn)])
        ## 采用共享的affine parameter
        self.affine_weight = nn.Parameter(torch.empty(out_chan))
        self.affine_bias = nn.Parameter(torch.empty(out_chan))
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, dataset, x, *other_x):
        ## TODO 此处可以优化，不同数据集的图像过卷积层可以拼接到一起，过BN层再分离
        ## 处理多数据集情况
        
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
            ## 在relu时再拼接后分离，测试发现fps反而有所下降
            # feats_out = torch.cat((feats[0], feats[1]))
            # for i in range(1, len(other_x)):
            #     feats_out = torch.cat((feats_out, feats[i]), 0)
            # feats_out = self.relu(feats_out)
            # feats = []
            # for i in range(0, len(other_x) + 1):
            #     feats.append(feats_out[i * batch_size: (i + 1) * batch_size])
        else:
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
        self.bn = nn.ModuleList([nn.BatchNorm2d(out_chan, affine=False) for i in range(0, n_bn)])
        ## 采用共享的affine parameter
        self.affine_weight = nn.Parameter(torch.empty(out_chan))
        self.affine_bias = nn.Parameter(torch.empty(out_chan))
        

    def forward(self, dataset, x, *other_x):
        ## TODO 此处可以优化，不同数据集的图像过卷积层可以拼接到一起，过BN层再分离
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



class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x, *other_x):
        feat = self.proj(x)
        feat = self.up(feat)

        ## 处理多数据集情况
        feats = [feat]
        for xs in other_x:
            feat = self.proj(xs)
            feat = self.up(feat)
            feats.append(feat)

        return feats

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)



class DetailBranch(nn.Module):

    def __init__(self, n_bn=1):
        ## n_bn bn层数量，对应混合的数据集数量

        super(DetailBranch, self).__init__()
        self.S1_1 = ConvBNReLU(3, 64, 3, stride=2, n_bn=n_bn)
        self.S1_2 = ConvBNReLU(64, 64, 3, stride=1, n_bn=n_bn)

        self.S2_1 = ConvBNReLU(64, 64, 3, stride=2, n_bn=n_bn)
        self.S2_2 = ConvBNReLU(64, 64, 3, stride=1, n_bn=n_bn)
        self.S2_3 = ConvBNReLU(64, 64, 3, stride=1, n_bn=n_bn)
        
        self.S3_1 = ConvBNReLU(64, 128, 3, stride=2, n_bn=n_bn)
        self.S3_2 = ConvBNReLU(128, 128, 3, stride=1, n_bn=n_bn)
        self.S3_3 = ConvBNReLU(128, 128, 3, stride=1, n_bn=n_bn)
        

    def forward(self, x, dataset=0, *other_x):
        ## other_x 其他数据集的输入
        ## 拆分列表传参
        # x = x.cuda()
        feats = self.S1_1(dataset, x, *other_x)
        feats = self.S1_2(dataset, *feats)

        feats = self.S2_1(dataset, *feats)
        feats = self.S2_2(dataset, *feats)
        feats = self.S2_3(dataset, *feats)

        feats = self.S3_1(dataset, *feats)
        feats = self.S3_2(dataset, *feats)
        feats = self.S3_3(dataset, *feats)

        return feats


class StemBlock(nn.Module):

    def __init__(self, n_bn=1):
        ## n_bn bn层数量，对应混合的数据集数量

        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2, n_bn=n_bn)

        # self.left = nn.Sequential(
        #     ConvBNReLU(16, 8, 1, stride=1, padding=0, n_bn=n_bn),
        #     ConvBNReLU(8, 16, 3, stride=2, n_bn=n_bn),
        # )

        self.left_1 = ConvBNReLU(16, 8, 1, stride=1, padding=0, n_bn=n_bn)
        self.left_2 = ConvBNReLU(8, 16, 3, stride=2, n_bn=n_bn)
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1, n_bn=n_bn)

    def forward(self, dataset, x, *other_x):
        # feat = self.conv(x)
        # feat_left = self.left(feat)
        # feat_right = self.right(feat)
        # feat = torch.cat([feat_left, feat_right], dim=1)
        # feat = self.fuse(feat)

        ## 修改为处理多数据集的情况
        feats = self.conv(dataset, x, *other_x)
        feat_lefts = self.left_1(dataset, *feats)
        feat_lefts = self.left_2(dataset, *feat_lefts)
        feat_rights = [self.right(feat) for feat in feats] 
        feats = [torch.cat([feat_lefts[i], feat_rights[i]], dim=1) for i in range(0, len(feat_lefts))]
        feats = self.fuse(dataset, *feats)

        return feats


class CEBlock(nn.Module):

    def __init__(self, n_bn=1):
        ## n_bn bn层数量，对应混合的数据集数量

        super(CEBlock, self).__init__()
        self.bn = nn.ModuleList([nn.BatchNorm2d(128) for i in range(0, n_bn)])
        # # 所有list的模型都需要手动.cuda()
        # for i in range(0, n_bn):
        #     self.bn[i].cuda()
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0, n_bn=n_bn)
        #TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1, n_bn=n_bn)

    def forward(self, dataset, x, *other_x):
        # feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = x.view(x.shape[0], x.shape[1], -1)
        feat = feat.mean(2)
        feat = feat.view(feat.shape[0], feat.shape[1], 1, 1)

        ## 多数据集处理部分
        feats = [feat]
        for xs in other_x:
            feat = xs.view(xs.shape[0], xs.shape[1], -1)
            feat = feat.mean(2)
            feat = feat.view(feat.shape[0], feat.shape[1], 1, 1)
            feats.append(feat)
        if (len(other_x) == 0):
            feats_bn = [self.bn[dataset](feats[0])]
        else:
            feats_bn = [self.bn[i](feat) for i, feat in enumerate(feats)] 
        feats_gap = self.conv_gap(dataset, *feats_bn)
        # feat = feat + x

        feats = [F.interpolate(feats_gap[0], size=(x.shape[2], x.shape[3])) + x]
        for i, xs in enumerate(other_x):
            feats.append(F.interpolate(feats_gap[i+1], size=(xs.shape[2], xs.shape[3])) + xs)

        feats = self.conv_last(dataset, *feats)
        return feats


class GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6, n_bn=1):
        super(GELayerS1, self).__init__()
        ## n_bn bn层数量，对应混合的数据集数量
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1, n_bn=n_bn)
        # self.dwconv = nn.Sequential(
        #     nn.Conv2d(
        #         in_chan, mid_chan, kernel_size=3, stride=1,
        #         padding=1, groups=in_chan, bias=False),
        #     nn.BatchNorm2d(mid_chan),
        #     nn.ReLU(inplace=True), # not shown in paper
        # )
        self.dwconv = ConvBNReLU(in_chan, mid_chan, 3, groups=in_chan, n_bn=n_bn)
        
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(
        #         mid_chan, out_chan, kernel_size=1, stride=1,
        #         padding=0, bias=False),
        #     nn.BatchNorm2d(out_chan),
        # )
        # self.conv2[1].last_bn = True
        self.conv2 = ConvBN(mid_chan, out_chan, ks=1, stride=1, padding=0, n_bn=n_bn)
        self.conv2.SetLastBNAttr(True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, dataset, x, *other_x):
        feats = self.conv1(dataset, x, *other_x)
        feats = self.dwconv(dataset, *feats)
        feats_conv = self.conv2(dataset, *feats)
        feat = feats_conv[0] + x
        feats = [feat]
        
        for i, xs in enumerate(other_x):
            feats.append(feats_conv[i+1] + xs)

        feats = [self.relu(feat) for feat in feats]
        return feats


class GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6, n_bn=1):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1, n_bn=n_bn)
        # self.dwconv1 = nn.Sequential(
        #     nn.Conv2d(
        #         in_chan, mid_chan, kernel_size=3, stride=2,
        #         padding=1, groups=in_chan, bias=False),
        #     nn.BatchNorm2d(mid_chan),
        # )
        # self.dwconv2 = nn.Sequential(
        #     nn.Conv2d(
        #         mid_chan, mid_chan, kernel_size=3, stride=1,
        #         padding=1, groups=mid_chan, bias=False),
        #     nn.BatchNorm2d(mid_chan),
        #     nn.ReLU(inplace=True), # not shown in paper
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(
        #         mid_chan, out_chan, kernel_size=1, stride=1,
        #         padding=0, bias=False),
        #     nn.BatchNorm2d(out_chan),
        # )
        # self.conv2[1].last_bn = True
        # self.shortcut = nn.Sequential(
        #         nn.Conv2d(
        #             in_chan, in_chan, kernel_size=3, stride=2,
        #             padding=1, groups=in_chan, bias=False),
        #         nn.BatchNorm2d(in_chan),
        #         nn.Conv2d(
        #             in_chan, out_chan, kernel_size=1, stride=1,
        #             padding=0, bias=False),
        #         nn.BatchNorm2d(out_chan),
        # )
        self.dwconv1 = ConvBN(in_chan, mid_chan, ks=3, stride=2, 
                            padding=1, groups=in_chan, bias=False, n_bn=n_bn)
        self.dwconv2 = ConvBN(mid_chan, mid_chan, ks=3, stride=1, 
                            padding=1, groups=mid_chan, bias=False, n_bn=n_bn)
        self.conv2 = ConvBN(mid_chan, out_chan, ks=1, stride=1,
                            padding=0, n_bn=n_bn)
        self.conv2.SetLastBNAttr(True)
        self.shortcut_1 = ConvBN(in_chan, in_chan, ks=3, stride=2, padding=1, groups=in_chan, bias=False, n_bn=n_bn)
        self.shortcut_2 = ConvBN(in_chan, out_chan, ks=1, stride=1, padding=0, bias=False, n_bn=n_bn)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, dataset, x, *other_x):
        ## 修改为多数据集模式
        feats = self.conv1(dataset, x, *other_x)
        feats = self.dwconv1(dataset, *feats)
        feats = self.dwconv2(dataset, *feats)
        feats = self.conv2(dataset, *feats)
        shortcuts = self.shortcut_1(dataset, x, *other_x)
        shortcuts = self.shortcut_2(dataset, *shortcuts)
        feats = [feat + shortcuts[i] for i, feat in enumerate(feats)]
        feats = [self.relu(feat) for feat in feats]
        return feats


class SegmentBranch(nn.Module):

    def __init__(self, n_bn=1):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock(n_bn=n_bn)
        self.S3_1 = GELayerS2(16, 32, n_bn=n_bn)
        self.S3_2 = GELayerS1(32, 32, n_bn=n_bn)
        
        self.S4_1 = GELayerS2(32, 64, n_bn=n_bn)
        self.S4_2 = GELayerS1(64, 64, n_bn=n_bn)
        
        self.S5_4_1 = GELayerS2(64, 128, n_bn=n_bn)
        self.S5_4_2 = GELayerS1(128, 128, n_bn=n_bn)
        self.S5_4_3 = GELayerS1(128, 128, n_bn=n_bn)
        self.S5_4_4 = GELayerS1(128, 128, n_bn=n_bn)
        
        self.S5_5 = CEBlock(n_bn=n_bn)

    def forward(self, x, dataset=0, *other_x):
        # x = x.cuda()
        feat2 = self.S1S2(dataset, x, *other_x)

        feat3 = self.S3_1(dataset, *feat2)
        feat3 = self.S3_2(dataset, *feat3)

        feat4 = self.S4_1(dataset, *feat3)
        feat4 = self.S4_2(dataset, *feat4)

        feat5_4 = self.S5_4_1(dataset, *feat4)
        feat5_4 = self.S5_4_2(dataset, *feat5_4)
        feat5_4 = self.S5_4_3(dataset, *feat5_4)
        feat5_4 = self.S5_4_4(dataset, *feat5_4)

        feat5_5 = self.S5_5(dataset, *feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(nn.Module):

    def __init__(self, n_bn=1):
        super(BGALayer, self).__init__()
        # self.left1 = nn.Sequential(
        #     nn.Conv2d(
        #         128, 128, kernel_size=3, stride=1,
        #         padding=1, groups=128, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.Conv2d(
        #         128, 128, kernel_size=1, stride=1,
        #         padding=0, bias=False),
        # )
        # self.left2 = nn.Sequential(
        #     nn.Conv2d(
        #         128, 128, kernel_size=3, stride=2,
        #         padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        # )
        # self.right1 = nn.Sequential(
        #     nn.Conv2d(
        #         128, 128, kernel_size=3, stride=1,
        #         padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        # )
        # self.right2 = nn.Sequential(
        #     nn.Conv2d(
        #         128, 128, kernel_size=3, stride=1,
        #         padding=1, groups=128, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.Conv2d(
        #         128, 128, kernel_size=1, stride=1,
        #         padding=0, bias=False),
        # )
        self.left1_convbn = ConvBN(128, 128, ks=3, groups=128, n_bn=n_bn)
        self.left1_conv = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.left2_convbn = ConvBN(128, 128, ks=3, stride=2, n_bn=n_bn)
        self.left2_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        
        self.right1 = ConvBN(128, 128, ks=3, n_bn=n_bn)
        
        self.right2_convbn = ConvBN(128, 128, ks=3, groups=128, n_bn=n_bn)
        self.right2_conv = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)

        ##TODO: does this really has no relu?
        self.conv = ConvBNReLU(128, 128, ks=3, n_bn=n_bn)
        

    def forward(self, x_d, x_s, dataset=0):
        ## x_d, x_s都是多数据集的list
        ## TODO 实现不定长参数版本
        # dsize = x_d.size()[2:]
        # x_d = [x.cuda() for x in x_d]
        # x_s = [x.cuda() for x in x_s]

        left1 = self.left1_convbn(dataset, *x_d)
        left1 = [self.left1_conv(x) for x in left1]

        left2 = self.left2_convbn(dataset, *x_d)
        left2 = [self.left2_pool(x) for x in left2]

        right1 = self.right1(dataset, *x_s)

        right2 = self.right2_convbn(dataset, *x_s)
        right2 = [self.right2_conv(x) for x in right2]

        right1 = [self.up1(x) for x in right1]

        left = [left1[i] * F.sigmoid(right1[i]) for i, x in enumerate(left1)]
        right = [left2[i] * F.sigmoid(right2[i]) for i, x in enumerate(left2)]
        right = [self.up2(x) for x in right]

        feats = [left[i] + right[i] for i, x in enumerate(left)]
        out = self.conv(dataset, *feats)
        return out



class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor

        # self.conv_out = nn.Sequential(
        #     nn.Sequential(
        #         nn.Upsample(scale_factor=2),
        #         ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
        #         ) if aux else nn.Identity(),
        #     nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
        #     nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        # )

        self.aux = aux
        self.up_sample1 = nn.Upsample(scale_factor=2)
        self.conv1 = ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)

        self.conv2 = nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True)
        self.up_sample2 = nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        

    def forward(self, x):
        # print(x.size())
        
        # 采用多分割头，所以不应该返回list
        feats = self.conv(0, x)
        feat = self.drop(feats[0])

        if self.aux is True:
            feat = self.up_sample1(feat)
            feats = self.conv1(0, feat)
            feat = self.conv2(feats[0])
        else:
            feat = self.conv2(feat)
            
        feat = self.up_sample2(feat)

        return feat


class BiSeNetV2(nn.Module):

    def __init__(self, n_classes, aux_mode='train', n_bn=1, *other_n_classes):
        super(BiSeNetV2, self).__init__()
        self.aux_mode = aux_mode
        self.detail = DetailBranch(n_bn=n_bn)
        self.segment = SegmentBranch(n_bn=n_bn)
        self.bga = BGALayer(n_bn=n_bn)

        ## TODO: what is the number of mid chan ?
        self.head = nn.ModuleList([SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)])
        if self.aux_mode == 'train':
            self.aux2 = nn.ModuleList([SegmentHead(16, 128, n_classes, up_factor=4)])
            self.aux3 = nn.ModuleList([SegmentHead(32, 128, n_classes, up_factor=8)])
            self.aux4 = nn.ModuleList([SegmentHead(64, 128, n_classes, up_factor=16)])
            self.aux5_4 = nn.ModuleList([SegmentHead(128, 128, n_classes, up_factor=32)])

        ## 多数据集的头
        # self.n_head = len(other_n_classes) + 1
        # if self.n_head > 1:
        for n in other_n_classes:
            self.head.append(SegmentHead(128, 1024, n, up_factor=8, aux=False))
            if self.aux_mode == 'train':
                self.aux2.append(SegmentHead(16, 128, n, up_factor=4))
                self.aux3.append(SegmentHead(32, 128, n, up_factor=8))
                self.aux4.append(SegmentHead(64, 128, n, up_factor=16))
                self.aux5_4.append(SegmentHead(128, 128, n, up_factor=32))
        # # 所有list的模型都需要手动.cuda()
        # for i in range(0, n_bn):
        #     self.head[i].cuda()
        #     if self.aux_mode == 'train':
        #         self.aux2[i].cuda()
        #         self.aux3[i].cuda()
        #         self.aux4[i].cuda()
        #         self.aux5_4[i].cuda()

        self.init_weights()

    def forward(self, x, dataset=0, *other_x):
        # x = x.cuda()
        ## other_x 其他数据集的输入
        # size = x.size()[2:]

        
        feat_d = self.detail(x, dataset, *other_x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x, dataset, *other_x)
        feat_head = self.bga(feat_d, feat_s, dataset)

        # logits = self.head[0](feat_head[0])
        ## 修改为多数据集模式，返回list
        
        # logits = [logit(feat_head[i]) for i, logit in enumerate(self.head)]
        ## 修改后支持单张图片输入
        if (len(other_x) == 0):
            logits = [self.head[dataset](feat_head[0])]
        else:
            logits = [self.head[i](feat_head[i]) for i in range(0, len(other_x) + 1)]
        ## TODO 修改下面的多数据集模式
        if (len(other_x) == 0):
            if self.aux_mode == 'train':
                # logits_aux2 = self.aux2(feat2)
                # logits_aux3 = self.aux3(feat3)
                # logits_aux4 = self.aux4(feat4)
                # logits_aux5_4 = self.aux5_4(feat5_4)
                ## 多数据集模式
                ## 修改后支持单张图片输入
                logits_aux2 = [self.aux2[dataset](feat2[0])]
                logits_aux3 = [self.aux3[dataset](feat3[0])]
                logits_aux4 = [self.aux4[dataset](feat4[0])]
                logits_aux5_4 = [self.aux5_4[dataset](feat5_4[0])]
                return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
            elif self.aux_mode == 'eval':
                return logits,
            elif self.aux_mode == 'pred':
                # pred = logits.argmax(dim=1)
                pred = [logit.argmax(dim=1) for logit in logits]
                return pred
            else:
                raise NotImplementedError
        else:
            if self.aux_mode == 'train':
                # logits_aux2 = self.aux2(feat2)
                # logits_aux3 = self.aux3(feat3)
                # logits_aux4 = self.aux4(feat4)
                # logits_aux5_4 = self.aux5_4(feat5_4)
                ## 多数据集模式
                ## 修改后支持单张图片输入
                logits_aux2 = [self.aux2[i](feat2[i]) for i in range(0, len(other_x) + 1)]
                logits_aux3 = [self.aux3[i](feat3[i]) for i in range(0, len(other_x) + 1)]
                logits_aux4 = [self.aux4[i](feat4[i]) for i in range(0, len(other_x) + 1)]
                logits_aux5_4 = [self.aux5_4[i](feat5_4[i]) for i in range(0, len(other_x) + 1)]
                return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
            elif self.aux_mode == 'eval':
                return logits,
            elif self.aux_mode == 'pred':
                # pred = logits.argmax(dim=1)
                pred = [logit.argmax(dim=1) for logit in logits]
                return pred
            else:
                raise NotImplementedError

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

        # self.load_pretrain()


    def load_pretrain(self):
        # state = modelzoo.load_url(backbone_url)
        state = torch.load(backbone_url)
        detail_state = {}
        def loadConvBNRelu(srcDict, src_name, targetdict, target_name):
            targetdict[target_name+'.conv.weight'] = srcDict[src_name+'.conv.weight']
            targetdict[target_name+'.affine_weight'] = srcDict[src_name+'.bn.weight']
            targetdict[target_name+'.affine_bias'] = srcDict[src_name+'.bn.bias']

        loadConvBNRelu(state['detail'], 'S1.0', detail_state, 'S1_1')
        loadConvBNRelu(state['detail'], 'S1.1', detail_state, 'S1_2')
        loadConvBNRelu(state['detail'], 'S2.0', detail_state, 'S2_1')
        loadConvBNRelu(state['detail'], 'S2.1', detail_state, 'S2_2')
        loadConvBNRelu(state['detail'], 'S2.2', detail_state, 'S2_3')
        loadConvBNRelu(state['detail'], 'S3.0', detail_state, 'S3_1')
        loadConvBNRelu(state['detail'], 'S3.1', detail_state, 'S3_2')
        loadConvBNRelu(state['detail'], 'S3.2', detail_state, 'S3_3')

        segment_state = {}
        loadConvBNRelu(state['segment'], 'S1S2.conv', segment_state, 'S1S2.conv')
        loadConvBNRelu(state['segment'], 'S1S2.left.0', segment_state, 'S1S2.left_1')
        loadConvBNRelu(state['segment'], 'S1S2.left.1', segment_state, 'S1S2.left_2')
        loadConvBNRelu(state['segment'], 'S1S2.fuse', segment_state, 'S1S2.fuse')

        def loadGELayerS2(srcDict, src_name, targetdict, target_name):
            loadConvBNRelu(srcDict, src_name+'.conv1', targetdict, target_name+'.conv1')
            
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
            loadConvBNRelu(srcDict, src_name+'.conv1', targetdict, target_name+'.conv1')
            
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
        loadConvBNRelu(state['segment'], 'S5_5.conv_gap', segment_state, 'S5_5.conv_gap')
        loadConvBNRelu(state['segment'], 'S5_5.conv_last', segment_state, 'S5_5.conv_last')
        # segment_state['S5_5.conv_gap.conv.weight'] = state['segment']['S5_5.conv_gap.conv.weight']
        # segment_state['S5_5.conv_last.conv.weight'] = state['segment']['S5_5.conv_last.conv.weight']

        bga_state = {}
        bga_state['left1_convbn.conv.weight'] = state['bga']['left1.0.weight']
        bga_state['left1_convbn.affine_weight'] = state['bga']['left1.1.weight']
        bga_state['left1_convbn.affine_bias'] = state['bga']['left1.1.bias']

        bga_state['left1_conv.weight'] = state['bga']['left1.2.weight']

        bga_state['left2_convbn.conv.weight'] = state['bga']['left2.0.weight']
        bga_state['left2_convbn.affine_weight'] = state['bga']['left2.1.weight']
        bga_state['left2_convbn.affine_bias'] = state['bga']['left2.1.bias']

        bga_state['right1.conv.weight'] = state['bga']['right1.0.weight']
        bga_state['right1.affine_weight'] = state['bga']['right1.1.weight']
        bga_state['right1.affine_bias'] = state['bga']['right1.1.bias']

        bga_state['right2_convbn.conv.weight'] = state['bga']['right2.0.weight']
        bga_state['right2_convbn.affine_weight'] = state['bga']['right2.1.weight']
        bga_state['right2_convbn.affine_bias'] = state['bga']['right2.1.bias']

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