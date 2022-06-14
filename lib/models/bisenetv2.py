
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

# backbone_url = 'https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth'
backbone_url = './pth/backbone_v2.pth'


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, n_bn=1):
        ## n_bn bn层数量，对应混合的数据集数量
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = [nn.BatchNorm2d(out_chan) for i in range(0, n_bn)]
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, *other_x):
        ## TODO 此处可以优化，不同数据集的图像过卷积层可以拼接到一起，过BN层再分离

        feat = self.conv(x)
        feat = self.bn[0](feat)
        feat = self.relu(feat)

        ## 处理多数据集情况
        feats = [feat]
        for i, xs in enumerate(other_x):
            feat = self.conv(xs)
            feat = self.bn[i+1](feat)
            feat = self.relu(feat)
            feats.append(feat)

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
        self.bn = [nn.BatchNorm2d(out_chan) for i in range(0, n_bn)]
        

    def forward(self, x, *other_x):
        ## TODO 此处可以优化，不同数据集的图像过卷积层可以拼接到一起，过BN层再分离

        feat = self.conv(x)
        feat = self.bn[0](feat)  

        ## 处理多数据集情况
        feats = [feat]
        for i, xs in enumerate(other_x):
            feat = self.conv(xs)
            feat = self.bn[i+1](feat)
            
            feats.append(feat)

        return feats

    def SetLastBNAttr(self, attr):
        for bn in self.bn:
            bn.last_bn = attr



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
        

    def forward(self, x, *other_x):
        ## other_x 其他数据集的输入
        ## 拆分列表传参
        feats = self.S1_1(x, *other_x)
        feats = self.S1_2(*feats)

        feats = self.S2_1(*feats)
        feats = self.S2_2(*feats)
        feats = self.S2_3(*feats)

        feats = self.S3_1(*feats)
        feats = self.S3_2(*feats)
        feats = self.S3_3(*feats)

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

    def forward(self, x, *other_x):
        # feat = self.conv(x)
        # feat_left = self.left(feat)
        # feat_right = self.right(feat)
        # feat = torch.cat([feat_left, feat_right], dim=1)
        # feat = self.fuse(feat)

        ## 修改为处理多数据集的情况
        feats = self.conv(x, *other_x)
        feat_lefts = self.left_1(*feats)
        feat_lefts = self.left_2(*feat_lefts)
        feat_rights = [self.right(feat) for feat in feats] 
        feats = [torch.cat([feat_lefts[i], feat_rights[i]], dim=1) for i in range(0, len(feat_lefts))]
        feats = self.fuse(*feats)

        return feats


class CEBlock(nn.Module):

    def __init__(self, n_bn=1):
        ## n_bn bn层数量，对应混合的数据集数量

        super(CEBlock, self).__init__()
        self.bn = [nn.BatchNorm2d(128) for i in range(0, n_bn)]
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0, n_bn=n_bn)
        #TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1, n_bn=n_bn)

    def forward(self, x, *other_x):
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

        feats_bn = [self.bn[i](feat) for i, feat in enumerate(feats)] 
        feats_gap = self.conv_gap(*feats_bn)
        # feat = feat + x

        feats = [F.interpolate(feats_gap[0], size=(x.shape[2], x.shape[3])) + x]
        for i, xs in enumerate(other_x):
            feats.append(F.interpolate(feats_gap[i+1], size=(xs.shape[2], xs.shape[3])) + xs)

        feats = self.conv_last(*feats)
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

    def forward(self, x, *other_x):
        feats = self.conv1(x, *other_x)
        feats = self.dwconv(*feats)
        feats_conv = self.conv2(*feats)
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

    def forward(self, x, *other_x):
        ## 修改为多数据集模式
        feats = self.conv1(x, *other_x)
        feats = self.dwconv1(*feats)
        feats = self.dwconv2(*feats)
        feats = self.conv2(*feats)
        shortcuts = self.shortcut_1(x, *other_x)
        shortcuts = self.shortcut_2(*shortcuts)
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

    def forward(self, x, *other_x):
        feat2 = self.S1S2(x, *other_x)

        feat3 = self.S3_1(*feat2)
        feat3 = self.S3_2(*feat3)

        feat4 = self.S4_1(*feat3)
        feat4 = self.S4_2(*feat4)

        feat5_4 = self.S5_4_1(*feat4)
        feat5_4 = self.S5_4_2(*feat5_4)
        feat5_4 = self.S5_4_3(*feat5_4)
        feat5_4 = self.S5_4_4(*feat5_4)

        feat5_5 = self.S5_5(*feat5_4)
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
        

    def forward(self, x_d, x_s):
        ## x_d, x_s都是多数据集的list
        ## TODO 实现不定长参数版本
        # dsize = x_d.size()[2:]
        left1 = self.left1_convbn(*x_d)
        left1 = [self.left1_conv(x) for x in left1]

        left2 = self.left2_convbn(*x_d)
        left2 = [self.left2_pool(x) for x in left2]

        right1 = self.right1(*x_s)

        right2 = self.right2_convbn(*x_s)
        right2 = [self.right2_conv(x) for x in right2]

        right1 = [self.up1(x) for x in right1]

        left = [left1[i] * F.sigmoid(right1[i]) for i, x in enumerate(left1)]
        right = [left2[i] * F.sigmoid(right2[i]) for i, x in enumerate(left2)]
        right = [self.up2(x) for x in right]

        feats = [left[i] + right[i] for i, x in enumerate(left)]
        out = self.conv(*feats)
        return out



class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True, n_bn=1):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1, n_bn=n_bn)
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
        self.conv1 = ConvBNReLU(mid_chan, mid_chan2, 3, stride=1, n_bn=n_bn)

        self.conv2 = nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True)
        self.up_sample2 = nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        

    def forward(self, x, *other_x):
        # print(x.size())
        feats = self.conv(x, *other_x)
        feats = [self.drop(feat) for feat in feats]

        if self.aux is True:
            feats = [self.up_sample1(feat) for feat in feats]
            feats = self.conv1(*feats)

        feats = [self.conv2(feat) for feat in feats]
        feats = [self.up_sample2(feat) for feat in feats]

        return feats


class BiSeNetV2(nn.Module):

    def __init__(self, n_classes, aux_mode='train', n_bn=1, *other_n_classes):
        super(BiSeNetV2, self).__init__()
        self.aux_mode = aux_mode
        self.detail = DetailBranch(n_bn=n_bn)
        self.segment = SegmentBranch(n_bn=n_bn)
        self.bga = BGALayer(n_bn=n_bn)

        ## TODO: what is the number of mid chan ?
        self.head = [SegmentHead(128, 1024, n_classes, up_factor=8, aux=False, n_bn=n_bn)]
        if self.aux_mode == 'train':
            self.aux2 = [SegmentHead(16, 128, n_classes, up_factor=4, n_bn=n_bn)]
            self.aux3 = [SegmentHead(32, 128, n_classes, up_factor=8, n_bn=n_bn)]
            self.aux4 = [SegmentHead(64, 128, n_classes, up_factor=16, n_bn=n_bn)]
            self.aux5_4 = [SegmentHead(128, 128, n_classes, up_factor=32, n_bn=n_bn)]

        ## 多数据集的头
        # self.n_head = len(other_n_classes) + 1
        # if self.n_head > 1:
        for n in other_n_classes:
            self.head.append(SegmentHead(128, 1024, n, up_factor=8, aux=False, n_bn=n_bn))
            if self.aux_mode == 'train':
                self.aux2.append(SegmentHead(16, 128, n, up_factor=4, n_bn=n_bn))
                self.aux3.append(SegmentHead(32, 128, n, up_factor=8, n_bn=n_bn))
                self.aux4.append(SegmentHead(64, 128, n, up_factor=16, n_bn=n_bn))
                self.aux5_4.append(SegmentHead(128, 128, n, up_factor=32, n_bn=n_bn))

        self.init_weights()

    def forward(self, x, *other_x):
        ## other_x 其他数据集的输入
        size = x.size()[2:]
        
        feat_d = self.detail(x, *other_x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x, *other_x)
        feat_head = self.bga(feat_d, feat_s)


        # logits = self.head(feat_head)
        ## 修改为多数据集模式，返回list
        logits = [logit(feat_head[i]) for i, logit in enumerate(self.head)]
        ## TODO 修改下面的多数据集模式
        if self.aux_mode == 'train':
            # logits_aux2 = self.aux2(feat2)
            # logits_aux3 = self.aux3(feat3)
            # logits_aux4 = self.aux4(feat4)
            # logits_aux5_4 = self.aux5_4(feat5_4)
            ## 多数据集模式
            logits_aux2 = [aux2(feat2[i]) for i, aux2 in enumerate(self.aux2)]
            logits_aux3 = [aux3(feat3[i]) for i, aux3 in enumerate(self.aux3)]
            logits_aux4 = [aux4(feat4[i]) for i, aux4 in enumerate(self.aux4)]
            logits_aux5_4 = [aux5_4(feat5_4[i]) for i, aux5_4 in enumerate(self.aux5_4)]
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
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # self.load_pretrain()


    def load_pretrain(self):
        # state = modelzoo.load_url(backbone_url)
        state = torch.load(backbone_url, map_location='cpu')
        for name, child in self.named_children():
            if name in state.keys():
                child.load_state_dict(state[name], strict=True)

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
    #  x = torch.randn(16, 3, 1024, 2048)
    #  detail = DetailBranch()
    #  feat = detail(x)
    #  print('detail', feat.size())
    #
    #  x = torch.randn(16, 3, 1024, 2048)
    #  stem = StemBlock()
    #  feat = stem(x)
    #  print('stem', feat.size())
    #
    #  x = torch.randn(16, 128, 16, 32)
    #  ceb = CEBlock()
    #  feat = ceb(x)
    #  print(feat.size())
    #
    #  x = torch.randn(16, 32, 16, 32)
    #  ge1 = GELayerS1(32, 32)
    #  feat = ge1(x)
    #  print(feat.size())
    #
    #  x = torch.randn(16, 16, 16, 32)
    #  ge2 = GELayerS2(16, 32)
    #  feat = ge2(x)
    #  print(feat.size())
    #
    #  left = torch.randn(16, 128, 64, 128)
    #  right = torch.randn(16, 128, 16, 32)
    #  bga = BGALayer()
    #  feat = bga(left, right)
    #  print(feat.size())
    #
    #  x = torch.randn(16, 128, 64, 128)
    #  head = SegmentHead(128, 128, 19)
    #  logits = head(x)
    #  print(logits.size())
    #
    #  x = torch.randn(16, 3, 1024, 2048)
    #  segment = SegmentBranch()
    #  feat = segment(x)[0]
    #  print(feat.size())
    #
    import time
    x = torch.randn(16, 3, 512, 1024) #.cuda()
    model = BiSeNetV2(n_classes=19)
    # model.cuda()
    model.eval()
    # outs = model(x)
    # for i in range(50):
    #     t0 = time.time()
    #     outs = model(x)
    #     print((time.time() - t0) * 1000)
    outs = model(x)
    # for out in outs:
    #     print(out.size())
    #  print(logits.size())

    #  for name, param in model.named_parameters():
    #      if len(param.size()) == 1:
    #          print(name)
    total = sum([param.nelement() for param in model.parameters()])
    print(total / 1e6)
    d_total = sum([param.nelement() for param in model.bga.parameters()])
    print(d_total / 1e6)
