import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import warnings
import numpy as np

from lib.module.util import _BNReluConv, upsample, BNReLUConv
from lib.models.resnet_pyramid import resnet18, resnet18_mulbn
from timm.models.layers import trunc_normal_

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
        

    def forward(self, x, dataset_id):
        feat = self.conv(x)
        feat_list = []
        cur_pos = 0
        for i in range(0, len(dataset_id)):
            if dataset_id[i] != dataset_id[cur_pos]:
                feat_ = self.bn[dataset_id[cur_pos]](feat[cur_pos:i])
                feat_list.append(feat_)
                cur_pos = i
        feat_ = self.bn[dataset_id[cur_pos]](feat[cur_pos:])
        feat_list.append(feat_)
        feat = torch.cat(feat_list, dim=0)
        feat = feat * self.affine_weight.reshape(1,-1,1,1) + self.affine_bias.reshape(1,-1,1,1) 
        feat = self.relu(feat)
        return feat



class SemsegModel_mulbn(nn.Module):
    def __init__(self, configer, num_inst_classes=None, use_bn=True, k=1, bias=True,
                 loss_ret_additional=False, upsample_logits=True,
                 multiscale_factors=(.5, .75, 1.5, 2.)):
        super(SemsegModel_mulbn, self).__init__()
        self.configer = configer
        self.aux_mode = self.configer.get('aux_mode')
        self.n_datasets = configer.get("n_datasets")
        self.backbone = resnet18_mulbn(configer, pretrained=True,
                    pyramid_levels=3,
                    k_upsample=3,
                    k_bneck=1,
                    output_stride=4,
                    efficient=True)
        self.total_cats = 0
        self.datasets_cats = []
        for i in range(0, self.n_datasets):
            self.datasets_cats.append(self.configer.get('dataset'+str(i+1), 'n_cats'))
            self.total_cats += self.datasets_cats[i]
        
        self.output_feat_dim = self.configer.get('GNN', 'output_feat_dim')
        self.max_num_unify_class = int(self.configer.get('GNN', 'unify_ratio') * self.total_cats)
        
        self.logits = BNReLUConv(self.backbone.num_features, self.output_feat_dim, ks=1, padding=0, bias=True, n_bn=self.n_datasets) 
        self.bipartite_graphs = nn.ParameterList([])
        for i in range(0, self.n_datasets):
            self.bipartite_graphs.append(nn.Parameter(
                torch.zeros(self.configer.get('dataset'+str(i+1), 'n_cats'), self.max_num_unify_class), requires_grad=False
                ))
            
        self.unify_prototype = nn.Parameter(torch.zeros(self.max_num_unify_class, self.output_feat_dim),
                                requires_grad=False)
        trunc_normal_(self.unify_prototype, std=0.02)
        
        self.num_classes = self.max_num_unify_class
        # self.logits = logit_class(self.backbone.num_features, self.num_classes, batch_norm=use_bn, k=k, bias=bias)
        if num_inst_classes is not None:
            raise NotImplementedError
            # self.border_logits = _BNReluConv(self.backbone.num_features, num_inst_classes, batch_norm=use_bn,
            #                                  k=k, bias=bias)
        self.criterion = None
        self.loss_ret_additional = loss_ret_additional
        self.img_req_grad = loss_ret_additional
        self.upsample_logits = upsample_logits
        self.multiscale_factors = multiscale_factors

        self.init_weights()

    def forward(self, image, dataset_id, dataset=0):
        features, _ = self.backbone(image, dataset_id)
        features = self.logits.forward(features, dataset_id)
        if self.aux_mode == 'train':
            if self.training:
                logits = torch.einsum('bchw, nc -> bnhw', features, self.unify_prototype)
                return {'seg':logits}
            else:
                return {'seg':features}
        elif self.aux_mode == 'eval':
            # logits = torch.einsum('bchw, nc -> bnhw', emb, self.unify_prototype[cur_cat:cur_cat+self.datasets_cats[dataset]])   
            logits = torch.einsum('bchw, nc -> bnhw', features, self.unify_prototype)
            # return logits
            remap_logits = torch.einsum('bchw, nc -> bnhw', logits, self.bipartite_graphs[dataset])
            return remap_logits
        elif self.aux_mode == 'pred':
            logits = torch.einsum('bchw, nc -> bnhw', features, self.unify_prototype)
            # logits = torch.einsum('bchw, nc -> bnhw', logits, self.bipartite_graphs[dataset][:self.datasets_cats[dataset]-1])
            logits = torch.einsum('bchw, nc -> bnhw', logits, self.bipartite_graphs[dataset])
            logits = F.interpolate(logits, size=(logits.size(2)*4, logits.size(3)*4), mode="bilinear", align_corners=True)
            
            pred = logits.argmax(dim=1)
            
            return pred
        elif self.aux_mode == 'clip':
            cur_cat=0
            for i in range(0, dataset):
                cur_cat += self.datasets_cats[i]
            
            logits = torch.einsum('bchw, nc -> bnhw', features, self.unify_prototype[cur_cat:cur_cat+self.datasets_cats[dataset]])   
            return logits
        elif self.aux_mode == 'uni_eval':
            logits = torch.einsum('bchw, nc -> bnhw', features, self.unify_prototype)
            return logits
        elif self.aux_mode == 'unseen':
            logits = torch.einsum('bchw, nc -> bnhw', features, self.unify_prototype)

            max_index = torch.argmax(logits, dim=1)
            temp = torch.eye(logits.size(1)).cuda()
            one_hot = temp[max_index]
            remap_logits = torch.einsum('bhwc, nc -> bnhw', one_hot, self.bipartite_graphs[dataset])
            return remap_logits
            
        else:
            logits = torch.einsum('bchw, nc -> bnhw', features, self.unify_prototype)
            # logits = torch.einsum('bchw, nc -> bnhw', logits, self.bipartite_graphs[dataset])
            logits = F.interpolate(logits, size=(logits.size(2)*4, logits.size(3)*4), mode="bilinear", align_corners=True)
            
            pred = logits.argmax(dim=1)
            
            return pred

    def forward_down(self, image, target_size, image_size):
        return self.backbone.forward_down(image), target_size, image_size

    def forward_up(self, feats, target_size, image_size):
        feats, additional = self.backbone.forward_up(feats)
        features = upsample(feats, target_size)
        logits = self.logits.forward(features)
        logits = upsample(logits, image_size)
        return logits, additional

    def prepare_data(self, batch, image_size, device=torch.device('cuda'), img_key='image'):
        if image_size is None:
            image_size = batch['target_size']
        warnings.warn(f'Image requires grad: {self.img_req_grad}', UserWarning)
        image = batch[img_key].detach().requires_grad_(self.img_req_grad).to(device)
        return {
            'image': image,
            'image_size': image_size,
            'target_size': batch.get('target_size_feats')
        }

    def do_forward(self, batch, image_size=None):
        data = self.prepare_data(batch, image_size)
        logits, additional = self.forward(**data)
        additional['model'] = self
        additional = {**additional, **data}
        return logits, additional

    def loss(self, batch):
        assert self.criterion is not None
        labels = batch['labels'].cuda()
        logits, additional = self.do_forward(batch, image_size=labels.shape[-2:])
        if self.loss_ret_additional:
            return self.criterion(logits, labels, batch=batch, additional=additional), additional
        return self.criterion(logits, labels, batch=batch, additional=additional)

    def random_init_params(self):
        params = [self.backbone.random_init_params(), self.logits.parameters()]
        if self.unify_prototype.require_grad:
            return params.append(self.unify_prototype)        
        # self.logits.parameters(), 
        # if hasattr(self, 'border_logits'):
        #     params += [self.border_logits.parameters()]
        return chain(*(params))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()

    def ms_forward(self, batch, image_size=None):
        image_size = batch.get('target_size', image_size if image_size is not None else batch['image'].shape[-2:])
        ms_logits = None
        pyramid = [batch['image'].cuda()]
        pyramid += [
            F.interpolate(pyramid[0], scale_factor=sf, mode=self.backbone.pyramid_subsample,
                          align_corners=self.backbone.align_corners) for sf in self.multiscale_factors
        ]
        for image in pyramid:
            batch['image'] = image
            logits, additional = self.do_forward(batch, image_size=image_size)
            if ms_logits is None:
                ms_logits = torch.zeros(logits.size()).to(logits.device)
            ms_logits += F.softmax(logits, dim=1)
        batch['image'] = pyramid[0].cpu()
        return ms_logits / len(pyramid), {}
    
    def set_bipartite_graphs(self, bi_graphs):
        
        if len(bi_graphs) == 2 * self.n_datasets:
            for i in range(0, self.n_datasets):
                self.bipartite_graphs[i] = nn.Parameter(
                    bi_graphs[2*i], requires_grad=False
                    )
        else:
            # print("bi_graphs len:", len(bi_graphs))
            for i in range(0, self.n_datasets):
                # print("i: ", i)
                self.bipartite_graphs[i] = nn.Parameter(
                    bi_graphs[i], requires_grad=False
                    )
                
    def set_unify_prototype(self, unify_prototype, grad=False):
        self.unify_prototype.data = unify_prototype
        self.unify_prototype.requires_grad=grad
        
    def get_optim_params(self):
        fine_tune_factor = 4
        optim_params = [
            {'params': self.random_init_params(), 'lr': self.configer.get('lr', 'seg_lr_start'), 'weight_decay': self.configer.get('lr', 'weight_decay')},
            {'params': self.fine_tune_params(), 'lr': self.configer.get('lr', 'seg_lr_start') / fine_tune_factor,
            'weight_decay': self.configer.get('lr', 'weight_decay') / fine_tune_factor},
        ]
        return optim_params
    
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
                    
    
    
    
class SemsegModel(nn.Module):
    def __init__(self, configer, num_inst_classes=None, use_bn=True, k=1, bias=True,
                 loss_ret_additional=False, upsample_logits=True, logit_class=_BNReluConv,
                 multiscale_factors=(.5, .75, 1.5, 2.)):
        super(SemsegModel, self).__init__()
        self.configer = configer
        self.aux_mode = self.configer.get('aux_mode')
        self.n_datasets = configer.get("n_datasets")
        self.with_datasets_aux = self.configer.get('loss', 'with_datasets_aux')
        self.backbone = resnet18(pretrained=True,
                    pyramid_levels=3,
                    k_upsample=3,
                    k_bneck=1,
                    output_stride=4,
                    efficient=True)
        self.total_cats = 0
        self.datasets_cats = []
        for i in range(0, self.n_datasets):
            self.datasets_cats.append(self.configer.get('dataset'+str(i+1), 'n_cats'))
            self.total_cats += self.datasets_cats[i]

        self.output_feat_dim = self.configer.get('GNN', 'output_feat_dim')
        self.max_num_unify_class = int(self.configer.get('GNN', 'unify_ratio') * self.total_cats)

        self.logits = logit_class(self.backbone.num_features, self.output_feat_dim, batch_norm=use_bn, k=k, bias=bias) 
        self.bipartite_graphs = nn.ParameterList([])
        for i in range(0, self.n_datasets):
            self.bipartite_graphs.append(nn.Parameter(
                torch.zeros(self.configer.get('dataset'+str(i+1), 'n_cats'), self.max_num_unify_class), requires_grad=False
                ))

        self.unify_prototype = nn.Parameter(torch.zeros(self.max_num_unify_class, self.output_feat_dim),
                                requires_grad=False)

        trunc_normal_(self.unify_prototype, std=0.02)

        if self.with_datasets_aux:
            self.aux_prototype = nn.ParameterList([])
            for i in range(0, self.n_datasets):
                self.aux_prototype.append(nn.Parameter(torch.zeros(self.datasets_cats[i], self.output_feat_dim),
                                        requires_grad=False))
                trunc_normal_(self.aux_prototype[i], std=0.02)




        self.num_classes = self.max_num_unify_class
        # self.logits = logit_class(self.backbone.num_features, self.num_classes, batch_norm=use_bn, k=k, bias=bias)
        if num_inst_classes is not None:
            raise NotImplementedError
            # self.border_logits = _BNReluConv(self.backbone.num_features, num_inst_classes, batch_norm=use_bn,
            #                                  k=k, bias=bias)
        self.criterion = None
        self.loss_ret_additional = loss_ret_additional
        self.img_req_grad = loss_ret_additional
        self.upsample_logits = upsample_logits
        self.multiscale_factors = multiscale_factors

    def forward(self, image, dataset=0):
        features, _ = self.backbone(image)
        features = self.logits.forward(features)
        if self.aux_mode == 'train':
            if self.training:
                logits = torch.einsum('bchw, nc -> bnhw', features, self.unify_prototype)
                if self.with_datasets_aux:
                    cur_cat = 0
                    aux_logits = []
                    for i in range(self.n_datasets):
                        aux_logits.append(torch.einsum('bchw, nc -> bnhw', features, self.aux_prototype[i]))
                        cur_cat += self.datasets_cats[i]
                        
                    return {'seg':logits, 'aux':aux_logits}
                return {'seg':logits}
            else:
                return {'seg':features}
        elif self.aux_mode == 'eval':
            # logits = torch.einsum('bchw, nc -> bnhw', emb, self.unify_prototype[cur_cat:cur_cat+self.datasets_cats[dataset]])   
            logits = torch.einsum('bchw, nc -> bnhw', features, self.unify_prototype)
            # return logits
            remap_logits = torch.einsum('bchw, nc -> bnhw', logits, self.bipartite_graphs[dataset])
            return remap_logits
        elif self.aux_mode == 'pred':
            logits = torch.einsum('bchw, nc -> bnhw', features, self.unify_prototype)
            # logits = torch.einsum('bchw, nc -> bnhw', logits, self.bipartite_graphs[dataset][:self.datasets_cats[dataset]-1])
            logits = torch.einsum('bchw, nc -> bnhw', logits, self.bipartite_graphs[dataset])
            logits = F.interpolate(logits, size=(logits.size(2)*4, logits.size(3)*4), mode="bilinear", align_corners=True)

            pred = logits.argmax(dim=1)

            return pred
        elif self.aux_mode == 'clip':
            cur_cat=0
            for i in range(0, dataset):
                cur_cat += self.datasets_cats[i]

            logits = torch.einsum('bchw, nc -> bnhw', features, self.unify_prototype[cur_cat:cur_cat+self.datasets_cats[dataset]])   
            return logits
        elif self.aux_mode == 'uni_eval':
            logits = torch.einsum('bchw, nc -> bnhw', features, self.unify_prototype)
            return logits
        elif self.aux_mode == 'unseen':
            logits = torch.einsum('bchw, nc -> bnhw', features, self.unify_prototype)

            max_index = torch.argmax(logits, dim=1)
            temp = torch.eye(logits.size(1)).cuda()
            one_hot = temp[max_index]
            remap_logits = torch.einsum('bhwc, nc -> bnhw', one_hot, self.bipartite_graphs[dataset])
            return remap_logits

        else:
            logits = torch.einsum('bchw, nc -> bnhw', features, self.unify_prototype)
            # logits = torch.einsum('bchw, nc -> bnhw', logits, self.bipartite_graphs[dataset])
            logits = F.interpolate(logits, size=(logits.size(2)*4, logits.size(3)*4), mode="bilinear", align_corners=True)

            pred = logits.argmax(dim=1)

            return pred

    def forward_down(self, image, target_size, image_size):
        return self.backbone.forward_down(image), target_size, image_size

    def forward_up(self, feats, target_size, image_size):
        feats, additional = self.backbone.forward_up(feats)
        features = upsample(feats, target_size)
        logits = self.logits.forward(features)
        logits = upsample(logits, image_size)
        return logits, additional

    def prepare_data(self, batch, image_size, device=torch.device('cuda'), img_key='image'):
        if image_size is None:
            image_size = batch['target_size']
        warnings.warn(f'Image requires grad: {self.img_req_grad}', UserWarning)
        image = batch[img_key].detach().requires_grad_(self.img_req_grad).to(device)
        return {
            'image': image,
            'image_size': image_size,
            'target_size': batch.get('target_size_feats')
        }

    def do_forward(self, batch, image_size=None):
        data = self.prepare_data(batch, image_size)
        logits, additional = self.forward(**data)
        additional['model'] = self
        additional = {**additional, **data}
        return logits, additional

    def loss(self, batch):
        assert self.criterion is not None
        labels = batch['labels'].cuda()
        logits, additional = self.do_forward(batch, image_size=labels.shape[-2:])
        if self.loss_ret_additional:
            return self.criterion(logits, labels, batch=batch, additional=additional), additional
        return self.criterion(logits, labels, batch=batch, additional=additional)

    def random_init_params(self):
        params = [self.backbone.random_init_params()]
        if self.unify_prototype.require_grad:
            return params.append(self.unify_prototype)        
        # self.logits.parameters(), 
        # if hasattr(self, 'border_logits'):
        #     params += [self.border_logits.parameters()]
        return chain(*(params))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()

    def ms_forward(self, batch, image_size=None):
        image_size = batch.get('target_size', image_size if image_size is not None else batch['image'].shape[-2:])
        ms_logits = None
        pyramid = [batch['image'].cuda()]
        pyramid += [
            F.interpolate(pyramid[0], scale_factor=sf, mode=self.backbone.pyramid_subsample,
                          align_corners=self.backbone.align_corners) for sf in self.multiscale_factors
        ]
        for image in pyramid:
            batch['image'] = image
            logits, additional = self.do_forward(batch, image_size=image_size)
            if ms_logits is None:
                ms_logits = torch.zeros(logits.size()).to(logits.device)
            ms_logits += F.softmax(logits, dim=1)
        batch['image'] = pyramid[0].cpu()
        return ms_logits / len(pyramid), {}

    def set_bipartite_graphs(self, bi_graphs):

        if len(bi_graphs) == 2 * self.n_datasets:
            for i in range(0, self.n_datasets):
                self.bipartite_graphs[i] = nn.Parameter(
                    bi_graphs[2*i], requires_grad=False
                    )
        else:
            # print("bi_graphs len:", len(bi_graphs))
            for i in range(0, self.n_datasets):
                # print("i: ", i)
                self.bipartite_graphs[i] = nn.Parameter(
                    bi_graphs[i], requires_grad=False
                    )

    def set_unify_prototype(self, unify_prototype, grad=False):
        if self.with_datasets_aux:
            self.unify_prototype.data = unify_prototype[self.total_cats:]
            self.unify_prototype.requires_grad=grad
            cur_cat = 0
            for i in range(self.n_datasets):
                self.aux_prototype[i].data = unify_prototype[cur_cat:cur_cat+self.datasets_cats[i]]
                cur_cat += self.datasets_cats[i]
                self.aux_prototype[i].requires_grad=grad
        else:
            self.unify_prototype.data = unify_prototype
            self.unify_prototype.requires_grad=grad

    def get_optim_params(self):
        fine_tune_factor = 4
        optim_params = [
            {'params': self.random_init_params(), 'lr': self.configer.get('lr', 'seg_lr_start'), 'weight_decay': self.configer.get('lr', 'weight_decay')},
            {'params': self.fine_tune_params(), 'lr': self.configer.get('lr', 'seg_lr_start') / fine_tune_factor,
            'weight_decay': self.configer.get('lr', 'weight_decay') / fine_tune_factor},
        ]
        return optim_params