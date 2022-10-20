from cmath import inf
from distutils.command.config import config
from traceback import print_tb
from lib.loss_contrast_mem import PixelContrastLoss, PixelPrototypeDistanceLoss, PixelContrastLossOnlyNeg
from lib.loss_helper import NLLPlusLoss, WeightedNLLPlusLoss, MultiLabelCrossEntropyLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.class_remap import ClassRemap, ClassRemapOneHotLabel



# class CrossDatasetsLoss(nn.Module):
#     def __init__(self, configer=None):
#         super(CrossDatasetsLoss, self).__init__()
        
#         self.configer = configer
#         self.classRemapper = eval(self.configer.get('class_remaper'))(configer=self.configer)
        
#         # self.ignore_index = -1
#         # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
#         #     self.ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
            
#         self.loss_weight = self.configer.get('contrast', 'loss_weight')    
            
#         # self.seg_criterion_warmup = NLLPlusLoss(configer=self.configer)   
#         self.seg_criterion = eval(self.configer.get('loss', 'type'))(configer=self.configer)   
            
#         self.with_aux = self.configer.get('loss', 'with_aux')
#         if self.with_aux:
#             self.aux_num = self.configer.get('loss', 'aux_num')
#             self.segLoss_aux = [eval(self.configer.get('loss', 'type'))(configer=self.configer) for _ in range(self.aux_num)]
        
#         self.use_contrast = self.configer.get('contrast', 'use_contrast')
#         if self.use_contrast:
#             self.contrast_criterion = PixelContrastLoss(configer=configer)
        
#             self.with_ppd = self.configer.get('contrast', 'with_ppd')
#             if self.with_ppd:
#                 self.ppd_loss_weight = self.configer.get('contrast', 'ppd_loss_weight')
#                 self.ppd_criterion = PixelPrototypeDistanceLoss(configer=configer)
        
#         self.upsample = self.configer.get('contrast', 'upsample')
#         self.network_stride = self.configer.get('network', 'stride')
        
#     def forward(self, preds, target, dataset_id, is_warmup=False):
#         assert "seg" in preds
        
        
#         logits, *logits_aux = preds['seg']
#         if self.use_contrast:
#             embedding = preds['embed']
        

#         b, h, w = logits[0].shape

#         lb = target
#         # 对标签下采样
#         if not self.upsample:
#             lb = lb[:, ::self.network_stride, ::self.network_stride]

#         if "segment_queue" in preds:
#             segment_queue = preds['segment_queue']
#         else:
#             segment_queue = None

#         seg_label = self.classRemapper.SegRemapping(lb, dataset_id)
    

#         if is_warmup or not self.use_contrast:
#             # warm up不计算contrast 损失
#             weight_mask = self.classRemapper.GetEqWeightMask(lb, dataset_id)

#             # pred = F.interpolate(input=logits, size=(h, w), mode='bilinear', align_corners=True)

#             loss_seg = self.seg_criterion(logits, weight_mask)
#             loss = loss_seg
#             loss_aux = None
#             if self.with_aux:
#                 pred_aux = [F.interpolate(input=logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logits_aux]
#                 loss_aux = [aux_criterion(aux, weight_mask) for aux, aux_criterion in zip(pred_aux, self.segLoss_aux)]
                
#                 # if torch.isnan(loss_aux[0]):
#                 #     # print(pred_aux)
#                 #     # print(seg_label)
#                 #     # print(loss_aux)
#                 #     print("***************")
#                 loss = loss_seg + sum(loss_aux)

#             return loss, loss_seg, loss_aux
#         else:
#             # emb_logits = torch.einsum('bchw,nc->bnhw', embedding, segment_queue)
#             contrast_lable, weight_mask = self.classRemapper.ContrastRemapping(lb, embedding, segment_queue, dataset_id)
#             pred = F.interpolate(input=logits, size=embedding.shape[-2:], mode='bilinear', align_corners=True)

#             _, predict = torch.max(pred, 1)

#             # if self.configer.get('contrast', 'upsample') is False:
#             #     network_stride = self.configer.get('network', 'stride')
#             #     predict = predict[:, ::network_stride, ::network_stride]

#             loss_contrast = self.contrast_criterion(embedding, contrast_lable, predict, segment_queue)

#             loss_seg = self.seg_criterion(logits, weight_mask)
#             loss = loss_seg
#             loss_aux = None
            
#             if self.with_aux:
#                 aux_weight_mask = self.classRemapper.GetEqWeightMask(lb, dataset_id)
#                 pred_aux = [F.interpolate(input=logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logits_aux]
#                 loss_aux = [aux_criterion(aux, aux_weight_mask) for aux, aux_criterion in zip(pred_aux, self.segLoss_aux)]

                
#                 loss = loss_seg + sum(loss_aux)
                
#             if self.with_ppd:
#                 loss_ppd = self.ppd_criterion(embedding, contrast_lable, segment_queue)
#                 loss_contrast = loss_contrast + self.ppd_loss_weight * loss_ppd
                
            
#             return loss + self.loss_weight * loss_contrast, loss_seg, loss_aux, loss_contrast
    

class CrossDatasetsLoss(nn.Module):
    def __init__(self, configer=None):
        super(CrossDatasetsLoss, self).__init__()
        
        self.configer = configer
        self.classRemapper = eval(self.configer.get('class_remaper'))(configer=self.configer)
        
        # self.ignore_index = -1
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     self.ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
            
        self.loss_weight = self.configer.get('contrast', 'loss_weight')    
            
        # self.seg_criterion_warmup = NLLPlusLoss(configer=self.configer)   
        self.seg_criterion = eval(self.configer.get('loss', 'type'))(configer=self.configer)   
            
        self.with_aux = self.configer.get('loss', 'with_aux')
        if self.with_aux:
            self.aux_num = self.configer.get('loss', 'aux_num')
            self.aux_weight = self.configer.get('loss', 'aux_weight')
            self.segLoss_aux = [eval(self.configer.get('loss', 'type'))(configer=self.configer) for _ in range(self.aux_num)]
        
        self.use_contrast = self.configer.get('contrast', 'use_contrast')
        if self.use_contrast:
            self.contrast_criterion = PixelContrastLoss(configer=configer)
        
            self.with_ppd = self.configer.get('contrast', 'with_ppd')
            if self.with_ppd:
                self.ppd_loss_weight = self.configer.get('contrast', 'ppd_loss_weight')
                self.ppd_criterion = PixelPrototypeDistanceLoss(configer=configer)
                
            self.with_hard_lb_contrast = self.configer.get('contrast', 'with_hard_lb_contrast')
            if self.with_hard_lb_contrast:
                self.hard_lb_contrast_loss_weight = self.configer.get('contrast', 'hard_lb_contrast_loss_weight')
                self.hard_lb_contrast_loss = PixelContrastLossOnlyNeg(configer=configer)
            
            
        
        self.upsample = self.configer.get('contrast', 'upsample')
        self.network_stride = self.configer.get('network', 'stride')
        
        self.with_domain_adversarial = self.configer.get('network', 'with_domain_adversarial')
        if self.with_domain_adversarial:
            self.n_datasets = self.configer.get('n_datasets')
            batch_sizes = torch.tensor([self.configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, self.n_datasets+1)])
            batch_size_sum = torch.sum(batch_sizes)
            
            weight_vector = F.normalize(batch_size_sum / batch_sizes, p=1, dim=0).cuda()
            
            self.domain_loss = torch.nn.CrossEntropyLoss(weight=weight_vector)
            self.domain_loss_weight = self.configer.get('loss', 'domain_loss_weight')
        
        
    def forward(self, preds, target, dataset_id, is_warmup=False):
        assert "seg" in preds
        
        
        logits, *logits_aux = preds['seg']
        if self.use_contrast:
            embedding = preds['embed']
        
        if self.with_domain_adversarial:
            domain_pred = preds['domain']        

        b, c, h, w = logits.shape

        lb = target
        # # 对标签下采样
        # if not self.upsample:
        #     lb = lb[:, ::self.network_stride, ::self.network_stride]

        if "segment_queue" in preds:
            segment_queue = preds['segment_queue']
        else:
            segment_queue = None

        seg_label = self.classRemapper.SegRemapping(lb, dataset_id)
    

        if is_warmup or not self.use_contrast:
            # pred = F.interpolate(input=logits, size=(h, w), mode='bilinear', align_corners=True)

            loss_seg = self.seg_criterion(logits, seg_label)
            loss = loss_seg
            loss_aux = None
            loss_domain = None
            if self.with_aux:
                pred_aux = [F.interpolate(input=logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logits_aux]
                loss_aux = [aux_criterion(aux, seg_label) for aux, aux_criterion in zip(pred_aux, self.segLoss_aux)]
                

                loss = loss_seg + self.aux_num * sum(loss_aux)
                
                
            if self.with_domain_adversarial:
                domain_label = torch.ones(b, dtype=torch.long) * dataset_id

                if domain_pred.is_cuda:
                    domain_label = domain_label.cuda()
                    
                    
                loss_domain = self.domain_loss(domain_pred, domain_label)
                loss = loss + self.domain_loss_weight * loss_domain
                

            return loss, loss_seg, loss_aux, loss_domain
        else:
            # emb_logits = torch.einsum('bchw,nc->bnhw', embedding, segment_queue)
            contrast_lable, seg_mask, hard_lb_mask = self.classRemapper.ContrastRemapping(lb, embedding, segment_queue, dataset_id)
            pred = F.interpolate(input=logits, size=embedding.shape[-2:], mode='bilinear', align_corners=True)

            _, predict = torch.max(pred, 1)

            # if self.configer.get('contrast', 'upsample') is False:
            #     network_stride = self.configer.get('network', 'stride')
            #     predict = predict[:, ::network_stride, ::network_stride]

            loss_contrast = self.contrast_criterion(embedding, contrast_lable, predict, segment_queue) + self.hard_lb_contrast_loss(embedding, hard_lb_mask, segment_queue)
             

            loss_seg = self.seg_criterion(logits, seg_mask)
            loss = loss_seg
            loss_aux = None
            
            if self.with_aux:
                # aux_weight_mask = self.classRemapper.GetEqWeightMask(lb, dataset_id)
                pred_aux = [F.interpolate(input=logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logits_aux]
                loss_aux = [aux_criterion(aux, seg_mask) for aux, aux_criterion in zip(pred_aux, self.segLoss_aux)]

                
                loss = loss_seg + self.aux_num * sum(loss_aux)
                
            if self.with_ppd:
                loss_ppd = self.ppd_criterion(embedding, contrast_lable, segment_queue)
                loss_contrast = loss_contrast + self.ppd_loss_weight * loss_ppd
                
            if self.with_domain_adversarial:
                domain_label = torch.ones(b, dtype=torch.long) * dataset_id
                if domain_pred.is_cuda:
                    domain_label = domain_label.cuda()
                    
                loss_domain = self.domain_loss(domain_pred, domain_label)
                loss = loss + self.domain_loss_weight * loss_domain
            
            
            return loss + self.loss_weight * loss_contrast, loss_seg, loss_aux, loss_contrast, loss_domain

if __name__ == "__main__":
    loss_fuc = PixelPrototypeDistanceLoss()
    a = torch.randn(2,4,3,2)
    print(a)
    lb = torch.tensor([[[0,1],[2,0],[255,0]],[[2,1],[1,255],[255,255]]])
    seq = torch.randn(3,4)
    print(seq)
    print(loss_fuc(a,lb,seq))
        