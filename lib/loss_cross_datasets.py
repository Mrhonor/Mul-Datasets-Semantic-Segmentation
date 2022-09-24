from cmath import inf
from traceback import print_tb
from lib.loss_contrast_mem import PixelContrastLoss
from lib.loss_helper import NLLPlusLoss, WeightedNLLPlusLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.class_remap import ClassRemap

class CrossDatasetsLoss(nn.Module):
    def __init__(self, configer=None):
        super(CrossDatasetsLoss, self).__init__()
        
        self.configer = configer
        self.classRemapper = ClassRemap(configer=self.configer)
        
        # self.ignore_index = -1
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     self.ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
            
        self.loss_weight = self.configer.get('contrast', 'loss_weight')    
            
        # self.seg_criterion_warmup = NLLPlusLoss(configer=self.configer)   
        self.seg_criterion = WeightedNLLPlusLoss(configer=self.configer)   
            
        self.with_aux = self.configer.get('loss', 'with_aux')
        if self.with_aux:
            self.aux_num = self.configer.get('loss', 'aux_num')
            self.segLoss_aux = [WeightedNLLPlusLoss(configer=self.configer) for _ in range(self.aux_num)]
        
        self.use_contrast = self.configer.get('contrast', 'use_contrast')
        if self.use_contrast:
            self.contrast_criterion = PixelContrastLoss(configer=configer)
        
        self.upsample = self.configer.get('contrast', 'upsample')
        
    def forward(self, preds, target, dataset_id, is_warmup=False):
        assert "seg" in preds
        
        
        logits, *logits_aux = preds['seg']
        if self.use_contrast:
            embedding = preds['embed']
        
        b, h, w = logits[0].shape

        lb = target
        # 对标签下采样
        if not self.upsample:
            lb = F.interpolate(input=target, size=(h, w), mode='nearest')

        if "segment_queue" in preds:
            segment_queue = preds['segment_queue']
        else:
            segment_queue = None

        seg_label = self.classRemapper.SegRemapping(lb, dataset_id)
    

        if is_warmup or not self.use_contrast:
            # warm up不计算contrast 损失
            weight_mask = self.classRemapper.GetEqWeightMask(lb, dataset_id)

            # pred = F.interpolate(input=logits, size=(h, w), mode='bilinear', align_corners=True)

            loss_seg = self.seg_criterion(logits, weight_mask)
            loss = loss_seg
            loss_aux = None
            if self.with_aux:
                pred_aux = [F.interpolate(input=logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logits_aux]
                loss_aux = [aux_criterion(aux, weight_mask) for aux, aux_criterion in zip(pred_aux, self.segLoss_aux)]
                
                # if torch.isnan(loss_aux[0]):
                #     # print(pred_aux)
                #     # print(seg_label)
                #     # print(loss_aux)
                #     print("***************")
                loss = loss_seg + sum(loss_aux)

            return loss, loss_seg, loss_aux
        else:
            contrast_lable, weight_mask = self.classRemapper.ContrastRemapping(lb, embedding, segment_queue, dataset_id)
            # pred = F.interpolate(input=logits, size=(h, w), mode='bilinear', align_corners=True)

            _, predict = torch.max(logits, 1)

            if self.configer.get('contrast', 'upsample') is False:
                network_stride = self.configer.get('network', 'stride')
                predict = predict[:, ::network_stride, ::network_stride]
            
            loss_contrast = self.contrast_criterion(embedding, contrast_lable, predict, segment_queue)

            loss_seg = self.seg_criterion(logits, weight_mask)
            loss = loss_seg
            loss_aux = None
            
            if self.with_aux:
                aux_weight_mask = self.classRemapper.GetEqWeightMask(lb, dataset_id)
                pred_aux = [F.interpolate(input=logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logits_aux]
                loss_aux = [aux_criterion(aux, aux_weight_mask) for aux, aux_criterion in zip(pred_aux, self.segLoss_aux)]

                
                loss = loss_seg + sum(loss_aux)
                
            
            return loss + self.loss_weight * loss_contrast, loss_seg, loss_aux, loss_contrast
    

        
        