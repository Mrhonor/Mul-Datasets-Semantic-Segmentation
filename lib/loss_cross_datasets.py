from cmath import inf
from distutils.command.config import config
from traceback import print_tb
from lib.loss_contrast_mem import PixelContrastLoss, PixelPrototypeDistanceLoss
from lib.loss_helper import NLLPlusLoss, WeightedNLLPlusLoss, MultiLabelCrossEntropyLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.class_remap import ClassRemap, ClassRemapOneHotLabel



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
            self.segLoss_aux = [eval(self.configer.get('loss', 'type'))(configer=self.configer) for _ in range(self.aux_num)]
        
        self.use_contrast = self.configer.get('contrast', 'use_contrast')
        if self.use_contrast:
            self.contrast_criterion = PixelContrastLoss(configer=configer)
        
            self.with_ppd = self.configer.get('contrast', 'with_ppd')
            if self.with_ppd:
                self.ppd_loss_weight = self.configer.get('contrast', 'ppd_loss_weight')
                self.ppd_criterion = PixelPrototypeDistanceLoss(configer=configer)
        
        self.upsample = self.configer.get('contrast', 'upsample')
        self.network_stride = self.configer.get('network', 'stride')
        
    def forward(self, preds, target, dataset_id, is_warmup=False):
        assert "seg" in preds
        
        
        logits, *logits_aux = preds['seg']
        if self.use_contrast:
            embedding = preds['embed']
        

        b, h, w = logits[0].shape

        lb = target
        # 对标签下采样
        if not self.upsample:
            # lb = lb[:, ::self.network_stride, ::self.network_stride]
            lb = F.interpolate(input=target.unsqueeze(1).float(), size=(h, w), mode='nearest').squeeze().long()

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
            # emb_logits = torch.einsum('bchw,nc->bnhw', embedding, segment_queue)
            contrast_lable, weight_mask = self.classRemapper.ContrastRemapping(lb, embedding, segment_queue, dataset_id)
            pred = F.interpolate(input=logits, size=embedding.shape[-2:], mode='bilinear', align_corners=True)

            _, predict = torch.max(pred, 1)

            # if self.configer.get('contrast', 'upsample') is False:
            #     network_stride = self.configer.get('network', 'stride')
            #     predict = predict[:, ::network_stride, ::network_stride]

            loss_contrast = self.contrast_criterion(embedding, contrast_lable, predict, segment_queue)

            loss_seg = self.seg_criterion(logits, weight_mask)
            loss = loss_seg
            loss_aux = None
            
            if self.with_aux:
                aux_weight_mask = self.classRemapper.GetEqWeightMask(lb, dataset_id)
                pred_aux = [F.interpolate(input=logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logits_aux]
                loss_aux = [aux_criterion(aux, aux_weight_mask) for aux, aux_criterion in zip(pred_aux, self.segLoss_aux)]

                
                loss = loss_seg + sum(loss_aux)
                
            if self.with_ppd:
                loss_ppd = self.ppd_criterion(embedding, contrast_lable, segment_queue)
                loss_contrast = loss_contrast + self.ppd_loss_weight * loss_ppd
                
            
            return loss + self.loss_weight * loss_contrast, loss_seg, loss_aux, loss_contrast
    

if __name__ == "__main__":
    loss_fuc = PixelPrototypeDistanceLoss()
    a = torch.randn(2,4,3,2)
    print(a)
    lb = torch.tensor([[[0,1],[2,0],[255,0]],[[2,1],[1,255],[255,255]]])
    seq = torch.randn(3,4)
    print(seq)
    print(loss_fuc(a,lb,seq))
        