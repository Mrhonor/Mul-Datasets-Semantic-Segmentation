from lib.loss_contrast_mem import PixelContrastLoss
from lib.loss_helper import NLLPlusLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.class_remap import ClassRemap

class CrossDatasetsLoss(nn.Module):
    def __init__(self, configer=None):
        super(CrossDatasetsLoss, self).__init__()
        
        self.configer = configer
        self.classRemapper = ClassRemap(configer=self.configer)
        
        self.ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
            
        self.loss_weight = self.configer.get('contrast', 'loss_weight')    
            
        self.seg_criterion = NLLPlusLoss(configer=self.configer)    
            
        self.with_aux = self.configer.get('loss', 'with_aux')
        if self.with_aux:
            self.aux_num = self.configer.get('loss', 'aux_num')
            self.segLoss_aux = [NLLPlusLoss(configer=self.configer) for _ in range(self.aux_num)]
        
        self.contrast_criterion = PixelContrastLoss(configer=configer)
        
    def forward(self, preds, target, dataset_id, is_warmup=False):
        h, w = target.size(1), target.size(2)
        

        assert "seg" in preds
        assert "embed" in preds
        
        logits, *logits_aux = preds['seg']
        embedding = preds['embed']
        

        if "segment_queue" in preds:
            segment_queue = preds['segment_queue']
        else:
            segment_queue = None

        seg_label = self.classRemapper.SegRemapping(target, dataset_id)
        contrast_lable, weight_mask = self.classRemapper.ContrastRemapping(target, embedding, segment_queue, dataset_id)

        pred = F.interpolate(input=logits, size=(h, w), mode='bilinear', align_corners=True)

        _, predict = torch.max(logits, 1)
        loss_contrast = self.contrast_criterion(embedding, contrast_lable, predict, segment_queue)

        loss_seg = self.seg_criterion(pred, weight_mask)

        loss_aux = None
        if self.with_aux:
            pred_aux = [F.interpolate(input=logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logits_aux]
            loss_aux = [aux_criterion(aux, seg_label) for aux, aux_criterion in zip(pred_aux, self.segLoss_aux)]
            loss = loss_seg + sum(loss_aux)

        if is_warmup is False:
            return loss, loss_seg, loss_contrast, loss_aux
        else:
            return loss + self.loss_weight * loss_contrast, loss_seg, loss_contrast, loss_aux
    

        
        