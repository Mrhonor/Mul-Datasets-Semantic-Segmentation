from lib.loss_contrast_mem import PixelContrastLoss
from lib.loss_helper import NLLPlusLoss

import torch
import torch.nn as nn
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
        
    def forward(self, preds, target, dataset_id):
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
        

        pred = F.interpolate(input=logits, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, seg_label)

        if self.with_aux:
            pred_aux = [F.interpolate(input=logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logits_aux]
            loss_aux = [aux_criterion(aux, seg_label) for aux, aux_criterion in zip(pred_aux, self.segLoss_aux)]
            loss +=  sum(loss_aux)

        _, predict = torch.max(logits, 1)
        loss_contrast = self.contrast_criterion(embedding, target, predict, segment_queue)


        return loss + self.loss_weight * loss_contrast
    

        
        