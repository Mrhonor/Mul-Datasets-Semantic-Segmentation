##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
# from lib.utils.tools.logger import Logger as Log

# from lib.rmi_loss import RMILoss

# from lib.lovasz_loss import lovasz_softmax_flat, flatten_probas


class FSCERMILoss(nn.Module):
    def __init__(self, configer=None):
        super(FSCERMILoss, self).__init__()
        self.configer = configer
        weight = None
        if self.configer.exists('loss', 'params') and 'ce_weight' in self.configer.get('loss', 'params'):
            weight = self.configer.get('loss', 'params')['ce_weight']
            weight = torch.FloatTensor(weight).cuda()

        reduction = 'elementwise_mean'
        if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
            reduction = self.configer.get('loss', 'params')['ce_reduction']

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.rmi_loss = RMILoss(self.configer)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        if isinstance(inputs, dict) and 'seg' in inputs:
            inputs = inputs['seg']
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

            loss_rmi = self.rmi_loss(inputs, target)

            loss = loss + loss_rmi

        return loss


class FSCELOVASZLoss(nn.Module):
    def __init__(self, configer=None):
        super(FSCELOVASZLoss, self).__init__()
        self.configer = configer
        weight = None
        if self.configer.exists('loss', 'params') and 'ce_weight' in self.configer.get('loss', 'params'):
            weight = self.configer.get('loss', 'params')['ce_weight']
            weight = torch.FloatTensor(weight).cuda()

        reduction = 'elementwise_mean'
        if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
            reduction = self.configer.get('loss', 'params')['ce_reduction']

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        if isinstance(inputs, dict) and 'seg' in inputs:
            inputs = inputs['seg']
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

            pred = F.softmax(input=inputs, dim=1)
            loss_lovasz = lovasz_softmax_flat(*flatten_probas(pred, target, self.ignore_index),
                                              only_present=True)

            loss = loss + loss_lovasz

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


class WeightedFSOhemCELoss(nn.Module):
    def __init__(self, configer):
        super().__init__()
        self.configer = configer
        self.thresh = self.configer.get('loss', 'params')['ohem_thresh']
        self.reduction = 'elementwise_mean'
        if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
            self.reduction = self.configer.get('loss', 'params')['ce_reduction']

    def forward(self, predict, target, min_kept=1, weight=None, ignore_index=-1, **kwargs):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
        """
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == ignore_index] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1, ) != ignore_index
        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()
        min_threshold = sort_prob[min(min_kept, sort_prob.numel() - 1)]
        threshold = max(min_threshold, self.thresh)
        loss_matrix = F.cross_entropy(predict, target, weight=weight, ignore_index=ignore_index,
                                      reduction='none').contiguous().view(-1, )
        sort_loss_matrix = loss_matrix[mask][sort_indices]
        select_loss_matrix = sort_loss_matrix[sort_prob < threshold]
        if self.reduction == 'sum':
            return select_loss_matrix.sum()
        elif self.reduction == 'elementwise_mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')


# Cross-entropy Loss
class FSCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSCELoss, self).__init__()
        self.configer = configer
        weight = None
        if self.configer.exists('loss', 'params') and 'ce_weight' in self.configer.get('loss', 'params'):
            weight = self.configer.get('loss', 'params')['ce_weight']
            weight = torch.FloatTensor(weight).cuda()

        reduction = 'elementwise_mean'
        if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
            reduction = self.configer.get('loss', 'params')['ce_reduction']

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']

        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


class FSOhemCELoss(nn.Module):
    def __init__(self, configer):
        super(FSOhemCELoss, self).__init__()
        self.configer = configer
        self.thresh = self.configer.get('loss', 'params')['ohem_thresh']
        self.min_kept = max(1, self.configer.get('loss', 'params')['ohem_minkeep'])
        weight = None
        if self.configer.exists('loss', 'params') and 'ce_weight' in self.configer.get('loss', 'params'):
            weight = self.configer.get('loss', 'params')['ce_weight']
            weight = torch.FloatTensor(weight).cuda()

        self.reduction = 'elementwise_mean'
        if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
            self.reduction = self.configer.get('loss', 'params')['ce_reduction']

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']

        self.ignore_label = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')

    def forward(self, predict, target, **kwargs):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1, ) != self.ignore_label
        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()
        min_threshold = sort_prob[min(self.min_kept, sort_prob.numel() - 1)]
        threshold = max(min_threshold, self.thresh)
        loss_matirx = self.ce_loss(predict, target).contiguous().view(-1, )
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == 'sum':
            return select_loss_matrix.sum()
        elif self.reduction == 'elementwise_mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')


class FSAuxOhemCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxOhemCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)
        if self.configer.get('loss', 'loss_type') == 'fs_auxohemce_loss':
            self.ohem_ce_loss = FSOhemCELoss(self.configer)
        else:
            assert self.configer.get('loss', 'loss_type') == 'fs_auxslowohemce_loss'
            self.ohem_ce_loss = FSSlowOhemCELoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out, seg_out = inputs
        seg_loss = self.ohem_ce_loss(seg_out, targets)
        aux_loss = self.ce_loss(aux_out, targets)
        loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
        loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss
        return loss


class FSAuxCELossDSN(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxCELossDSN, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux1, aux2, aux3, seg_out = inputs
        seg_loss = self.ce_loss(seg_out, targets)
        aux1_loss = self.ce_loss(aux1, targets)
        aux2_loss = self.ce_loss(aux2, targets)
        aux3_loss = self.ce_loss(aux3, targets)
        loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
        loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * (aux1_loss + aux2_loss + aux3_loss) / 3
        return loss


class FSAuxCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out, seg_out = inputs
        seg_loss = self.ce_loss(seg_out, targets)
        aux_loss = self.ce_loss(aux_out, targets)
        loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
        loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss
        return loss


class FSAuxRMILoss(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxRMILoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)
        self.rmi_loss = RMILoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out, seg_out = inputs
        aux_loss = self.ce_loss(aux_out, targets)
        seg_loss = self.rmi_loss(seg_out, targets)
        loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
        loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss
        return loss


class MSFSAuxRMILoss(nn.Module):
    def __init__(self, configer=None):
        super(MSFSAuxRMILoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)
        self.rmi_loss = RMILoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out = inputs['aux']
        seg_out = inputs['pred']
        pred_05x = inputs['pred_05x']
        pred_10x = inputs['pred_10x']

        aux_loss = self.ce_loss(aux_out, targets)
        seg_loss = self.rmi_loss(seg_out, targets)
        loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
        loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss

        scaled_pred_05x = torch.nn.functional.interpolate(pred_05x, size=(seg_out.size(2), seg_out.size(3)),
                                                          mode='bilinear', align_corners=False)
        loss_lo = self.ce_loss(scaled_pred_05x, targets)
        loss_hi = self.ce_loss(pred_10x, targets)
        loss += 0.05 * loss_lo
        loss += 0.05 * loss_hi

        return loss


class FSRMILoss(nn.Module):
    def __init__(self, configer=None):
        super(FSRMILoss, self).__init__()
        self.configer = configer
        self.rmi_loss = RMILoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        seg_out = inputs
        loss = self.rmi_loss(seg_out, targets)
        return loss


class SegFixLoss(nn.Module):
    """
    We predict a binary mask to categorize the boundary pixels as class 1 and otherwise as class 0
    Based on the pixels predicted as 1 within the binary mask, we further predict the direction for these
    pixels.
    """

    def __init__(self, configer=None):
        super().__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)

    def calc_weights(self, label_map, num_classes):

        weights = []
        for i in range(num_classes):
            weights.append((label_map == i).sum().data)
        weights = torch.FloatTensor(weights)
        weights_sum = weights.sum()
        return (1 - weights / weights_sum).cuda()

    def forward(self, inputs, targets, **kwargs):

        from lib.utils.helpers.offset_helper import DTOffsetHelper

        pred_mask, pred_direction = inputs

        seg_label_map, distance_map, angle_map = targets[0], targets[1], targets[2]
        gt_mask = DTOffsetHelper.distance_to_mask_label(distance_map, seg_label_map, return_tensor=True)

        gt_size = gt_mask.shape[1:]
        mask_weights = self.calc_weights(gt_mask, 2)

        pred_direction = F.interpolate(pred_direction, size=gt_size, mode="bilinear", align_corners=True)
        pred_mask = F.interpolate(pred_mask, size=gt_size, mode="bilinear", align_corners=True)
        mask_loss = F.cross_entropy(pred_mask, gt_mask, weight=mask_weights, ignore_index=-1)

        mask_threshold = float(os.environ.get('mask_threshold', 0.5))
        binary_pred_mask = torch.softmax(pred_mask, dim=1)[:, 1, :, :] > mask_threshold

        gt_direction = DTOffsetHelper.angle_to_direction_label(
            angle_map,
            seg_label_map=seg_label_map,
            extra_ignore_mask=(binary_pred_mask == 0),
            return_tensor=True
        )

        direction_loss_mask = gt_direction != -1
        direction_weights = self.calc_weights(gt_direction[direction_loss_mask], pred_direction.size(1))
        direction_loss = F.cross_entropy(pred_direction, gt_direction, weight=direction_weights, ignore_index=-1)

        if self.training \
                and self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0 \
                and torch.cuda.current_device() == 0:
            Log.info('mask loss: {} direction loss: {}.'.format(mask_loss, direction_loss))

        mask_weight = float(os.environ.get('mask_weight', 1))
        direction_weight = float(os.environ.get('direction_weight', 1))

        return mask_weight * mask_loss + direction_weight * direction_loss

class NLLPlusLoss(nn.Module):
    def __init__(self, configer=None):
        super(NLLPlusLoss, self).__init__()
        
        self.configer = configer
        self.ignore_index = self.configer.get('loss', 'ignore_index')
        
        self.softmax = nn.Softmax(dim=1)
        self.nllloss = nn.NLLLoss(ignore_index=self.ignore_index, reduction='mean')
    
    def forward(self, x, labels):
        # labels为 k x batch size x H x W
        # k为最大映射数量，对于输入的x需要计算k次nll loss后求和，
        # 对于没有k个映射对象的类别，如只有n个的类别，后n-k个labels值均为ignore_index
        pred = self.softmax(x)
        probs = None
        for lb in labels:
            val = -self.nllloss(pred, lb)
            probs = val if probs is None else probs+val
        
        # prob = torch.sum(probs)
        loss = -torch.log(probs)
        return loss
    
class WeightedNLLPlusLoss(NLLPlusLoss):
    def __init__(self, configer=None):
        super(WeightedNLLPlusLoss, self).__init__(configer=configer)
        
    def forward(self, x, weighted_mask):
        # labels: k x batch size x H x W, weighted_mask: batch size x H x W x num_of_class
        batch_size, n, h, w = x.shape
        
        pred = self.softmax(x)
        probs = torch.einsum('bnhw,bhwn->bhw', pred, weighted_mask)
        
            
        prob = torch.sum(probs) / (batch_size * h * w)
        loss = -torch.log(prob)
        

            
        return loss

class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp, sn):
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss

class MultiLabelCrossEntropyLoss(nn.Module):
    def __init__(self, configer):
        super(MultiLabelCrossEntropyLoss, self).__init__()
        self.configer = configer
        self.m = 0
        self.gamma = 1
        if configer:
            self.m = self.configer.get('loss', 'm')
            self.gamma = self.configer.get('loss', 'gamma')
        self.soft_plus = nn.Softplus()
    
    def forward(self, x, labels):
        # x: batch size x c x h x w
        # labels: batch x h x w x c, 1表示目标标签，0表示非目标标签
        b, c, h, w = x.shape
        
        pred = x.permute(1,0,2,3)
        pred = pred.contiguous().view(c, -1)
        lb = labels.permute(3, 0, 1, 2)
        lb = lb.contiguous().view(c, -1)
        
        pred = F.sigmoid(pred)
        pos_lb = lb
        
        neg_lb = 1- lb
        # print("x: ", x.shape)
        # print("pred: ",pred.shape)
        # print("lb: ", lb.shape)
        # pos_pred = pred * pos_lb
        # ## 保持矩阵形式，忽略为0项
        # pos_pred[pos_pred == 0] = -1e12 


        pos_pred = torch.exp(-pred * self.gamma) * pos_lb
        # neg_pred = pred * neg_lb
        # neg_pred[neg_pred == 0] = -1e12
        neg_pred = torch.exp((pred + self.m) * self.gamma) * neg_lb
        # loss = self.soft_plus(torch.logsumexp(pos_pred, dim=0) + torch.logsumexp(neg_pred, dim=0))
        loss = torch.log(torch.sum(pos_pred, dim=0)*torch.sum(neg_pred, dim=0) + 1)
        
        lb_num = torch.sum(loss != 0)
        if lb_num ==0:
            return 0
        
        # print(torch.max(loss))# / lb_num)
        loss = torch.sum(loss) / lb_num 
        
        # loss = torch.sum(loss) / torch.sum(loss != 0)
    
        # print(loss)
        return loss
        
    
def test_MultiLabelCrossEntropyLoss():
    loss_fuc = MultiLabelCrossEntropyLoss(None)
    x = torch.tensor([[-1, 1],
                      [2, 3], 
                      [4, 4]])
    labels = torch.tensor([[0,1],
                           [1,0],
                           [0,1]])
    print(loss_fuc(x, labels))
    # print("true value: 0.7111")
    
        # print("pred: ")
        # print(pos_pred)
        # print(neg_pred)
        # print("logsumexp: ")
        # print(torch.logsumexp(pos_pred, dim=0))
        # print(torch.logsumexp(neg_pred, dim=0))
        # print(torch.logsumexp(pos_pred, dim=0) + torch.logsumexp(neg_pred, dim=0))
        # print(self.soft_plus(torch.logsumexp(pos_pred, dim=0) + torch.logsumexp(neg_pred, dim=0)))
    

if __name__ == "__main__":
    test_MultiLabelCrossEntropyLoss()
    # a = torch.randn(2, 3, 2, 2)
    
    # b = torch.tensor([[[[0,0],[1,1]],[[2,2],[0,0]]]])
    # weighted_mask = torch.tensor([[[[0.5,0,0.5], [0.3,0.7,0]],[[0.3,0.4,0.3], [0,1,0]]],[[[0,0,1], [0,0,1]],[[1,0,0],[1,0,0]]]], dtype=torch.float)
    # loss1 = NLLPlusLoss()
    # loss2 = WeightedNLLPlusLoss()
    # print(loss1(a,b))
    # print(loss2(a,weighted_mask))
    # print(loss2(a,weighted_mask).item())