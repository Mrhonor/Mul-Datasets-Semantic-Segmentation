
import sys
sys.path.insert(0, '/root/mr/Mul-Datasets-Semantic-Segmentation')
from cmath import inf
from distutils.command.config import config
from traceback import print_tb
from lib.loss.loss_contrast_mem import PixelContrastLoss, PixelPrototypeDistanceLoss, PixelContrastLossOnlyNeg, PixelContrastLossMulProto
from lib.loss.loss_helper import NLLPlusLoss, WeightedNLLPlusLoss, MultiLabelCrossEntropyLoss, CircleLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.module.memory_bank_helper import memory_bank_push
from lib.class_remap import ClassRemap, ClassRemapOneHotLabel
from lib.prototype_learning import prototype_learning, KmeansProtoLearning
from lib.module.kmeans import kmeans
from einops import rearrange, repeat, einsum


def LabelToOneHot(LabelVector, nClass, ignore_index=-1):
    
    ## 输入的label应该是一维tensor向量
    OutOneHot = torch.zeros(len(LabelVector), nClass, dtype=torch.bool)
    if LabelVector.is_cuda:
        OutOneHot = OutOneHot.cuda()
        
    OutOneHot[LabelVector!=ignore_index, LabelVector[LabelVector!=ignore_index]]=1
    return OutOneHot

class CrossDatasetsLoss(nn.Module):
    def __init__(self, configer=None):
        super(CrossDatasetsLoss, self).__init__()
        self.configer = configer
        self.n_datasets = self.configer.get('n_datasets')
        self.classRemapper = eval(self.configer.get('class_remaper'))(configer=self.configer)
        self.num_unify_classes = self.configer.get('num_unify_classes')
        self.num_prototype = self.configer.get('contrast', 'num_prototype')
        self.temperature = self.configer.get('contrast', 'temperature')
        self.with_mulbn = self.configer.get('contrast', 'with_mulbn')
        self.reweight = self.configer.get('loss', 'reweight')
        self.use_ema = self.configer.get('use_ema')
        
        # self.ignore_index = -1
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     self.ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
            
        self.loss_weight = self.configer.get('contrast', 'loss_weight')    
            
        
        # 处理多标签
        self.seg_criterion_mul = eval(self.configer.get('loss', 'type'))(configer=self.configer)   
        # 处理单标签
        # self.seg_criterion_sig = eval(self.configer.get('loss', 'type'))()   
            
        self.with_aux = self.configer.get('loss', 'with_aux')
        if self.with_aux:
            self.aux_num = self.configer.get('loss', 'aux_num')
            self.aux_weight = self.configer.get('loss', 'aux_weight')
            self.segLoss_aux_Mul = [eval(self.configer.get('loss', 'type'))(configer=self.configer) for _ in range(self.aux_num)]
            # self.segLoss_aux_Sig = [eval(self.configer.get('loss', 'type'))() for _ in range(self.aux_num)]
        
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
                self.hard_lb_contrast_loss = PixelContrastLossMulProto(configer=configer)
            
            self.with_consistence = self.configer.get('contrast', 'with_consistence')
            if self.with_consistence:
                self.consistent_criterion = nn.KLDivLoss(reduction='mean')
                self.consistent_loss_weight = self.configer.get('contrast', 'consistent_loss_weight')
            
        
        self.upsample = self.configer.get('contrast', 'upsample')
        self.network_stride = self.configer.get('network', 'stride')
        
        self.with_domain_adversarial = self.configer.get('network', 'with_domain_adversarial')
        if self.with_domain_adversarial:
            batch_sizes = torch.tensor([self.configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, self.n_datasets+1)])
            batch_size_sum = torch.sum(batch_sizes)
            
            weight_vector = F.normalize(batch_size_sum / batch_sizes, p=1, dim=0).cuda()
            
            self.domain_loss1 = torch.nn.CrossEntropyLoss(weight=weight_vector)
            self.domain_loss2 = torch.nn.CrossEntropyLoss(weight=weight_vector)
            self.domain_loss_weight = self.configer.get('loss', 'domain_loss_weight')
        
        
    def forward(self, preds, target, dataset_ids, is_warmup=False):
        assert "seg" in preds
        
        if self.with_aux:
            logits, *logits_aux = preds['seg']
        else:
            logits = preds['seg']
        
        if self.use_contrast:
            if self.with_mulbn:
                embedding, embedding_others = preds['embed']
            else:
                embedding = preds['embed']
            # embedding = torch.cat((embedding_ori, embedding_bn), dim=0)
        
        if self.with_domain_adversarial:
            domain_pred1, domain_pred2 = preds['domain']        

        if self.use_ema:
            ema_pred = preds['ema']

        b, c, h, w = logits.shape

        lb = target
 
        if "prototypes" in preds:
            prototypes = preds['prototypes']
        else:
            prototypes = None

        test = False

        contrast_lb = lb[:, ::self.network_stride, ::self.network_stride]
        
        
        new_proto = None
        max_index_others = None
        if self.use_contrast:
            rearr_emb = rearrange(embedding, 'b c h w -> (b h w) c')
            proto_mask = self.AdaptiveSingleSegRemapping(contrast_lb, dataset_ids)
            proto_logits, proto_target, new_proto = prototype_learning(self.configer, prototypes, rearr_emb, logits, proto_mask, update_prototype=True)

            ## n: num of class; k: num of prototype per class
            ## proto_logits: (b h_c w_c) * (nk) 每个通道输出分别与prototype的内积
            ## proto_target: 每个通道输出所分配到的prototype的index
            
            proto_targetOntHot = LabelToOneHot(proto_target, self.num_unify_classes*self.num_prototype)
            
            proto_targetOntHot = rearrange(proto_targetOntHot, '(b h w) n -> b h w n', b=contrast_lb.shape[0], h=contrast_lb.shape[1], w=contrast_lb.shape[2])
            
            if self.with_mulbn:
                rearr_emb_others = [rearrange(emb, 'b c h w -> (b h w) c') for emb in embedding_others] 
                max_index_others = [torch.max(emb, dim=1)[1] for emb in rearr_emb_others]
                # proto_targetOntHot_others = [LabelToOneHot(tgt, self.num_unify_classes*self.num_prototype) for tgt in max_index_others]
                # proto_targetOntHot_others = [rearrange(tgt, '(b h w) n -> b h w n', b=contrast_lb.shape[0], h=contrast_lb.shape[1], w=contrast_lb.shape[2]) for tgt in proto_targetOntHot_others] 
                
                # cosine_similarity = torch.mm(_c, prototypes.view(-1, prototypes.shape[-1]).t())


        loss_aux = None
        loss_domain = None
        loss_contrast = None
        KLloss = None

        if is_warmup or not self.use_contrast:
            # pred = F.interpolate(input=logits, size=(h, w), mode='bilinear', align_corners=True)
            seg_label_mul = self.AdaptiveSegRemapping(lb, dataset_ids)
            loss_seg_mul = self.seg_criterion_mul(logits, seg_label_mul)
            
            # loss_seg_mul = self.seg_criterion_mul(logits, seg_label_mul + seg_label_sig)
            loss_seg = loss_seg_mul
            loss = loss_seg
            if self.with_aux:
                # pred_aux = [F.interpolate(input=logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logits_aux]
                pred_aux = [F.interpolate(input=logit, size=(h, w), mode='bilinear', align_corners=True)
                            for logit in logits_aux]
                loss_aux = [aux_criterion_mul(aux, seg_label_mul) for aux, aux_criterion_mul in zip(pred_aux, self.segLoss_aux_Mul)]
                # loss_aux = [aux_criterion_mul(aux, seg_label_mul+ seg_label_sig) for aux, aux_criterion_mul, aux_criterion_sig in zip(pred_aux, self.segLoss_aux_Mul, self.segLoss_aux_Sig)]
                
                loss = loss + self.aux_weight * sum(loss_aux)
                
             
        else:
            reweight_matrix = None
            if self.reweight:
                reweight_matrix = self.AdaptiveGetReweightMatrix(lb, dataset_ids).contiguous().view(-1)
                
            # loss_contrast = self.contrast_criterion(embedding, contrast_mask_label, predict, segment_queue) + self.hard_lb_contrast_loss(embedding, hard_lb_mask, segment_queue)
            
            # proto_targetOntHot 单标签， contrast_mask_label 多标签
            if not self.use_ema:
                contrast_mask_label, seg_mask_mul = self.AdaptiveMultiProtoRemapping(lb, proto_logits, dataset_ids, max_index_others)
            else:
                contrast_mask_label, _ = self.AdaptiveMultiProtoRemapping(lb, proto_logits, dataset_ids, max_index_others)
                rearr_ema = rearrange(ema_pred, 'b c h w -> (b h w) c')
                _, seg_mask_mul = self.AdaptiveMultiProtoRemapping(lb, rearr_ema, dataset_ids)
                
                
            loss_contrast = self.hard_lb_contrast_loss(proto_logits, contrast_mask_label+proto_targetOntHot)
            # if self.with_mulbn:
            #     for i in range(0, self.n_datasets):
            #         proto_logits_other = torch.mm(rearr_emb_others[i], prototypes.view(-1, prototypes.shape[-1]).t())
            #         loss_contrast += self.hard_lb_contrast_loss(proto_logits_other, contrast_mask_label+proto_targetOntHot) 


            loss_seg_mul = self.seg_criterion_mul(logits, seg_mask_mul, reweight_matrix)
            loss_seg = loss_seg_mul 
            loss = loss_seg

            
            if self.with_aux:
                # aux_weight_mask = self.classRemapper.GetEqWeightMask(lb, dataset_id)

                pred_aux = [F.interpolate(input=logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logits_aux]
                # loss_aux = [aux_criterion_sig(aux[0], seg_mask_sig) + aux_criterion_mul(aux[1], seg_mask_mul) for aux, aux_criterion_mul, aux_criterion_sig in zip(pred_aux, self.segLoss_aux_Mul, self.segLoss_aux_Sig)]
                loss_aux = [aux_criterion_mul(aux, seg_mask_mul, reweight_matrix) for aux, aux_criterion_mul in zip(pred_aux, self.segLoss_aux_Mul)]
                
                loss = loss + self.aux_weight * sum(loss_aux)
                
            # if self.with_ppd:
            #     loss_ppd = self.ppd_criterion(embedding, contrast_mask_label, segment_queue)
            #     loss_contrast = loss_contrast + self.ppd_loss_weight * loss_ppd
                
            loss += self.loss_weight * loss_contrast
            
            if self.with_consistence:
                
                KLloss = self.consistent_criterion(F.log_softmax(logits[int(b/2):] / self.temperature, dim=1), F.softmax(logits[:int(b/2)].detach() / self.temperature, dim=1))
    
                loss += self.consistent_loss_weight * KLloss  
            
            

        if self.with_domain_adversarial:
            domain_label = torch.ones(b, dtype=torch.int) 

            if domain_pred1.is_cuda:
                domain_label = domain_label.cuda()
                
            domain_label = domain_label * dataset_ids
                
            
            loss_domain1 = self.domain_loss1(domain_pred1, domain_label)
            loss_domain2 = self.domain_loss2(domain_pred2, domain_label)
            loss_domain = loss_domain1 + loss_domain2
            loss = loss + self.domain_loss_weight * loss_domain
            
            
        return loss, loss_seg, loss_aux, loss_contrast, loss_domain, KLloss, new_proto


    def AdaptiveSingleSegRemapping(self, lb, dataset_ids):

        proto_mask = torch.zeros_like(lb)

        for i in range(0, self.n_datasets):
            if not (dataset_ids == i).any():
                continue
            
            proto_mask[dataset_ids==i] = self.classRemapper.SingleSegRemapping(lb[dataset_ids==i], i)
        
        return proto_mask.contiguous().view(-1)
    
    def AdaptiveSegRemapping(self, lb, dataset_ids):
        b, h, w = lb.shape
        seg_label_mul = torch.zeros(b,h,w,self.num_unify_classes, dtype=torch.bool)
        if lb.is_cuda:
            seg_label_mul = seg_label_mul.cuda()
            
        for i in range(0, self.n_datasets):
            if not (dataset_ids == i).any():
                continue
            
            seg_label_mul[dataset_ids] = self.classRemapper.SegRemapping(lb[dataset_ids==i], i)
            
        return seg_label_mul

    def AdaptiveMultiProtoRemapping(self, lb, proto_logits, dataset_ids, max_index_others=None):
        b, h, w = lb.shape
        seg_mask_mul = torch.zeros(b,h,w,self.num_unify_classes, dtype=torch.bool)
        contrast_mask_label = torch.zeros(b, int(h/self.network_stride), int(w/self.network_stride), self.num_unify_classes*self.num_prototype, dtype=torch.bool)
        if lb.is_cuda:
            seg_mask_mul = seg_mask_mul.cuda()
            contrast_mask_label = contrast_mask_label.cuda()
            
        for i in range(0, self.n_datasets):
            if not (dataset_ids == i).any():
                continue
            re_proto_logits = rearrange(proto_logits, '(b h w) n -> b h w n', b=contrast_mask_label.shape[0], h=contrast_mask_label.shape[1], w=contrast_mask_label.shape[2])
            this_proto_logits = rearrange(re_proto_logits[dataset_ids==i], 'b h w n -> (b h w) n')
            out_contrast_mask, out_seg_mask = self.classRemapper.MultiProtoRemapping(lb[dataset_ids==i], this_proto_logits, i, max_index_others)
            contrast_mask_label[dataset_ids==i] = out_contrast_mask
            seg_mask_mul[dataset_ids==i] = out_seg_mask
            
        return contrast_mask_label, seg_mask_mul
    
    def AdaptiveGetReweightMatrix(self, lb, dataset_ids):
        b, h, w = lb.shape
        ReweightMatrix_mul = torch.ones_like(lb)
            
        for i in range(0, self.n_datasets):
            if not (dataset_ids == i).any():
                continue
            
            ReweightMatrix_mul[dataset_ids==i] = self.classRemapper.getReweightMatrix(lb[dataset_ids==i], i)
            
        return ReweightMatrix_mul
        
class CrossDatasetsCELoss(nn.Module):
    def __init__(self, configer=None):
        super(CrossDatasetsCELoss, self).__init__()
        self.configer = configer
        self.n_datasets = self.configer.get('n_datasets')
        self.classRemapper = eval(self.configer.get('class_remaper'))(configer=self.configer)
        self.num_unify_classes = self.configer.get('num_unify_classes')
        self.num_prototype = self.configer.get('contrast', 'num_prototype')
        self.temperature = self.configer.get('contrast', 'temperature')
        self.with_mulbn = self.configer.get('contrast', 'with_mulbn')
        self.reweight = self.configer.get('loss', 'reweight')
        self.ignore_index = self.configer.get('loss', 'ignore_index')

        self.n_cats = []
        for i in range(1, self.n_datasets+1):
            self.n_cats.append(self.configer.get('dataset'+str(i), 'n_cats'))

        self.CELoss = torch.nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, preds, target, dataset_ids, is_warmup=False):
        self.with_aux = self.configer.get('loss', 'with_aux')
        
        if self.with_aux:
            logits, *logits_aux = preds['seg']
        else:
            logits = preds['seg']
        
        
        loss = None
        for i in range(0, self.n_datasets):
            if not (dataset_ids == i).any():
                continue
            
            RemapMatrix = self.classRemapper.getRemapMatrix(i)
            if logits.is_cuda:
                RemapMatrix = RemapMatrix.cuda()
            

            remap_logits = torch.einsum('bchw, nc -> bnhw', logits[dataset_ids==i], RemapMatrix)
            if loss is None:
                loss = self.CELoss(remap_logits, target[dataset_ids==i])
            else:
                loss = loss + self.CELoss(remap_logits, target[dataset_ids==i])
                
        return loss
            
class CrossDatasetsCELoss_KMeans(nn.Module):
    def __init__(self, configer=None):
        super(CrossDatasetsCELoss_KMeans, self).__init__()
        self.configer = configer
        self.n_datasets = self.configer.get('n_datasets')
        self.classRemapper = eval(self.configer.get('class_remaper'))(configer=self.configer)
        self.num_unify_classes = self.configer.get('num_unify_classes')
        self.num_prototype = self.configer.get('contrast', 'num_prototype')
        self.coefficient = self.configer.get('contrast', 'coefficient')
        
        self.ignore_index = -1
        if self.configer.exists('loss', 'ignore_index'):
            self.ignore_index = self.configer.get('loss', 'ignore_index')
            
        self.loss_weight = self.configer.get('contrast', 'loss_weight')    
                          
     
        self.seg_criterion_mul = eval(self.configer.get('loss', 'type'))(configer=self.configer)   
        ## 处理多标签
        # 处理单标签
        # self.seg_criterion_sig = OhemCELoss(0.7, ignore_lb=self.ignore_index)
            
        self.with_aux = self.configer.get('loss', 'with_aux')
        if self.with_aux:
            self.aux_num = self.configer.get('loss', 'aux_num')
            self.aux_weight = self.configer.get('loss', 'aux_weight')
            self.segLoss_aux_Mul = [eval(self.configer.get('loss', 'type'))(configer=self.configer) for _ in range(self.aux_num)]
            
        
        self.use_contrast = self.configer.get('contrast', 'use_contrast')
        if self.use_contrast:
            self.contrast_criterion = PixelContrastLoss(configer=configer)
            self.hard_lb_contrast_loss = PixelContrastLossMulProto(configer=configer)
        
        self.upsample = self.configer.get('contrast', 'upsample')
        self.network_stride = self.configer.get('network', 'stride')
        
        
    def forward(self, preds, target, dataset_ids, is_warmup=False):
        assert "seg" in preds
        
        if self.with_aux: 
            logits, *logits_aux = preds['seg']
        else:
            logits = preds['seg']
        
        if self.use_contrast:
            embedding = preds['embed']
        
        b, c, h, w = logits.shape

        lb = target
 

        if "prototypes" in preds:
            memory_bank, memory_bank_ptr, memory_bank_init, prototypes = preds['prototypes']
        else:
            memory_bank, memory_bank_ptr, memory_bank_init, prototypes = None, None, None, None

        contrast_lb = lb[:, ::self.network_stride, ::self.network_stride]
        new_proto = None
        if self.use_contrast:
            rearr_emb = rearrange(embedding, 'b c h w -> (b h w) c')
            proto_target = self.AdaptiveSingleSegRemapping(contrast_lb, dataset_ids)
            memory_bank_push(self.configer, memory_bank, memory_bank_ptr, rearr_emb.detach(), proto_target, memory_bank_init, random_pick_ratio=0.1)
            if self.IsInitMemoryBank(memory_bank_init):
                # 完成初始化
                self.AdaptiveKMeansProtoLearning(contrast_lb, memory_bank, memory_bank_ptr, memory_bank_init, embedding.detach(), dataset_ids)
                new_proto_mean = torch.mean(memory_bank, dim=1)
                new_proto_mean = F.normalize(new_proto_mean, p=2, dim=-1)
                
                prototypes = F.normalize(new_proto_mean * (1 - self.coefficient) + prototypes * self.coefficient, p=2, dim=-1)
                                                
                
            else:
                return 

        loss_aux = None
        loss_domain = None
        loss_contrast = None
        kl_loss = None

        if is_warmup or not self.use_contrast:
            # pred = F.interpolate(input=logits, size=(h, w), mode='bilinear', align_corners=True)
            seg_label_mul = self.AdaptiveSegRemapping(lb, dataset_ids)
            loss_seg_mul = self.seg_criterion_mul(logits, seg_label_mul)
            
            # loss_seg_mul = self.seg_criterion_mul(logits, seg_label_mul + seg_label_sig)
            loss_seg = loss_seg_mul
            loss = loss_seg
            if self.with_aux:
                # pred_aux = [F.interpolate(input=logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logits_aux]
                pred_aux = [F.interpolate(input=logit, size=(h, w), mode='bilinear', align_corners=True)
                            for logit in logits_aux]
                loss_aux = [aux_criterion_mul(aux, seg_label_mul) for aux, aux_criterion_mul in zip(pred_aux, self.segLoss_aux_Mul)]
                # loss_aux = [aux_criterion_mul(aux, seg_label_mul+ seg_label_sig) for aux, aux_criterion_mul, aux_criterion_sig in zip(pred_aux, self.segLoss_aux_Mul, self.segLoss_aux_Sig)]
                
                loss = loss + self.aux_weight * sum(loss_aux)
                
        else:            
            proto_logits = torch.mm(rearr_emb, prototypes.view(-1, prototypes.shape[-1]).t())
            proto_targetOneHot = self.AdaptiveSingleSegRemappingOneHot(contrast_lb, dataset_ids)
            
            # proto_targetOntHot 单标签， contrast_mask_label 多标签
            contrast_mask_label, seg_mask_mul = self.AdaptiveMultiProtoRemapping(lb, proto_logits.detach(), dataset_ids)

            loss_contrast = self.hard_lb_contrast_loss(proto_logits, contrast_mask_label+proto_targetOneHot)
            
            loss_seg_mul = self.seg_criterion_mul(logits, seg_mask_mul)
            loss_seg = loss_seg_mul 
            loss = loss_seg

            
            if self.with_aux:
                # aux_weight_mask = self.classRemapper.GetEqWeightMask(lb, dataset_id)
                pred_aux = [F.interpolate(input=logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logits_aux]
                # loss_aux = [aux_criterion_sig(aux[0], seg_mask_sig) + aux_criterion_mul(aux[1], seg_mask_mul) for aux, aux_criterion_mul, aux_criterion_sig in zip(pred_aux, self.segLoss_aux_Mul, self.segLoss_aux_Sig)]
                loss_aux = [aux_criterion_mul(aux, seg_mask_mul) for aux, aux_criterion_mul in zip(pred_aux, self.segLoss_aux_Mul)]
                 
                loss = loss + self.aux_weight * sum(loss_aux)
                
            # if self.with_ppd:
            #     loss_ppd = self.ppd_criterion(embedding, contrast_mask_label, segment_queue)
            #     loss_contrast = loss_contrast + self.ppd_loss_weight * loss_ppd
                
            loss = loss + self.loss_weight * loss_contrast

        return loss, loss_seg, loss_aux, loss_contrast, loss_domain, kl_loss, prototypes
            



    def AdaptiveSingleSegRemapping(self, lb, dataset_ids):
        proto_mask = torch.zeros_like(lb)
        
        for i in range(0, self.n_datasets):
            if not (dataset_ids == i).any():
                continue
            
            proto_mask[dataset_ids==i] = self.classRemapper.SingleSegRemapping(lb[dataset_ids==i], i)
        
        return proto_mask.contiguous().view(-1)

    def AdaptiveSingleSegRemappingOneHot(self, lb, dataset_ids):
        b,h,w = lb.shape
        proto_mask = torch.zeros(b,h,w,self.num_unify_classes, dtype=torch.bool)
        if lb.is_cuda:
            proto_mask = proto_mask.cuda()

        for i in range(0, self.n_datasets):
            if not (dataset_ids == i).any():
                continue
            
            proto_mask[dataset_ids==i] = self.classRemapper.SingleSegRemappingOneHot(lb[dataset_ids==i], i)
        
        return proto_mask
    
    def AdaptiveSegRemapping(self, lb, dataset_ids):
        b, h, w = lb.shape
        seg_label_mul = torch.zeros(b,h,w,self.num_unify_classes, dtype=torch.bool)
        if lb.is_cuda:
            seg_label_mul = seg_label_mul.cuda()
            
        for i in range(0, self.n_datasets):
            if not (dataset_ids == i).any():
                continue
            
            seg_label_mul[dataset_ids] = self.classRemapper.SegRemapping(lb[dataset_ids==i], i)
            
        return seg_label_mul

    def AdaptiveMultiProtoRemapping(self, lb, proto_logits, dataset_ids):
        b, h, w = lb.shape
        seg_mask_mul = torch.zeros(b,h,w,self.num_unify_classes, dtype=torch.bool)
        contrast_mask_label = torch.zeros(b, int(h/self.network_stride), int(w/self.network_stride), self.num_unify_classes*self.num_prototype, dtype=torch.bool)
        if lb.is_cuda:
            seg_mask_mul = seg_mask_mul.cuda()
            contrast_mask_label = contrast_mask_label.cuda()
            
        for i in range(0, self.n_datasets):
            if not (dataset_ids == i).any():
                continue
            re_proto_logits = rearrange(proto_logits, '(b h w) n -> b h w n', b=contrast_mask_label.shape[0], h=contrast_mask_label.shape[1], w=contrast_mask_label.shape[2])
            this_proto_logits = rearrange(re_proto_logits[dataset_ids==i], 'b h w n -> (b h w) n')
            out_contrast_mask, out_seg_mask = self.classRemapper.MultiProtoRemapping(lb[dataset_ids==i], this_proto_logits, i)
            contrast_mask_label[dataset_ids==i] = out_contrast_mask
            seg_mask_mul[dataset_ids==i] = out_seg_mask
            
        return contrast_mask_label, seg_mask_mul
        
    def AdaptiveKMeansRemapping(self, lb, dataset_ids):
        cluster_mask = torch.zeros_like(lb).bool()
        constraint_mask = torch.zeros((*(lb.shape), self.num_unify_classes), dtype=torch.bool)
        if lb.is_cuda:
            constraint_mask = constraint_mask.cuda()

        for i in range(0, self.n_datasets):
            if not (dataset_ids == i).any():
                continue
            
            cluster_mask[dataset_ids==i], constraint_mask[dataset_ids==i] = self.classRemapper.KMeansRemapping(lb[dataset_ids==i], i)
        
        out_constrain_mask =  constraint_mask[cluster_mask].logical_not()
        cluster_mask = cluster_mask.contiguous().view(-1)
        
        return cluster_mask, out_constrain_mask
    
    def AdaptiveUpsampleProtoTarget(self, lb, proto_target, dataset_ids):
        # b, h, w = lb.shape
        seg_mask_mul = torch.zeros(*(lb.shape), self.num_unify_classes, dtype=torch.bool)
        
        if lb.is_cuda:
            seg_mask_mul = seg_mask_mul.cuda()
            
        for i in range(0, self.n_datasets):
            if not (dataset_ids == i).any():
                continue
            re_proto_target = rearrange(proto_target, '(b h w) -> b h w', b=lb.shape[0], h=int(lb.shape[1]/self.network_stride), w=int(lb.shape[2]/self.network_stride))
            # this_proto_target = rearrange(re_proto_target[dataset_ids==i], 'b h w n -> (b h w) n')
            out_seg_mask = self.classRemapper.UpsampleProtoTarget(lb[dataset_ids==i], re_proto_target[dataset_ids==i], i)
            seg_mask_mul[dataset_ids==i] = out_seg_mask
            
        return 

    def IsInitMemoryBank(self, init_datas):
        for lb_id, i in enumerate(init_datas):
            if self.classRemapper.IsSingleRemaplb(lb_id) and (i is False):
                return False
        return True

    def AdaptiveKMeansProtoLearning(self, lb, memory_bank, memory_bank_ptr, memory_bank_init, emb, dataset_ids):
        proj_dim = self.configer.get('contrast', 'proj_dim')
        memory_size = self.configer.get('contrast', 'memory_bank_size')
        b, h, w = lb.shape
        # out_kmeans_lb = torch.zeros(b,h,w,self.num_unify_classes, dtype=torch.bool)
        # if lb.iscuda:
        #     out_kmeans_lb = out_kmeans_lb.cuda()
        re_emb = rearrange(emb, 'b c h w -> b h w c')
             
        for ds_id in range(0, self.n_datasets):
            if not (dataset_ids == ds_id).any():
                continue
                
            this_lb = lb[dataset_ids==ds_id]
            this_emb = re_emb[dataset_ids==ds_id]
            
            # this_out_kmeans_lb = out_kmeans_lb[dataset_ids == ds_id]
            len_dataset = self.configer.get('dataset'+str(ds_id+1), 'n_cats')
            for lb_id in range(0, len_dataset):
                if not (this_lb == lb_id).any():
                    continue
                remap_lbs = self.classRemapper.getAnyClassRemap(lb_id, ds_id)
                remap_lbs = torch.tensor(remap_lbs)
                if lb.is_cuda:
                    remap_lbs = remap_lbs.cuda()
                    
                if len(remap_lbs) == 1:
                    continue

                # 找到无单标签映射类别，生成对应的带memory的聚类特征集
                for i in remap_lbs:
                    if self.classRemapper.IsSingleRemaplb(i):
                        continue
                    
                    in_x = this_emb[this_lb==lb_id]
                    len_in_x = len(in_x)
                    inited_lb = remap_lbs[memory_bank_init[remap_lbs]]
                    if inited_lb.any():
                        this_memory = memory_bank[inited_lb]
                        this_memory = rearrange(this_memory, 'b n d -> (b n) d')
                        in_x = torch.cat((in_x, this_memory), dim=0)
                    cluster_centers = torch.zeros((len(remap_lbs), proj_dim), dtype=torch.float32)
                    if lb.is_cuda:
                        cluster_centers = cluster_centers.cuda()
                    
                    index = 0
                    for k in range(0, len(remap_lbs)):
                        if memory_bank_init[remap_lbs[k]]:
                            cluster_centers[k] = torch.mean(memory_bank[remap_lbs[k]], dim=0)
                        else:
                            cluster_centers[k] = in_x[index]
                            index += 1 
                    # 生成约束矩阵
                    constraint_matrix = torch.zeros(in_x.shape[0], len(remap_lbs), dtype=torch.bool)
                    index = 0
                    for k in range(0, len(remap_lbs)):
                        if inited_lb.any() and inited_lb[index] == remap_lbs[k]:
                            constraint_vector = torch.zeros(len(remap_lbs), dtype=torch.bool)
                            constraint_vector[k] = True
                            constraint_matrix[len_in_x+index*memory_size:len_in_x+(index+1)*memory_size] = constraint_vector
                            index += 1
                            if index >= len(inited_lb):
                                break
                    
                    target_device = 'cpu'
                    out_cluster = torch.zeros(len_in_x, self.num_unify_classes, dtype=torch.bool)
                    if lb.is_cuda:
                        constraint_matrix = constraint_matrix.cuda()
                        target_device = 'cuda' 
                        out_cluster = out_cluster.cuda()
                            
                    choice_cluster, initial_state = kmeans(in_x, len(remap_lbs), cluster_centers=cluster_centers, distance='cosine', device=target_device, constraint_matrix=constraint_matrix)
                    memory_bank_push(self.configer, memory_bank, memory_bank_ptr, in_x[:len_in_x], remap_lbs[choice_cluster[:len_in_x]], memory_bank_init, 0.1)
                    # out_cluster[remap_lbs[choice_cluster]] = True
                    # this_out_kmeans_lb[this_lb==i] = out_cluster
                    break
            
            # out_kmeans_lb[dataset_ids==ds_id] = this_out_kmeans_lb
        
        # # 标签中需要kmeans的像素被分到了对应的标签，其他元素为0
        # return out_kmeans_lb             
        

class CrossDatasetsCELoss_CLIP(nn.Module):
    def __init__(self, configer=None):
        super(CrossDatasetsCELoss_CLIP, self).__init__()
        self.configer = configer
        self.n_datasets = self.configer.get('n_datasets')
        self.num_unify_classes = self.configer.get('num_unify_classes')
        self.num_prototype = self.configer.get('contrast', 'num_prototype')
        self.temperature = self.configer.get('contrast', 'temperature')
        self.with_mulbn = self.configer.get('contrast', 'with_mulbn')
        self.reweight = self.configer.get('loss', 'reweight')
        self.ignore_index = self.configer.get('loss', 'ignore_index')
        self.with_unify_label = self.configer.get('loss', 'with_unify_label')
        if self.with_unify_label:
            self.classRemapper = eval(self.configer.get('class_remaper'))(configer=self.configer)

        self.n_cats = []
        for i in range(1, self.n_datasets+1):
            self.n_cats.append(self.configer.get('dataset'+str(i), 'n_cats'))

        self.CELoss = torch.nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, preds, target, dataset_ids, is_warmup=False):
        logits = preds['seg']
        text_feature_vecs = preds['prototypes']
        
        loss = None
        if self.with_unify_label:
            logits = remap_logits = torch.einsum('bchw, nc -> bnhw', logits, text_feature_vecs[self.n_datasets])
        
        for i in range(0, self.n_datasets):
            if not (dataset_ids == i).any():
                continue
            
            RemapMatrix = self.classRemapper.getRemapMatrix(i)
            if logits.is_cuda:
                RemapMatrix = RemapMatrix.cuda()
            
            if self.with_unify_label:
                remap_logits = torch.einsum('bchw, nc -> bnhw', logits[dataset_ids==i], RemapMatrix)
                remap_logits = F.interpolate(remap_logits, size=(target.size(1), target.size(2)), mode="bilinear", align_corners=True)
            else:
                remap_logits = torch.einsum('bchw, nc -> bnhw', logits[dataset_ids==i], text_feature_vecs[i])
                remap_logits = F.interpolate(remap_logits, size=(target.size(1), target.size(2)), mode="bilinear", align_corners=True)
            
            if loss is None:
                loss = self.CELoss(remap_logits, target[dataset_ids==i])
            else:
                loss = loss + self.CELoss(remap_logits, target[dataset_ids==i])
                
        return loss
    
class CrossDatasetsCELoss_GNN(nn.Module):
    def __init__(self, configer=None):
        super(CrossDatasetsCELoss_GNN, self).__init__()
        self.configer = configer
        self.n_datasets = self.configer.get('n_datasets')
        self.num_prototype = self.configer.get('contrast', 'num_prototype')
        self.temperature = self.configer.get('contrast', 'temperature')
        self.with_mulbn = self.configer.get('contrast', 'with_mulbn')
        self.reweight = self.configer.get('loss', 'reweight')
        self.ignore_index = self.configer.get('loss', 'ignore_index')
        self.with_unify_label = self.configer.get('loss', 'with_unify_label')
        self.with_spa = self.configer.get('loss', 'with_spa')
        self.spa_loss_weight = self.configer.get('loss', 'spa_loss_weight')
        self.with_max_enc = self.configer.get('loss', 'with_max_enc')
        self.max_enc_weight = self.configer.get('loss', 'max_enc_weight')

        self.n_cats = []
        for i in range(1, self.n_datasets+1):
            self.n_cats.append(self.configer.get('dataset'+str(i), 'n_cats'))

        self.CELoss = torch.nn.CrossEntropyLoss(ignore_index=255)
        
        if self.with_max_enc:
            self.MSE_loss = torch.nn.MSELoss()

    def forward(self, preds, target, dataset_ids, is_warmup=False):
        logits = preds['seg']
        unify_prototype = preds['unify_prototype']
        bi_graphs = preds['bi_graphs']
        
        loss = None
        logits = torch.einsum('bchw, nc -> bnhw', logits, unify_prototype)
        # print("logits_max : {}, logits_min : {}".format(torch.max(logits), torch.min(logits)))
        
        for i in range(0, self.n_datasets):
            if not (dataset_ids == i).any():
                continue
            
            
            remap_logits = torch.einsum('bchw, nc -> bnhw', logits[dataset_ids==i], bi_graphs[i])
            remap_logits = F.interpolate(remap_logits, size=(target.size(1), target.size(2)), mode="bilinear", align_corners=True)
            # print("remap_logits_max : {}, remap_logits_min : {}".format(torch.max(remap_logits), torch.min(remap_logits)))
            
            # a = target[dataset_ids==i].clone()
            # a[a == 255] = 0
            # print("i : {}, a_max : {}, a_min : {}".format(i, torch.max(a), torch.min(a)))
            
            # print(torch.sum(bi_graphs[i]))
            # print("remap_logits: ", remap_logits.shape)
            if loss is None:
                loss = self.CELoss(remap_logits, target[dataset_ids==i])
            else:
                loss = loss + self.CELoss(remap_logits, target[dataset_ids==i])

            if self.with_spa:
                spa_loss = self.spa_loss_weight * torch.pow(torch.norm(bi_graphs[i], p='fro'), 2)
                loss = loss + spa_loss
            
            if self.with_max_enc:
                max_enc_loss = self.max_enc_weight * self.MSE_loss(torch.max(bi_graphs[i], dim=0)[0], torch.ones(bi_graphs[i].size(1)).cuda())
                loss = loss + max_enc_loss
                   
        return loss
    
class CrossDatasetsCELoss_AdvGNN(nn.Module):
    def __init__(self, configer=None):
        super(CrossDatasetsCELoss_AdvGNN, self).__init__()
        self.configer = configer
        self.n_datasets = self.configer.get('n_datasets')
        self.num_prototype = self.configer.get('contrast', 'num_prototype')
        self.temperature = self.configer.get('contrast', 'temperature')
        self.with_mulbn = self.configer.get('contrast', 'with_mulbn')
        self.reweight = self.configer.get('loss', 'reweight')
        self.ignore_index = self.configer.get('loss', 'ignore_index')
        self.with_unify_label = self.configer.get('loss', 'with_unify_label')
        self.with_spa = self.configer.get('loss', 'with_spa')
        self.spa_loss_weight = self.configer.get('loss', 'spa_loss_weight')
        self.with_max_enc = self.configer.get('loss', 'with_max_enc')
        self.max_enc_weight = self.configer.get('loss', 'max_enc_weight')
        
        self.n_cats = []
        for i in range(1, self.n_datasets+1):
            self.n_cats.append(self.configer.get('dataset'+str(i), 'n_cats'))

        self.CELoss = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.advloss = nn.BCELoss()
        self.adv_loss_weight = self.configer.get('loss', 'adv_loss_weight')
        
        if self.with_max_enc:
            self.MSE_loss = torch.nn.MSELoss()

    def forward(self, preds, target, dataset_ids, is_adv=True):
        logits = preds['seg']
        unify_prototype = preds['unify_prototype']
        bi_graphs = preds['bi_graphs']
        adv_out = preds['adv_out']
        
        label_real = torch.zeros(adv_out['ADV1'][0].shape[0], 1)
        label_fake = torch.ones(adv_out['ADV1'][0].shape[0], 1)
        
        if adv_out['ADV1'][0].is_cuda:
            label_real = label_real.cuda()
            label_fake = label_fake.cuda()
        
        loss = None
        adv_loss = None
        logits = torch.einsum('bchw, nc -> bnhw', logits, unify_prototype)
        # print("logits_max : {}, logits_min : {}".format(torch.max(logits), torch.min(logits)))
        
        for i in range(0, self.n_datasets):
            if not (dataset_ids == i).any():
                continue
            
            remap_logits = torch.einsum('bchw, nc -> bnhw', logits[dataset_ids==i], bi_graphs[i])
            remap_logits = F.interpolate(remap_logits, size=(target.size(1), target.size(2)), mode="bilinear", align_corners=True)
            # print("remap_logits_max : {}, remap_logits_min : {}".format(torch.max(remap_logits), torch.min(remap_logits)))
            # a = target[dataset_ids==i].clone()
            # a[a == 255] = 0
            # print("i : {}, a_max : {}, a_min : {}".format(i, torch.max(a), torch.min(a)))
            
            # print(torch.sum(bi_graphs[i]))
            if loss is None:
                loss = self.CELoss(remap_logits, target[dataset_ids==i])
            else:
                loss = loss + self.CELoss(remap_logits, target[dataset_ids==i])

            if self.with_spa:
                spa_loss = self.spa_loss_weight * torch.pow(torch.norm(bi_graphs[i], p='fro'), 2)
                loss = loss + spa_loss
            
            if self.with_max_enc:
                max_enc_loss = self.max_enc_weight * self.MSE_loss(torch.max(bi_graphs[i], dim=0)[0], torch.ones(bi_graphs[i].size(1)).cuda())
                loss = loss + max_enc_loss
              
        if is_adv:  
            real_out = self.advloss(adv_out['ADV1'][0], label_real) + self.advloss(adv_out['ADV2'][0], label_real)
            fake_out = self.advloss(adv_out['ADV1'][1], label_fake) + self.advloss(adv_out['ADV2'][1], label_fake)
            
            adv_loss = real_out + fake_out
            G_fake_out = self.advloss(adv_out['ADV1'][2], label_real) + self.advloss(adv_out['ADV2'][2], label_real)
            loss = loss + self.adv_loss_weight * G_fake_out
                   
        return loss, adv_loss
    
if __name__ == "__main__":
    test_CrossDatasetsCELoss()
    # test_LabelToOneHot()
    # loss_fuc = PixelPrototypeDistanceLoss()
    # a = torch.randn(2,4,3,2)
    # print(a)
    # lb = torch.tensor([[[0,1],[2,0],[255,0]],[[2,1],[1,255],[255,255]]])
    # seq = torch.randn(3,4)
    # print(seq)
    # print(loss_fuc(a,lb,seq))
       