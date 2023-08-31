import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.HRNet_backbone import HRNetBackbone
from lib.models.hrnet_backbone_ori import HRNetBackbone_ori
from lib.module.module_helper import ConvBNReLU, ModuleHelper
from lib.module.projection import ProjectionHead, ProjectionHeadOri
from lib.module.domain_classifier_head import DomainClassifierHead
from timm.models.layers import trunc_normal_
from lib.class_remap import ClassRemap
import clip

backbone_url = './res/hrnetv2_w48_imagenet_pretrained.pth'

class SegmentHead(nn.Module):

    def __init__(self, in_chan, n_classes, up_factor=4, n_bn=1):
        super(SegmentHead, self).__init__()
         
        self.up_factor = int(up_factor)
        self.n_bn = n_bn

        self.conv1 = ConvBNReLU(in_chan, in_chan, ks=3, stride=1, padding=1, n_bn=self.n_bn)
        self.dropout = nn.Dropout2d(0.10)
        self.conv2 = nn.Conv2d(in_chan, n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        
        
    def forward(self, dataset, x, *other_x):
        _, _, h, w = x.shape
        
        # 采用多分割头，所以不应该返回list
        feats = self.conv1(dataset, x, *other_x)
        feats = [self.dropout(feat) for feat in feats]
        feats = [self.conv2(feat) for feat in feats]
            
        if self.up_factor > 1:
            feats = [F.interpolate(input=feat, size=(h*self.up_factor, w*self.up_factor), mode='bilinear', align_corners=True) for feat in feats]

        return feats



class HRNet_W48_CONTRAST(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, configer):
        super(HRNet_W48_CONTRAST, self).__init__()
        self.configer = configer
        self.aux_mode = self.configer.get('aux_mode')
        self.n_bn = self.configer.get('n_bn')
        self.num_unify_classes = self.configer.get('num_unify_classes')
        self.n_datasets = self.configer.get('n_datasets')
        self.backbone = HRNetBackbone(configer)
        self.proj_dim = self.configer.get('contrast', 'proj_dim')
        self.full_res_stem = self.configer.get('hrnet', 'full_res_stem')
        self.num_prototype = self.configer.get('contrast', 'num_prototype')
        
        if self.full_res_stem:
            up_fac = 1
        else:
            up_fac = 4

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384
        self.cls_head = SegmentHead(in_channels, self.num_unify_classes, up_factor=up_fac, n_bn=self.n_bn)

        self.use_contrast = self.configer.get('contrast', 'use_contrast')
        if self.use_contrast:
            self.proj_head = ProjectionHead(dim_in=in_channels, proj_dim=self.proj_dim)
            
        self.prototypes = nn.Parameter(torch.zeros(self.num_unify_classes, self.num_prototype, self.proj_dim),
                                       requires_grad=False)

        trunc_normal_(self.prototypes, std=0.02)
        self.init_weights()    
       
        self.with_memory_bank = self.configer.get('contrast', 'memory_bank')
        if self.with_memory_bank:
            self.memory_bank_size = self.configer.get('contrast', 'memory_bank_size')
            self.register_buffer("memory_bank", torch.randn(self.num_unify_classes, self.memory_bank_size, self.proj_dim))
            self.memory_bank = nn.functional.normalize(self.memory_bank, p=2, dim=2)
            self.register_buffer("memory_bank_ptr", torch.zeros(self.num_unify_classes, dtype=torch.long))
            self.register_buffer("memory_bank_init", torch.zeros(self.num_unify_classes, dtype=torch.bool))
            
        # self.with_domain_adversarial = self.configer.get('network', 'with_domain_adversarial')
        # if self.with_domain_adversarial:
        #     self.DomainCls_head = DomainClassifierHead(in_channels, n_domain=self.n_datasets, )
        

    def forward(self, x_, *other_x, dataset=0):
        x = self.backbone(x_, *other_x, dataset=0)
        _, _, h, w = x[0][0].size()

        feat1 = x[0]
        feat2 = [F.interpolate(x_data, size=(h, w), mode="bilinear", align_corners=True) for x_data in x[1]]
        feat3 = [F.interpolate(x_data, size=(h, w), mode="bilinear", align_corners=True) for x_data in x[2]]
        feat4 = [F.interpolate(x_data, size=(h, w), mode="bilinear", align_corners=True) for x_data in x[3]]

        feats = [torch.cat([feat1_data, feat2_data, feat3_data, feat4_data], 1) 
                for feat1_data, feat2_data, feat3_data, feat4_data in zip(feat1, feat2, feat3, feat4)]
        
        out = self.cls_head(dataset, *feats)

        if self.aux_mode == 'train':
            emb = None
            if self.use_contrast:
                emb = self.proj_head(dataset, *feats)
            return {'seg': out, 'embed': emb}
        elif self.aux_mode == 'eval':
            return out[0]
        elif self.aux_mode == 'pred':
            pred = out[0].argmax(dim=1)
            return pred
        else:
            raise NotImplementedError
        
    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.requires_grad == False:
                    continue
                
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4:
                    wd_params.append(param)
                else:
                    nowd_params.append(param)
                    # print(param.dim())
                    # print(param)
                    print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
    
    def set_train_dataset_aux(self, new_val=False):
        self.train_dataset_aux = new_val

    def switch_require_grad_state(self, require_grad_state=True):
        for p in self.detail.parameters():
            p.requires_grad = require_grad_state
            
        for p in self.segment.parameters():
            p.requires_grad = require_grad_state
            
        for p in self.bga.parameters():
            p.requires_grad = require_grad_state
        
    def PrototypesUpdate(self, new_proto):
        self.prototypes = nn.Parameter(F.normalize(new_proto, p=2, dim=-1),
                                        requires_grad=False)
        
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
                
        self.load_pretrain()
                
    def load_pretrain(self):
        state = torch.load(backbone_url)
        newstate = {}
        def loadConvBN(srcDict, src_name, targetdict, target_name, ID):
            targetdict[target_name+'conv.weight'] = srcDict[src_name+'conv' + str(ID)+ '.weight']
            targetdict[target_name+'affine_weight'] = srcDict[src_name+'bn' + str(ID)+ '.weight']
            targetdict[target_name+'affine_bias'] = srcDict[src_name+'bn' + str(ID)+ '.bias']
        
        def loadLayer(srcDict, src_name, targetdict, num):
            for i in range(1, num+1):
                loadConvBN(state, src_name, targetdict, src_name+'conv'+str(i)+'.', i)
        
        def loadConvBN_NoName(srcDict, src_name, targetdict):
            targetdict[src_name+'conv.weight'] = srcDict[src_name+'0.weight']
            targetdict[src_name+'affine_weight'] = srcDict[src_name+'1.weight']
            targetdict[src_name+'affine_bias'] = srcDict[src_name+'1.bias']
            
        def loadBranch(srcDict, src_name, targetdict):
            loadLayer(state, src_name + '0.', newstate, 2)
            loadLayer(state, src_name + '1.', newstate, 2)
            loadLayer(state, src_name + '2.', newstate, 2)
            loadLayer(state, src_name + '3.', newstate, 2)
            
        def loadStage3fuse(srcDict, src_name, targetdict):
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.0.1.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.0.2.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.1.0.0.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.1.2.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.2.0.0.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.2.0.1.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.2.1.0.', targetdict)
        
        def loadStage(srcDict, src_name, targetdict, num):
            for i in range(0, num):
                loadBranch(srcDict, src_name+'branches.'+str(i)+'.', targetdict)
                
        def loadStage4fuse(srcDict, src_name, targetdict):
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.0.1.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.0.2.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.0.3.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.1.0.0.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.1.2.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.1.3.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.2.0.0.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.2.0.1.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.2.1.0.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.2.3.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.3.0.0.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.3.0.1.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.3.0.2.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.3.1.0.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.3.1.1.', targetdict)
            loadConvBN_NoName(srcDict, src_name+'fuse_layers.3.2.0.', targetdict)
        
        loadConvBN(state, '', newstate, 'conv1.', 1)
        loadConvBN(state, '', newstate, 'conv2.', 2)
        loadLayer(state, 'layer1.0.', newstate, 3)
        loadConvBN_NoName(state, 'layer1.0.downsample.', newstate)
        loadLayer(state, 'layer1.1.', newstate, 3)
        loadLayer(state, 'layer1.2.', newstate, 3)
        loadLayer(state, 'layer1.3.', newstate, 3)
        loadConvBN_NoName(state, 'transition1.0.', newstate)
        loadConvBN_NoName(state, 'transition1.1.0.', newstate)
        loadStage(state, 'stage2.0.', newstate, 2)
        loadConvBN_NoName(state, 'stage2.0.fuse_layers.0.1.', newstate)
        loadConvBN_NoName(state, 'stage2.0.fuse_layers.1.0.0.', newstate)
        loadConvBN_NoName(state, 'transition2.2.0.', newstate)
        loadStage(state, 'stage3.0.', newstate, 3)
        loadStage3fuse(state, 'stage3.0.', newstate)
        loadStage(state, 'stage3.1.', newstate, 3)
        loadStage3fuse(state, 'stage3.1.', newstate)
        loadStage(state, 'stage3.2.', newstate, 3)
        loadStage3fuse(state, 'stage3.2.', newstate)
        loadStage(state, 'stage3.3.', newstate, 3)
        loadStage3fuse(state, 'stage3.3.', newstate)
        loadConvBN_NoName(state, 'transition3.3.0.', newstate)
        loadStage(state, 'stage4.0.', newstate, 4)
        loadStage4fuse(state, 'stage4.0.', newstate)
        loadStage(state, 'stage4.1.', newstate, 4)
        loadStage4fuse(state, 'stage4.1.', newstate)
        loadStage(state, 'stage4.2.', newstate, 4)
        loadStage4fuse(state, 'stage4.2.', newstate)
        
        self.backbone.load_state_dict(newstate, strict=False)
        
    def PrototypesUpdate(self, new_proto):
        self.prototypes = nn.Parameter(F.normalize(new_proto, p=2, dim=-1),
                                        requires_grad=False)

    # def GenMemoryBank(self):
    #     classRemapper = ClassRemap(self.configer)
    #     self.memory_bank_size = self.configer.get('contrast', 'memory_bank_size')
    #     self.memory_bank_map = []
    #     for dataset_id in range(0, self.n_datasets):
    #         memory_bank_map_this_dataset = []
    #         len_dataset = self.configer.get('dataset'+str(dataset_id+1), 'n_cats')
    #         for lb_id in range(0, len_dataset):
    #             remap_lbs = lassRemapper.getAnyClassRemap(lb_id, dataset_id)
    #             if len(remap_lbs) == 1:
    #                 continue
    #             else:
    #                 for i in remap_lbs:
    #                     if classRemapper.IsSingleRemaplb(i):
                            
                            
        
    #     self.register_buffer("memory_bank", torch.randn(self.num_unify_classes, self.memory_bank_size, self.proj_dim))
    #     self.memory_bank = nn.functional.normalize(self.memory_bank, p=2, dim=2)
    #     self.register_buffer("memory_bank_ptr", torch.zeros(self.num_unify_classes, dtype=torch.long))
    #     self.register_buffer("memory_bank_init", torch.zeros(self.num_unify_classes, dtype=torch.bool))
        

    # newstate = loadpretrain(state)
    # net.backbone.load_state_dict(newstate, strict=False)
       
class HRNet_W48(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, configer):
        super(HRNet_W48, self).__init__()
        self.configer = configer
        self.aux_mode = self.configer.get('aux_mode')
        self.n_bn = self.configer.get('n_bn')
        self.num_unify_classes = self.configer.get('num_unify_classes')
        self.n_datasets = self.configer.get('n_datasets')
        self.backbone = HRNetBackbone_ori(configer)
        self.proj_dim = self.configer.get('contrast', 'proj_dim')
        self.full_res_stem = self.configer.get('hrnet', 'full_res_stem')
        self.num_prototype = self.configer.get('contrast', 'num_prototype')
        
        
        if self.full_res_stem:
            up_fac = 1
        else:
            up_fac = 4

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10),
            nn.Conv2d(in_channels, self.num_unify_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.use_contrast = self.configer.get('contrast', 'use_contrast')
        if self.use_contrast:
            self.proj_head = ProjectionHeadOri(dim_in=in_channels, proj_dim=self.proj_dim, bn_type=self.configer.get('network', 'bn_type'))
            
        self.prototypes = nn.Parameter(torch.zeros(self.num_unify_classes, self.proj_dim),
                                       requires_grad=False)

        trunc_normal_(self.prototypes, std=0.02)
        # self.init_weights()    
       
        self.with_memory_bank = self.configer.get('contrast', 'memory_bank')
        if self.with_memory_bank:
            self.memory_bank_size = self.configer.get('contrast', 'memory_bank_size')
            self.register_buffer("memory_bank", torch.randn(self.num_unify_classes, self.memory_bank_size, self.proj_dim))
            self.memory_bank = nn.functional.normalize(self.memory_bank, p=2, dim=2)
            self.register_buffer("memory_bank_ptr", torch.zeros(self.num_unify_classes, dtype=torch.long))
            self.register_buffer("memory_bank_init", torch.zeros(self.num_unify_classes, dtype=torch.bool))

    def forward(self, x_, dataset=0):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out = self.cls_head(feats)
        out = F.interpolate(out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        if self.aux_mode == 'train':
            emb = None
            if self.use_contrast:
                emb = self.proj_head(feats)
            return {"seg": out, 'embed': emb}
        elif self.aux_mode == 'eval':
            return out
        elif self.aux_mode == 'pred':
            pred = out.argmax(dim=1)
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
                
        self.load_pretrain()

        
    def load_pretrain(self):
        state = torch.load(backbone_url)
        self.backbone.load_state_dict(state, strict=False)

    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.requires_grad == False:
                    continue
                
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4:
                    wd_params.append(param)
                else:
                    nowd_params.append(param)
                    # print(param.dim())
                    # print(param)
                    print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

    def PrototypesUpdate(self, new_proto):
        self.prototypes = nn.Parameter(F.normalize(new_proto, p=2, dim=-1),
                                        requires_grad=False)


        


class HRNet_W48_CLIP(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, configer):
        super(HRNet_W48_CLIP, self).__init__()
        self.configer = configer
        self.aux_mode = self.configer.get('aux_mode')
        self.n_bn = self.configer.get('n_bn')
        # self.num_unify_classes = self.configer.get('num_unify_classes')
        self.n_datasets = self.configer.get('n_datasets')
        self.backbone = HRNetBackbone_ori(configer)
        self.proj_dim = self.configer.get('contrast', 'proj_dim')
        self.full_res_stem = self.configer.get('hrnet', 'full_res_stem')
        self.num_prototype = self.configer.get('contrast', 'num_prototype')
        self.with_unify_lb = self.configer.get('loss', 'with_unify_label')
        
        if self.full_res_stem:
            up_fac = 1
        else:
            up_fac = 4

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384

        self.proj_head = ProjectionHeadOri(dim_in=in_channels, proj_dim=512, bn_type=self.configer.get('network', 'bn_type'))
            
        self.init_weights()    
       

    def forward(self, x_, dataset=0):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        emb = self.proj_head(feats)
        if self.aux_mode == 'train':
            return {'seg':emb}
        elif self.aux_mode == 'eval':
            logits = torch.einsum('bchw, nc -> bnhw', emb, self.text_feature_vecs[dataset])
            # logits = torch.einsum('bchw, nc -> bnhw', emb, self.text_feature_vecs[self.n_datasets])
            
            return logits
        elif self.aux_mode == 'pred':
            # logits = torch.einsum('bchw, nc -> bnhw', emb, self.text_feature_vecs[self.n_datasets])
            logits = torch.einsum('bchw, nc -> bnhw', emb, self.text_feature_vecs[dataset])
            logits = F.interpolate(logits, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
            pred = logits.argmax(dim=1)
            return pred
        elif self.aux_mode == 'test':
            logits = [torch.einsum('bchw, nc -> bnhw', emb, self.text_feature_vecs[i]) for i in range(0, self.n_datasets)] 
            
            return logits
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
                        
        self.get_encode_lb_vec()
        self.load_pretrain()

    def get_encode_lb_vec(self):
        self.text_feature_vecs = []
        with torch.no_grad():
            clip_model, _ = clip.load("ViT-B/32", device="cuda")
            for i in range(0, self.n_datasets):
                lb_name = self.configer.get("dataset"+str(i+1), "label_names")
                lb_name = ["a photo of " + name + "." for name in lb_name]
                text = clip.tokenize(lb_name).cuda()
                text_features = clip_model.encode_text(text).type(torch.float32)
                self.text_feature_vecs.append(text_features)
                
                
            if self.with_unify_lb:
                lb_name = self.configer.get("unify_classes_name")
                lb_name = ["a photo of " + name + "." for name in lb_name]
                text = clip.tokenize(lb_name).cuda()
                text_features = clip_model.encode_text(text).type(torch.float32)
                self.text_feature_vecs.append(text_features)
                
            
        
    def load_pretrain(self):
        state = torch.load(backbone_url)
        self.backbone.load_state_dict(state, strict=False)

    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.requires_grad == False:
                    continue
                
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4:
                    wd_params.append(param)
                else:
                    nowd_params.append(param)
                    # print(param.dim())
                    # print(param)
                    print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

class HRNet_W48_GNN(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, configer):
        super(HRNet_W48_GNN, self).__init__()
        self.configer = configer
        self.aux_mode = self.configer.get('aux_mode')
        self.n_bn = self.configer.get('n_bn')
        # self.num_unify_classes = self.configer.get('num_unify_classes')
        self.n_datasets = self.configer.get('n_datasets')
        self.backbone = HRNetBackbone_ori(configer)
        self.proj_dim = self.configer.get('contrast', 'proj_dim')
        self.full_res_stem = self.configer.get('hrnet', 'full_res_stem')
        self.num_prototype = self.configer.get('contrast', 'num_prototype')
        # self.output_feat_dim = self.configer.get('GNN', 'output_feat_dim')
        self.output_feat_dim = self.configer.get('GNN', 'output_feat_dim')
        
        if self.full_res_stem:
            up_fac = 1
        else:
            up_fac = 4

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384

        self.proj_head = ProjectionHeadOri(dim_in=in_channels, proj_dim=self.output_feat_dim, bn_type=self.configer.get('network', 'bn_type'))

        self.total_cats = 0
        
        for i in range(0, self.n_datasets):
            self.total_cats += self.configer.get('dataset'+str(i+1), 'n_cats')
        print("self.total_cats:", self.total_cats)
        
        self.max_num_unify_class = int(self.configer.get('GNN', 'unify_ratio') * self.total_cats)
        self.bipartite_graphs = nn.ParameterList([])
        for i in range(0, self.n_datasets):
            self.bipartite_graphs.append(nn.Parameter(
                torch.zeros(self.configer.get('dataset'+str(i+1), 'n_cats'), self.max_num_unify_class), requires_grad=False
                ))
            

        self.unify_prototype = nn.Parameter(torch.zeros(self.max_num_unify_class, self.output_feat_dim),
                                requires_grad=True)
        trunc_normal_(self.unify_prototype, std=0.02)
            
        self.init_weights()    
       

    def forward(self, x_, dataset=0):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        emb = self.proj_head(feats)
        if self.aux_mode == 'train':
            if self.training:
                logits = torch.einsum('bchw, nc -> bnhw', emb, self.unify_prototype)
                return {'seg':logits}
            else:
                return {'seg':emb}
        elif self.aux_mode == 'eval':
            logits = torch.einsum('bchw, nc -> bnhw', emb, self.unify_prototype)
            remap_logits = torch.einsum('bchw, nc -> bnhw', logits, self.bipartite_graphs[dataset])
            # remap_logits = F.interpolate(remap_logits, size=(target.size(1), target.size(2)), mode="bilinear", align_corners=True)
            return remap_logits
        elif self.aux_mode == 'pred':
            logits = torch.einsum('bchw, nc -> bnhw', emb, self.unify_prototype)
            logits = torch.einsum('bchw, nc -> bnhw', logits, self.bipartite_graphs[dataset])
            logits = F.interpolate(logits, size=(logits.size(2)*4, logits.size(3)*4), mode="bilinear", align_corners=True)
            
            pred = logits.argmax(dim=1)
            
            return pred
        else:
            logits = torch.einsum('bchw, nc -> bnhw', emb, self.unify_prototype)
            # logits = torch.einsum('bchw, nc -> bnhw', logits, self.bipartite_graphs[dataset])
            logits = F.interpolate(logits, size=(logits.size(2)*4, logits.size(3)*4), mode="bilinear", align_corners=True)
            
            pred = logits.argmax(dim=1)
            
            return pred

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
        state = torch.load(backbone_url)
        self.backbone.load_state_dict(state, strict=False)

    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.requires_grad == False:
                    continue
                
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4 or param.dim() == 2:
                    wd_params.append(param)
                else:
                    nowd_params.append(param)
                    print(param.dim())
                    # print(param)
                    print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
    
    def set_bipartite_graphs(self, bi_graphs):
        for i in range(0, self.n_datasets):
            self.bipartite_graphs[i] = nn.Parameter(
                bi_graphs[i], requires_grad=False
                )
        
    def set_unify_prototype(self, unify_prototype, grad=False):
        self.unify_prototype = nn.Parameter(unify_prototype,
                                requires_grad=grad)
        


class HRNet_W48_GNN_MulHead(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, configer):
        super(HRNet_W48_GNN_MulHead, self).__init__()
        self.configer = configer
        self.aux_mode = self.configer.get('aux_mode')
        self.n_bn = self.configer.get('n_bn')
        # self.num_unify_classes = self.configer.get('num_unify_classes')
        self.n_datasets = self.configer.get('n_datasets')
        self.backbone = HRNetBackbone_ori(configer)
        self.proj_dim = self.configer.get('contrast', 'proj_dim')
        self.full_res_stem = self.configer.get('hrnet', 'full_res_stem')
        self.num_prototype = self.configer.get('contrast', 'num_prototype')
        # self.output_feat_dim = self.configer.get('GNN', 'output_feat_dim')
        self.output_feat_dim = self.configer.get('GNN', 'output_feat_dim')
        
        if self.full_res_stem:
            up_fac = 1
        else:
            up_fac = 4

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384

        self.proj_head = ProjectionHeadOri(dim_in=in_channels, proj_dim=self.output_feat_dim, bn_type=self.configer.get('network', 'bn_type'))

        self.total_cats = 0
        self.cats = []
        for i in range(0, self.n_datasets):
            self.cats.append(self.configer.get('dataset'+str(i+1), 'n_cats'))
            self.total_cats += self.configer.get('dataset'+str(i+1), 'n_cats')
        print("self.total_cats:", self.total_cats)
        
        self.max_num_unify_class = int(self.configer.get('GNN', 'unify_ratio') * self.total_cats)
        

        self.unify_prototype = nn.Parameter(torch.zeros(self.max_num_unify_class, self.output_feat_dim),
                                requires_grad=True)
        trunc_normal_(self.unify_prototype, std=0.02)
            
        self.init_weights()    
       

    def forward(self, x_, dataset=0):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        emb = self.proj_head(feats)
        if self.aux_mode == 'train':
            if self.training:
                logits = torch.einsum('bchw, nc -> bnhw', emb, self.unify_prototype)
                return {'seg':logits}
            else:
                return {'seg':emb}
        elif self.aux_mode == 'eval':
            logits = torch.einsum('bchw, nc -> bnhw', emb, self.unify_prototype)
            # remap_logits = F.interpolate(remap_logits, size=(target.size(1), target.size(2)), mode="bilinear", align_corners=True)
            this_index = sum(self.cats[:dataset])
            return logits[this_index:this_index+self.cats[dataset]]
        else:
            logits = torch.einsum('bchw, nc -> bnhw', emb, self.unify_prototype)
            # logits = torch.einsum('bchw, nc -> bnhw', logits, self.bipartite_graphs[dataset])
            logits = F.interpolate(logits, size=(logits.size(2)*4, logits.size(3)*4), mode="bilinear", align_corners=True)
            # remap_logits = torch.einsum('bchw, nc -> bnhw', logits, self.bipartite_graphs[dataset])
            pred = logits.argmax(dim=1)
            
            return pred

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
        state = torch.load(backbone_url)
        self.backbone.load_state_dict(state, strict=False)

    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.requires_grad == False:
                    continue
                
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4 or param.dim() == 2:
                    wd_params.append(param)
                else:
                    nowd_params.append(param)
                    print(param.dim())
                    # print(param)
                    print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
    
    def set_bipartite_graphs(self, bi_graphs):
        self.bipartite_graphs = nn.ModuleList(bi_graphs)
        
    def set_unify_prototype(self, unify_prototype, grad=False):
        self.unify_prototype = nn.Parameter(unify_prototype,
                                requires_grad=grad)
        

