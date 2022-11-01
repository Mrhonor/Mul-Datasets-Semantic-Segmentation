import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassRemap():
    def __init__(self, configer=None):
        self.configer = configer
        self.ignore_index = self.configer.get('loss', 'ignore_index')
        self.temperature = self.configer.get('contrast', 'temperature')
        self.remapList = []
        self.maxMapNums = [] # 最大映射类别数
        self.num_unify_classes = self.configer.get('num_unify_classes')
        self.softmax = nn.Softmax(dim=1)
        self.network_stride = self.configer.get('network', 'stride')
        self.Upsample = nn.Upsample(scale_factor=self.network_stride, mode='nearest')
        self._unpack()
        
        
    def SegRemapping(self, labels, dataset_id):
        ## 输入 batchsize x H x W 输出 k x batch size x H x W
        ## dataset_id指定映射方案
        outLabels = []
        
        for i in range(0, self.maxMapNums[dataset_id]):
            mask = torch.ones_like(labels) * self.ignore_index
            
            for k, v in self.remapList[dataset_id].items():
                if len(v) <= i: 
                    continue

                mask[labels==int(k)] = v[i]
                
            outLabels.append(mask)
            
        return outLabels
        
    
    def ContrastRemapping(self, labels, embed, proto, dataset_id):
        is_emb_upsampled = self.configer.get('contrast', 'upsample')
        
        # if is_emb_upsampled:
        #     contrast_lb = labels[:, ::self.network_stride, ::self.network_stride]
        # else:
        contrast_lb = labels
            
        # contrast_lb = labels
        B, H, W = contrast_lb.shape
        
        mask = torch.ones_like(contrast_lb) * self.ignore_index
        weight_mask = torch.zeros([B, H, W, self.num_unify_classes], dtype=torch.float32)
        if embed.is_cuda:
            weight_mask = weight_mask.cuda()
        
        for k, v in self.remapList[dataset_id].items():
            # 判断是否为空
            if not (contrast_lb==int(k)).any():
                continue
            
            
            if len(v) == 1: 
                mask[contrast_lb==int(k)] = v[0]
                
                weight_vector = torch.zeros(self.num_unify_classes, dtype=torch.float32)
                if weight_mask.is_cuda:
                    weight_vector = weight_vector.cuda()
                    
                weight_vector[v[0]] = 1
                weight_mask[contrast_lb==int(k)] = weight_vector
            else:
                # 在多映射情况下，找到内积最大的一项的标签
                
                shapeEmbed = embed.permute(0,2,3,1).detach()
                # shapeEmbed = F.normalize(shapeEmbed, p=2, dim=-1)
                # print(shapeEmbed[labels==int(k)].shape)
                # print(proto.shape)
                # print(proto[v].shape)
                # n: b x h x w, d: dim, c:  num of class
                simScore = torch.div(torch.einsum('nd,cd->nc', shapeEmbed[contrast_lb==int(k)], proto[v]), self.temperature)
                MaxSimIndex = torch.max(simScore, dim=1)[1]
                if mask.is_cuda:
                    mask[contrast_lb==int(k)] = torch.tensor(v, dtype=torch.long)[MaxSimIndex].cuda()
                else:
                    mask[contrast_lb==int(k)] = torch.tensor(v, dtype=torch.long)[MaxSimIndex]
                
                expend_vector = torch.zeros([simScore.shape[0], self.num_unify_classes], dtype=torch.float32)
                if weight_mask.is_cuda:
                    expend_vector = expend_vector.cuda()
                    
                
                expend_vector[:, v] = self.softmax(simScore)
                
                weight_mask[contrast_lb==int(k)] = expend_vector
                
        # if is_emb_upsampled:
        #     weight_mask = self.Upsample(weight_mask.permute(0,3,1,2)).permute(0,2,3,1)
          
        return mask, weight_mask
        
    def GetEqWeightMask(self, labels, dataset_id):
        B, H, W = labels.shape
        weight_mask = torch.zeros([B, H, W, self.num_unify_classes], dtype=torch.float32)
        if labels.is_cuda:
            weight_mask = weight_mask.cuda()
            
        for k, v in self.remapList[dataset_id].items():
            map_num = len(v)
            weight_vector = torch.zeros(self.num_unify_classes, dtype=torch.float32)
            if weight_mask.is_cuda:
                weight_vector = weight_vector.cuda()
                
            for val in v:
                weight_vector[val] = 1  # / map_num
                
            weight_mask[labels==int(k)] = weight_vector

        
        return weight_mask
    
    
    def _unpack(self):
        self.n_datasets = -1
        if self.configer.exists('n_datasets'):
            self.n_datasets = self.configer.get('n_datasets')
        else:
            raise NotImplementedError("read json errror! no  n_datasets")    

        # 读取class remap info
        for i in range(1, self.n_datasets+1):
            if not self.configer.exists('class_remap' + str(i)):
                raise  NotImplementedError("read json errror! no class_remap"+str(i))
            class_id = 0
            maxMapNum = 0
            class_remap = {}
            while str(class_id) in self.configer.get('class_remap' + str(i)):
                class_remap[class_id] = self.configer.get('class_remap' + str(i))[str(class_id)]
                maxMapNum = len(class_remap[class_id]) if len(class_remap[class_id]) > maxMapNum else maxMapNum
                class_id += 1
            self.remapList.append(class_remap)
            self.maxMapNums.append(maxMapNum)
            
    def getAnyClassRemap(self, lb_id, dataset_id):
        return self.remapList[dataset_id][lb_id]
    
    def ReverseSegRemap(self, preds, dataset_id):
        # logits : B x h x w
        Remap_pred = torch.ones_like(preds) * self.ignore_index
        for k, v in self.remapList[dataset_id].items():
            for lb in v:
                Remap_pred[preds == int(lb)] = int(k)
                
        return Remap_pred
        
class ClassRemapOneHotLabel(ClassRemap):
    def __init__(self, configer=None):
        super(ClassRemapOneHotLabel, self).__init__(configer)
        self.update_sim_thresh = self.configer.get('contrast', 'update_sim_thresh')

    def SegRemapping(self, labels, dataset_id):
        ## 输入 batchsize x H x W 输出 n x batch size x H x W
        ## 1表示目标类， 0表示非目标类
        ## dataset_id指定映射方案
        b, h, w = labels.shape
        outLabels = torch.zeros(self.num_unify_classes, b, h, w)
        if labels.is_cuda:
            outLabels = outLabels.cuda()
        
        for k, v in self.remapList[dataset_id].items():
            for i in v:
                outLabels[i,labels==int(k)] = 1

        return outLabels
    
    def ContrastRemapping(self, labels, embed, proto, dataset_id):
        # seg_mask 监督分割头的多类别标签
        # contrast_mask 监督投影头的标签信息
        is_emb_upsampled = self.configer.get('contrast', 'upsample')
        if is_emb_upsampled:
            raise Exception("Not Imp")
        # if is_emb_upsampled:
        #     contrast_lb = labels[:, ::self.network_stride, ::self.network_stride]
        # else:
        #     contrast_lb = labels
        
        contrast_lb = labels[:, ::self.network_stride, ::self.network_stride]
        # contrast_lb = labels
        B, H, W = labels.shape
        _, h_c, w_c = contrast_lb.shape
        contrast_mask = torch.ones_like(contrast_lb) * self.ignore_index
        
        # seg_mask = torch.zeros([B, H, W, self.num_unify_classes], dtype=torch.int)
        temp_mask = torch.zeros([B, h_c, w_c, self.num_unify_classes], dtype=torch.long)
        hard_lb_mask = torch.zeros([B, h_c, w_c, self.num_unify_classes], dtype=torch.long)
        if labels.is_cuda:
            # seg_mask = seg_mask.cuda()
            temp_mask = temp_mask.cuda()
            hard_lb_mask = hard_lb_mask.cuda()
        
        ## 循环两遍，单标签覆盖多标签
        for k, v in self.remapList[dataset_id].items():
            # 判断是否为空
            if not (labels==int(k)).any():
                continue
            
            if len(v) != 1: 
                # 在多映射情况下，找到内积最大的一项的标签
                if not (contrast_lb==int(k)).any():
                    continue
                
                shapeEmbed = embed.permute(0,2,3,1).detach()
                # shapeEmbed = F.normalize(shapeEmbed, p=2, dim=-1)
                # print(shapeEmbed[labels==int(k)].shape)
                # print(proto.shape)
                # print(proto[v].shape)
                # n: b x h x w, d: dim, c:  num of class
                simScore = torch.div(torch.einsum('nd,cd->nc', shapeEmbed[contrast_lb==int(k)], proto), self.temperature)
                # print("simScore")
                # print(simScore)
                simScore = self.softmax(simScore)
                
                MaxSim, MaxSimIndex = torch.max(simScore, dim=1)
                outputIndex = torch.ones_like(MaxSimIndex) * self.ignore_index
                
                # 困难样本
                hardLbIndex = torch.zeros([MaxSimIndex.shape[0], self.num_unify_classes], dtype=torch.long)
                # 筛选出目标类的index。对于最大值不在目标类的情况以及最大值没超过阈值的pixel，标签设置为忽略
                tensor_v = torch.tensor(v)
                if labels.is_cuda:
                    tensor_v = tensor_v.cuda()
                
                targetIndex = MaxSimIndex.unsqueeze(1).eq(tensor_v)
                targetIndex = torch.unique(targetIndex.nonzero(as_tuple=True)[0])
                MaxSimIndex[MaxSim < self.update_sim_thresh] = self.ignore_index
                
                outputIndex[targetIndex] = MaxSimIndex[targetIndex]
                
                hardLd = torch.zeros(self.num_unify_classes, dtype=torch.long)
                hardLd[tensor_v] = 1
               
                hardLbIndex[outputIndex==self.ignore_index] = hardLd
                
                if contrast_mask.is_cuda:
                    contrast_mask[contrast_lb==int(k)] = outputIndex.cuda()
                    hard_lb_mask[contrast_lb==int(k)] = hardLbIndex.cuda()
                else:
                    contrast_mask[contrast_lb==int(k)] = outputIndex
                    hard_lb_mask[contrast_lb==int(k)] = hardLbIndex
                
                expend_vector = torch.zeros([simScore.shape[0], self.num_unify_classes], dtype=torch.long)
                if temp_mask.is_cuda:
                    expend_vector = expend_vector.cuda()
                
                for i in range(0, outputIndex.shape[0]):
                    if outputIndex[i] == self.ignore_index:
                        expend_vector[i, v] = 1
                        # continue
                    else:
                        expend_vector[i, outputIndex[i]] = 1
                
                temp_mask[contrast_lb==int(k)] = expend_vector
        
        # print(temp_mask)
        # 对temp_mask进行上采样
        seg_mask = temp_mask.permute(0, 3, 1, 2)
        seg_mask = F.interpolate(seg_mask.float(), size=(H,W), mode='nearest').long()
        seg_mask = seg_mask.permute(0, 2, 3, 1)
        # print(seg_mask)
        for k, v in self.remapList[dataset_id].items():
            # 判断是否为空
            if not (labels==int(k)).any():
                continue
            
            
            if len(v) == 1: 
                contrast_mask[contrast_lb==int(k)] = v[0]
                
                seg_vector = torch.zeros(self.num_unify_classes, dtype=torch.long)
                if seg_mask.is_cuda:
                    seg_vector = seg_vector.cuda()
                    
                seg_vector[v[0]] = 1
                seg_mask[labels==int(k)] = seg_vector
            else:
                expend_vector = torch.zeros(self.num_unify_classes, dtype=torch.long)
                if seg_mask.is_cuda:
                    expend_vector = expend_vector.cuda()

                expend_vector[v] = 1
                # 寻找为0项
                lb_k_mask = seg_mask[labels==int(k)]
                maxValue = torch.max(lb_k_mask, dim=1)[0]
                lb_k_mask[maxValue==0] = expend_vector
                seg_mask[labels==int(k)] = lb_k_mask
    
        zero_vector = torch.zeros(self.num_unify_classes, dtype=torch.long)
        if seg_mask.is_cuda:
            zero_vector = zero_vector.cuda()
            
        seg_mask[labels==self.ignore_index] = zero_vector
        # if is_emb_upsampled:
        #     weight_mask = self.Upsample(weight_mask.permute(0,3,1,2)).permute(0,2,3,1)
          
        return contrast_mask, seg_mask, hard_lb_mask
    
    
def test_ContrastRemapping():
    configer = Configer(configs='configs/test.json')
    classRemap = ClassRemapOneHotLabel(configer)
    labels = torch.tensor([[2, 0, 0, 0],
                           [2, 1, 1, 1],
                           [2, 2, 1, 2],
                           [0, 0, 0, 2]]).unsqueeze(0)
    segment_queue = torch.tensor([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=torch.float) # 19 x 2
    embed = torch.tensor([[[-0.2, -0.8],[0.9, 0.1]],
                          [[-0.8, 0.2],[-0.1, 0.9]]]).unsqueeze(0)
    # print(embed.shape)
    embed = embed.permute(0, 3, 1, 2)
    # print(embed)
    contrast_mask, seg_mask, hard_lb_mask = classRemap.ContrastRemapping(labels, embed, segment_queue, 0)
    print(contrast_mask)
    
        
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/cxh/mr/BiSeNet')
    from tools.configer import *
    from math import sin, cos
    
    test_ContrastRemapping()
    # pi = 3.14

    # configer = Configer(configs='configs/bisenetv2_city_cam.json')
    # classRemap = ClassRemap(configer=configer)
    # labels = torch.tensor([[[0, 6, 0], [1, 0, 1]]], dtype=torch.float) # 1 x 2 x 3
    # embed = torch.tensor([[[[cos(pi/4), cos(pi/3), cos(2*pi/3)], [cos(7*pi/6), cos(3*pi/2), cos(5*pi/3)]], 
    #                        [[sin(pi/4), sin(pi/3), sin(2*pi/3)], [sin(7*pi/6), sin(3*pi/2), sin(5*pi/3)]]]], requires_grad=True) # 1 x 1 x 2 x 3
    # proto = torch.tensor([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=torch.float) # 19 x 2
    # # print(classRemap.ContrastRemapping(labels, embed, proto, 2))
    # print(classRemap.GetEqWeightMask(labels, 1))
    # # print(classRemap.getAnyClassRemap(3, 0))
    

    