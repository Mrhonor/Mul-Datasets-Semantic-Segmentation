import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassRemap():
    def __init__(self, configer=None):
        self.configer = configer
        self.ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        self.remapList = []
        self.maxMapNums = [] # 最大映射类别数
        self.num_unify_classes = self.configer.get('num_unify_classes')
        self.softmax = nn.Softmax(dim=1)
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
        
    def ContrastRemapping(self, target, embed, proto, dataset_id):
        network_stride = self.configer.get('network', 'stride')
        labels = target[:, ::network_stride, ::network_stride]
        B, H, W = labels.shape
        print(labels.shape)
        print(embed.shape)
        mask = torch.ones_like(labels) * self.ignore_index
        weight_mask = torch.zeros([B, H, W, self.num_unify_classes], dtype=torch.float32)
        if embed.is_cuda:
            weight_mask = weight_mask.cuda()
        
        for k, v in self.remapList[dataset_id].items():
            if len(v) == 1: 
                mask[labels==int(k)] = v[0]
            else:
                # 在多映射情况下，找到内积最大的一项的标签
                shapeEmbed = embed.permute(0,2,3,1)
                shapeEmbed = F.normalize(shapeEmbed, p=2, dim=-1)
                # n: b x h x w, d: dim, c:  num of class
                simScore = torch.einsum('nd,cd->nc', shapeEmbed[labels==int(k)], proto[v])
                MaxSimIndex = torch.max(simScore, dim=1)[1]
                mask[labels==int(k)] = torch.Tensor(v)[MaxSimIndex]
                
                expend_vector = torch.zeros([simScore.shape[0], self.num_unify_classes], dtype=torch.float32)
                expend_vector[:, v] = self.softmax(simScore)
                
                weight_mask[labels==int(k)] = expend_vector
                
                  
        return mask, weight_mask
        
    
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
        
        
if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'D:/Study/code/BiSeNet')
    from tools.configer import *
    from math import sin, cos

    pi = 3.14

    configer = Configer(configs='../configs/bisenetv2_city.json')
    classRemap = ClassRemap(configer=configer)
    labels = torch.tensor([[[0, 1, 0], [1, 0, 1]]], dtype=torch.float) # 1 x 2 x 3
    embed = torch.tensor([[[[cos(pi/4), cos(pi/3), cos(2*pi/3)], [cos(7*pi/6), cos(3*pi/2), cos(5*pi/3)]], 
                           [[sin(pi/4), sin(pi/3), sin(2*pi/3)], [sin(7*pi/6), sin(3*pi/2), sin(5*pi/3)]]]], requires_grad=True) # 1 x 1 x 2 x 3
    proto = torch.tensor([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=torch.float) # 19 x 2
    print(classRemap.ContrastRemapping(labels, embed, proto, 2))
    # print(classRemap.Remaping(labels, 1))
    # print(classRemap.getAnyClassRemap(3, 0))
    

    