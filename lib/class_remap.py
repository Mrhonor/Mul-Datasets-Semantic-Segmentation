import torch

class ClassRemap():
    def __init__(self, configer=None):
        self.configer = configer
        self.ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        self.remapList = []
        self.maxMapNums = [] # 最大映射类别数
        self._unpack()
        
        
    def SegRemapping(self, labels, dataset_id):
        ## 输入 batchsize x H x W 输出 k x batch size x H x W
        ## dataset_id指定映射方案
        outLabels = []
        
        for i in range(0, self.maxMapNums[dataset_id]):
            mask = torch.ones_like(labels) * self.ignore_lb
            
            for k, v in self.remapList[dataset_id].items():
                if len(v) <= i: 
                    continue

                mask[labels==int(k)] = v[i]
                
            outLabels.append(mask)
            
        return outLabels
        
    def ContrastRemapping(self, label, embed, proto):
        mask = torch.ones_like(labels) * self.ignore_lb
        
        for k, v in self.remapList[dataset_id].items():
            if len(v) == 1: 
                mask[labels==int(k)] = v[0]
            else:
                shapeEmbed = embed.permute(0,2,3,1)
                mask[labels==int(k)] = torch.einsum('nd,cd->nc', shapeEmbed[labels==int(k)], proto)
                
  
        return mask
        
    
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
            

        
        
if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'D:/Study/code/BiSeNet')
    from tools.configer import *
    configer = Configer(configs='../configs/bisenetv2_city.json')
    classRemap = ClassRemap(configer=configer)
    labels = torch.Tensor([3, 1, 10])
    print(classRemap.Remaping(labels, 0))
    print(classRemap.Remaping(labels, 1))
    

    