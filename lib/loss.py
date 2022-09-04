import torch.nn as nn
import torch

class NLLPlusLoss(nn.Module):
    def __init__(self, ignore_lb=255):
        super(NLLPlusLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.nllloss = nn.NLLLoss(ignore_index=ignore_lb, reduction='mean')
    
    def forward(self, x, labels):
        # labels为 k x batch size x H x W
        # k为最大映射数量，对于输入的x需要计算k次nll loss后求和，
        # 对于没有k个映射对象的类别，如只有n个的类别，后n-k个labels值均为ignore_index
        pred = self.softmax(x)
        probs = None
        for lb in labels:
            val = -self.nllloss(pred, lb)
            probs = val if probs is None else probs+val
            
        prob = torch.sum(probs)
        loss = -torch.log(prob)
        return loss
    
    
if __name__ == '__main__':
    x = torch.Tensor([[2,1], [1,2]])
    label = torch.tensor([[0, 1],[1, 2]])
    
    # x = torch.randn((4, 19, 512, 1024)).cuda()
    # label = torch.ones((2, 4, 512, 1024)).long().cuda()
    lossfunc = NLLPlusLoss(ignore_lb=2)
    print(lossfunc(x, label))
