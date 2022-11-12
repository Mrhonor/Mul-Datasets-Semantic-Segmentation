import torch
import torch.nn as nn

class ConvNorm(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super(ConvNorm, self).__init__()
        
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, x):
        feat = self.conv(x)
        
        # if self.training:
        normX = torch.norm(x, dim=1, keepdim=True)

        normW = torch.norm(self.conv.weight.squeeze(), dim=1, keepdim=True)
        normVal = torch.einsum('on,bnhw->bohw', normW, normX) + 1e-8

        feat_sim = feat / normVal

        return feat_sim
        # else:
        #     return feat
            
        
        
if __name__ == '__main__':
    x = torch.randn(6,2,4,4)
    convnorm = ConvNorm(2,3)
    print(convnorm(x))
    
    # x = 