import torch
import torch.nn as nn
import torch.nn.functional as F

pixel_num = torch.tensor([7930364.0, 10283520.0])

def clac_class_weight(pixel_num):
    norm_num = F.normalize(pixel_num, p=1, dim=0)
    print(norm_num)
    norm_num = torch.exp(-norm_num)+1
    print(norm_num)
    return norm_num

if __name__ == "__main__":
    test = torch.tensor([3.0, 4.0])
    clac_class_weight(test) 