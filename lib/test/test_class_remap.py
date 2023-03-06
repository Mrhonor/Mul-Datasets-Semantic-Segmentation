import os
import sys
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
import pytest
from lib.class_remap import *
from tools.configer import Configer
from einops import rearrange, repeat, einsum

class Test_ClassRemap:
        
    def test_isSingleRemaplb(self):
        configer = Configer(configs='configs/test/test_isSingleRemaplb.json')
        classRemaper = ClassRemap(configer)
        out1 = classRemaper.IsSingleRemaplb(1)
        assert out1 == True
        out2 = classRemaper.IsSingleRemaplb(3)
        assert out2 == False

    # def test_ContrastRemapping(self):
    #     configer = Configer(configs='configs/test/test.json')
    #     classRemaper = ClassRemap(configer)
    #     labels = torch.tensor([[2, 0, 0, 0],
    #                         [2, 1, 1, 1],
    #                         [2, 2, 1, 2],
    #                         [0, 0, 0, 2]]).unsqueeze(0)
    #     segment_queue = torch.tensor([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=torch.float) # 19 x 2
    #     embed = torch.tensor([[[-0.1, 0.9],[0.9, 0.1]],
    #                         [[-0.8, 0.2],[-0.1, 0.9]]]).unsqueeze(0)
    #     # 
    #     # embed = embed.permute(0, 3, 1, 2)
    #     Shapedembed = rearrange(embed, 'b h w c -> b c h w')
    #     # print(proto_logits)

        
    #     # contrast_mask, seg_mask = classRemap.ContrastRemapping(labels, embed, segment_queue, 0)
    #     contrast_mask, seg_mask = classRemaper.ContrastRemapping(labels, Shapedembed, segment_queue, 0)
    #     print(contrast_mask)
    #     print(seg_mask) 

class Test_ClassRemapOneHotLabel:

    def test_MultiProtoRemapping(self):
        configer = Configer(configs='configs/test/test.json')
        classRemap = ClassRemapOneHotLabel(configer)
        labels = torch.tensor([[2, 0, 0, 0],
                            [2, 1, 1, 1],
                            [2, 2, 1, 2],
                            [0, 0, 0, 2]]).unsqueeze(0)
        segment_queue = torch.tensor([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=torch.float) # 19 x 2
        embed = torch.tensor([[[-0.1, 0.9],[0.9, 0.1]],
                            [[-0.8, 0.2],[-0.1, 0.9]]]).unsqueeze(0)
        # 
        # embed = embed.permute(0, 3, 1, 2)
        Shapedembed = embed.contiguous().view(-1, 2)
        mul_segment_queue = torch.tensor([[[-1, 0], [0.9, 0.1], [-0.1, 1], [0, -1]],
                                        [[-0.9, 0.1], [1, 0], [0, 1], [-0.1, -1.9]]], dtype=torch.float) # 19 x 2 x 2
        proto_logits = torch.mm(Shapedembed, mul_segment_queue.view(-1, 2).T)
        # print(mul_segment_queue.view(-1, 2))
        # print(proto_logits)
        rearrange_logit = torch.zeros_like(proto_logits)

        for i in range(0, 2):
            rearrange_logit[:, i::2] = proto_logits[:, i*4:(i+1)*4]
        # print(rearrange_logit)
        
        # contrast_mask, seg_mask = classRemap.ContrastRemapping(labels, embed, segment_queue, 0)
        contrast_mask, seg_mask = classRemap.MultiProtoRemapping(labels, rearrange_logit, 0)
        out_contrast_mask = torch.tensor([[[[False, False, False, False,  True, False, False, False],
                                        [False,  True, False, False, False, False, False, False]],

                                        [[False, False, False, False,  True,  True,  True,  True],
                                        [False, False,  True, False, False, False, False, False]]]])
        out_seg_mask = torch.tensor([[[[False, False,  True, False],
                                    [ True, False, False, False],
                                    [ True, False, False, False],
                                    [ True, False, False, False]],

                                    [[False, False,  True, False],
                                    [False,  True, False, False],
                                    [False,  True, False, False],
                                    [False,  True, False, False]],

                                    [[False, False,  True,  True],
                                    [False, False,  True,  True],
                                    [False,  True, False, False],
                                    [False, False,  True,  True]],

                                    [[ True, False, False, False],
                                    [ True, False, False, False],
                                    [ True, False, False, False],
                                    [False, False,  True,  True]]]])
        
        assert not (out_contrast_mask == contrast_mask).logical_not().any()
        assert not (out_seg_mask == seg_mask).logical_not().any()
        
    
# def test_getReweightMatrix():
#     configer = Configer(configs='configs/test.json')
#     classRemap = ClassRemapOneHotLabel(configer)
#     labels = torch.tensor([[2, 0, 0, 0],
#                            [2, 1, 1, 1],
#                            [2, 2, 1, 3],
#                            [0, 0, 0, 2]]).unsqueeze(0)

#     print(classRemap.getReweightMatrix(labels, 1))

        
# if __name__ == "__main__":
#     import sys
#     sys.path.insert(0, '/home/cxh/mr/BiSeNet')
#     from tools.configer import *
#     from math import sin, cos
#     from einops import rearrange
    
#     # test_ContrastRemapping()
#     test_get_reweight_matrix()
#     # pi = 3.14

#     # configer = Configer(configs='configs/bisenetv2_city_cam.json')
#     # classRemap = ClassRemap(configer=configer)
#     # labels = torch.tensor([[[0, 6, 0], [1, 0, 1]]], dtype=torch.float) # 1 x 2 x 3
#     # embed = torch.tensor([[[[cos(pi/4), cos(pi/3), cos(2*pi/3)], [cos(7*pi/6), cos(3*pi/2), cos(5*pi/3)]], 
#     #                        [[sin(pi/4), sin(pi/3), sin(2*pi/3)], [sin(7*pi/6), sin(3*pi/2), sin(5*pi/3)]]]], requires_grad=True) # 1 x 1 x 2 x 3
#     # proto = torch.tensor([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=torch.float) # 19 x 2
#     # # print(classRemap.ContrastRemapping(labels, embed, proto, 2))
#     # print(classRemap.GetEqWeightMask(labels, 1))
#     # # print(classRemap.getAnyClassRemap(3, 0))
    

    