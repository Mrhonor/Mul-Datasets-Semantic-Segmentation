import os
import sys
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(path)
import pytest
from lib.class_remap import *
from tools.configer import Configer
from einops import rearrange, repeat, einsum
from lib.loss.loss_cross_datasets import *
from tools.train_amp_contrast_single import set_model

class Test_CrossDatasetsLoss_KMeans:
    def test_LabelToOneHot(self):
        lb = torch.tensor([2, 1,2,-1])
        true_out = torch.tensor([[False, False,  True],
                [False,  True, False],
                [False, False,  True],
                [False, False, False]])
        assert not (true_out == LabelToOneHot(lb, 3)).logical_not().any()
        
    def test_IsInitMemoryBank(self):
        configer = Configer(configs='configs/test/test_isSingleRemaplb.json') 
        loss_fuc = CrossDatasetsCELoss_KMeans(configer)
        init_datas = [True, True, False, False]
        assert loss_fuc.IsInitMemoryBank(init_datas) == False
        init_datas = [True, True, True, False]
        assert loss_fuc.IsInitMemoryBank(init_datas) == True 
    
    def test_AdaptiveKMeansProtoLearning(self):
        configer = Configer(configs='configs/test/test_isSingleRemaplb.json') 
        loss_fuc = CrossDatasetsCELoss_KMeans(configer)
        lb = torch.tensor([[[0,2],
                            [2,1]],
                           [[2,0],
                            [0,1]]])
        
        memory_bank = torch.tensor([[[0,1], [0.6,0.8]],
                                    [[1,0], [0.8,0.6]],
                                    [[0,-1], [-0.6,-0.8]],
                                    [[0,0], [0,0]]])
        
        memory_bank_ptr = torch.tensor([0,0,0,0])
        memory_bank_init = torch.tensor([True, True, True, False])

        dataset_ids = torch.tensor([0,1])
        embedding = torch.tensor([[[[0,1],[-1, 0.3]],
                                   [[-0.3, -0.9],[1,0]]],
                                  [[[0.9, 0.3],[-0.99, -0.1]],
                                   [[0.9,0.3],[0.3, -0.9]]]])

        embedding = rearrange(embedding, 'b h w c -> b c h w')
         
        loss_fuc.AdaptiveKMeansProtoLearning(lb, memory_bank, memory_bank_ptr, memory_bank_init, embedding, dataset_ids)
        
        true_memory_bank = torch.tensor([[[ 0.0000,  1.0000],
                                        [ 0.6000,  0.8000]],

                                        [[-0.9900, -0.1000],
                                        [ 0.8000,  0.6000]],

                                        [[-0.3000, -0.9000],
                                        [-0.6000, -0.8000]],

                                        [[-1.0000,  0.3000],
                                        [ 0.9000,  0.3000]]])
        
        true_memory_bank_ptr = torch.tensor([0,1,1,0])
        true_memory_bank_init = torch.tensor([True, True, True, True]) 

        print(memory_bank)
        print(memory_bank_ptr)
        print(memory_bank_init)

        assert memory_bank.equal(true_memory_bank)
        assert memory_bank_ptr.equal(true_memory_bank_ptr)
        assert memory_bank_init.equal(true_memory_bank_init)
            
    # def test_AdaptiveSingleSegRemapping(self):
    #     configer = Configer(configs='configs/test/test_isSingleRemaplb.json') 
    #     loss_fuc = CrossDatasetsCELoss_KMeans(configer)
    #     lb = torch.tensor([[[0,1],[2,1]],[[3,1],[2,0]]])
    #     dataset_ids = torch.tensor([0,1])
    #     loss_fuc.AdaptiveSegRemapping(lb, dataset_ids)
        
        
    # def test_CrossDatasetsCELoss_KMeans(self):
    #     configer = Configer(configs='configs/test/test.json') 
    #     loss_fuc = CrossDatasetsCELoss_KMeans(configer)
    #     net = set_model(configer)
    #     adaptive_out = {}
    #     labels = torch.tensor([[2, 0, 0, 0],
    #                         [2, 1, 1, 1],
    #                         [2, 2, 1, 2],
    #                         [0, 0, 0, 2]]).unsqueeze(0)
    #     segment_queue = torch.tensor([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=torch.float) # 19 x 2
    #     embed = torch.tensor([[[-0.1, 0.9],[0.9, 0.1]],
    #                         [[-0.8, 0.2],[-0.1, 0.9]]]).unsqueeze(0)
    #     # 
    #     # embed = embed.permute(0, 3, 1, 2)
    #     Shapedembed = embed.contiguous().view(-1, 2)
    #     mul_segment_queue = torch.tensor([[[-1, 0], [0.9, 0.1], [-0.1, 1], [0, -1]],
    #                                     [[-0.9, 0.1], [1, 0], [0, 1], [-0.1, -1.9]]], dtype=torch.float) # 19 x 2 x 2
    #     proto_logits = torch.mm(Shapedembed, mul_segment_queue.view(-1, 2).T)
    #     print(mul_segment_queue.view(-1, 2))
    #     # print(proto_logits)
    #     rearrange_logit = torch.zeros_like(proto_logits)

    #     for i in range(0, 2):
    #         rearrange_logit[:, i::2] = proto_logits[:, i*4:(i+1)*4]
    #     print(rearrange_logit)
    #     # adaptive_out = 
        
    #     adaptive_out['prototypes'] = [net.memory_bank, net.memory_bank_ptr, net.memory_bank_init, net.prototypes]
    #     backward_loss, loss_seg, loss_aux, loss_contrast, loss_domain, kl_loss, new_proto = loss_fuc(adaptive_out, lb, dataset_lbs, is_warmup)
    #     assert False

class Test_CrossDatasetsCELoss:
    def test_CrossDatasetsCELoss(self):
        configer = Configer(configs='configs/test/test.json') 
        loss_fuc = CrossDatasetsCELoss(configer)
        lb = torch.tensor([[[2,1],[0,1]],[[2,1],[1,2]]])
        logits = torch.tensor([[[[1,2,3,4],[0,1,2,3]],[[2,3,4,1],[3,0,1,2]]],
                            [[[3,1,2,0],[2,4,1,0]],[[3,1,0,2],[2,4,3,1]]]], dtype=torch.float)
        logits = rearrange(logits, 'b h w c -> b c h w') 
        RemapMatrix = loss_fuc.classRemapper.getRemapMatrix(0)
        dataset_ids = torch.tensor([0,1])
        remap_logits = torch.einsum('bchw, nc -> bnhw', logits[0].unsqueeze(0), RemapMatrix)
        true_remap_logits = torch.tensor([[[[1, 2, 7],
                                    [0, 1, 5]],
                                    [[2, 3, 5],
                                    [3, 0, 3]]]])
        CELoss = torch.nn.CrossEntropyLoss(ignore_index=255)
        
        RemapMatrix = loss_fuc.classRemapper.getRemapMatrix(1)
        remap_logits = torch.einsum('bchw, nc -> bnhw', logits[1].unsqueeze(0), RemapMatrix)
        
        true_remap_logits = torch.tensor([[[[0,2,1,3],
                                    [0,1,4,2]],
                                    [[2,0,1,3],
                                    [1,3,4,2]]]])
        
        true_loss = 5.106813430786133
        pred = {}
        pred['seg'] = logits
        assert float(loss_fuc(pred, lb, dataset_ids)) == true_loss