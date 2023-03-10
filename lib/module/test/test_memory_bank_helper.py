import pytest
import os
import sys
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(path)
import torch

from lib.module.memory_bank_helper import *
from tools.configer import Configer
from einops import rearrange

def test_memory_bank_push():
    configer = Configer(configs='configs/test/test_isSingleRemaplb.json') 
    lb = torch.tensor([[[0,2],
                        [2,1]],
                        [[2,0],
                        [0,1]]])
    
    memory_bank = torch.tensor([[[0,0], [0,0]],
                                [[0,0], [0,0]],
                                [[0,0], [0,0]],
                                [[0,0], [0,0]]])
    
    memory_bank_ptr = torch.tensor([0,1,0,0])
    memory_bank_init = torch.tensor([False, False, False, False])

    dataset_ids = torch.tensor([0,1])
    embedding = torch.tensor([[[[0,1],[-1, 0]],
                            [[0.3, -0.9],[1,0]]],
                            [[[0.9, 0.3],[-0.99, -0.1]],
                            [[0.9,0.3],[0.3, -0.9]]]])
    memory_bank_push(configer, memory_bank, memory_bank_ptr, embedding, lb, memory_bank_init, 1)
            
    true_memory_bank = torch.tensor([[[0,1], [0,0]],
                                    [[0.9,0.3], [1,0]],
                                    [[0.3,-0.9], [0,0]],
                                    [[0,0], [0,0]]])

    true_memory_bank_ptr = torch.tensor([1,0,1,0])
    true_memory_bank_init = torch.tensor([False, True, False, False])
    
    