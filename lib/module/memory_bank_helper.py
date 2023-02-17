import torch

def memory_bank_push(configer, memory_bank, memory_bank_ptr, _c, gt_seg, memory_bank_init, random_pick_ratio=1):
    num_unify_classes = configer.get('num_unify_classes')
    num_prototype = configer.get('contrast', 'num_prototype')
    if random_pick_ratio > 1: 
        random_pick_ratio = 1
    elif random_pick_ratio < 0:
        random_pick_ratio = 0 

    for i in range(0, num_unify_classes):
        if not (gt_seg==int(i)).any():
            continue
        
        else:
            this_feat = _c[gt_seg==int(i)]
            # pixel enqueue and dequeue
            num_pixel = this_feat.shape[0]
            perm = torch.randperm(num_pixel)
            K = int(min(num_pixel, num_prototype) * random_pick_ratio)
            
            if K == 0:
                K += 1
            
            feat = this_feat[perm[:K], :]
            
            ptr = int(memory_bank_ptr[i])

            if ptr + K >= num_prototype:
                remain_num = num_prototype - ptr
                new_cir_num = ptr + K - num_prototype
                memory_bank[i, ptr:, :] = feat[:remain_num]
                memory_bank[i, :new_cir_num, :] = feat[remain_num:]
                memory_bank_ptr[i] = new_cir_num
            else:
                memory_bank[i, ptr:ptr + K, :] = feat
                memory_bank_ptr[i] = (memory_bank_ptr[i] + K) % num_prototype
                memory_bank_init[i] = True
                
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    
    from tools.configer import Configer
    configer = Configer(configs='configs/test_kmeans.json')
    
    memory_bank = torch.zeros(3, 2, 2)
    memory_bank_ptr = torch.zeros(3)
    # _c = torch.tensor([[[1,1],
    #                     [2,2]],
    #                     [[-1,1],
    #                     [-2,1]],
    #                     [[0,-1],
    #                     [1,-2]]], dtype=torch.float, requires_grad=True)
    # memory_bank = torch.tensor([[[1,1],
    #                              [2,2]],
    #                             [[-1,1],
    #                              [-2,1]],
    #                             [[0,-1],
    #                              [1,-2]]], dtype=torch.float)
    # memory_bank_ptr = torch.tensor([0,0,0])
    _c = torch.tensor([[0,0],
                       [1,2],
                       [-1,2],
                       [-1,-2],
                       [1,1],
                       [1,1],
                       [0,3],
                       [-1,2]], dtype=torch.float, requires_grad=True)
    # cluster_seg = torch.ones(6, dtype=torch.bool)
    # gt_seg = torch.tensor([[1,1,1],
    #                         [1,1,1],
    #                         [1,1,1],
    #                         [1,1,1],
    #                         [0,1,1],
    #                         [1,0,1]], dtype=torch.bool).logical_not()
    gt_seg = torch.tensor([-1, 0, 2, 1, -1, 1, 0, 1])
    memory_bank_push(configer, memory_bank, memory_bank_ptr, _c.detach(), gt_seg)
    print(memory_bank)
    print(memory_bank_ptr)
            