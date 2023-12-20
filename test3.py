import torch
from tools.get_bipartile import print_bipartite
from tools.configer import Configer
configer = Configer(config='configs/ltbgnn_3_datasets_snp.json')
state = torch.load('res/celoss/seg_model_480000.pth', map_location='cpu')
bi = []
for i in range(0, 3):
    bi.append(state['model_state_dict'][f'bipartite_graphs.{i}'])
print_bipartite(configer, 3, bi)