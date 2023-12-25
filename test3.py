import torch
from tools.get_bipartile import print_bipartite
from tools.configer import Configer
from lib.models import model_factory
from lib.get_dataloader import get_data_loader, get_single_data_loader
from lib.loss.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg
from lib.loss.loss_cross_datasets import CrossDatasetsLoss, CrossDatasetsCELoss, CrossDatasetsCELoss_KMeans, CrossDatasetsCELoss_CLIP, CrossDatasetsCELoss_GNN, CrossDatasetsCELoss_AdvGNN
from lib.class_remap import ClassRemap
from lib.module.gen_graph_node_feature import gen_graph_node_feature

def set_graph_model(configer):

    net = model_factory[configer.get('GNN','model_name')](configer)

    if configer.get('train', 'graph_finetune'):
        state = torch.load('res/celoss/seg_model_300000.pth', map_location='cpu')
        
        if 'model_state_dict' in state:
            net.load_state_dict(state['model_state_dict'], strict=False)
        else:
            net.load_state_dict(state, strict=False)
        # net.load_state_dict(torch.load(configer.get('train', 'graph_finetune_from'), map_location='cpu'), strict=False)

        
    if configer.get('use_sync_bn'): 
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    # net.cuda()
    net.train()
    return net

configer = Configer(configs='configs/ltbgnn_3_datasets_snp.json')
graph_node_features = gen_graph_node_feature(configer)
graph_net = set_graph_model(configer=configer)

_, gnn_bi_graphs = graph_net.get_optimal_matching(graph_node_features, True)
_, ori_bi_graphs, _, _ = graph_net(graph_node_features)    

# state = torch.load('res/celoss/seg_model_480000.pth', map_location='cpu')
# unify_prototype = unify_prototype.detach()
bi_graphs = []
if len(ori_bi_graphs) == 6:
    for j in range(0, len(ori_bi_graphs), 2):
        bi_graphs.append(ori_bi_graphs[j].detach())
else:
    bi_graphs = [bigh.detach() for bigh in ori_bi_graphs]
# bi = []
# for i in range(0, 3):
#     bi.append(state['model_state_dict'][f'bipartite_graphs.{i}'])
print_bipartite(configer, 3, bi_graphs)
print("_______________________________________")
print_bipartite(configer, 3, gnn_bi_graphs)