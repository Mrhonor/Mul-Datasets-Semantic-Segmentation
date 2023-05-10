import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from lib.module.module_helper import GraphAttentionLayer, SpGraphAttentionLayer
import numpy as np
import scipy.sparse as sp


class GAT(nn.Module):
    def __init__(self, configer):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        
        self.configer = configer
        self.nfeat = self.configer.get('GNN', 'nfeat')
        self.nfeat_out = self.configer.get('GNN', 'nfeat_out')
        self.nhid = self.configer.get('GNN', 'nhid')
        self.att_out_dim = self.configer.get('GNN', 'att_out_dim')
        self.alpha = self.configer.get('GNN', 'alpha')
        self.nheads = self.configer.get('GNN', 'nheads')
        self.mlp_dim = self.configer.get('GNN', 'mlp_dim')
        

        self.output_feat_dim = self.configer.get('GNN', 'output_feat_dim')
        self.dropout_rate = self.configer.get('GNN', 'dropout_rate')
        self.threshold_value = self.configer.get('GNN', 'threshold_value')
        self.fix_arch = False
        self.fix_architecture_alter_iter = self.configer.get('GNN', 'fix_architecture_alter_iter')

        self.linear_before = nn.Linear(self.feat, self.nfeat_out)
        self.relu = nn.ReLU()

        self.attentions_layer1 = nn.ModuleList([GraphAttentionLayer(self.nfeat_out, self.nhid, dropout=self.dropout_rate, alpha=self.alpha, concat=True) for _ in range(self.nheads)])

        self.attentions_layer2 = nn.ModuleList([GraphAttentionLayer(self.nhid * self.nheads, self.nhid, dropout=self.dropout_rate, alpha=self.alpha, concat=True) for _ in range(self.nheads)])

        self.out_att = GraphAttentionLayer(self.nhid * self.nheads, self.att_out_dim, dropout=self.dropout_rate, alpha=self.alpha, concat=False)
        
        self.linear1 = nn.Linear(self.att_out_dim, self.mlp_dim)
        
        self.linear2 = nn.Linear(self.mlp_dim, self.output_feat_dim) 

        
        ## datasets Node features
        self.n_datasets = self.configer.get('n_datasets')
        self.total_cats = 0
        self.dataset_cats = []
        for i in range(0, self.n_datasets):
            self.dataset_cats.append(self.configer.get('dataset'+str(i+1), 'n_cats'))
            self.total_cats += self.configer.get('dataset'+str(i+1), 'n_cats')
        
        self.max_num_unify_class = int(self.configer.get('GNN', 'unify_ratio') * self.total_cats)
        # self.register_buffer("fix_node_features", torch.randn(self.total_cats, self.nfeat))
        self.unify_node_features = nn.Parameter(torch.randn(self.max_num_unify_class, self.nfeat), requires_grad=True)
        trunc_normal_(self.unify_node_features, std=0.02)
        
        ## Graph adjacency matrix
        self.init_adjacency_matrix()
        
    def forward(self, x):
        x = self.linear_before(x)
        x = self.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = torch.cat([att(x, self.adj_matrix) for att in self.attentions_layer1], dim=1)
        # x = F.dropout(x, self.dropout_rate, training=self.training)
        x = torch.cat([att(x, self.adj_matrix) for att in self.attentions_layer2], dim=1)
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = F.elu(self.out_att(x, self.adj_matrix))
        x = self.linear1(x)
        arch_x = self.relu(x)
        arch_x = self.linear2(arch_x)
        
        return x[self.total_cats:], self.calc_bipartite_graph(arch_x)

    def calc_bipartite_graph(self, x):
        this_fix_arch = self.fix_arch
        cur_iter = self.configer.get('iter')
        if cur_iter / self.fix_architecture_alter_iter % 2 == 0:
            self.fix_arch == False
        else:
            self.fix_arch == True    
        
        if this_fix_arch:    
            return self.bipartite_graphs
        
        unify_feats = x[self.total_cats:]
        
        cur_cat = 0
        self.bipartite_graphs = []
        for i in range(0, self.n_datasets):
            this_feats = x[cur_cat:cur_cat+self.dataset_cats[i]]
            cur_cat += self.dataset_cats[i]
            similar_matrix = torch.einsum('nc, mc -> nm', this_feats, unify_feats)
            softmax_similar_matrix = F.softmax(similar_matrix / 0.07, dim=1)
            # softmax_similar_matrix[softmax_similar_matrix < self.threshold_value] = 0
            # max_value, max_index = torch.max(softmax_similar_matrix, dim=0)
            # bi_graph = torch.zeros(self.dataset_cats[i], self.max_num_unify_class)
            # if x.is_cuda:
            #     bi_graph = bi_graph.cuda()

            # bi_graph[max_index] = 1
            
            # this_iter_thresh = 0.3 + (self.threshold_value - 0.3) * self.configer.get('iter') / self.configer.get('lr', 'max_iter')
            # this_iter_thresh = self.threshold_value * self.configer.get('iter') / self.configer.get('lr', 'max_iter')
            # bi_graph[:, max_value < this_iter_thresh] = 0
            
            
            self.bipartite_graphs.append(softmax_similar_matrix)
            
        return self.bipartite_graphsS
            
            
    def init_adjacency_matrix(self):
        def normalize_adj(mx):
        
            rowsum = np.array(mx.sum(1))
            r_inv_sqrt = np.power(rowsum, -0.5).flatten()
            r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
            r_mat_inv_sqrt = torch.diag(torch.tensor(r_inv_sqrt))
            
            # r_mat_inv_sqrt = torch.diag(torch.tensor(r_inv_sqrt))
            # print(r_mat_inv_sqrt)
            return torch.mm(r_mat_inv_sqrt, torch.mm(mx, r_mat_inv_sqrt))
        
        # init adjacency matrix according to the number of categories of each dataset
        
        self.register_buffer("adj_matrix", torch.zeros(self.total_cats+self.max_num_unify_class, self.total_cats+self.max_num_unify_class))
        
        self.adj_matrix[self.total_cats:, :] = 1
        self.adj_matrix[:, self.total_cats:] = 1
        self.adj_matrix[:self.total_cats, :self.total_cats] = torch.eye(self.total_cats)
        self.adj_matrix[self.total_cats:, self.total_cats:] = torch.eye(self.max_num_unify_class)
        
        self.adj_matrix = normalize_adj(self.adj_matrix)
        

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
