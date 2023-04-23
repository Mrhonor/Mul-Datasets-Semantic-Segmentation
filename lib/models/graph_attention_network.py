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
        self.nhid = self.configer.get('GNN', 'nhid')
        self.att_out_dim = self.configer.get('GNN', 'att_out_dim')
        self.alpha = self.configer.get('GNN', 'alpha')
        self.nheads = self.configer.get('GNN', 'nheads')
        self.mlp_dim = self.configer.get('GNN', 'mlp_dim')
        

        self.output_feat_dim = self.configer.get('GNN', 'output_feat_dim')
        self.dropout_rate = self.configer.get('GNN', 'dropout_rate')
        self.threshold_value = self.configer.get('GNN', 'threshold_value')

        self.attentions_layer1 = nn.ModuleList([GraphAttentionLayer(self.nfeat, self.nhid, dropout=self.dropout, alpha=self.alpha, concat=True) for _ in range(self.nheads)])

        self.attentions_layer2 = nn.ModuleList([GraphAttentionLayer(self.nhid * self.nheads, self.nhid, dropout=self.dropout, alpha=self.alpha, concat=True) for _ in range(self.nheads)])

        self.out_att = GraphAttentionLayer(self.nhid * self.nheads, self.att_out_dim, dropout=self.dropout, alpha=self.alpha, concat=False)
        
        self.linear1 = nn.Linear(self.att_out_dim, self.mlp_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.mlp_dim, self.output_feat_dim) 
        
        ## datasets Node features
        self.n_datasets = self.configer.get('data', 'n_datasets')
        self.total_cats = 0
        self.dataset_cats = []
        for i in range(0, self.n_datasets):
            self.dataset_cats.append(self.configer.get('dataset'+str(i+1), 'n_cats'))
            self.total_cats += self.configer.get('dataset'+str(i+1), 'n_cats')
        
        self.max_num_unify_class = self.configer.get('GNN', 'unify_ratio') * self.total_cats        
        # self.register_buffer("fix_node_features", torch.randn(self.total_cats, self.nfeat))
        self.unify_node_features = nn.Parameter(torch.randn(self.max_num_unify_class, self.nfeat), requires_grad=True)
        trunc_normal_(self.unify_node_features, std=0.02)
        
        ## Graph adjacency matrix
        self.init_adjacency_matrix()
        
    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, self.adj_matrix) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.adj_matrix))
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x[self.total_cats:], self.calc_bipartite_graph(x)

    def calc_bipartite_graph(self, x):
        unify_feats = x[self.total_cats:]
        
        cur_cat = 0
        bipartite_graphs = []
        for i in range(0, self.n_datasets):
            this_feats = x[cur_cat:cur_cat+self.dataset_cats[i]]
            cur_cat += self.dataset_cats[i]
            similar_matrix = torch.einsum('nc, mc -> nm', this_feats, unify_feats)
            softmax_similar_matrix = F.softmax(similar_matrix, dim=1)
            # softmax_similar_matrix[softmax_similar_matrix < self.threshold_value] = 0
            max_index, max_value = torch.max(softmax_similar_matrix)
            bi_graph = torch.zero(self.dataset_cats[i], self.max_num_unify_class)
            bi_graph[max_index] = 1
            
            this_iter_thresh = 0.3 + (self.threshold_value - 0.3) * self.configer.get('iter') / self.configer.get('lr', 'max_iter')
            bi_graph[max_value < this_iter_thresh] = 0
            
            bipartite_graphs.append(bi_graph)
            
        return bipartite_graphs.T()
            
            
    def init_adjacency_matrix(self):
        def normalize_adj(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv_sqrt = np.power(rowsum, -0.5).flatten()
            r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
            r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
            return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
        
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

