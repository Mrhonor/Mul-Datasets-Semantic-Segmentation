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

        self.linear_before = nn.Linear(self.nfeat, self.nfeat_out)
        self.relu = nn.ReLU()

        self.attentions_layer1 = nn.ModuleList([GraphAttentionLayer(self.nfeat_out, self.nhid, dropout=self.dropout_rate, alpha=self.alpha, concat=True) for _ in range(self.nheads)])

        # self.attentions_layer2 = nn.ModuleList([GraphAttentionLayer(self.nhid * self.nheads, self.nhid, dropout=self.dropout_rate, alpha=self.alpha, concat=True) for _ in range(self.nheads)])

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
        
        # self.bipartite_graphs = []
        # for i in range(0, self.n_datasets):
        #     self.bipartite_graphs.append(nn.Parameter(torch.zeros(self.dataset_cats[i], self.max_num_unify_class), requires_grad=True))

        
        # self.register_buffer("fix_node_features", torch.randn(self.total_cats, self.nfeat))
        self.unify_node_features = nn.Parameter(torch.randn(self.max_num_unify_class, self.nfeat), requires_grad=True)
        trunc_normal_(self.unify_node_features, std=0.02)
        
        ## Graph adjacency matrix
        self.init_adjacency_matrix()
        
    def forward(self, x):
        x = self.linear_before(x)
        x = self.relu(x)
        # x = F.dropout(x, self.dropout_rate, training=self.training)
        feat = torch.cat([att(x, self.adj_matrix) for att in self.attentions_layer1], dim=1)
        x = feat + x
        # x = F.dropout(x, self.dropout_rate, training=self.training)
        # x = torch.cat([att(x, self.adj_matrix) for att in self.attentions_layer2], dim=1)
        # x = F.dropout(x, self.dropout_rate, training=self.training)
        x = F.elu(self.out_att(x, self.adj_matrix) + x)
        feat = self.linear1(x)
        arch_x = self.relu(x + feat)
        arch_x = self.linear2(arch_x)
        
        return feat[self.total_cats:], self.calc_bipartite_graph(arch_x)

    def calc_bipartite_graph(self, x):
        this_fix_arch = self.fix_arch
        cur_iter = self.configer.get('iter')
        if cur_iter < self.fix_architecture_alter_iter:
            self.linear2.requires_grad = False
            return self.pretrain_bipartite_graphs(is_cuda=x.is_cuda)
        
        if (cur_iter // self.fix_architecture_alter_iter) % 2 == 0:
            self.linear2.requires_grad = False
            self.fix_arch = False
        else:
            self.linear2.requires_grad = True
            self.fix_arch = True    
        
        if this_fix_arch:    
            return self.bipartite_graphs.detach()
        
        unify_feats = x[self.total_cats:]
        
        cur_cat = 0
        self.bipartite_graphs = []
        for i in range(0, self.n_datasets):
            this_feats = x[cur_cat:cur_cat+self.dataset_cats[i]]
            cur_cat += self.dataset_cats[i]
            similar_matrix = torch.einsum('nc, mc -> nm', this_feats, unify_feats)
            softmax_similar_matrix = F.softmax(similar_matrix / 0.05, dim=0)
            # softmax_similar_matrix[softmax_similar_matrix < self.threshold_value] = 0
            # max_value, max_index = torch.max(softmax_similar_matrix, dim=0)
            # self.bipartite_graphs[i] = torch.zeros(self.dataset_cats[i], self.max_num_unify_class, requires_grad=True)
            # if x.is_cuda:
            #     bi_graph = bi_graph.cuda()

            # self.bipartite_graphs[i][max_index] = 1
            
            # this_iter_thresh = 0.3 + (self.threshold_value - 0.3) * self.configer.get('iter') / self.configer.get('lr', 'max_iter')
            # this_iter_thresh = self.threshold_value * self.configer.get('iter') / self.configer.get('lr', 'max_iter')
            # bi_graph[:, max_value < this_iter_thresh] = 0
            
            
            self.bipartite_graphs.append(softmax_similar_matrix)

        return self.bipartite_graphs
       
    def pretrain_bipartite_graphs(self, is_cuda):
        self.bipartite_graphs = []
        cur_cat = 0
        for i in range(0, self.n_datasets):
            this_bigraph = torch.zeros(self.dataset_cats[i], self.max_num_unify_class)
            for j in range(0, self.dataset_cats[i]):
                this_bigraph[j][cur_cat+j] = 1
            cur_cat += self.dataset_cats[i]
            
            if is_cuda:
                this_bigraph = this_bigraph.cuda()
            self.bipartite_graphs.append(this_bigraph)
            
        return self.bipartite_graphs     
            
    def init_adjacency_matrix(self):
        def normalize_adj(mx):
            # I = torch.eye(mx.shape[0])
            # if mx.is_cuda:
            #     I = I.cuda()
                
            # mx = mx - I
            rowsum = np.array(mx.sum(1))
            r_inv_sqrt = np.power(rowsum, -0.5).flatten()
            r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
            r_mat_inv_sqrt = torch.diag(torch.tensor(r_inv_sqrt))
            
            # r_mat_inv_sqrt = torch.diag(torch.tensor(r_inv_sqrt))
            # print(r_mat_inv_sqrt)
            return torch.mm(r_mat_inv_sqrt, torch.mm(mx, r_mat_inv_sqrt))
        
        # init adjacency matrix according to the number of categories of each dataset
        
        self.adj_matrix = torch.zeros(self.total_cats+self.max_num_unify_class, self.total_cats+self.max_num_unify_class)
        
        # self.adj_matrix[self.total_cats:, :] = 1
        # self.adj_matrix[:, self.total_cats:] = 1
        # self.adj_matrix[:self.total_cats, :self.total_cats] = torch.eye(self.total_cats)
        # self.adj_matrix[self.total_cats:, self.total_cats:] = torch.eye(self.max_num_unify_class)
        
        self.adj_matrix[self.total_cats:, :] = 1
        self.adj_matrix[:, self.total_cats:] = 1
        cur_cat = 0
        for i in range(0, self.n_datasets):
            self.adj_matrix[cur_cat:cur_cat+self.dataset_cats[i], self.total_cats+cur_cat:self.total_cats+cur_cat+self.dataset_cats[i]] = torch.eye(self.dataset_cats[i])
            self.adj_matrix[self.total_cats+cur_cat:self.total_cats+cur_cat+self.dataset_cats[i], cur_cat:cur_cat+self.dataset_cats[i]] = torch.eye(self.dataset_cats[i])
            # for j in range(0, self.dataset_cats[i]):
            #     self.adj_matrix[cur_cat+j, self.total_cats] = 0
            #     self.adj_matrix[self.total_cats+j+cur_cat, cur_cat+1:cur_cat+self.dataset_cats[i]] = 0
            #     self.adj_matrix[cur_cat+1:cur_cat+self.dataset_cats[i], self.total_cats+j+cur_cat] = 0
                
            # self.adj_matrix[self.total_cats:, cur_cat] = 1
            # self.adj_matrix[self.total_cats+i*self.max_num_unify_class:self.total_cats+(i+1)*self.max_num_unify_class, self.total_cats:] = 1
            cur_cat += self.dataset_cats[i]

        self.adj_matrix[:self.total_cats, :self.total_cats] = torch.eye(self.total_cats)
        self.adj_matrix[self.total_cats:, self.total_cats:] = torch.eye(self.max_num_unify_class)
        # torch.set_printoptions(profile="full")
        # print(self.adj_matrix)
        # torch.set_printoptions(profile="default") 
        self.adj_matrix = normalize_adj(self.adj_matrix)
        self.adj_matrix = self.adj_matrix.cuda()

    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.requires_grad == False:
                    continue
                
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4 or param.dim() == 2:
                    wd_params.append(param)
                else:
                    nowd_params.append(param)
                    # print(param.dim())
                    # print(param)
                    # print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
        

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


class Heter_GAT(nn.Module):
    def __init__(self, configer):
        """Dense version of GAT."""
        super(Heter_GAT, self).__init__()
        
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

        self.linear_before = nn.Linear(self.nfeat, self.nfeat_out)
        self.relu = nn.ReLU()

        self.attentions_layer1 = nn.ModuleList([GraphAttentionLayer(self.nfeat_out, self.nhid, dropout=self.dropout_rate, alpha=self.alpha, concat=True) for _ in range(self.nheads)])

        # self.attentions_layer2 = nn.ModuleList([GraphAttentionLayer(self.nhid * self.nheads, self.nhid, dropout=self.dropout_rate, alpha=self.alpha, concat=True) for _ in range(self.nheads)])

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
        
        # self.bipartite_graphs = []
        # for i in range(0, self.n_datasets):
        #     self.bipartite_graphs.append(nn.Parameter(torch.zeros(self.dataset_cats[i], self.max_num_unify_class), requires_grad=True))

        
        # self.register_buffer("fix_node_features", torch.randn(self.total_cats, self.nfeat))
        self.unify_node_features = nn.Parameter(torch.randn(self.max_num_unify_class, self.nfeat), requires_grad=True)
        trunc_normal_(self.unify_node_features, std=0.02)
        
        ## Graph adjacency matrix
        self.init_adjacency_matrix()
        
    def forward(self, x):
        x = self.linear_before(x)
        x = self.relu(x)
        # x = F.dropout(x, self.dropout_rate, training=self.training)
        feat = torch.cat([att(x, self.adj_matrix) for att in self.attentions_layer1], dim=1)
        x = feat + x
        # x = F.dropout(x, self.dropout_rate, training=self.training)
        # x = torch.cat([att(x, self.adj_matrix) for att in self.attentions_layer2], dim=1)
        # x = F.dropout(x, self.dropout_rate, training=self.training)
        x = F.elu(self.out_att(x, self.adj_matrix) + x)
        feat = self.linear1(x)
        arch_x = self.relu(x + feat)
        arch_x = self.linear2(arch_x)
        
        return feat[self.total_cats:], self.calc_bipartite_graph(arch_x)

    def calc_bipartite_graph(self, x):
        this_fix_arch = self.fix_arch
        cur_iter = self.configer.get('iter')
        if cur_iter < self.fix_architecture_alter_iter:
            self.linear2.requires_grad = False
            return self.pretrain_bipartite_graphs(is_cuda=x.is_cuda)
        
        if (cur_iter // self.fix_architecture_alter_iter) % 2 == 0:
            self.linear2.requires_grad = False
            self.fix_arch = False
        else:
            self.linear2.requires_grad = True
            self.fix_arch = True    
        
        if this_fix_arch:    
            return self.bipartite_graphs.detach()
        
        unify_feats = x[self.total_cats:]
        
        cur_cat = 0
        self.bipartite_graphs = []
        for i in range(0, self.n_datasets):
            this_feats = x[cur_cat:cur_cat+self.dataset_cats[i]]
            cur_cat += self.dataset_cats[i]
            similar_matrix = torch.einsum('nc, mc -> nm', this_feats, unify_feats)
            softmax_similar_matrix = F.softmax(similar_matrix / 0.05, dim=0)
            # softmax_similar_matrix[softmax_similar_matrix < self.threshold_value] = 0
            # max_value, max_index = torch.max(softmax_similar_matrix, dim=0)
            # self.bipartite_graphs[i] = torch.zeros(self.dataset_cats[i], self.max_num_unify_class, requires_grad=True)
            # if x.is_cuda:
            #     bi_graph = bi_graph.cuda()

            # self.bipartite_graphs[i][max_index] = 1
            
            # this_iter_thresh = 0.3 + (self.threshold_value - 0.3) * self.configer.get('iter') / self.configer.get('lr', 'max_iter')
            # this_iter_thresh = self.threshold_value * self.configer.get('iter') / self.configer.get('lr', 'max_iter')
            # bi_graph[:, max_value < this_iter_thresh] = 0
            
            
            self.bipartite_graphs.append(softmax_similar_matrix)

        return self.bipartite_graphs
       
    def pretrain_bipartite_graphs(self, is_cuda):
        self.bipartite_graphs = []
        cur_cat = 0
        for i in range(0, self.n_datasets):
            this_bigraph = torch.zeros(self.dataset_cats[i], self.max_num_unify_class)
            for j in range(0, self.dataset_cats[i]):
                this_bigraph[j][cur_cat+j] = 1
            cur_cat += self.dataset_cats[i]
            
            if is_cuda:
                this_bigraph = this_bigraph.cuda()
            self.bipartite_graphs.append(this_bigraph)
            
        return self.bipartite_graphs     
            
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
        
        self.adj_matrix = torch.zeros(self.total_cats+self.max_num_unify_class, self.total_cats+self.max_num_unify_class)
        
        # self.adj_matrix[self.total_cats:, :] = 1
        # self.adj_matrix[:, self.total_cats:] = 1
        # self.adj_matrix[:self.total_cats, :self.total_cats] = torch.eye(self.total_cats)
        # self.adj_matrix[self.total_cats:, self.total_cats:] = torch.eye(self.max_num_unify_class)
        
        self.adj_matrix[self.total_cats:, :] = 1
        self.adj_matrix[:, self.total_cats:] = 1
        cur_cat = 0
        for i in range(0, self.n_datasets):
            self.adj_matrix[cur_cat:cur_cat+self.dataset_cats[i], self.total_cats+cur_cat:self.total_cats+cur_cat+self.dataset_cats[i]] = torch.eye(self.dataset_cats[i])
            self.adj_matrix[self.total_cats+cur_cat:self.total_cats+cur_cat+self.dataset_cats[i], cur_cat:cur_cat+self.dataset_cats[i]] = torch.eye(self.dataset_cats[i])
            # for j in range(0, self.dataset_cats[i]):
            #     self.adj_matrix[cur_cat+j, self.total_cats] = 0
            #     self.adj_matrix[self.total_cats+j+cur_cat, cur_cat+1:cur_cat+self.dataset_cats[i]] = 0
            #     self.adj_matrix[cur_cat+1:cur_cat+self.dataset_cats[i], self.total_cats+j+cur_cat] = 0
                
            # self.adj_matrix[self.total_cats:, cur_cat] = 1
            # self.adj_matrix[self.total_cats+i*self.max_num_unify_class:self.total_cats+(i+1)*self.max_num_unify_class, self.total_cats:] = 1
            cur_cat += self.dataset_cats[i]

        self.adj_matrix[:self.total_cats, :self.total_cats] = torch.eye(self.total_cats)
        self.adj_matrix[self.total_cats:, self.total_cats:] = torch.eye(self.max_num_unify_class)
        # torch.set_printoptions(profile="full")
        # print(self.adj_matrix)
        # torch.set_printoptions(profile="default") 
        self.adj_matrix = normalize_adj(self.adj_matrix)
        self.adj_matrix = self.adj_matrix.cuda()

    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.requires_grad == False:
                    continue
                
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4 or param.dim() == 2:
                    wd_params.append(param)
                else:
                    nowd_params.append(param)
                    # print(param.dim())
                    # print(param)
                    # print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

class Learnable_Topology_GAT(nn.Module):
    def __init__(self, configer):
        """Dense version of GAT."""
        super(Learnable_Topology_GAT, self).__init__()
        
        self.configer = configer
        self.nfeat = self.configer.get('GNN', 'nfeat')
        self.nfeat_out = self.configer.get('GNN', 'nfeat_out')
        self.nfeat_adj = self.configer.get('GNN', 'nfeat_adj')
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
        self.calc_bipartite = self.configer.get('GNN', 'calc_bipartite')


        self.linear_before = nn.Linear(self.nfeat, self.nfeat_out)
        self.linear_adj = nn.Linear(self.nfeat_out, self.nfeat_adj)
        self.relu = nn.ReLU()

        self.attentions_layer1 = nn.ModuleList([GraphAttentionLayer(self.nfeat_out, self.nhid, dropout=self.dropout_rate, alpha=self.alpha, concat=True) for _ in range(self.nheads)])

        # self.attentions_layer2 = nn.ModuleList([GraphAttentionLayer(self.nhid * self.nheads, self.nhid, dropout=self.dropout_rate, alpha=self.alpha, concat=True) for _ in range(self.nheads)])

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
        
        # self.bipartite_graphs = []
        # for i in range(0, self.n_datasets):
        #     self.bipartite_graphs.append(nn.Parameter(torch.zeros(self.dataset_cats[i], self.max_num_unify_class), requires_grad=True))

        
        # self.register_buffer("fix_node_features", torch.randn(self.total_cats, self.nfeat))
        self.unify_node_features = nn.Parameter(torch.randn(self.max_num_unify_class, self.nfeat), requires_grad=True)
        trunc_normal_(self.unify_node_features, std=0.02)
        
        ## Graph adjacency matrix
        self.adj_matrix = nn.Parameter(torch.zeros(self.total_cats+self.max_num_unify_class, self.total_cats+self.max_num_unify_class), requires_grad=True)
        # self.init_adjacency_matrix()
        
    def forward(self, x):
        x = self.linear_before(x)
        adj = self.calc_adjacency_matrix(x)
        x = self.relu(x)
        # x = F.dropout(x, self.dropout_rate, training=self.training)
        feat = torch.cat([att(x, adj) for att in self.attentions_layer1], dim=1)
        x = feat + x
        x = F.dropout(x, self.dropout_rate, training=self.training)
        # x = torch.cat([att(x, self.adj_matrix) for att in self.attentions_layer2], dim=1)
        # x = F.dropout(x, self.dropout_rate, training=self.training)
        x = F.elu(self.out_att(x, adj) + x)
        feat = self.linear1(x)
        if self.calc_bipartite:
            arch_x = self.relu(x + feat)
            arch_x = self.linear2(arch_x)
            
            return feat[self.total_cats:], self.calc_bipartite_graph(arch_x)
        else:
            return feat[self.total_cats:], self.sep_bipartite_graphs(adj)

    def sep_bipartite_graphs(self, adj):
        self.bipartite_graphs = []
        cur_cat = 0
        for i in range(0, self.n_datasets):
            this_bipartite_graph = adj[cur_cat:cur_cat+self.dataset_cats[i], self.total_cats:]
            this_bipartite_graph = F.softmax(this_bipartite_graph/0.05, dim=0)
            self.bipartite_graphs.append(this_bipartite_graph)
            cur_cat += self.dataset_cats[i]
        return self.bipartite_graphs

    def calc_bipartite_graph(self, x):
        this_fix_arch = self.fix_arch
        cur_iter = self.configer.get('iter')
        if cur_iter < self.fix_architecture_alter_iter:
            self.linear2.requires_grad = False
            return self.pretrain_bipartite_graphs(is_cuda=x.is_cuda)
        
        if (cur_iter // self.fix_architecture_alter_iter) % 2 == 0:
            self.linear2.requires_grad = False
            self.fix_arch = False
        else:
            self.linear2.requires_grad = True
            self.fix_arch = True    
        
        if this_fix_arch:    
            return self.bipartite_graphs.detach()
        
        unify_feats = x[self.total_cats:]
        
        cur_cat = 0
        self.bipartite_graphs = []
        for i in range(0, self.n_datasets):
            this_feats = x[cur_cat:cur_cat+self.dataset_cats[i]]
            cur_cat += self.dataset_cats[i]
            similar_matrix = torch.einsum('nc, mc -> nm', this_feats, unify_feats)
            softmax_similar_matrix = F.softmax(similar_matrix / 0.05, dim=0)
            # softmax_similar_matrix[softmax_similar_matrix < self.threshold_value] = 0
            # max_value, max_index = torch.max(softmax_similar_matrix, dim=0)
            # self.bipartite_graphs[i] = torch.zeros(self.dataset_cats[i], self.max_num_unify_class, requires_grad=True)
            # if x.is_cuda:
            #     bi_graph = bi_graph.cuda()

            # self.bipartite_graphs[i][max_index] = 1
            
            # this_iter_thresh = 0.3 + (self.threshold_value - 0.3) * self.configer.get('iter') / self.configer.get('lr', 'max_iter')
            # this_iter_thresh = self.threshold_value * self.configer.get('iter') / self.configer.get('lr', 'max_iter')
            # bi_graph[:, max_value < this_iter_thresh] = 0
            
            
            self.bipartite_graphs.append(softmax_similar_matrix)

        return self.bipartite_graphs
       
    def pretrain_bipartite_graphs(self, is_cuda):
        self.bipartite_graphs = []
        cur_cat = 0
        for i in range(0, self.n_datasets):
            this_bigraph = torch.zeros(self.dataset_cats[i], self.max_num_unify_class)
            for j in range(0, self.dataset_cats[i]):
                this_bigraph[j][cur_cat+j] = 1
            cur_cat += self.dataset_cats[i]
            
            if is_cuda:
                this_bigraph = this_bigraph.cuda()
            self.bipartite_graphs.append(this_bigraph)
            
        return self.bipartite_graphs     
        
    def calc_adjacency_matrix(self, x):    

        adj_feat = self.linear_adj(x)
        norm_adj_feat = F.normalize(adj_feat, p=2, dim=1)
        similar_matrix = torch.einsum('nc, mc -> nm', norm_adj_feat, norm_adj_feat)
        
        def normalize_adj(mx):
        
            rowsum = mx.sum(1)
            r_inv_sqrt = torch.diag(1 / rowsum)
            r_inv_sqrt[r_inv_sqrt==torch.inf] = 0.
            
            if mx.is_cuda:
                r_inv_sqrt = r_inv_sqrt.cuda()
            
            # r_mat_inv_sqrt = torch.diag(torch.tensor(r_inv_sqrt))
            # print(r_mat_inv_sqrt)
            return torch.mm(r_inv_sqrt, mx)
        
        similar_matrix = normalize_adj(similar_matrix)
        return similar_matrix

    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.requires_grad == False:
                    continue
                
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4 or param.dim() == 2:
                    wd_params.append(param)
                else:
                    nowd_params.append(param)
                    # print(param.dim())
                    # print(param)
                    # print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
