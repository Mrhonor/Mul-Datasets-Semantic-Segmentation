import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from lib.module.module_helper import GraphAttentionLayer, SpGraphAttentionLayer, GraphConvolution, Discriminator, MultiHeadedAttention, AttentionalPropagation
from lib.module.sinkhorn import solve_optimal_transport
import numpy as np
import scipy.sparse as sp
from munkres import Munkres
import ot

class GCN(nn.Module):
    def __init__(self, infeat, outfeat):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(infeat, outfeat)

    def forward(self, x, adj):
        x = torch.tanh(self.gc1(x, adj))
        return x

    def aggregation(self, x, adj):
        x = self.gc1(x, adj)
        return x

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
        self.output_feat_dim = self.configer.get('GNN', 'output_feat_dim')
        
        self.adj_feat_dim = self.configer.get('GNN', 'adj_feat_dim')
        self.dropout_rate = self.configer.get('GNN', 'dropout_rate')
        self.threshold_value = self.configer.get('GNN', 'threshold_value')
        self.fix_arch = False
        self.fix_architecture_alter_iter = self.configer.get('GNN', 'fix_architecture_alter_iter')

        self.linear_before = nn.Linear(self.nfeat, self.nfeat_out)
        self.relu = nn.ReLU()

        self.attentions_layer1 = nn.ModuleList([GraphAttentionLayer(self.nfeat_out, self.nhid, dropout=self.dropout_rate, alpha=self.alpha, concat=True) for _ in range(self.nheads)])

        # self.attentions_layer2 = nn.ModuleList([GraphAttentionLayer(self.nhid * self.nheads, self.nhid, dropout=self.dropout_rate, alpha=self.alpha, concat=True) for _ in range(self.nheads)])

        self.out_att = GraphAttentionLayer(self.nhid * self.nheads, self.att_out_dim, dropout=self.dropout_rate, alpha=self.alpha, concat=False)
        
        self.linear1 = nn.Linear(self.att_out_dim, self.output_feat_dim)
        
        self.linear2 = nn.Linear(self.att_out_dim, self.adj_feat_dim) 
        ## datasets Node features
        self.n_datasets = self.configer.get('n_datasets')
        self.total_cats = 0
        self.dataset_cats = []
        for i in range(0, self.n_datasets):
            self.dataset_cats.append(self.configer.get('dataset'+str(i+1), 'n_cats'))
            self.total_cats += self.configer.get('dataset'+str(i+1), 'n_cats')
        print("self.total_cats:", self.total_cats)
        
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
        x = torch.cat([x, self.unify_node_features], dim=0)
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
        arch_x = self.relu(x)
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
            return [bigh.detach() for bigh in self.bipartite_graphs]
        
        unify_feats = x[self.total_cats:]
        
        cur_cat = 0
        self.bipartite_graphs = []
        for i in range(0, self.n_datasets):
            this_feats = x[cur_cat:cur_cat+self.dataset_cats[i]]
            cur_cat += self.dataset_cats[i]
            similar_matrix = torch.einsum('nc, mc -> nm', this_feats, unify_feats)
            # print("similar_matrix.shape:", similar_matrix.shape)
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
        
class Self_Attention_GNN(nn.Module):
    def __init__(self, configer):
        """Dense version of GAT."""
        super(Self_Attention_GNN, self).__init__()
        
        self.configer = configer
        self.nfeat = self.configer.get('GNN', 'nfeat')
        self.nfeat_out = self.configer.get('GNN', 'nfeat_out')
        self.nhid = self.configer.get('GNN', 'nhid')
        self.att_out_dim = self.configer.get('GNN', 'att_out_dim')
        self.alpha = self.configer.get('GNN', 'alpha')
        self.nheads = self.configer.get('GNN', 'nheads')
        self.adj_feat_dim = self.configer.get('GNN', 'adj_feat_dim')
        
        self.output_feat_dim = self.configer.get('GNN', 'output_feat_dim')
        self.dropout_rate = self.configer.get('GNN', 'dropout_rate')
        self.threshold_value = self.configer.get('GNN', 'threshold_value')
        self.fix_arch = False
        self.fix_architecture_alter_iter = self.configer.get('GNN', 'fix_architecture_alter_iter')

        self.linear_before = nn.Linear(self.nfeat, self.nfeat_out)
        self.relu = nn.ReLU()

        # self.attentions_layer1 = nn.ModuleList([GraphAttentionLayer(self.nfeat_out, self.nhid, dropout=self.dropout_rate, alpha=self.alpha, concat=True) for _ in range(self.nheads)])
        self.attentions_layer1 = AttentionalPropagation(self.nfeat_out, self.nheads)

        # self.attentions_layer2 = nn.ModuleList([GraphAttentionLayer(self.nhid * self.nheads, self.nhid, dropout=self.dropout_rate, alpha=self.alpha, concat=True) for _ in range(self.nheads)])

        self.out_att = AttentionalPropagation(self.nfeat_out, self.nheads)
        
        self.linear1 = nn.Linear(self.nfeat_out, self.output_feat_dim)
        
        self.arch_linear = nn.Linear(self.nfeat_out, self.adj_feat_dim) 
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
        x = torch.cat([x, self.unify_node_features], dim=0)
        x = self.linear_before(x)
        x = self.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)
        feat = self.attentions_layer1(x, x, self.adj_matrix)
        # x = feat + x
        x = F.dropout(feat, self.dropout_rate, training=self.training)
        # x = torch.cat([att(x, self.adj_matrix) for att in self.attentions_layer2], dim=1)
        # x = F.dropout(x, self.dropout_rate, training=self.training)
        x = F.elu(self.out_att(x, x, self.adj_matrix) + x)
        feat = self.linear1(x)
        arch_x = self.relu(x)
        arch_x = self.arch_linear(arch_x)
        
        return feat[self.total_cats:], self.calc_bipartite_graph(arch_x)

    def calc_bipartite_graph(self, x):
        this_fix_arch = self.fix_arch
        cur_iter = self.configer.get('iter')
        if cur_iter < self.fix_architecture_alter_iter:
            self.arch_linear.requires_grad = False
            return self.pretrain_bipartite_graphs(is_cuda=x.is_cuda)
        
        if (cur_iter // self.fix_architecture_alter_iter) % 2 == 0:
            self.arch_linear.requires_grad = False
            self.fix_arch = False
        else:
            self.arch_linear.requires_grad = True
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
        x = torch.cat([x, self.unify_node_features], dim=0)
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
        """learnable-topology version of GAT. Output Adjacency Matrix is learnable"""
        super(Learnable_Topology_GAT, self).__init__()
        
        self.configer = configer
        self.nfeat = self.configer.get('GNN', 'nfeat')
        self.nfeat_out = self.configer.get('GNN', 'nfeat_out')
        self.nfeat_adj = self.configer.get('GNN', 'nfeat_adj')
        self.nhid = self.configer.get('GNN', 'nhid')
        self.att_out_dim = self.configer.get('GNN', 'att_out_dim')
        self.alpha = self.configer.get('GNN', 'alpha')
        self.nheads = self.configer.get('GNN', 'nheads')
        self.adj_feat_dim = self.configer.get('GNN', 'adj_feat_dim')
        
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
        
        self.linear1 = nn.Linear(self.att_out_dim, self.output_feat_dim)
        
        self.linear2 = nn.Linear(self.output_feat_dim, self.adj_feat_dim) 
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
        x = torch.cat([x, self.unify_node_features], dim=0)
        x = self.linear_before(x)
        adj, non_norm_adj = self.calc_adjacency_matrix(x)
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
            return feat[self.total_cats:], self.sep_bipartite_graphs(non_norm_adj)

    def sep_bipartite_graphs(self, adj):
        self.bipartite_graphs = []
        cur_cat = 0
        for i in range(0, self.n_datasets):
            this_bipartite_graph = adj[cur_cat:cur_cat+self.dataset_cats[i], self.total_cats:]
            this_bipartite_graph = F.softmax(this_bipartite_graph/0.07, dim=0)
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
        
        norm_matrix = normalize_adj(similar_matrix)

        return norm_matrix, similar_matrix

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


class Learnable_Topology_BGNN(nn.Module):
    def __init__(self, configer):
        """Dense version of GAT."""
        super(Learnable_Topology_BGNN, self).__init__()
        
        self.configer = configer
        self.nfeat = self.configer.get('GNN', 'nfeat')
        self.nfeat_out = self.configer.get('GNN', 'nfeat_out')
        self.nfeat_adj = self.configer.get('GNN', 'nfeat_adj')
        self.nhid = self.configer.get('GNN', 'nhid')
        self.att_out_dim = self.configer.get('GNN', 'att_out_dim')
        self.alpha = self.configer.get('GNN', 'alpha')
        self.nheads = self.configer.get('GNN', 'nheads')
        self.adj_feat_dim = self.configer.get('GNN', 'adj_feat_dim')
        
        self.output_feat_dim = self.configer.get('GNN', 'output_feat_dim')
        self.dropout_rate = self.configer.get('GNN', 'dropout_rate')
        self.threshold_value = self.configer.get('GNN', 'threshold_value')
        self.fix_arch = False
        self.fix_architecture_alter_iter = self.configer.get('GNN', 'fix_architecture_alter_iter')
        self.calc_bipartite = self.configer.get('GNN', 'calc_bipartite')
        self.output_max_adj = self.configer.get('GNN', 'output_max_adj')
        self.output_softmax_and_max_adj = self.configer.get('GNN', 'output_softmax_and_max_adj')

        self.linear_before = nn.Linear(self.nfeat, self.nfeat_out)
        self.linear_adj = nn.Linear(self.nfeat_out, self.nfeat_adj)
        if self.calc_bipartite:
            self.linear_adj2 = nn.Linear(self.adj_feat_dim, self.adj_feat_dim)
            
            
        self.relu = nn.ReLU()

        self.GCN_layer1 = GCN(self.nfeat_out, self.nfeat_out)

        # self.attentions_layer2 = nn.ModuleList([GraphAttentionLayer(self.nhid * self.nheads, self.nhid, dropout=self.dropout_rate, alpha=self.alpha, concat=True) for _ in range(self.nheads)])

        self.GCN_layer2 = GCN(self.nfeat_out, self.nfeat_out)
        
        self.linear1 = nn.Linear(self.nfeat_out, self.output_feat_dim)
        
        self.linear2 = nn.Linear(self.output_feat_dim, self.adj_feat_dim) 
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
        self.adj_matrix = nn.Parameter(torch.zeros(self.total_cats+self.max_num_unify_class, self.total_cats+self.max_num_unify_class), requires_grad=True)
        # self.init_adjacency_matrix()
        self.netD1 = Discriminator(self.nfeat_out, 128, 1, self.dropout_rate)
        self.netD1.weights_init()
        self.netD2 = Discriminator(self.nfeat_out, 128, 1, self.dropout_rate)
        self.netD2.weights_init()
        
        self.use_km = True
        if self.use_km:
            self.km_algorithms = Munkres()
        # else:
            
        self.beta = [ot.unif(self.dataset_cats[i]) for i in range(0, self.n_datasets)]
        
        
    def forward(self, x, pretraining=False):
        x = torch.cat([x, self.unify_node_features], dim=0)
        
        feat1 = self.linear_before(x)
        adj_mI, non_norm_adj_mI = self.calc_adjacency_matrix(feat1)
        feat1_relu = self.relu(feat1)
        
        before_gcn1_x = F.dropout(feat1_relu, self.dropout_rate, training=self.training)
        feat_gcn1 = self.GCN_layer1(before_gcn1_x, adj_mI)
        out_real_1 = self.netD1(before_gcn1_x.detach())
        out_fake_1 = self.netD1(feat_gcn1.detach())
        g_out_fake_1 = self.netD1(feat_gcn1)
        
        feat2 = feat_gcn1 + before_gcn1_x
        before_gcn2_x = F.dropout(feat2, self.dropout_rate, training=self.training)
        feat_gcn2 = self.GCN_layer2(before_gcn2_x, adj_mI)
        out_real_2 = self.netD2(before_gcn2_x.detach())
        out_fake_2 = self.netD2(feat_gcn2.detach())
        g_out_fake_2 = self.netD2(feat_gcn2)
        
        feat3 = F.elu(feat_gcn2 + before_gcn2_x)
        # feat3_drop = F.dropout(feat3, self.dropout_rate, training=self.training)
        feat_out = self.linear1(feat3)

        adv_out = {}
        adv_out['ADV1'] = [out_real_1, out_fake_1, g_out_fake_1]
        adv_out['ADV2'] = [out_real_2, out_fake_2, g_out_fake_2]
        if pretraining:
            return feat_out[self.total_cats:], self.sep_bipartite_graphs(non_norm_adj_mI), adv_out, non_norm_adj_mI
        elif self.calc_bipartite:
            arch_x = self.relu(feat3 + feat_out)
            arch_x = self.linear2(arch_x)
            _, non_norm_adj_mI_after = self.calc_adjacency_matrix(arch_x)
            
            return feat_out[self.total_cats:], self.sep_bipartite_graphs(non_norm_adj_mI_after), adv_out, non_norm_adj_mI_after
        else:
            return feat_out[self.total_cats:], self.sep_bipartite_graphs(non_norm_adj_mI), adv_out, non_norm_adj_mI

    def sep_bipartite_graphs(self, adj):
        self.bipartite_graphs = []
        cur_cat = 0
        for i in range(0, self.n_datasets):
            this_bipartite_graph = adj[cur_cat:cur_cat+self.dataset_cats[i], self.total_cats:]

            if self.output_max_adj:
                # 找到每列的最大值
                max_values, _ = torch.max(this_bipartite_graph, dim=0)

                # 创建掩码矩阵，将每列的最大值位置置为1，其余位置置为0
                mask = torch.zeros_like(this_bipartite_graph)
                mask[this_bipartite_graph == max_values] = 1
                max_bipartite_graph = this_bipartite_graph * mask
                self.bipartite_graphs.append(max_bipartite_graph)
                
            if self.output_softmax_and_max_adj or not self.output_max_adj:
                softmax_bipartite_graph = F.softmax(this_bipartite_graph/0.07, dim=0)

                self.bipartite_graphs.append(softmax_bipartite_graph)
            
            cur_cat += self.dataset_cats[i]
        
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
        
        if x.size(1) == self.nfeat_out:
            adj_feat = self.linear_adj(x)
        else:
            adj_feat = self.linear_adj2(x)
        norm_adj_feat = F.normalize(adj_feat, p=2, dim=1)
        similar_matrix = torch.einsum('nc, mc -> nm', norm_adj_feat, norm_adj_feat)
        adj_mI = similar_matrix - torch.diag(torch.diag(similar_matrix))
        
        def normalize_adj(mx):
        
            rowsum = mx.sum(1)
            r_inv_sqrt = torch.diag(1 / rowsum)
            r_inv_sqrt[r_inv_sqrt==torch.inf] = 0.
            
            if mx.is_cuda:
                r_inv_sqrt = r_inv_sqrt.cuda()
            
            # r_mat_inv_sqrt = torch.diag(torch.tensor(r_inv_sqrt))
            # print(r_mat_inv_sqrt)
            return torch.mm(r_inv_sqrt, mx)
        
        # similar_matrix = normalize_adj(similar_matrix)
        norm_adj_mI = normalize_adj(adj_mI)
        return norm_adj_mI, adj_mI
    
    def get_optimal_matching(self, x, init=False):
        x = torch.cat([x, self.unify_node_features], dim=0)
        
        feat1 = self.linear_before(x)
        adj_mI, non_norm_adj_mI = self.calc_adjacency_matrix(feat1)
        feat1_relu = self.relu(feat1)
        
        feat_gcn1 = self.GCN_layer1(feat1_relu, adj_mI)
        
        feat2 = feat_gcn1 + feat1_relu
        feat_gcn2 = self.GCN_layer2(feat2, adj_mI)
        
        feat3 = F.elu(feat_gcn2 + feat2)
        feat_out = self.linear1(feat3)

        if init:
            if self.calc_bipartite:
                arch_x = self.relu(feat3 + feat_out)
                arch_x = self.linear2(arch_x)
                _, non_norm_adj_mI_after = self.calc_adjacency_matrix(arch_x)
                
                return feat_out[self.total_cats:], self.sep_bipartite_graphs_by_km(non_norm_adj_mI_after)
            else:
                # return feat_out[self.total_cats:], self.sep_bipartite_graphs(non_norm_adj_mI)
                return feat_out[self.total_cats:], self.sep_bipartite_graphs_by_uot(non_norm_adj_mI)
                # return feat_out[self.total_cats:], self.sep_bipartite_graphs_by_km(non_norm_adj_mI)
        else:
            return feat_out[self.total_cats:], self.pretrain_bipartite_graphs(x.is_cuda)

    def sep_bipartite_graphs_by_km(self, adj):
        self.bipartite_graphs = []
        cur_cat = 0
        for i in range(0, self.n_datasets):
            this_bipartite_graph = adj[cur_cat:cur_cat+self.dataset_cats[i], self.total_cats:]
            this_bipartite_graph = this_bipartite_graph.detach()
            if self.use_km:
                indexes = self.km_algorithms.compute(-this_bipartite_graph.cpu().numpy())
                out_bipartite_graphs = torch.zeros_like(this_bipartite_graph)
                
                for j in range(0, self.max_num_unify_class):
                    flag = False
                    for row, col in indexes:
                        if col == j:
                            flag = True
                            out_bipartite_graphs[row, col] = 1
                            
                    if not flag:
                        max_index = torch.argmax(this_bipartite_graph[:,j])
                        out_bipartite_graphs[max_index, j] = 1
            else:

                res = solve_optimal_transport(this_bipartite_graph[None], 100, -10)

                indexes = res['matches1']

                out_bipartite_graphs = torch.zeros_like(this_bipartite_graph)
                for j, idx in enumerate(indexes[0]):
                    if idx == -1:
                        max_index = torch.argmax(this_bipartite_graph[:,j])
                        out_bipartite_graphs[max_index, j] = 1
                    else:
                        out_bipartite_graphs[idx, j] = 1


            self.bipartite_graphs.append(out_bipartite_graphs) 
                
            cur_cat += self.dataset_cats[i]
        
        return self.bipartite_graphs 
    
    def sep_bipartite_graphs_by_uot(self, adj):
        self.bipartite_graphs = []
        cur_cat = 0
        for i in range(0, self.n_datasets):
            this_bipartite_graph = adj[cur_cat:cur_cat+self.dataset_cats[i], self.total_cats:]
            this_bipartite_graph = (-this_bipartite_graph.detach().clone()+1 + 1e-8)/2
            out_bipartite_graphs = torch.zeros_like(this_bipartite_graph)

            alpha = ot.unif(self.max_num_unify_class)
                
            Q_st = ot.unbalanced.sinkhorn_knopp_unbalanced(alpha, self.beta[i], this_bipartite_graph.T.cpu().numpy(), 
                                                            reg=0.01, reg_m=0.5, stopThr=1e-4) 

            Q_st = torch.from_numpy(Q_st).float().cuda()

            # make sum equals to 1
            sum_pi = torch.sum(Q_st)
            Q_st_bar = Q_st/sum_pi
            # print(Q_st_bar.shape)
            # print(out_bipartite_graphs.shape)
            
            # # highly confident target samples selected by statistics mean
            # if mode == 'minibatch':
            #     Q_anchor = Q_st_bar[fake_size+fill_size:, :]
            # if mode == 'all':
            #     Q_anchor = Q_st_bar

            # # confidence score w^t_i
            wt_i, pseudo_label = torch.max(Q_st_bar, 1)
            # print(pseudo_label)
            for col, index in enumerate(pseudo_label):
                out_bipartite_graphs[index, col] = 1

            for row in range(0, self.dataset_cats[i]):
                if torch.sum(out_bipartite_graphs[row]) == 0:
                    # print(f'find miss one in UOT, datasets:{i}, row:{row}')
                    sorted_tensor, indices = torch.sort(Q_st_bar.T[row])

                    flag = False
                    for ori_index in indices:
                        # map_lb = pseudo_label[ori_index]
                        map_lb = [i for i, x in enumerate(out_bipartite_graphs[:, ori_index]) if x == 1]
                        if len(map_lb) != 1:
                            print("!")
                        map_lb = map_lb[0]
                        if torch.sum(out_bipartite_graphs[map_lb, :]) > 1:

                            # print(ori_index)
                            # print(row)
                            # print(map_lb)
                            # print(out_bipartite_graphs[:,ori_index])
                            out_bipartite_graphs[row, ori_index] = 1
                            out_bipartite_graphs[map_lb, ori_index] = 0
                            # print(out_bipartite_graphs[:,ori_index])
                            # print(out_bipartite_graphs[row, ori_index])
                            # print(out_bipartite_graphs[map_lb, ori_index])
                            flag = True
                            break
                    if flag is False:
                        print("error don't find correct one")
                    
            new_beta = torch.sum(Q_st_bar,0).cpu().numpy()


            mu = 0.7
            self.beta[i] = mu*self.beta[i] + (1-mu)*new_beta
            # print(out_bipartite_graphs)
            # print(torch.max(out_bipartite_graphs, dim=0))
            self.bipartite_graphs.append(out_bipartite_graphs) 
                
            cur_cat += self.dataset_cats[i]
        
        return self.bipartite_graphs 

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
            if 'netD' in name:
                continue
            elif 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

    def get_discri_params(self):
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
            print("out_name: ", name)
            if 'netD' in name:
                print(name)
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

    def set_unify_node_features(self, unify_node_features, grad=True):
        self.unify_node_features = nn.Parameter(unify_node_features, requires_grad=grad)

# class Learnable_Topology_BGNN_2(nn.Module):
    # def __init__(self, configer):
    #     """Dense version of GAT."""
    #     super(Learnable_Topology_BGNN_2, self).__init__()
        
    #     self.configer = configer
    #     self.nfeat = self.configer.get('GNN', 'nfeat')
    #     self.nfeat_out = self.configer.get('GNN', 'nfeat_out')
    #     self.nfeat_adj = self.configer.get('GNN', 'nfeat_adj')
    #     self.nhid = self.configer.get('GNN', 'nhid')
    #     self.att_out_dim = self.configer.get('GNN', 'att_out_dim')
    #     self.alpha = self.configer.get('GNN', 'alpha')
    #     self.nheads = self.configer.get('GNN', 'nheads')
    #     self.adj_feat_dim = self.configer.get('GNN', 'adj_feat_dim')
        
    #     self.output_feat_dim = self.configer.get('GNN', 'output_feat_dim')
    #     self.dropout_rate = self.configer.get('GNN', 'dropout_rate')
    #     self.threshold_value = self.configer.get('GNN', 'threshold_value')
    #     self.fix_arch = False
    #     self.fix_architecture_alter_iter = self.configer.get('GNN', 'fix_architecture_alter_iter')
    #     self.calc_bipartite = self.configer.get('GNN', 'calc_bipartite')
    #     self.output_max_adj = self.configer.get('GNN', 'output_max_adj')
    #     self.output_softmax_and_max_adj = self.configer.get('GNN', 'output_softmax_and_max_adj')

    #     self.linear_before = nn.Linear(self.nfeat, self.nfeat_out)
    #     self.linear_adj = nn.Linear(self.nfeat_out, self.nfeat_adj)
    #     self.relu = nn.ReLU()

    #     # self.GCN_layer1 = GCN(self.nfeat_out, self.nfeat_out)

    #     # # self.attentions_layer2 = nn.ModuleList([GraphAttentionLayer(self.nhid * self.nheads, self.nhid, dropout=self.dropout_rate, alpha=self.alpha, concat=True) for _ in range(self.nheads)])

    #     # self.GCN_layer2 = GCN(self.nfeat_out, self.nfeat_out)

    #     self.GCN_layer1 = AttentionalPropagation(self.nfeat_out, self.nheads)

    #     # self.attentions_layer2 = nn.ModuleList([GraphAttentionLayer(self.nhid * self.nheads, self.nhid, dropout=self.dropout_rate, alpha=self.alpha, concat=True) for _ in range(self.nheads)])

    #     self.GCN_layer2 = AttentionalPropagation(self.nfeat_out, self.nheads)
        
    #     self.linear1 = nn.Linear(self.nfeat_out, self.output_feat_dim)
        
    #     self.linear2 = nn.Linear(self.output_feat_dim, self.adj_feat_dim) 
    #     ## datasets Node features
    #     self.n_datasets = self.configer.get('n_datasets')
    #     self.total_cats = 0
    #     self.dataset_cats = []
    #     for i in range(0, self.n_datasets):
    #         self.dataset_cats.append(self.configer.get('dataset'+str(i+1), 'n_cats'))
    #         self.total_cats += self.configer.get('dataset'+str(i+1), 'n_cats')
        
    #     self.max_num_unify_class = int(self.configer.get('GNN', 'unify_ratio') * self.total_cats)
        
    #     # self.register_buffer("fix_node_features", torch.randn(self.total_cats, self.nfeat))
    #     self.unify_node_features = nn.Parameter(torch.randn(self.max_num_unify_class, self.nfeat), requires_grad=True)
    #     trunc_normal_(self.unify_node_features, std=0.02)
        
    #     ## Graph adjacency matrix
    #     self.adj_matrix = nn.Parameter(torch.zeros(self.total_cats+self.max_num_unify_class, self.total_cats+self.max_num_unify_class), requires_grad=True)
    #     # self.init_adjacency_matrix()
    #     self.netD1 = Discriminator(self.nfeat_out, 128, 1, self.dropout_rate)
    #     self.netD1.weights_init()
    #     self.netD2 = Discriminator(self.nfeat_out, 128, 1, self.dropout_rate)
    #     self.netD2.weights_init()
        
    #     self.km_algorithms = Munkres()
        
        
    # def forward(self, x):
    #     x = torch.cat([x, self.unify_node_features], dim=0)
        
    #     feat1 = self.linear_before(x)
    #     adj_mI, non_norm_adj_mI = self.calc_adjacency_matrix(feat1)
    #     feat1_relu = self.relu(feat1)
        
    #     before_gcn1_x = F.dropout(feat1_relu, self.dropout_rate, training=self.training)
    #     feat_gcn1 = self.GCN_layer1(before_gcn1_x, before_gcn1_x, adj_mI)
    #     out_real_1 = self.netD1(before_gcn1_x.detach())
    #     out_fake_1 = self.netD1(feat_gcn1.detach())
    #     g_out_fake_1 = self.netD1(feat_gcn1)
        
    #     feat2 = self.relu(feat_gcn1) + before_gcn1_x
    #     before_gcn2_x = F.dropout(feat2, self.dropout_rate, training=self.training)
    #     feat_gcn2 = self.GCN_layer2(before_gcn2_x, before_gcn2_x, adj_mI)
    #     out_real_2 = self.netD2(before_gcn2_x.detach())
    #     out_fake_2 = self.netD2(feat_gcn2.detach())
    #     g_out_fake_2 = self.netD2(feat_gcn2)
        
    #     feat3 = F.elu(self.relu(feat_gcn2) + before_gcn2_x)
    #     feat3_drop = F.dropout(feat3, self.dropout_rate, training=self.training)
    #     feat_out = self.linear1(feat3_drop)

    #     adv_out = {}
    #     adv_out['ADV1'] = [out_real_1, out_fake_1, g_out_fake_1]
    #     adv_out['ADV2'] = [out_real_2, out_fake_2, g_out_fake_2]
    #     if self.calc_bipartite:
    #         arch_x = self.relu(feat3_drop + feat_out)
    #         arch_x = self.linear2(arch_x)
            
    #         return feat_out[self.total_cats:], self.calc_bipartite_graph(arch_x), adv_out
    #     else:
    #         return feat_out[self.total_cats:], self.sep_bipartite_graphs(non_norm_adj_mI), adv_out

    # def sep_bipartite_graphs(self, adj):
    #     self.bipartite_graphs = []
    #     cur_cat = 0
    #     for i in range(0, self.n_datasets):
    #         this_bipartite_graph = adj[cur_cat:cur_cat+self.dataset_cats[i], self.total_cats:]
    #         if self.output_max_adj:
    #             # 找到每列的最大值
    #             max_values, _ = torch.max(this_bipartite_graph, dim=0)

    #             # 创建掩码矩阵，将每列的最大值位置置为1，其余位置置为0
    #             mask = torch.zeros_like(this_bipartite_graph)
    #             mask[this_bipartite_graph == max_values] = 1
    #             max_bipartite_graph = this_bipartite_graph * mask
    #             self.bipartite_graphs.append(max_bipartite_graph)
                
    #         if self.output_softmax_and_max_adj or not self.output_max_adj:
    #             softmax_bipartite_graph = F.softmax(this_bipartite_graph/0.07, dim=0)
    #             self.bipartite_graphs.append(softmax_bipartite_graph)
            
    #         cur_cat += self.dataset_cats[i]
        
    #     return self.bipartite_graphs

    # def calc_bipartite_graph(self, x):
    #     this_fix_arch = self.fix_arch
    #     cur_iter = self.configer.get('iter')
    #     if cur_iter < self.fix_architecture_alter_iter:
    #         self.linear2.requires_grad = False
    #         return self.pretrain_bipartite_graphs(is_cuda=x.is_cuda)
        
    #     if (cur_iter // self.fix_architecture_alter_iter) % 2 == 0:
    #         self.linear2.requires_grad = False
    #         self.fix_arch = False
    #     else:
    #         self.linear2.requires_grad = True
    #         self.fix_arch = True    
        
    #     if this_fix_arch:    
    #         return self.bipartite_graphs.detach()
        
    #     unify_feats = x[self.total_cats:]
        
    #     cur_cat = 0
    #     self.bipartite_graphs = []
    #     for i in range(0, self.n_datasets):
    #         this_feats = x[cur_cat:cur_cat+self.dataset_cats[i]]
    #         cur_cat += self.dataset_cats[i]
    #         similar_matrix = torch.einsum('nc, mc -> nm', this_feats, unify_feats)
    #         softmax_similar_matrix = F.softmax(similar_matrix / 0.05, dim=0)
    #         # softmax_similar_matrix[softmax_similar_matrix < self.threshold_value] = 0
    #         # max_value, max_index = torch.max(softmax_similar_matrix, dim=0)
    #         # self.bipartite_graphs[i] = torch.zeros(self.dataset_cats[i], self.max_num_unify_class, requires_grad=True)
    #         # if x.is_cuda:
    #         #     bi_graph = bi_graph.cuda()

    #         # self.bipartite_graphs[i][max_index] = 1
            
    #         # this_iter_thresh = 0.3 + (self.threshold_value - 0.3) * self.configer.get('iter') / self.configer.get('lr', 'max_iter')
    #         # this_iter_thresh = self.threshold_value * self.configer.get('iter') / self.configer.get('lr', 'max_iter')
    #         # bi_graph[:, max_value < this_iter_thresh] = 0
            
            
    #         self.bipartite_graphs.append(softmax_similar_matrix)

    #     return self.bipartite_graphs
       
    # def pretrain_bipartite_graphs(self, is_cuda):
    #     self.bipartite_graphs = []
    #     cur_cat = 0
    #     for i in range(0, self.n_datasets):
    #         this_bigraph = torch.zeros(self.dataset_cats[i], self.max_num_unify_class)
    #         for j in range(0, self.dataset_cats[i]):
    #             this_bigraph[j][cur_cat+j] = 1
    #         cur_cat += self.dataset_cats[i]
            
    #         if is_cuda:
    #             this_bigraph = this_bigraph.cuda()
    #         self.bipartite_graphs.append(this_bigraph)
            
    #     return self.bipartite_graphs     
        
    # def calc_adjacency_matrix(self, x):    

    #     adj_feat = self.linear_adj(x)
    #     norm_adj_feat = F.normalize(adj_feat, p=2, dim=1)
    #     similar_matrix = torch.einsum('nc, mc -> nm', norm_adj_feat, norm_adj_feat)
    #     adj_mI = similar_matrix - torch.diag(torch.diag(similar_matrix))
        
    #     def normalize_adj(mx):
        
    #         rowsum = mx.sum(1)
    #         r_inv_sqrt = torch.diag(1 / rowsum)
    #         r_inv_sqrt[r_inv_sqrt==torch.inf] = 0.
            
    #         if mx.is_cuda:
    #             r_inv_sqrt = r_inv_sqrt.cuda()
            
    #         # r_mat_inv_sqrt = torch.diag(torch.tensor(r_inv_sqrt))
    #         # print(r_mat_inv_sqrt)
    #         return torch.mm(r_inv_sqrt, mx)
        
    #     # similar_matrix = normalize_adj(similar_matrix)
    #     norm_adj_mI = normalize_adj(adj_mI)
    #     return norm_adj_mI, adj_mI
    
    # def get_optimal_matching(self, x):
    #     x = torch.cat([x, self.unify_node_features], dim=0)
        
    #     feat1 = self.linear_before(x)
    #     adj_mI, non_norm_adj_mI = self.calc_adjacency_matrix(feat1)
    #     feat1_relu = self.relu(feat1)
        
    #     feat_gcn1 = self.GCN_layer1(feat1_relu, feat1_relu, adj_mI)
        
    #     feat2 = feat_gcn1 + feat1_relu
    #     feat_gcn2 = self.GCN_layer2(feat2, feat2, adj_mI)
        
    #     feat3 = F.elu(feat_gcn2 + feat2)
    #     feat_out = self.linear1(feat3)

    #     return feat_out[self.total_cats:], self.sep_bipartite_graphs_by_km(non_norm_adj_mI)

    # def sep_bipartite_graphs_by_km(self, adj):
    #     self.bipartite_graphs = []
    #     cur_cat = 0
    #     for i in range(0, self.n_datasets):
    #         this_bipartite_graph = adj[cur_cat:cur_cat+self.dataset_cats[i], self.total_cats:]
    #         ## TODO: use km to get the bipartite graph
    #         indexes = self.km_algorithms.compute(-this_bipartite_graph.detach().cpu().numpy())
    #         out_bipartite_graphs = torch.zeros_like(this_bipartite_graph)
            
    #         for j in range(0, self.max_num_unify_class):
    #             flag = False
    #             for row, col in indexes:
    #                 if col == j:
    #                     flag = True
    #                     out_bipartite_graphs[row, col] = 1
                        
    #             if not flag:
    #                 max_index = torch.argmax(this_bipartite_graph[:,j])
    #                 out_bipartite_graphs[max_index, j] = 1
                
    #         self.bipartite_graphs.append(out_bipartite_graphs) 
                
    #         cur_cat += self.dataset_cats[i]
        
    #     return self.bipartite_graphs 

    # def get_params(self):
    #     def add_param_to_list(mod, wd_params, nowd_params):
    #         for param in mod.parameters():
    #             if param.requires_grad == False:
    #                 continue
                
    #             if param.dim() == 1:
    #                 nowd_params.append(param)
    #             elif param.dim() == 4 or param.dim() == 2:
    #                 wd_params.append(param)
    #             else:
    #                 nowd_params.append(param)
    #                 # print(param.dim())
    #                 # print(param)
    #                 # print(name)

    #     wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
    #     for name, child in self.named_children():
    #         if 'netD' in name:
    #             continue
    #         elif 'head' in name or 'aux' in name:
    #             add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
    #         else:
    #             add_param_to_list(child, wd_params, nowd_params)
    #     return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

    # def get_discri_params(self):
    #     def add_param_to_list(mod, wd_params, nowd_params):
    #         for param in mod.parameters():
    #             if param.requires_grad == False:
    #                 continue
                
    #             if param.dim() == 1:
    #                 nowd_params.append(param)
    #             elif param.dim() == 4 or param.dim() == 2:
    #                 wd_params.append(param)
    #             else:
    #                 nowd_params.append(param)
    #                 # print(param.dim())
    #                 # print(param)
    #                 # print(name)

    #     wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
    #     for name, child in self.named_children():
    #         if 'netD' in name:
    #             add_param_to_list(child, wd_params, nowd_params)
    #     return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params