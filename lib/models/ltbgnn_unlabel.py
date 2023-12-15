import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from lib.module.module_helper import GraphAttentionLayer, SpGraphAttentionLayer, GraphConvolution, Discriminator, MultiHeadedAttention, AttentionalPropagation, GraphSAGEConvolution
from lib.module.sinkhorn import solve_optimal_transport
import numpy as np
import scipy.sparse as sp
from munkres import Munkres
import ot
import random

class GCN(nn.Module):
    def __init__(self, infeat, outfeat):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(infeat, outfeat)

    def forward(self, x, adj):
        x = torch.tanh(self.gc1(x, adj)+x)
        return x

    def aggregation(self, x, adj):
        x = self.gc1(x, adj)
        return x

class GSAGE(nn.Module):
    def __init__(self, infeat, outfeat):
        super(GSAGE, self).__init__()

        self.gc1 = GraphSAGEConvolution(infeat, outfeat)

    def forward(self, x, adj):
        x = torch.tanh(self.gc1(x, adj))
        return x

    def aggregation(self, x, adj):
        x = self.gc1(x, adj)
        return x


class Learnable_Topology_BGNN_unlabel(nn.Module):
    def __init__(self, configer):
        """Dense version of GAT."""
        super(Learnable_Topology_BGNN_unlabel, self).__init__()
        # print('Learnable_Topology_BGNN_unlabel')
        
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
        self.uot_ratio = self.configer.get('GNN', 'uot_ratio')

        self.mse_or_adv = self.configer.get('GNN', 'mse_or_adv')
        self.GNN_type = self.configer.get('GNN', 'GNN_type')

        self.linear_before = nn.Linear(self.nfeat, self.nfeat_out)
        self.linear_adj = nn.Linear(self.nfeat_out, self.nfeat_adj)
        if self.calc_bipartite:
            self.linear_adj2 = nn.Linear(self.adj_feat_dim, self.adj_feat_dim)
            
            
        self.relu = nn.ReLU()


        if self.GNN_type == 'GCN':
            self.GCN_layer1 = GCN(self.nfeat_out, self.nfeat_out)
            self.GCN_layer2 = GCN(self.nfeat_out, self.nfeat_out)
            self.GCN_layer3 = GCN(self.nfeat_out, self.nfeat_out)
            self.GCN_layer4 = GCN(self.nfeat_out, self.nfeat_out)
        elif self.GNN_type == 'GSAGE':
            self.GCN_layer1 = GSAGE(self.nfeat_out, self.nfeat_out)
            self.GCN_layer2 = GSAGE(self.nfeat_out, self.nfeat_out)
            self.GCN_layer3 = GSAGE(self.nfeat_out, self.nfeat_out)   
            self.GCN_layer4 = GSAGE(self.nfeat_out, self.nfeat_out)   
        
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
        
        self.unlable_node_features = nn.Parameter(torch.randn(self.n_datasets, self.nfeat), requires_grad=True)
        
        trunc_normal_(self.unify_node_features, std=0.02)
        trunc_normal_(self.unlable_node_features, std=0.02)
        
        ## Graph adjacency matrix
        self.adj_matrix = nn.Parameter(torch.zeros(self.total_cats+self.max_num_unify_class, self.total_cats+self.max_num_unify_class), requires_grad=True)
        # self.init_adjacency_matrix()
        if self.mse_or_adv == 'adv':
            self.netD1 = Discriminator(self.nfeat_out, 128, 1, self.dropout_rate)
            self.netD1.weights_init()
            self.netD2 = Discriminator(self.nfeat_out, 128, 1, self.dropout_rate)
            self.netD2.weights_init()
            self.netD3 = Discriminator(self.nfeat_out, 128, 1, self.dropout_rate)
            self.netD3.weights_init()
            self.netD4 = Discriminator(self.nfeat_out, 128, 1, self.dropout_rate)
            self.netD4.weights_init()
        
        self.use_km = False
        if self.use_km:
            self.km_algorithms = Munkres()
        # else:
            
        self.beta = [ot.unif(self.dataset_cats[i]) for i in range(0, self.n_datasets)]
        self.uot_update = 0
        self.uot_bi = None
        
    def forward(self, x, pretraining=False):
        cur_cat = 0
        dataset_feats = []
        for i in range(0, self.n_datasets):
            dataset_feats.append(x[cur_cat:cur_cat+self.dataset_cats[i]-1])
            dataset_feats.append(self.unlable_node_features[i].unsqueeze(0))
            cur_cat += self.dataset_cats[i]-1
        
        dataset_feats = torch.cat(dataset_feats, dim=0)
        input_x = torch.cat([dataset_feats, self.unify_node_features], dim=0)
        # input_x = torch.cat([x, self.unify_node_features], dim=0)
        
        feat1 = self.linear_before(input_x)
        adj_mI, non_norm_adj_mI = self.calc_adjacency_matrix(feat1)
        feat1_relu = self.relu(feat1)
        
        before_gcn1_x = F.dropout(feat1_relu, self.dropout_rate, training=self.training)
        feat_gcn1 = self.GCN_layer1(before_gcn1_x, adj_mI)
        if self.mse_or_adv == 'adv':
            out_real_1 = self.netD1(before_gcn1_x.detach())
            out_fake_1 = self.netD1(feat_gcn1.detach())
            g_out_fake_1 = self.netD1(feat_gcn1)
        
        # feat2 = feat_gcn1 + before_gcn1_x
        before_gcn2_x = F.dropout(feat_gcn1, self.dropout_rate, training=self.training)
        feat_gcn2 = self.GCN_layer2(before_gcn2_x, adj_mI)
        if self.mse_or_adv == 'adv':
            out_real_2 = self.netD2(before_gcn2_x.detach())
            out_fake_2 = self.netD2(feat_gcn2.detach())
            g_out_fake_2 = self.netD2(feat_gcn2)
        
        # feat3 = feat_gcn2 + before_gcn2_x
        before_gcn3_x = F.dropout(feat_gcn2, self.dropout_rate, training=self.training)
        feat_gcn3 = self.GCN_layer3(before_gcn3_x, adj_mI)
        if self.mse_or_adv == 'adv':
            out_real_3 = self.netD3(before_gcn3_x.detach())
            out_fake_3 = self.netD3(feat_gcn3.detach())
            g_out_fake_3 = self.netD3(feat_gcn3)
        
        before_gcn4_x = F.dropout(feat_gcn3, self.dropout_rate, training=self.training)
        feat_gcn4 = self.GCN_layer4(before_gcn4_x, adj_mI)
        if self.mse_or_adv == 'adv':
            out_real_4 = self.netD4(before_gcn4_x.detach())
            out_fake_4 = self.netD4(feat_gcn4.detach())
            g_out_fake_4 = self.netD4(feat_gcn4)
        # before_gcn3_x = F.dropout(feat_gcn3, self.dropout_rate, training=self.training)
        # feat_gcn3 = self.GCN_layer4(before_gcn3_x, adj_mI)
        # feat4 = F.elu(feat_gcn3 + before_gcn3_x)
        # feat3_drop = F.dropout(feat3, self.dropout_rate, training=self.training)
        feat_out = self.linear1(feat_gcn4)

        adv_out = {}
        if self.mse_or_adv == 'adv':
            adv_out['ADV1'] = [out_real_1, out_fake_1, g_out_fake_1]
            adv_out['ADV2'] = [out_real_2, out_fake_2, g_out_fake_2]
            adv_out['ADV3'] = [out_real_3, out_fake_3, g_out_fake_3]
            adv_out['ADV4'] = [out_real_4, out_fake_4, g_out_fake_4]
        elif self.mse_or_adv == 'mse':
            adv_out['ADV1'] = [feat1_relu.detach(), feat_gcn1]
            adv_out['ADV2'] = [feat_gcn1.detach(), feat_gcn2]
            adv_out['ADV3'] = [feat_gcn2.detach(), feat_gcn3]
            adv_out['ADV4'] = [feat_gcn3.detach(), feat_gcn4]
            
        if pretraining:
            return feat_out[self.total_cats:], self.sep_bipartite_graphs(non_norm_adj_mI), adv_out, non_norm_adj_mI
        elif self.calc_bipartite:
            arch_x = self.relu(feat_out)
            arch_x = self.linear2(arch_x)
            _, non_norm_adj_mI_after = self.calc_adjacency_matrix(arch_x)
            
            return feat_out[self.total_cats:], self.sep_bipartite_graphs(non_norm_adj_mI_after), adv_out, non_norm_adj_mI_after
        else:
            return feat_out[self.total_cats:], self.sep_bipartite_graphs(non_norm_adj_mI), adv_out, non_norm_adj_mI

    def sep_bipartite_graphs(self, adj):
        self.bipartite_graphs = []
        cur_cat = 0
        
        # if self.uot_update == 0:
        #     self.uot_bi = self.sep_bipartite_graphs_by_uot(adj.detach())
        #     self.uot_update = 100
        # else:
        #     self.uot_update -= 1

            
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
                # self.bipartite_graphs.append(self.uot_bi[i].detach())
                
            if self.output_softmax_and_max_adj or not self.output_max_adj:
                # softmax_bipartite_graph = F.softmax(this_bipartite_graph/0.07, dim=0)
                
                self.bipartite_graphs.append(this_bipartite_graph)
            
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
        # adj_mI = similar_matrix - torch.diag(torch.diag(similar_matrix))
        adj_mI = similar_matrix
        mask = torch.ones(adj_mI.size(), dtype=torch.bool)
        if adj_mI.is_cuda:
            mask = mask.cuda()
        
        mask[:self.total_cats, :self.total_cats] = 0
        mask[self.total_cats:, self.total_cats:] = 0
        adj_mI = adj_mI * mask
            
        # adj_mI[:self.total_cats, :self.total_cats] -= similar_matrix[:self.total_cats, :self.total_cats] 
        # adj_mI[self.total_cats:, self.total_cats:] -= similar_matrix[self.total_cats:, self.total_cats:]
        

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
        cur_cat = 0
        for i in range(0, self.n_datasets):
            this_bipartite_graph = adj_mI[cur_cat:cur_cat+self.dataset_cats[i], self.total_cats:]

            softmax_bipartite_graph = F.softmax(this_bipartite_graph/0.07, dim=0)
            adj_mI[cur_cat:cur_cat+self.dataset_cats[i], self.total_cats:] = softmax_bipartite_graph
            cur_cat = cur_cat+self.dataset_cats[i]

        # softmax_adj = F.softmax(adj_mI/0.07, dim=0)
        norm_adj_mI = normalize_adj(adj_mI)
        return norm_adj_mI, adj_mI
    
    def get_optimal_matching(self, x, init=False):
        cur_cat = 0
        dataset_feats = []
        for i in range(0, self.n_datasets):
            dataset_feats.append(x[cur_cat:cur_cat+self.dataset_cats[i]-1])
            dataset_feats.append(self.unlable_node_features[i].unsqueeze(0))
            cur_cat += self.dataset_cats[i]-1
        
        dataset_feats = torch.cat(dataset_feats, dim=0)
        x = torch.cat([dataset_feats, self.unify_node_features], dim=0)
        # x = torch.cat([x, self.unify_node_features], dim=0)
        
        feat1 = self.linear_before(x)
        adj_mI, non_norm_adj_mI = self.calc_adjacency_matrix(feat1)
        feat1_relu = self.relu(feat1)
        
        feat_gcn1 = self.GCN_layer1(feat1_relu, adj_mI)
        
        # feat2 = feat_gcn1 + feat1_relu
        feat_gcn2 = self.GCN_layer2(feat_gcn1, adj_mI)
        
        # feat3 = feat_gcn2 + feat2
        feat_gcn3 = self.GCN_layer3(feat_gcn2, adj_mI)

        feat_gcn4 = self.GCN_layer4(feat_gcn3, adj_mI)
        # feat4 = F.elu(feat_gcn3 + feat3)
        feat_out = self.linear1(feat_gcn3)

        if init:
            if self.calc_bipartite:
                arch_x = self.relu(feat_gcn3 + feat_out)
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
        bipartite_graphs = []
        cur_cat = 0
        for i in range(0, self.n_datasets):
            this_bipartite_graph = adj[cur_cat:cur_cat+self.dataset_cats[i], self.total_cats:]
            this_bipartite_graph = (-this_bipartite_graph.detach().clone()+1 + 1e-8)/2
            out_bipartite_graphs = torch.zeros_like(this_bipartite_graph)

            alpha = ot.unif(self.max_num_unify_class)
                
            Q_st = ot.unbalanced.sinkhorn_knopp_unbalanced(alpha, self.beta[i], this_bipartite_graph.T.cpu().numpy(), 
                                                            reg=0.01, reg_m=5, stopThr=1e-6) 

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
                    
                    
            for row in range(0, self.dataset_cats[i]):
                if torch.sum(out_bipartite_graphs[row]) > 1:
                    max_v, max_i = 0, 0
                    for index, val in enumerate(out_bipartite_graphs[row]):
                        if val == 1:
                            if max_v < Q_st_bar[index, row]:
                                max_v = Q_st_bar[index, row]
                                max_i = index
                            if Q_st_bar[index, row] < self.uot_ratio/(Q_st_bar.shape[0]*Q_st_bar.shape[1]):
                                out_bipartite_graphs[row, index] = 0
                        
                            
                    if torch.sum(out_bipartite_graphs[row]) == 0:
                        out_bipartite_graphs[row, max_i] = 1
                
            new_beta = torch.sum(Q_st_bar,0).cpu().numpy()


            mu = 0.7
            self.beta[i] = mu*self.beta[i] + (1-mu)*new_beta
            # print(out_bipartite_graphs)
            # print(torch.max(out_bipartite_graphs, dim=0))
            bipartite_graphs.append(out_bipartite_graphs) 
                
            cur_cat += self.dataset_cats[i]
        
        # temp_bipartite_graphs = torch.cat(self.bipartite_graphs, dim=0)
        # unique_cols, unique_indices = torch.unique(temp_bipartite_graphs, sorted=False, dim=1, return_inverse=True)
        
        # for j in range(0, len(unique_indices)):
        #     if sum(unique_indices == unique_indices[j]) == 1:
        #         continue
                
        #     flag = True
        #     while flag:
        #         new_col = torch.zeros_like(temp_bipartite_graphs[:,j])
        #         cur_cat = 0
        #         for s_id in range(0, self.n_datasets):
        #             rint = random.randint(cur_cat, cur_cat+self.dataset_cats[s_id]-1)
        #             cur_cat += self.dataset_cats[s_id]
        #             new_col[rint] = 1
                    
        #         flag = False
        #         for col in range(0, temp_bipartite_graphs.shape[1]):
        #             if (temp_bipartite_graphs[:, col] == new_col).all():
        #                 flag = True
        #                 break
                    
        #         if flag == False:
        #             print(f"cats: {j}")
        #             temp_bipartite_graphs[:, j] = new_col
        #             unique_indices[j] = -1
                    
                    
                
        # self.bipartite_graphs = []
        # cur_cat = 0
        # for s_id in range(0, self.n_datasets):
        #     self.bipartite_graphs.append(temp_bipartite_graphs[cur_cat:cur_cat+self.dataset_cats[s_id], :])
        #     cur_cat += self.dataset_cats[s_id]
        
        return bipartite_graphs 

    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            # for param in mod.parameters():
            if param.requires_grad == False:
                return
                # continue
            
            if param.dim() == 1:
                nowd_params.append(param)
            elif param.dim() == 4 or param.dim() == 2:
                wd_params.append(param)
            else:
                nowd_params.append(param)
                print(param.dim())
                # print(param)
                print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        # for name, child in self.named_children():
        for name, param in self.named_parameters():
            
            if 'head' in name or 'aux' in name:
                add_param_to_list(param, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(param, wd_params, nowd_params)
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