import sys

import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.metrics import f1_score, roc_auc_score

class SBGRL(nn.Module):
    def __init__(self, args, num_a, num_b, emb_size_a=32, emb_size_b=32, layer_num=2):
        super(SBGRL,self).__init__()
        self.emb_size_a = args.dim_embs
        self.emb_size_b = args.dim_embs
        self.num_a = num_a
        self.num_b = num_b
        self.layer_num = layer_num
        self.layers_a_b_pos = nn.ModuleList([GATConv(self.emb_size_a, self.emb_size_b)
                                             for _ in range(layer_num)])
        self.layers_a_b_neg = nn.ModuleList([GATConv(self.emb_size_a, self.emb_size_b)
                                             for _ in range(layer_num)])
        self.layers_a_a_pos = nn.ModuleList([GATConv(self.emb_size_a, self.emb_size_a)
                                             for _ in range(layer_num)])
        self.layers_a_a_neg = nn.ModuleList([GATConv(self.emb_size_a, self.emb_size_a)
                                             for _ in range(layer_num)])
        self.args = args

        # initial feature transformer
        self.trans_g_pos_1 = nn.Linear(self.emb_size_a, self.emb_size_a, bias=False)
        self.trans_g_neg_1 = nn.Linear(self.emb_size_a, self.emb_size_a, bias=False)
        self.trans_g_pos_2 = nn.Linear(self.emb_size_a, self.emb_size_a, bias=False)
        self.trans_g_neg_2 = nn.Linear(self.emb_size_a, self.emb_size_a, bias=False)
        self.trans_g_pos_3 = nn.Linear(self.emb_size_a, self.emb_size_a, bias=False)
        self.trans_g_neg_3 = nn.Linear(self.emb_size_a, self.emb_size_a, bias=False)
        self.trans_g_pos_4 = nn.Linear(self.emb_size_a, self.emb_size_a, bias=False)
        self.trans_g_neg_4 = nn.Linear(self.emb_size_a, self.emb_size_a, bias=False)

        # self.trans_g_neg_1 = self.trans_g_pos_1
        # self.trans_g_pos_2 = self.trans_g_pos_1
        # self.trans_g_neg_2 = self.trans_g_pos_1
        # self.trans_g_pos_3 = self.trans_g_pos_1
        # self.trans_g_neg_3 = self.trans_g_pos_1
        # self.trans_g_pos_4 = self.trans_g_pos_1
        # self.trans_g_neg_4 = self.trans_g_pos_1



        # combined h^0, h^1, h^2, h^3
        self.mlp_g_pos_1 = nn.Linear(self.emb_size_a*(1+layer_num), self.emb_size_a, bias=False)
        self.mlp_g_neg_1 = nn.Linear(self.emb_size_a*(1+layer_num), self.emb_size_a, bias=False)
        self.mlp_g_pos_2 = nn.Linear(self.emb_size_a*(1+layer_num), self.emb_size_a, bias=False)
        self.mlp_g_neg_2 = nn.Linear(self.emb_size_a*(1+layer_num), self.emb_size_a, bias=False)
        self.mlp_g_pos_3 = nn.Linear(self.emb_size_a*(1+layer_num), self.emb_size_a, bias=False)
        self.mlp_g_neg_3 = nn.Linear(self.emb_size_a*(1+layer_num), self.emb_size_a, bias=False)
        self.mlp_g_pos_4 = nn.Linear(self.emb_size_a*(1+layer_num), self.emb_size_a, bias=False)
        self.mlp_g_neg_4 = nn.Linear(self.emb_size_a*(1+layer_num), self.emb_size_a, bias=False)

        # emb trans
        self.mlp_emb = nn.Linear(self.emb_size_a*8, self.emb_size_a, bias=False)
        # self.mlp_emb = nn.Sequential(
        #     nn.Dropout(args.dropout),
        #     nn.Linear(emb_size_a*8, emb_size_a*4),
        #     nn.PReLU(),
        #     nn.Linear(emb_size_a*4, emb_size_a)
        # )
        # self.combine_type = args.combine_type
        # if self.combine_type == 'concat':
        #     self.transform = nn.Sequential(nn.Linear(transform_type*args.dim_embs, args.dim_embs))
        #     # self.link_predictor = ScorePredictor(args, dim_embs = args.dim_embs)
        # elif self.combine_type == 'attn':
        #     self.attention = nn.Sequential(nn.Linear(args.dim_embs, args.dim_query), nn.Tanh(), nn.Linear(args.dim_query, 1, bias=False))
        #     # self.link_predictor = ScorePredictor(args, dim_embs=args.dim_embs)
        # self.mlp_p1 = nn.Linear(4*args.dim_embs, args.dim_embs)
        # self.mlp_p2 = nn.Linear(4*args.dim_embs, args.dim_embs)

        self.update_func = nn.Sequential(
            nn.Dropout(args.dropout),
            # nn.Linear(emb_size_a , emb_size_a),
            # nn.PReLU(),
            # nn.Linear(emb_size_a, emb_size_a)

        )
        self.activation = nn.PReLU()
        self.link_predictor = ScorePredictor(args, dim_embs = args.dim_embs)


    def forward(self, edges, x):
        pos_index_a_b_1, neg_index_a_b_1, pos_index_a_b_2, neg_index_a_b_2, \
        pos_index_a_1, neg_index_a_1, pos_index_b_1, neg_index_b_1, \
        pos_index_a_2, neg_index_a_2, pos_index_b_2, neg_index_b_2 = edges


        # Graph 1 and 2 pos
        x_pos_a_b_1_list = []
        x_pos_index_a_b_1 = self.trans_g_pos_1(x)
        x_pos_a_b_1_list.append(x_pos_index_a_b_1)
        x_pos_index_a_b_1 = self.layers_a_b_pos[0](x_pos_index_a_b_1, pos_index_a_b_1)
        x_pos_index_a_b_1 = self.activation(x_pos_index_a_b_1)
        x_pos_a_b_1_list.append(x_pos_index_a_b_1)

        x_pos_a_b_2_list = []
        x_pos_index_a_b_2 = self.trans_g_pos_2(x)
        x_pos_a_b_2_list.append(x_pos_index_a_b_2)
        x_pos_index_a_b_2 = self.layers_a_b_pos[0](x_pos_index_a_b_2, pos_index_a_b_2)
        x_pos_index_a_b_2 = self.activation(x_pos_index_a_b_2)
        x_pos_a_b_2_list.append(x_pos_index_a_b_2)
        for i in range(1, self.layer_num):
            x_pos_index_a_b_1 = self.layers_a_b_pos[i](x_pos_index_a_b_1, pos_index_a_b_1)
            x_pos_index_a_b_1 = self.activation(x_pos_index_a_b_1)
            x_pos_a_b_1_list.append(x_pos_index_a_b_1)

            x_pos_index_a_b_2 = self.layers_a_b_pos[i](x_pos_index_a_b_2, pos_index_a_b_2)
            x_pos_index_a_b_2 = self.activation(x_pos_index_a_b_2)
            x_pos_a_b_2_list.append(x_pos_index_a_b_2)

        # Graph 1 and 2 neg
        x_neg_a_b_1_list = []
        x_neg_index_a_b_1 =  self.trans_g_neg_1(x)
        x_neg_a_b_1_list.append(x_neg_index_a_b_1)
        x_neg_index_a_b_1 = self.layers_a_b_neg[0](x_neg_index_a_b_1, neg_index_a_b_1)
        x_neg_index_a_b_1 = self.activation(x_neg_index_a_b_1)
        x_neg_a_b_1_list.append(x_neg_index_a_b_1)

        x_neg_a_b_2_list = []
        x_neg_index_a_b_2 = self.trans_g_neg_2(x)
        x_neg_a_b_2_list.append(x_neg_index_a_b_2)
        x_neg_index_a_b_2 = self.layers_a_b_neg[0](x_neg_index_a_b_2, neg_index_a_b_2)
        x_neg_index_a_b_2 = self.activation(x_neg_index_a_b_2)
        x_neg_a_b_2_list.append(x_neg_index_a_b_2)
        for i in range(1, self.layer_num):
            x_neg_index_a_b_1 = self.layers_a_b_neg[i](x_neg_index_a_b_1, neg_index_a_b_1)
            x_neg_index_a_b_1 = self.activation(x_neg_index_a_b_1)
            x_neg_a_b_1_list.append(x_neg_index_a_b_1)

            x_neg_index_a_b_2 = self.layers_a_b_neg[i](x_neg_index_a_b_2, neg_index_a_b_2)
            x_neg_index_a_b_2 = self.activation(x_neg_index_a_b_2)
            x_neg_a_b_2_list.append(x_neg_index_a_b_2)

        # Graph 3 and 4 pos
        x_pos_a_1_list = []
        x_pos_index_a_1 = self.trans_g_pos_3(x)
        x_pos_a_1_list.append(x_pos_index_a_1)
        x_pos_index_a_1 = self.layers_a_a_pos[0](x_pos_index_a_1, pos_index_a_1)
        x_pos_index_a_1 = self.activation(x_pos_index_a_1)
        x_pos_a_1_list.append(x_pos_index_a_1)

        x_pos_b_1_list = []
        x_pos_index_b_1 = self.trans_g_pos_3(x)
        x_pos_b_1_list.append(x_pos_index_b_1)
        x_pos_index_b_1 = self.layers_a_a_pos[0](x, pos_index_b_1)
        x_pos_index_b_1 = self.activation(x_pos_index_b_1)
        x_pos_b_1_list.append(x_pos_index_b_1)

        x_pos_a_2_list = []
        x_pos_index_a_2 = self.trans_g_pos_4(x)
        x_pos_a_2_list.append(x_pos_index_a_2)
        x_pos_index_a_2 = self.layers_a_a_pos[0](x_pos_index_a_2, pos_index_a_2)
        x_pos_index_a_2 = self.activation(x_pos_index_a_2)
        x_pos_a_2_list.append(x_pos_index_a_2)

        x_pos_b_2_list = []
        x_pos_index_b_2 = self.trans_g_pos_4(x)
        x_pos_b_2_list.append(x_pos_index_b_2)
        x_pos_index_b_2 = self.layers_a_a_pos[0](x_pos_index_b_2, pos_index_b_2)
        x_pos_index_b_2 = self.activation(x_pos_index_b_2)
        x_pos_b_2_list.append(x_pos_index_b_2)

        for i in range(1, self.layer_num):
            x_pos_index_a_1 = self.layers_a_a_pos[i](x_pos_index_a_1, pos_index_a_1)
            x_pos_index_a_1 = self.activation(x_pos_index_a_1)
            x_pos_a_1_list.append(x_pos_index_a_1)

            x_pos_index_b_1 = self.layers_a_a_pos[i](x_pos_index_b_1, pos_index_b_1)
            x_pos_index_b_1 = self.activation(x_pos_index_b_1)
            x_pos_b_1_list.append(x_pos_index_b_1)

            x_pos_index_a_2 = self.layers_a_a_pos[i](x_pos_index_a_2, pos_index_a_2)
            x_pos_index_a_2 =self.activation(x_pos_index_a_2)
            x_pos_a_2_list.append(x_pos_index_a_2)

            x_pos_index_b_2 = self.layers_a_a_pos[i](x_pos_index_b_2, pos_index_b_2)
            x_pos_index_b_2 =self.activation(x_pos_index_b_2)
            x_pos_b_2_list.append(x_pos_index_b_2)

        # Graph 3 and 4 neg
        x_neg_a_1_list = []
        x_neg_index_a_1 = self.trans_g_neg_3(x)
        x_neg_a_1_list.append(x_neg_index_a_1)
        x_neg_index_a_1 = self.layers_a_a_neg[0](x_neg_index_a_1, neg_index_a_1)
        x_neg_index_a_1 = self.activation(x_neg_index_a_1)
        x_neg_a_1_list.append(x_neg_index_a_1)

        x_neg_b_1_list = []
        x_neg_index_b_1 = self.trans_g_neg_3(x)
        x_neg_b_1_list.append(x_neg_index_b_1)
        x_neg_index_b_1 = self.layers_a_a_neg[0](x_neg_index_b_1, neg_index_b_1)
        x_neg_index_b_1 =self.activation(x_neg_index_b_1)
        x_neg_b_1_list.append(x_neg_index_b_1)

        x_neg_a_2_list = []
        x_neg_index_a_2 = self.trans_g_neg_4(x)
        x_neg_a_2_list.append(x_neg_index_a_2)
        x_neg_index_a_2 = self.layers_a_a_neg[0](x_neg_index_a_2, neg_index_a_2)
        x_neg_index_a_2 = self.activation(x_neg_index_a_2)
        x_neg_a_2_list.append(x_neg_index_a_2)

        x_neg_b_2_list = []
        x_neg_index_b_2 = self.trans_g_neg_4(x)
        x_neg_b_2_list.append(x_neg_index_b_2)
        x_neg_index_b_2 = self.layers_a_a_neg[0](x_neg_index_b_2, neg_index_b_2)
        x_neg_index_b_2 = self.activation(x_neg_index_b_2)
        x_neg_b_2_list.append(x_neg_index_b_2)

        for i in range(1, self.layer_num):
            x_neg_index_a_1 = self.layers_a_a_neg[i](x_neg_index_a_1, neg_index_a_1)
            x_neg_index_a_1 =self.activation(x_neg_index_a_1)
            x_neg_a_1_list.append(x_neg_index_a_1)

            x_neg_index_b_1 = self.layers_a_a_neg[i](x_neg_index_b_1, neg_index_b_1)
            x_neg_index_b_1 = self.activation(x_neg_index_b_1)
            x_neg_b_1_list.append(x_neg_index_b_1)

            x_neg_index_a_2 = self.layers_a_a_neg[i](x_neg_index_a_2, neg_index_a_2)
            x_neg_index_a_2 = self.activation(x_neg_index_a_2)
            x_neg_a_2_list.append(x_neg_index_a_2)

            x_neg_index_b_2 = self.layers_a_a_neg[i](x_neg_index_b_2, neg_index_b_2)
            x_neg_index_b_2 = self.activation(x_neg_index_b_2)
            x_neg_b_2_list.append(x_neg_index_b_2)

        x_pos_index_a_b_1 = self.mlp_g_pos_1(torch.cat(x_pos_a_b_1_list, dim=1))
        x_pos_index_a_b_1 = self.update_func(x_pos_index_a_b_1)

        x_neg_index_a_b_1 = self.mlp_g_neg_1(torch.cat(x_neg_a_b_1_list, dim=1))
        x_neg_index_a_b_1 = self.update_func(x_neg_index_a_b_1)

        x_pos_index_a_b_2 = self.mlp_g_pos_2(torch.cat(x_pos_a_b_2_list, dim=1))
        x_pos_index_a_b_2 = self.update_func(x_pos_index_a_b_2)

        x_neg_index_a_b_2 = self.mlp_g_neg_2(torch.cat(x_neg_a_b_2_list, dim=1))
        x_neg_index_a_b_2 = self.update_func(x_neg_index_a_b_2)

        x_pos_index_a_1 = torch.cat(x_pos_a_1_list, dim=1)
        x_pos_index_b_1 = torch.cat(x_pos_b_1_list, dim=1)
        x_pos_index_a_b_3 = torch.cat([x_pos_index_a_1[:self.num_a],x_pos_index_b_1[self.num_a:]], dim=0)
        x_pos_index_a_b_3 = self.mlp_g_pos_3(x_pos_index_a_b_3)
        x_pos_index_a_b_3 = self.update_func(x_pos_index_a_b_3)

        x_neg_index_a_1 = torch.cat(x_neg_a_1_list, dim=1)
        x_neg_index_b_1 = torch.cat(x_neg_b_1_list, dim=1)
        x_neg_index_a_b_3 = torch.cat([x_neg_index_a_1[:self.num_a],x_neg_index_b_1[self.num_a:]], dim=0)
        x_neg_index_a_b_3 = self.mlp_g_neg_3(x_neg_index_a_b_3)
        x_neg_index_a_b_3 = self.update_func(x_neg_index_a_b_3)

        # x_pos_index_a_1 = self.update_func(x_pos_index_a_1)
        # x_neg_index_a_1 = self.update_func(x_neg_index_a_1)
        # x_pos_index_b_1 = self.update_func(x_pos_index_b_1)
        # x_neg_index_b_1 = self.update_func(x_neg_index_b_1)

        x_pos_index_a_2 = torch.cat(x_pos_a_2_list, dim=1)
        x_pos_index_b_2 = torch.cat(x_pos_b_2_list, dim=1)
        x_pos_index_a_b_4 = torch.cat([x_pos_index_a_2[:self.num_a], x_pos_index_b_2[self.num_a:]], dim=0)
        x_pos_index_a_b_4 = self.mlp_g_pos_4(x_pos_index_a_b_4)
        x_pos_index_a_b_4 = self.update_func(x_pos_index_a_b_4)

        x_neg_index_a_2 = torch.cat(x_neg_a_2_list, dim=1)
        x_neg_index_b_2 = torch.cat(x_neg_b_2_list, dim=1)
        x_neg_index_a_b_4 = torch.cat([x_neg_index_a_2[:self.num_a], x_neg_index_b_2[self.num_a:]], dim=0)
        x_neg_index_a_b_4 = self.mlp_g_neg_4(x_neg_index_a_b_4)
        x_neg_index_a_b_4 = self.update_func(x_neg_index_a_b_4)

        # x_pos_index_a_2 = self.update_func(x_pos_index_a_2)
        # x_neg_index_a_2 = self.update_func(x_neg_index_a_2)
        # x_pos_index_b_2 = self.update_func(x_pos_index_b_2)
        # x_neg_index_b_2 = self.update_func(x_neg_index_b_2)


        return x_pos_index_a_b_1, x_neg_index_a_b_1, \
               x_pos_index_a_b_2, x_neg_index_a_b_2, \
               x_pos_index_a_b_3, x_neg_index_a_b_3, \
               x_pos_index_a_b_4, x_neg_index_a_b_4


    def computer_contrastive_loss(self, x):

        def inter_contrastive(emb_1, emb_2):
            pos = torch.exp(torch.div(torch.bmm(emb_1.view(emb_1.shape[0], 1, emb_1.shape[1]),
                                                emb_2.view(emb_2.shape[0], emb_2.shape[1], 1)),
                                      self.args.tau))
            def generate_neg_score(emb_1, emb_2):
                neg_similarity = torch.mm(emb_1.view(emb_1.shape[0], emb_1.shape[1]), emb_2.transpose(0, 1))
                neg_similarity[np.arange(emb_1.shape[0]), np.arange(emb_1.shape[0])] = 0
                return torch.sum(torch.exp(torch.div(neg_similarity, self.args.tau)), dim=1)

            neg = generate_neg_score(emb_1, emb_2)
            return torch.mean(- (torch.log(torch.div(pos, neg))))

        def intra_contrastive(emb, emb_1_pos, emb_2_pos, emb_1_neg, emb_2_neg):
            pos_score_1 = torch.exp(torch.div(torch.bmm(emb.view(emb.shape[0], 1, emb.shape[1]),
                                                        emb_1_pos.view(emb_1_pos.shape[0], emb_1_pos.shape[1], 1)),
                                              self.args.tau))
            pos_score_2 = torch.exp(torch.div(torch.bmm(emb.view(emb.shape[0], 1, emb.shape[1]),
                                                        emb_2_pos.view(emb_2_pos.shape[0], emb_2_pos.shape[1], 1)),
                                              self.args.tau))

            pos =  pos_score_1 + pos_score_2

            def generate_neg_score(emb, emb_1_neg, emb_2_neg):
                neg_score_1 = torch.bmm(emb.view(emb.shape[0], 1, emb.shape[1]),
                                        emb_1_neg.view(emb_1_neg.shape[0], emb_1_neg.shape[1], 1))
                neg_score_2 = torch.bmm(emb.view(emb.shape[0], 1, emb.shape[1]),
                                        emb_2_neg.view(emb_2_neg.shape[0], emb_2_neg.shape[1], 1))
                return torch.exp(torch.div(neg_score_1, self.args.tau)) + \
                       torch.exp(torch.div(neg_score_2, self.args.tau))
            neg = generate_neg_score(emb, emb_1_neg, emb_2_neg)
            return torch.mean(- torch.log(torch.div(pos, neg)))

        x_pos_index_a_b_1, x_neg_index_a_b_1, \
        x_pos_index_a_b_2, x_neg_index_a_b_2, \
        x_pos_index_a_b_3, x_neg_index_a_b_3, \
        x_pos_index_a_b_4, x_neg_index_a_b_4 = x

        x_pos_index_a_b_1 = F.normalize(x_pos_index_a_b_1, p=2, dim=1)
        x_neg_index_a_b_1 = F.normalize(x_neg_index_a_b_1, p=2, dim=1)

        x_pos_index_a_b_2 = F.normalize(x_pos_index_a_b_2, p=2, dim=1)
        x_neg_index_a_b_2 = F.normalize(x_neg_index_a_b_2, p=2, dim=1)

        x_pos_index_a_b_3 = F.normalize(x_pos_index_a_b_3, p=2, dim=1)
        x_neg_index_a_b_3 = F.normalize(x_neg_index_a_b_3, p=2, dim=1)

        x_pos_index_a_b_4 = F.normalize(x_pos_index_a_b_4, p=2, dim=1)
        x_neg_index_a_b_4 = F.normalize(x_neg_index_a_b_4, p=2, dim=1)



        # x_pos_index_a_1 = F.normalize(x_pos_index_a_1, p=2, dim=1)
        # x_neg_index_a_1 = F.normalize(x_neg_index_a_1, p=2, dim=1)
        # x_pos_index_b_1 = F.normalize(x_pos_index_b_1, p=2, dim=1)
        # x_neg_index_b_1 = F.normalize(x_neg_index_b_1, p=2, dim=1)
        #
        # x_pos_index_a_2 = F.normalize(x_pos_index_a_2, p=2, dim=1)
        # x_neg_index_a_2 = F.normalize(x_neg_index_a_2, p=2, dim=1)
        # x_pos_index_b_2 = F.normalize(x_pos_index_b_2, p=2, dim=1)
        # x_neg_index_b_2 = F.normalize(x_neg_index_b_2, p=2, dim=1)

        inter_loss_pos_1 = inter_contrastive(x_pos_index_a_b_1, x_pos_index_a_b_2)
        inter_loss_neg_1 = inter_contrastive(x_neg_index_a_b_1, x_neg_index_a_b_2)

        inter_loss_pos_2 = inter_contrastive(x_pos_index_a_b_3, x_pos_index_a_b_4)
        inter_loss_neg_2 = inter_contrastive(x_neg_index_a_b_3, x_neg_index_a_b_4)




        # inter_loss_pos_a = inter_contrastive(x_pos_index_a_1, x_pos_index_a_2)
        # inter_loss_neg_a = inter_contrastive(x_neg_index_a_1, x_neg_index_a_2)
        #
        # inter_loss_pos_b = inter_contrastive(x_pos_index_b_1, x_pos_index_b_2)
        # inter_loss_neg_b = inter_contrastive(x_neg_index_b_1, x_neg_index_b_2)

        inter = inter_loss_pos_1 + inter_loss_neg_1 + \
                inter_loss_pos_2 + inter_loss_neg_2

        # embs_p1 = torch.cat([x_pos_index_a_b_1, x_pos_index_a_b_2, x_neg_index_a_b_1, x_neg_index_a_b_2], dim=1)
        #
        #
        # embs_p2_a = torch.cat([x_pos_index_a_1[:self.num_a], x_pos_index_a_2[:self.num_a], x_neg_index_a_1[:self.num_a], x_neg_index_a_2[:self.num_a]], dim=1)
        # embs_p2_b = torch.cat([x_pos_index_b_1[self.num_a:], x_pos_index_b_2[self.num_a:], x_neg_index_b_1[self.num_a:], x_neg_index_b_2[self.num_a:]], dim=1)
        #
        # embs_p2 = torch.cat([embs_p2_a, embs_p2_b], dim=0)
        # embs_p1_mlp = self.mlp_p1(embs_p1)
        # embs_p2_mlp = self.mlp_p2(embs_p2)
        #
        # embs_p1_mlp = F.normalize(embs_p1_mlp, p=2, dim=1)
        # embs_p2_mlp = F.normalize(embs_p2_mlp, p=2, dim=1)

        self.embs = torch.cat([x_pos_index_a_b_1, x_pos_index_a_b_2, x_pos_index_a_b_3, x_pos_index_a_b_4,
                               x_neg_index_a_b_1, x_neg_index_a_b_2, x_neg_index_a_b_3 ,x_neg_index_a_b_4], dim=1)
        self.embs = self.mlp_emb(self.embs)
        self.embs = F.normalize(self.embs, p=2, dim=1)

        # self.embs = torch.cat([embs_p1_mlp, embs_p2_mlp], dim=1)

        intra_p1 = intra_contrastive(self.embs,
                                     x_pos_index_a_b_1, x_pos_index_a_b_2,
                                     x_neg_index_a_b_1, x_neg_index_a_b_2)
        intra_p2 = intra_contrastive(self.embs, x_pos_index_a_b_3,x_pos_index_a_b_4,
                                     x_neg_index_a_b_3, x_neg_index_a_b_4 )
        #
        # intra_p2_a = intra_contrastive(embs_p2_mlp,
        #                                x_pos_index_a_1, x_pos_index_a_2,
        #                                x_neg_index_a_1, x_neg_index_a_2)
        # intra_p2_b = intra_contrastive(embs_p2_mlp,
        #                                x_pos_index_b_1, x_pos_index_b_2,
        #                                x_neg_index_b_1, x_neg_index_b_2)
        intra = intra_p1 + intra_p2
        return (1-self.args.alpha) * inter + self.args.alpha * intra

    def predict_combine(self, embs, uids, vids):
        # u_embs = self.combine(embs, uids)
        # v_embs = self.combine(embs, vids)
        u_embs = embs[uids]
        v_embs = embs[vids]
        score = self.link_predictor(u_embs, v_embs)
        return score

    def compute_label_loss(self, score, y_label):
        pos_weight = torch.tensor([(y_label==0).sum().item()/(y_label==1).sum().item()]*y_label.shape[0]).to(score.device)
        return F.binary_cross_entropy_with_logits(score, y_label, pos_weight=pos_weight)

    def compute_attention(self, embs):
        attn = self.attention(embs).softmax(dim=0)
        return attn

    def combine(self, embs, nids, device):
        if self.args.sign_conv == 'sign':
            if self.args.sign_aggre == 'pos':
                embs = (embs[0], embs[1])
            elif self.args.sign_aggre == 'neg':
                embs = (embs[2], embs[3])

        if self.combine_type == 'concat':
            embs = torch.cat(embs, dim=-1)
            sub_embs = embs[nids].to(device)
            out_embs = self.transform(sub_embs)
            return out_embs
        elif self.combine_type == 'attn':
            embs = torch.stack(embs, dim=0)
            sub_embs = embs[:, nids].to(device)
            attn = self.compute_attention(sub_embs)
            # attn: (2,n,1)   sub_embs: (2,n,feature)
            out_embs = (attn * sub_embs).sum(dim=0)
            return out_embs
        elif self.combine_type == 'mean':
            embs = torch.stack(embs, dim=0).mean(dim=0)
            sub_embs = embs[nids].to(device)
            return sub_embs
        elif self.combine_type == 'pos':
            sub_embs = embs[0][nids].to(device)
            return sub_embs


class ScorePredictor(nn.Module):
    def __init__(self, args, **params):
        super().__init__()
        self.args = args

        if self.args.predictor == 'dot':
            pass
        elif self.args.predictor == '1-linear':
            self.predictor = nn.Linear(self.args.dim_embs * 2, 1)
        elif self.args.predictor == '2-linear':
            self.predictor = nn.Sequential(nn.Linear(self.args.dim_embs * 2, self.args.dim_embs),
                                           nn.LeakyReLU(),
                                           nn.Linear(self.args.dim_embs, 1))
        elif self.args.predictor == '3-linear':
            self.predictor = nn.Sequential(nn.Linear(self.args.dim_embs * 2, self.args.dim_embs),
                                           nn.LeakyReLU(),
                                           nn.Linear(self.args.dim_embs, self.args.dim_embs),
                                           nn.LeakyReLU(),
                                           nn.Linear(self.args.dim_embs, 1)
                                           )
        elif self.args.predictor == '4-linear':
            self.predictor = nn.Sequential(nn.Linear(self.args.dim_embs * 2, self.args.dim_embs),
                                           nn.LeakyReLU(),
                                           nn.Linear(self.args.dim_embs, self.args.dim_embs),
                                           nn.LeakyReLU(),
                                           nn.Linear(self.args.dim_embs, self.args.dim_embs),
                                           nn.LeakyReLU(),
                                           nn.Linear(self.args.dim_embs, 1)
                                           )
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, u_e, u_v):
        if self.args.predictor == 'dot':
            score = u_e.mul(u_v).sum(dim=-1)
        else:
            x = torch.cat([u_e, u_v], dim=-1)
            score = self.predictor(x).flatten()
        return score


@torch.no_grad()
def test_and_val(pred_y, y, mode='val', epoch=0):
    preds = pred_y.cpu().numpy()
    y = y.cpu().numpy()

    preds[preds >= 0.5]  = 1
    preds[preds < 0.5] = 0
    test_y = y

    auc = roc_auc_score(test_y, preds)
    f1 = f1_score(test_y, preds)
    macro_f1 = f1_score(test_y, preds, average='macro')
    micro_f1 = f1_score(test_y, preds, average='micro')
    pos_ratio = np.sum(test_y) /  len(test_y)
    res = {
        f'{mode}_auc': auc,
        f'{mode}_f1' : f1,
        f'{mode}_pos_ratio': pos_ratio,
        f'{mode}_epoch': epoch,
        f'{mode}_macro_f1' : macro_f1,
        f'{mode}_micro_f1' : micro_f1,
    }
    return res





