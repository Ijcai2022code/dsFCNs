import torch
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import random
import argparse
from scipy.io import loadmat
from scipy.linalg import fractional_matrix_power, inv

class GCNLayer(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.ReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, feat, adj):
        feat = self.fc(feat)
        out = torch.bmm(adj, feat)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, num_layers):
        super(GCN, self).__init__()
        n_h = out_ft
        self.layers = []
        self.num_layers = num_layers
        self.layers.append(GCNLayer(in_ft,  n_h))
        self.act = nn.ReLU()
        for __ in range(num_layers - 1):
            self.layers.append(GCNLayer(n_h, n_h))

    def forward(self, feat, adj):
        h_1 = self.layers[0](feat, adj)
        h_1g = torch.sum(h_1, 1)
        for idx in range(self.num_layers - 1):
            h_1 = self.layers[idx + 1](h_1, adj)
            h_1g = torch.cat((h_1g, torch.sum(h_1, 1)), -1)
        return h_1, h_1g


class MLP(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(MLP, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_ft, out_ft)

    def forward(self, x):
        return self.ffn(x) + self.linear_shortcut(x)


class Model(nn.Module):
    def __init__(self, n_in, n_h, num_layers):
        super(Model, self).__init__()
        self.mlp2 = MLP(num_layers * n_h, n_h)
        self.gnn1 = GCN(n_in, n_h, num_layers)
        self.w_list = nn.ModuleList([nn.Linear(2 * n_h,2 * n_h, bias=False) for _ in range(5)])
        self.y_list = nn.ModuleList([nn.Linear(2 * n_h, 1) for _ in range(5)])
        self.att_act1 = nn.Tanh()
        self.act = nn.PReLU()
        self.att_act2 = nn.Softmax(dim=-1)
        self.mlp = nn.Linear(n_h, 2)
        self.mlp_cat = nn.Linear(10 * n_h, 2)
    def concat_att(self, h_list):
        h_combine_list = []
        for i, h in enumerate(h_list):
            h = self.w_list[i](h)
            h = self.y_list[i](h)
            h_combine_list.append(h)
        score = torch.cat(h_combine_list, -1)
        score = self.att_act1(score)
        score = self.att_act2(score)
        score = torch.unsqueeze(score, -1)
        h = torch.stack(h_list, dim=1)
        h = score * h
        h = torch.reshape(h, [h.shape[0], h.shape[1] * h.shape[2]])
        return h
    def forward(self, fea1, fea2, fea3, fea4, fea5, adj1,adj2, adj3, adj4, adj5, diff1, diff2,diff3,diff4,diff5):
        lv1, gv1= self.gnn1(fea1, adj1)
        lv2, gv2= self.gnn1(fea2, adj2)
        lv3, gv3= self.gnn1(fea3, adj3)
        lv4, gv4= self.gnn1(fea4, adj4)
        lv5, gv5= self.gnn1(fea5, adj5)

        difl1, difg1 = self.gnn1(fea1, diff1)
        difl2, difg2 = self.gnn1(fea2, diff2)
        difl3, difg3 = self.gnn1(fea3, diff3)
        difl4, difg4 = self.gnn1(fea4, diff4)
        difl5, difg5 = self.gnn1(fea5, diff5)


        gv1 = self.mlp2(gv1)
        gv2 = self.mlp2(gv2)
        gv3 = self.mlp2(gv3)
        gv4 = self.mlp2(gv4)
        gv5 = self.mlp2(gv5)

        difg1 = self.mlp2(difg1)
        difg2 = self.mlp2(difg2)
        difg3 = self.mlp2(difg3)
        difg4 = self.mlp2(difg4)
        difg5 = self.mlp2(difg5)

        g_list = []
        g_list.append(gv1)
        g_list.append(gv2)
        g_list.append(gv3)
        g_list.append(gv4)
        g_list.append(gv5)
        view_fusion = []

        view_fusion.append(torch.cat([gv1, difg1],dim=1))
        view_fusion.append(torch.cat([gv2, difg2],dim=1))
        view_fusion.append(torch.cat([gv3, difg3],dim=1))
        view_fusion.append(torch.cat([gv4, difg4],dim=1))
        view_fusion.append(torch.cat([gv5, difg5],dim=1))

        g_fusion = self.concat_att(view_fusion)
        g_fusion = self.mlp_cat(g_fusion)

        return lv1, lv2, lv3, lv4, lv5, gv1, gv2,gv3, gv4,gv5, g_fusion, difl1,difl2,difl3,difl4,difl5,difg1,difg2,difg3,difg4,difg5, view_fusion

    def embed(self, fea1, fea2, fea3, fea4, fea5, adj1,adj2, adj3, adj4, adj5, diff1, diff2,diff3,diff4,diff5):
        __, __, __, __, __, gv1, gv2, gv3, gv4,gv5, g_fusion,__, __, __, __, __,__, __, __, __, __,view_fusion = self.forward(fea1, fea2, fea3, fea4, fea5, adj1,adj2, adj3, adj4, adj5, diff1, diff2,diff3,diff4,diff5)
        g_sum = (gv1 + gv2 + gv3 + gv4 + gv5) / 5
        g_sum = self.mlp(g_sum)
        return (g_sum).detach(), g_fusion.detach()