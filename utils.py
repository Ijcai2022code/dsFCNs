import torch
import numpy as np
import os
from scipy.io import loadmat
from scipy.linalg import fractional_matrix_power, inv
import random
def compute_ppr(a, alpha=0.2, self_loop=True):
    # a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + 3 * np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1

def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.)+eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A =  deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A


def load(args):
    imagelist = os.listdir('/home/myj/code/Machine-leaning-in-action/multiplex/ourcode/FTD/')
    basedir = os.path.dirname('/home/myj/code/Machine-leaning-in-action/multiplex/ourcode/FTD/')
    if args.dataset == 'FDT':
        datadir1 = os.path.join(basedir, imagelist[0])
        datadir2 = os.path.join(basedir, imagelist[1])
        datadir3 = os.path.join(basedir, imagelist[2])
        datadir4 = os.path.join(basedir, imagelist[3])
        #
        adj = loadmat(datadir2)
        fea = loadmat(datadir3)
        graph = loadmat(datadir4)
        label = loadmat(datadir1)
        A = fea['FTD']
        B = adj['FTDadj']
        C = graph['FTgraph']
        D = label['label']
        AA = np.array(A.tolist()[0]).reshape(-1, 5, )
        BB = np.array(B.tolist()[0]).reshape(-1, 5, )
        CC = np.array(C.tolist()[0]).reshape(-1, 5, )
        label = np.array(D.tolist()[0])
        lable2 = torch.from_numpy(label)

        fea1 = np.array(AA[:, 0].tolist()).reshape(-1, 90, 90)
        fea2 = np.array(AA[:, 1].tolist()).reshape(-1, 90, 90)
        fea3 = np.array(AA[:, 2].tolist()).reshape(-1, 90, 90)
        fea4 = np.array(AA[:, 3].tolist()).reshape(-1, 90, 90)
        fea5 = np.array(AA[:, 4].tolist()).reshape(-1, 90, 90)

        fea1 = torch.from_numpy(fea1)
        fea2 = torch.from_numpy(fea2)
        fea3 = torch.from_numpy(fea3)
        fea4 = torch.from_numpy(fea4)
        fea5 = torch.from_numpy(fea5)

        adj1 = np.array(BB[:, 0].tolist()).reshape(-1, 90, 90)
        adj2 = np.array(BB[:, 1].tolist()).reshape(-1, 90, 90)
        adj3 = np.array(BB[:, 2].tolist()).reshape(-1, 90, 90)
        adj4 = np.array(BB[:, 3].tolist()).reshape(-1, 90, 90)
        adj5 = np.array(BB[:, 4].tolist()).reshape(-1, 90, 90)
        diff1 = []
        diff2 = []
        diff3 = []
        diff4 = []
        diff5 = []

        for i in range(0, adj1.shape[0]):
            diff1.append(compute_ppr(adj1[i], alpha=0.2))
            diff2.append(compute_ppr(adj2[i], alpha=0.2))
            diff3.append(compute_ppr(adj3[i], alpha=0.2))
            diff4.append(compute_ppr(adj4[i], alpha=0.2))
            diff5.append(compute_ppr(adj5[i], alpha=0.2))

        diff1 = torch.from_numpy(np.array(diff1))
        diff2 = torch.from_numpy(np.array(diff2))
        diff3 = torch.from_numpy(np.array(diff3))
        diff4 = torch.from_numpy(np.array(diff4))
        diff5 = torch.from_numpy(np.array(diff5))

        adj1 = torch.from_numpy(adj1)
        adj2 = torch.from_numpy(adj2)
        adj3 = torch.from_numpy(adj3)
        adj4 = torch.from_numpy(adj4)
        adj5 = torch.from_numpy(adj5)



        fea1 = fea1.to(torch.float32)
        fea2 = fea2.to(torch.float32)
        fea3 = fea3.to(torch.float32)
        fea4 = fea4.to(torch.float32)
        fea5 = fea5.to(torch.float32)

        adj1 = adj1.to(torch.float32)
        adj2 = adj2.to(torch.float32)
        adj3 = adj3.to(torch.float32)
        adj4 = adj4.to(torch.float32)
        adj5 = adj5.to(torch.float32)

        diff1 = diff1.to(torch.float32)
        diff2 = diff2.to(torch.float32)
        diff3 = diff3.to(torch.float32)
        diff4 = diff4.to(torch.float32)
        diff5 = diff5.to(torch.float32)

        adj1 = normalize_graph(adj1 + 3 * torch.eye(adj1[0].shape[1]))
        adj2 = normalize_graph(adj2 + 3 * torch.eye(adj2[0].shape[1]))
        adj3 = normalize_graph(adj3 + 3 * torch.eye(adj3[0].shape[1]))
        adj4 = normalize_graph(adj4 + 3 * torch.eye(adj4[0].shape[1]))
        adj5 = normalize_graph(adj5 + 3 * torch.eye(adj5[0].shape[1]))
    return fea1,fea2,fea3,fea4,fea5,adj1, adj2, adj3, adj4, adj5,label,lable2, diff1, diff2, diff3, diff4, diff5



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sen_spe(a,b):
    TP=0
    FN=0
    TN=0
    FP=0
    for i in range(len(a)):
        if a[i]==1 and b[i]==1:
            TP=TP+1
        elif a[i]==1 and b[i]==0:
            FN=FN+1
        elif a[i]==0 and b[i]==1:
            FP=FP+1
        elif a[i]==0 and b[i]==0:
            TN=TN+1
        else: pass

    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    return TPR,TNR

def train_test(data_label, label_rate):
    if label_rate == 0.2:
        index = list(range(0, len(data_label)))
        n = int(len(index) * label_rate)
        random.shuffle(index)
        train_0 = index[:n]
        train_1 = index[n:2*n]
        train_2 = index[2*n:3*n]
        train_3 = index[3*n:4*n]
        train_4 = index[4*n:]
        train_index = [train_0, train_1, train_2, train_3, train_4]
        test_0 = train_1 + train_2 + train_3 + train_4
        test_1 = train_0 + train_2 + train_3 + train_4
        test_2 = train_0 + train_1 + train_3 + train_4
        test_3 = train_0 + train_1 + train_2 + train_4
        test_4 = train_0 + train_1 + train_2 + train_3
        test_index = [test_0, test_1, test_2, test_3, test_4]
    elif label_rate == 0.8:
        index = list(range(0, len(data_label)))
        n = int(len(index) * 0.2)
        random.shuffle(index)
        train_0 = index[:n]
        train_1 = index[n:2 * n]
        train_2 = index[2 * n:3 * n]
        train_3 = index[3 * n:4 * n]
        train_4 = index[4 * n:]
        test_index = [train_0, train_1, train_2, train_3, train_4]
        test_0 = train_1 + train_2 + train_3 + train_4
        test_1 = train_0 + train_2 + train_3 + train_4
        test_2 = train_0 + train_1 + train_3 + train_4
        test_3 = train_0 + train_1 + train_2 + train_4
        test_4 = train_0 + train_1 + train_2 + train_3
        train_index = [test_0, test_1, test_2, test_3, test_4]
    return train_index, test_index