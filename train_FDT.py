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
from utils import load
from utils import accuracy, sen_spe, train_test
from model import Model
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score, recall_score,  average_precision_score


def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples

    if average:
        return Eq.mean()
    else:
        return Eq


def local_global_loss_(l_enc, g_enc, batch, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]
    # max_nodes = num_nodes // num_graphs

    pos_mask = torch.zeros((num_nodes, num_graphs))
    neg_mask = torch.ones((num_nodes, num_graphs))

    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.


    res = torch.mm(l_enc, g_enc.t())
    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    return E_neg - E_pos


def train_FTD(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(args.gpu_num)

    fea1, fea2, fea3, fea4, fea5, adj1, adj2, adj3, adj4, adj5, label,label2,  diff1, diff2, diff3, diff4, diff5 = load(args)


    idx_train, idx_test = train_test(label, args.train_rate)

    nb_epochs = args.nb_epochs
    batch_size = args.batch_size
    lr = args.lr
    l2_coef = args.l2_coef
    hid_units = args.hid_units
    num_layer = args.num_layer
    ft_size = fea1[0].shape[1]
    max_nodes = fea1[0].shape[0]

    model = Model(ft_size, hid_units, num_layer)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    itr = (adj1.shape[0] // batch_size) + 1


    label[86:] = 0
    label = torch.LongTensor(label)
    acc_list = []
    sen_list = []
    spe_list = []
    AUC_list = []
    acc = []
    auc = []
    sen = []
    spe = []
    celoss = torch.nn.CrossEntropyLoss()
    my_margin = 0.8
    margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduce=False)

    for id_train, id_test in zip(idx_train, idx_test):
        for epoch in tqdm(range(nb_epochs+1)):
            epoch_loss = 0.0
            train_idx = np.arange(adj1.shape[0])
            np.random.shuffle(train_idx)

            train_idx = torch.LongTensor(id_train)
            test_idx = torch.LongTensor(id_test)

            for idx in range(0, len(train_idx), batch_size):
                model.train()
                optimiser.zero_grad()
                batch = train_idx[idx: idx + batch_size]
                idx_list = []
                for i in range(1):
                    idx_0 = np.random.permutation(len(batch))
                    idx_list.append(idx_0)
                batch_repeat = torch.LongTensor(np.repeat(np.arange(batch.shape[0]), 90)).cuda()

                lv1, lv2, lv3, lv4, lv5, gv1, gv2,gv3, gv4,gv5, g_fusion, difl1,difl2,difl3,difl4,difl5,difg1,difg2,difg3,difg4,difg5,view_fusion = model(fea1[batch], fea2[batch], fea3[batch], fea4[batch], fea5[batch], adj1[batch], adj2[batch],
                                 adj3[batch], adj4[batch], adj4[batch], diff1[batch], diff2[batch], diff3[batch], diff4[batch], diff5[batch])

                lv1 = lv1.view(batch.shape[0] * max_nodes, -1)
                lv2 = lv2.view(batch.shape[0] * max_nodes, -1)
                lv3 = lv3.view(batch.shape[0] * max_nodes, -1)
                lv4 = lv4.view(batch.shape[0] * max_nodes, -1)
                lv5 = lv5.view(batch.shape[0] * max_nodes, -1)

                difl1 = difl1.view(batch.shape[0] * max_nodes, -1)
                difl2 = difl2.view(batch.shape[0] * max_nodes, -1)
                difl3 = difl3.view(batch.shape[0] * max_nodes, -1)
                difl4 = difl4.view(batch.shape[0] * max_nodes, -1)
                difl5 = difl5.view(batch.shape[0] * max_nodes, -1)

                measure = 'JSD'
                loss6 = local_global_loss_(lv1, difg1, batch_repeat, measure)
                loss6f = local_global_loss_(difl1, gv1, batch_repeat, measure)
                loss7 = local_global_loss_(lv2, difg2, batch_repeat, measure)
                loss7f = local_global_loss_(difl2, gv2, batch_repeat, measure)
                loss8 = local_global_loss_(lv3, difg3, batch_repeat, measure)
                loss8f = local_global_loss_(difl3, gv3, batch_repeat, measure)
                loss9 = local_global_loss_(lv4, difg4, batch_repeat, measure)
                loss9f = local_global_loss_(difl4, gv4, batch_repeat, measure)
                loss10 = local_global_loss_(lv5, difg5, batch_repeat, measure)
                loss10f = local_global_loss_(difl5, gv5, batch_repeat, measure)
                loss_mar_0 = 0
                loss_mar_1 = 0
                loss_mar_2 = 0
                loss_mar_3 = 0
                margin_label = -1 * torch.ones(len(batch))
                s_p_0 = F.pairwise_distance(view_fusion[0], view_fusion[1])
                s_n_0 = F.pairwise_distance(view_fusion[0], view_fusion[1][idx_list[0]])
                loss_mar_0 += (margin_loss(s_p_0, s_n_0, margin_label)).mean()

                s_p_1 = F.pairwise_distance(view_fusion[1], view_fusion[2])
                s_n_1 = F.pairwise_distance(view_fusion[1], view_fusion[2][idx_list[0]])
                loss_mar_1 += (margin_loss(s_p_1, s_n_1, margin_label)).mean()

                s_p_2 = F.pairwise_distance(view_fusion[2], view_fusion[3])
                s_n_2 = F.pairwise_distance(view_fusion[2], view_fusion[3][idx_list[0]])
                loss_mar_2 += (margin_loss(s_p_2, s_n_2, margin_label)).mean()

                s_p_3 = F.pairwise_distance(view_fusion[3], view_fusion[4])
                s_n_3 = F.pairwise_distance(view_fusion[3], view_fusion[4][idx_list[0]])
                loss_mar_3 += (margin_loss(s_p_3, s_n_3, margin_label)).mean()
                loss_ce = celoss(g_fusion,label[batch])


                loss_c =  args.w_ce * loss_ce +  args.w_gl * (loss6 + loss7 + loss8 + loss9
                        + loss10 + loss6f + loss7f + loss8f + loss9f + loss10f) + args.w_gg * ( loss_mar_0 + loss_mar_1 + loss_mar_2 + loss_mar_3)


                epoch_loss += loss_c
                loss_c.backward()
                optimiser.step()

            epoch_loss /= itr
            if epoch % args.test_epoch == 0 and epoch !=0 :
                model.eval()
                batch_test = test_idx
                embeds, g_fusion = model.embed(fea1[batch_test], fea2[batch_test], fea3[batch_test], fea4[batch_test], fea5[batch_test], adj1[batch_test], adj2[batch_test],
                                 adj3[batch_test], adj4[batch_test], adj4[batch_test], diff1[batch_test], diff2[batch_test], diff3[batch_test], diff4[batch_test], diff5[batch_test])
                acc_test = accuracy(g_fusion, label[id_test])
                preds = torch.argmax(g_fusion, dim=1)
                try:
                    auc_test = roc_auc_score(preds, label[id_test])
                except:
                    auc_test = acc_test * 0
                try:
                    sen_test, spe_test = sen_spe(preds, label[id_test])
                except:
                    sen_test = acc_test * 0
                    spe_test = acc_test * 0
                sen_list.append(sen_test)
                spe_list.append(spe_test)
                AUC_list.append(auc_test)
                acc_list.append(acc_test)
                print('acc',acc_test)
                print('AUC', auc)
                print('SEN', sen)
                print('SPE', spe)
    # torch.save(model.state_dict(), f'{22}-{13}.pkl')
    if args.train_rate == 0.8 or args.train_rate == 0.2:
        acc.append(float(acc_list[args.test_epoch]))
        acc.append(float(acc_list[args.test_epoch]))
        acc.append(float(acc_list[args.test_epoch]))
        acc.append(float(acc_list[args.test_epoch]))
        acc.append(float(acc_list[args.test_epoch]))

        auc.append(float(AUC_list[args.test_epoch]))
        auc.append(float(AUC_list[args.test_epoch]))
        auc.append(float(AUC_list[args.test_epoch]))
        auc.append(float(AUC_list[args.test_epoch]))
        auc.append(float(AUC_list[args.test_epoch]))

        sen.append(float(sen_list[args.test_epoch]))
        sen.append(float(sen_list[args.test_epoch]))
        sen.append(float(sen_list[args.test_epoch]))
        sen.append(float(sen_list[args.test_epoch]))
        sen.append(float(sen_list[args.test_epoch]))

        spe.append(float(spe_list[args.test_epoch]))
        spe.append(float(spe_list[args.test_epoch]))
        spe.append(float(spe_list[args.test_epoch]))
        spe.append(float(spe_list[args.test_epoch]))
        spe.append(float(spe_list[args.test_epoch]))

    print('acc', acc, 'acc_mean', np.mean(acc))
    print('auc', auc, 'auc_mean', np.mean(auc))
    print('sen', sen, 'sen_mean', np.mean(sen))
    print('spe', spe, 'spe_mean', np.mean(spe))
    return  acc, acc_list, auc, AUC_list, sen, sen_list, spe, spe_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedder', nargs='?', default='multiview_FTD')
    parser.add_argument('--dataset', nargs='?', default='FDT')
    parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
    parser.add_argument('--sparse', type=bool, default=True, help='sparse adjacency matrix')
    parser.add_argument('--nb_epochs', type=int, default=200, help='the number of epochs')
    parser.add_argument('--test_epoch', type=int, default=20, help='test_epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gpu_num', type=int, default=0, help='the id of gpu to use')
    parser.add_argument('--seed', type=int, default=0, help='the seed to use')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--hid_units', type=int, default=64, help='hid_units')
    parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
    parser.add_argument('--l2_coef', type=float, default=0.0, help='')
    parser.add_argument('--train_rate', type=float, default=0.8, help='')
    parser.add_argument('--num_run', type=int, default=1, help='the number of runs')
    parser.add_argument('--num_layer', type=int, default=3, help='the number of layers')
    parser.add_argument('--w_ce', type=float, default=10, help='weight of cross entropy loss')
    parser.add_argument('--w_gl', type=float, default=0.9, help='weight of global-local')
    parser.add_argument('--w_gg', type=float, default=0.1, help='weight of global-global')


    return parser.parse_known_args()


def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)



def main():
    args, unknown = parse_args()
    printConfig(args)

    acc_seedlist = []
    auc_seedlist = []
    sen_seedlist = []
    spe_seedlist = []
    for i in range(0, args.num_run):
        args.seed = i
        if args.embedder == "multiview_FTD":
            acc, acc_list, auc, AUC_list, sen, sen_list, spe, spe_list = train_FTD(
                args)
        acc_seedlist.append(np.mean(acc))
        auc_seedlist.append(np.mean(auc))
        sen_seedlist.append(np.mean(sen))
        spe_seedlist.append(np.mean(spe))
    acc_final = np.mean(np.array(acc_seedlist))
    auc_final = np.mean(np.array(auc_seedlist))
    sen_final = np.mean(np.array(sen_seedlist))
    spe_final = np.mean(np.array(spe_seedlist))
    filePath = "log"
    exp_ID = 0
    for filename in os.listdir(filePath):
        file_info = filename.split("_")
        file_dataname = file_info[0]
        if file_dataname == args.dataset:
            exp_ID = max(int(file_info[1]), exp_ID)
    exp_name = args.dataset + "_" + str(exp_ID + 1)
    exp_name = os.path.join(filePath, exp_name)
    os.makedirs(exp_name)
    arg_file = open(os.path.join(exp_name, "arg.txt"), "a")
    for k, v in sorted(args.__dict__.items()):
        arg_file.write("\n- {}: {}".format(k, v))
    if args.train_rate == 0.8 or args.train_rate == 0.2:
        os.rename(exp_name,
                  exp_name + "_" + '%.4f' % acc[0] + "_" + '%.4f' % acc[1] + "_" + '%.4f' % acc[
                      2] + "_" + '%.4f' % acc[3] \
                  + "_" + '%.4f' % acc[4] + "mean_" + '%.4f' % acc_final)
        arg_file.write(
            "\n- fold_maxacc:{},{},{},{},{},meanacc_:{},meanauc_:{},meansen_:{},meanspe_:{}".format(
                acc[0], acc[1],
                acc[2], acc[3],
                acc[4],
                acc_final,
                auc_final, sen_final, spe_final))
    arg_file.writelines(str(np.array(acc_list)))
    arg_file.writelines(str(np.array(AUC_list)))
    arg_file.writelines(str(np.array(sen_list)))
    arg_file.writelines(str(np.array(spe_list)))
    arg_file.close()


    return acc_max


if __name__ == '__main__':
       acc_max = main()






