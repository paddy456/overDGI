import numpy as np
import torch

from overlapDGI import evaluation


def make_adj(x, n):
    adj = np.zeros((n,n),dtype=float)
    for i in range(0,len(x[0])):
        adj[x[0][i]][x[1][i]] = 1
    return adj

def make_modularity_matrix(adj):
    adj = adj*(torch.ones(adj.shape[0], adj.shape[0]) - torch.eye(adj.shape[0]))
    degrees = adj.sum(dim=0).unsqueeze(1)
    mod = adj - degrees@degrees.t()/adj.sum()
    return mod

def count(label):
    cnt = [0] * 7
    for i  in label:
        cnt[i] += 1
    print(cnt)

def result(pred, labels):
    nmi = evaluation.NMI_helper(pred, labels)
    ac = evaluation.matched_ac(pred, labels)
    f1 = evaluation.cal_F_score(pred, labels)[0]
    return nmi,ac,f1

