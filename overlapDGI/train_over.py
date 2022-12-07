import argparse
import math
import warnings

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch.distributions import kl_divergence
from DGI import DeepGraphInfomax
from helpers import scores, Scores, kv_to_print_str, matrix_to_cnl_format
from overlapDGI.model import Summarizer, corruption, cluster_net, Encoder
from utils_from_vgraph import calc_f1, calc_jaccard
import networkx as nx
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")



from _nocd.utils import load_dataset

# channels = 16
community_pred_threshold = 0.3
# loader = load_dataset("../_nocd/data/facebook_ego/fb_0.npz")
loader = load_dataset("data/mag_eng.npz")

A, X, Z_gt = loader['A'], loader['X'], loader['Z']
num_communities = Z_gt.shape[1]
communities_cnl_format = matrix_to_cnl_format(Z_gt.T, num_communities)
x_norm = normalize(X)
x_norm = torch.tensor(x_norm.todense())

graph = nx.from_numpy_matrix(A)


def cal_sim(g, u, v):
    v_set = set(g.neighbors(v))
    u_set = set(g.neighbors(u))
    inter = v_set.intersection(u_set)
    if inter == 0:
        return 0
    sim = (len(inter) + 2) / (math.sqrt((len(v_set) + 1) * (len(u_set) + 1)))

    return sim


centers = []

for node in graph.nodes:
    sim = 0
    for nbr in graph.neighbors(node):
        if graph.degree(nbr) > graph.degree(node):
            sim_ = cal_sim(graph, node, nbr)
            sim = max(sim, sim_)
    if sim == 0:

        centers.append((node, graph.degree(node)))
    else:

        centers.append((node, graph.degree(node) * (1 - sim)))

centers = sorted(centers, key=lambda v: v[1], reverse=True)
k_centers = []

for k_c in centers[0:num_communities]:
    k_centers.append(k_c[0])

print(k_centers)

train_pos_edge_index = torch.tensor((A.tocoo().row, A.tocoo().col), dtype=torch.long)

if len(Z_gt.shape) < 2:
    N = A.shape[0]
    K = len(set(Z_gt))
else:
    N, K = Z_gt.shape

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate.')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--clustertemp', type=float, default=30,
                    help='how hard to make the softmax for the cluster assignments')
parser.add_argument('--num_cluster_iter', type=int, default=1,
                    help='number of iterations for clustering')
parser.add_argument('--seed', type=int, default=24, help='Random seed.')
args = parser.parse_args()
args.K = K
args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device('cpu')
# data = data.to(device)
# adj_all = torch.from_numpy(make_adj(data.edge_index.numpy(), data.num_nodes)).float()
# test_object = make_modularity_matrix(adj_all)

hidden_size = args.hidden
model = DeepGraphInfomax(
    # hidden_channels=hidden_size, encoder=Encoder(data.num_features, hidden_size),
    hidden_channels=hidden_size, encoder=Encoder(x_norm.size(1), hidden_size),
    summary=Summarizer(),
    corruption=corruption,
    args=args,
    cluster=cluster_net).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train():
    model.train()
    optimizer.zero_grad()
    # pos_z, neg_z, summary, mu, r, dist = model(data.x, data.edge_index)
    pos_z, neg_z, summary, mu, r, dist = model(x_norm, train_pos_edge_index, k_centers)

    dgi_loss = model.loss(pos_z, neg_z, summary)
    # modularity_loss = model.modularity(mu,r,pos_z,dist,adj_all,test_object, args)
    comm_loss = model.comm_loss(pos_z, mu)
    pc_given_Z, qc_given_ZA = model.community_dists_probs(dist, train_pos_edge_index, alpha=0.3)
    c = F.gumbel_softmax(qc_given_ZA.logits, tau=1, hard=True)
    # recon_loss = model.recon_loss((pos_z, c), data.edge_index, mu)
    kl_loss = 1.0 * kl_divergence(qc_given_ZA, pc_given_Z).mean()
    loss = dgi_loss + comm_loss + kl_loss
    # loss = dgi_loss
    loss.backward()
    optimizer.step()
    return loss.item()


maxf1 = 0
maxja = 0

for epoch in range(500):
    loss = train()

    if epoch % 1 == 0:
        model.eval()
        # pos_z, neg_z, summary, mu, r, dist = model(data.x, data.edge_index)
        pos_z, neg_z, summary, mu, r, dist = model(x_norm, train_pos_edge_index, k_centers)

        _, qc_given_ZA = model.community_dists_probs(dist, train_pos_edge_index)
        # pre_comm_scores_weighted = (qc_given_ZA.probs / qc_given_ZA.probs.max(dim=1, keepdim=True)[0])
        pre_comm_scores_weighted = (r / r.max(dim=1, keepdim=True)[0])
        pre_comm_scores_weighted_thresholded = (
                pre_comm_scores_weighted > community_pred_threshold).detach().cpu().numpy()
        # pre_comm_scores_weighted_thresholded = (
        #             pre_comm_scores_weighted > community_pred_threshold).detach().cpu().numpy()
        pre_cnl_format = matrix_to_cnl_format(pre_comm_scores_weighted_thresholded.T, num_communities)

        # labelled_idx = [x for r in communities_cnl_format for x in r]
        # for i in range(len(pre_cnl_format)):
        #     pre_cnl_format[i] = [x for x in pre_cnl_format[i] if x in labelled_idx]

        f1 = calc_f1(communities_cnl_format, pre_cnl_format)
        ja = calc_jaccard(communities_cnl_format, pre_cnl_format)

        maxf1 = max(f1, maxf1)
        maxja = max(maxja, ja)

        metrics = scores(
            [Scores.COMMUNITY_OVERLAPPING_F1, Scores.COMMUNITY_OVERLAPPING_JACCARD],
            print_down=False, match_labels=False, communities_cnl=communities_cnl_format,
            communities_cnl_pred=pre_cnl_format
        )

        print("Epoch: {}\t".format(epoch) + kv_to_print_str(metrics))

print(f"f1:{maxf1}\tja:{maxja}")
