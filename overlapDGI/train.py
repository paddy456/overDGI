import argparse
import numpy as np
import torch
from sklearn import cluster
from torch.distributions import kl_divergence
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from overlapDGI.DGI import DeepGraphInfomax
from overlapDGI.model import Encoder, Summarizer, corruption, cluster_net
from overlapDGI.utils import make_adj, make_modularity_matrix, result, count

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default='Cora',
                    help='which network to load')
parser.add_argument('--K', type=int, default=7,
                    help='How many partitions')
parser.add_argument('--clustertemp', type=float, default=30,
                    help='how hard to make the softmax for the cluster assignments')
parser.add_argument('--train_iters', type=int, default=1001,
                    help='number of training iterations')
parser.add_argument('--num_cluster_iter', type=int, default=1,
                    help='number of iterations for clustering')
parser.add_argument('--seed', type=int, default=24, help='Random seed.')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Loading data
dataset = Planetoid(root='/tmp/'+args.dataset, name=args.dataset)

device = torch.device('cpu')
data = dataset[0].to(device)
# adj_all = torch.from_numpy(make_adj(data.edge_index.numpy(), args)).float()
# test_object = make_modularity_matrix(adj_all)

hidden_size = args.hidden
model = DeepGraphInfomax(
    hidden_channels=hidden_size, encoder=Encoder(dataset.num_features, hidden_size),
    summary=Summarizer(),
    corruption=corruption,
    args=args,
    cluster=cluster_net).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)

def train():
    model.train()
    optimizer.zero_grad()
    pos_z, neg_z, summary, mu, r, dist = model(data.x, data.edge_index)
    dgi_loss = model.loss(pos_z, neg_z, summary)
    # modularity_loss = model.modularity(mu,r,pos_z,dist,adj_all,test_object, args)
    comm_loss = model.comm_loss(pos_z,mu)
    pc_given_Z, qc_given_ZA = model.community_dists_probs(dist,data.edge_index)
    c = F.gumbel_softmax(qc_given_ZA.logits, tau=1, hard=True)
    recon_loss = model.recon_loss((pos_z, c), data.edge_index, mu)
    kl_loss = 1.0 * kl_divergence(qc_given_ZA, pc_given_Z).mean()
    loss = kl_loss + comm_loss + recon_loss - modularity_loss + 5 * dgi_loss
    # loss = dgi_loss
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model):
    model.eval()

    with torch.no_grad():
        node_emb, _, _, _, r, _ = model(data.x, data.edge_index)
    r_assign = r.argmax(dim=1)
    cluster_model = cluster.KMeans(n_clusters=dataset.num_classes)
    cluster_model.fit(node_emb.cpu())
    pred = cluster_model.labels_
    labels = data.y.cpu().numpy()

    print('label is:')
    count(labels)
    print('result of kmeans is:')
    count(pred)
    nmi,ac,f1 = result(pred, labels)
    r_nmi,r_ac,r_f1 = result(r_assign.numpy(),labels)
    print("dgi_metrics: ",nmi,ac,f1)
    print("clusternet_METRICS: ",r_nmi,r_ac,r_f1)
    return max(nmi,r_nmi),max(ac,r_ac)

def node_classification_test(model):
    model.eval()
    z, _, _, _, _, _ = model(data.x, data.edge_index)
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask], max_iter=150)
    print('Accuracy of node classification is {}'.format(acc))
    return acc

print('Start training !!!')
stop_cnt = 0
best_idx = 0
patience = 200
min_loss = 1e9
real_epoch = 0
max_nmi = 0
max_ac = 0

for epoch in range(1001):
    loss = train()
    if epoch % 20 == 0 and epoch > 0:
        print('epoch = {}'.format(epoch))
        tmp_mx_nmi,tmp_max_ac = test(model)
        max_nmi = max(max_nmi,tmp_mx_nmi)
        max_ac = max(max_ac,tmp_max_ac)
        node_classification_test(model)
        if loss < min_loss:
            min_loss = loss
            best_idx = epoch
            stop_cnt = 0
            torch.save(model.state_dict(), 'best_model.pkl')
        else:
            stop_cnt += 1
        if stop_cnt >= patience:
            real_epoch = epoch
            break