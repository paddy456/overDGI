import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, APPNP
import math

def GELU(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(1433, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, 7, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class APPNP_model(torch.nn.Module):
    def __init__(self):
        super(APPNP_model, self).__init__()
        self.lin1 = Linear(1433, 64)
        self.lin2 = Linear(64, 7)
        self.prop1 = APPNP(10, 0.1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels) # , cached=True)
        self.prelu = nn.PReLU(hidden_channels)


    def forward(self, x, edge_index, k_centers):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        # x = self.prop(x, edge_index)
        return x

class Summarizer(nn.Module):
    def __init__(self):
        super(Summarizer, self).__init__()
    
    def forward(self, z):
        return torch.sigmoid(z.mean(dim=0))

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index





class Summarizer(nn.Module):
    def __init__(self):
        super(Summarizer, self).__init__()

    def forward(self, z):
        return torch.sigmoid(z.mean(dim=0))


def corruption(x, edge_index, k_centers):
    return x[torch.randperm(x.size(0))], edge_index


def cluster_net(data, k, temp, num_iter, cluster_temp, init):

    data = torch.diag(1. / torch.norm(data, p=2, dim=1)) @ data

    mu = init
    n = data.shape[0]
    d = data.shape[1]

    for t in range(num_iter):
        # get distances between all data points and cluster centers
        #        dist = torch.cosine_similarity(data[:, None].expand(n, k, d).reshape((-1, d)), mu[None].expand(n, k, d).reshape((-1, d))).reshape((n, k))
        dist = data @ mu.t()
        # cluster responsibilities via softmax
        r = torch.softmax(cluster_temp * dist, 1)
        # total responsibility of each cluster
        cluster_r = r.sum(dim=0)
        # mean of points in each cluster weighted by responsibility
        cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1)
        # update cluster means
        new_mu = torch.diag(1 / cluster_r) @ cluster_mean
        mu = new_mu
    dist = data @ mu.t()
    r = torch.softmax(cluster_temp * dist, 1)
    return mu, r, dist