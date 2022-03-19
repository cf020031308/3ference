# Usage: python3 code.py tfs-reuse cora 6 -1
import sys
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import dgl
import dgl.nn
from ogb.nodeproppred import NodePropPredDataset


if len(sys.argv) > 3:
    g_method, g_data, g_split, *gcard = sys.argv[1:]
    gcard.append('0')
    if g_split.lower() in '0 f false':
        g_split = '0'
else:
    g_method = 'tfs-reuse'
    g_data = 'cora'
    g_split = '6'
g_split = int(g_split)
epochs = 200
batch_size = 1024
hid = 64

# hid = int(gcard[0])
# gcard[0] = 1

dev = torch.device('cuda:%d' % int(gcard[0]))
gpu = lambda x: x.to(dev)


def optimize(params, lr=0.01):
    if run == 0:
        print('params:', sum(p.numel() for p in params))
    return optim.Adam(params, lr=lr)


def speye(n):
    return torch.sparse_coo_tensor(
        torch.arange(n).view(1, -1).repeat(2, 1), [1] * n)


def spnorm(A, eps=1e-5):
    D = (torch.sparse.sum(A, dim=1).to_dense() + eps) ** -0.5
    indices = A._indices()
    return torch.sparse_coo_tensor(indices, D[indices[0]] * D[indices[1]])


def dot(x, y):
    return (x.unsqueeze(-2) @ y.unsqueeze(-1)).squeeze(-1).squeeze(-1)


def count_subgraphs(src, dst, n):
    val = torch.arange(n)
    for _ in range(100):
        idx = val[src] < val[dst]
        val[src[idx]] = val[dst[idx]]
    return val.unique().shape[0]


def build_knn_graph(x, base, k=4, b=512, ignore_self=False):
    n = x.shape[0]
    weight = gpu(torch.zeros((n, k)))
    adj = gpu(torch.zeros((n, k), dtype=int))
    for i in range(0, n, b):
        knn = (
            (x[i:i+b].unsqueeze(1) - base.unsqueeze(0))
            .norm(dim=2)
            .topk(k + int(ignore_self), largest=False))
        val = knn.values[:, 1:] if ignore_self else knn.values
        idx = knn.indices[:, 1:] if ignore_self else knn.indices
        val = torch.softmax(-val, dim=-1)
        weight[i:i+b] = val
        adj[i:i+b] = idx
    return weight, adj


class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, A=None):
        super(self.__class__, self).__init__()
        self.conv1 = gpu(nn.Linear(in_feats, hid_feats))
        self.conv2 = gpu(nn.Linear(hid_feats, out_feats))
        self.A = A

    def forward(self, feats):
        h = self.conv1(self.A @ feats)
        h = F.leaky_relu(h)
        h = self.conv2(self.A @ h)
        return h


class MLP(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(self.__class__, self).__init__()
        self.pred = gpu(nn.Sequential(
            nn.Linear(in_feats, hid_feats),
            nn.LeakyReLU(),
            nn.Linear(hid_feats, hid_feats),
            nn.LeakyReLU(),
            nn.Linear(hid_feats, out_feats),
        ))

    def forward(self, x):
        return self.pred(x)


class Res(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(self.__class__, self).__init__()
        self.dec = gpu(nn.Linear(in_feats, hid_feats))
        self.pred = gpu(nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(2 * hid_feats, out_feats),
            nn.Tanh(),
        ))
        # self.w = nn.Parameter(gpu(torch.ones(1, out_feats)))
        # self.b = nn.Parameter(gpu(torch.zeros(1, out_feats)))

    def forward(self, x, y):
        h = torch.cat((x, y), dim=-1)
        return self.pred(h)
        # return self.w * self.pred(h) + self.b


class Res3(nn.Module):
    def __init__(self, din, hid, dout):
        super(self.__class__, self).__init__()
        self.enc = gpu(nn.Sequential(
            nn.Linear(din, hid),
            nn.LeakyReLU(),
        ))
        self.res = gpu(nn.Linear(hid * 2, dout))

    def forward(self, x, x0, y0):
        return self.res(torch.cat((
            self.enc(x).unsqueeze(1).repeat(1, x0.shape[1], 1),
            self.enc(x0)
        ), dim=-1))


class TF(nn.Module):
    def __init__(self, din, hid, dout):
        super(self.__class__, self).__init__()
        self.enc = gpu(nn.Sequential(
            nn.Linear(din, hid),
            nn.LeakyReLU(),
        ))
        self.res = gpu(nn.Linear(hid * 2 + dout, dout))

    def forward(self, x, x0, y0, logw=None):
        y = self.res(torch.cat((
            self.enc(x).unsqueeze(1).repeat(1, x0.shape[1], 1),
            self.enc(x0),
            y0,
        ), dim=-1))
        w = y0 * 0
        if logw is not None:
            w = w + logw.unsqueeze(-1)
        y = (y * torch.softmax(w, dim=1)).sum(dim=1)
        return y


class KD(nn.Module):
    def __init__(self, din, hid, dout):
        super(self.__class__, self).__init__()
        self.enc = gpu(nn.Sequential(
            nn.Linear(din, hid),
            nn.LeakyReLU(),
        ))
        self.res = gpu(nn.Linear(hid * 2, dout))
        self.W = nn.Parameter(gpu(torch.rand(dout, dout)))

    def forward(self, x, x0, y0):
        return y0 @ self.W + self.res(
            torch.cat((self.enc(x), self.enc(x0)), dim=-1))


if g_data == 'weekday':
    startdate = datetime.date(1980, 1, 1)
    enddate = datetime.date(2020, 1, 1)
    delta = datetime.timedelta(days=1)
    fmt = '%Y%m%d'
    node_features, node_labels = [], []
    while startdate < enddate:
        node_labels.append(startdate.weekday())
        node_features.append([float(c) for c in startdate.strftime(fmt)])
        startdate += delta
    node_features = gpu(torch.tensor(node_features))
    node_labels = gpu(torch.tensor(node_labels, dtype=int))
    n_nodes = node_features.shape[0]
    _, adj = build_knn_graph(
        node_features, node_features, ignore_self=True)
    src = torch.arange(adj.shape[0]).repeat(adj.shape[1])
    dst = torch.cat([adj[:, i] for i in range(adj.shape[1])], dim=0)
    graph = dgl.graph((src.cpu(), dst.cpu()))
    graph.ndata['feat'] = node_features.cpu()
    graph.ndata['label'] = node_labels.cpu()
    # print('\n'.join([
    #     'weekday graph generated.',
    #     '  NumNodes: %d' % n_nodes,
    #     '  NumEdges: %d' % src.shape[0],
    #     '  NumFeats: 8',
    #     '  NumClasses: 7',
    # ]))
elif g_data in ('proteins', 'arxiv'):
    dataset = NodePropPredDataset(name='ogbn-%s' % g_data)
    g = dataset[0][0]
    edge = g['edge_index']
    graph = dgl.graph((edge[0, :], edge[1, :]))
    if g_data == 'proteins':
        graph.ndata['feat'] = torch.rand(g['num_nodes'], 1)
        labels = {l: i for i, l in enumerate(set(g['node_species'][:, 0]))}
        graph.ndata['label'] = torch.tensor([
            labels[k] for k in g['node_species'][:, 0]])
    elif g_data == 'arxiv':
        graph.ndata['feat'] = torch.from_numpy(g['node_feat'])
        graph.ndata['label'] = torch.from_numpy(dataset[0][1][:, 0])
    n_nodes = g['num_nodes']
    train_mask = torch.zeros(n_nodes, dtype=bool)
    valid_mask = torch.zeros(n_nodes, dtype=bool)
    test_mask = torch.zeros(n_nodes, dtype=bool)
    idx = dataset.get_idx_split()
    train_mask[idx['train']] = True
    valid_mask[idx['valid']] = True
    test_mask[idx['test']] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = valid_mask
    graph.ndata['test_mask'] = test_mask
else:
    graph = (
        dgl.data.CoraGraphDataset()[0] if g_data == 'cora'
        else dgl.data.CiteseerGraphDataset()[0] if g_data == 'citeseer'
        else dgl.data.PubmedGraphDataset()[0] if g_data == 'pubmed'
        else dgl.data.CoauthorCSDataset()[0] if g_data == 'coauthor-cs'
        else dgl.data.CoauthorPhysicsDataset()[0] if g_data == 'coauthor-phy'
        else dgl.data.RedditDataset()[0] if g_data == 'reddit'
        else dgl.data.AmazonCoBuyComputerDataset()[0] if g_data == 'amazon-com'
        else dgl.data.AmazonCoBuyPhotoDataset()[0] if g_data == 'amazon-photo'
        else None
    )
    # if g_data.startswith('coauthor'):
    #     print('Fix bug of https://github.com/dmlc/dgl/issues/2553 @20210121')
    #     src, _ = graph.edges()
    #     e = src.shape[0] // 2
    #     g = dgl.graph((src[:e], src[e:]))
    #     g.ndata.update(graph.ndata)
    #     graph = g
node_features = gpu(graph.ndata['feat'])
node_labels = gpu(graph.ndata['label'])
src, dst = graph.edges()
# flt = src >= dst
# src = src[flt]
# dst = dst[flt]
n_nodes = node_features.shape[0]
n_edges = src.shape[0]
n_features = node_features.shape[1]
n_labels = int(node_labels.max().item() + 1)
print('nodes: %d' % n_nodes)
print('features: %d' % n_features)
print('classes: %d' % n_labels)
print('edges: %d' % ((n_edges - (src == dst).sum().item())))
print('degree: %.2f' % ((1 if g_data == 'weekday' else 2) * n_edges / n_nodes))
print('subgraphs:', count_subgraphs(src, dst, n_nodes))
print('intra_rate: %.2f%%' % (
    100 * (node_labels[src] == node_labels[dst]).sum().float() / n_edges))
# print('self-loops: %d' % src[src == dst].unique().shape[0])
# labelpair = torch.zeros(n_labels, n_labels)
# for s, d in zip(node_labels[src], node_labels[dst]):
#     labelpair[d, s] += 1
# labelpair = labelpair / labelpair.sum(dim=-1, keepdim=True)
# print('labelpair:', labelpair.tolist())
# exit()


class Stat(object):
    def __init__(self, name=''):
        self.name = name
        self.accs = []
        self.times = []
        self.best_accs = []
        self.best_times = []

    def __call__(self, logits, startfrom=0):
        self.accs.append([
            ((logits[mask].max(dim=1).indices == node_labels[mask]).sum()
             / gpu(mask).sum().float()).item()
            for mask in (train_mask, valid_mask, test_mask)
        ])
        self.times.append(time.time() - self.tick)

    def start_run(self):
        self.tick = time.time()

    def end_run(self):
        self.accs = torch.tensor(self.accs)
        # print('best:', self.accs.max(dim=0).values)
        idx = self.accs.max(dim=0).indices[1]
        self.best_accs.append((idx, self.accs[idx, 2]))
        self.best_times.append(self.times[idx])
        self.accs = []
        self.times = []
        # print('best:', self.best_accs[-1])

    def end_all(self):
        conv = 1.0 + torch.tensor([idx for idx, _ in self.best_accs])
        acc = 100 * torch.tensor([acc for _, acc in self.best_accs])
        tm = torch.tensor(self.best_times)
        print(self.name)
        print('time:%.3f±%.3f' % (tm.mean().item(), tm.std().item()))
        print('conv:%.3f±%.3f' % (conv.mean().item(), conv.std().item()))
        print('acc:%.2f±%.2f' % (acc.mean().item(), acc.std().item()))


evaluate = Stat(
    name='data: %s, method: %s, train: %d0%%' % (g_data, g_method, g_split))
for run in range(10):
    torch.manual_seed(run)
    if g_split:
        train_mask = torch.zeros(n_nodes, dtype=bool)
        valid_mask = torch.zeros(n_nodes, dtype=bool)
        test_mask = torch.zeros(n_nodes, dtype=bool)
        idx = torch.randperm(n_nodes)
        val_num = test_num = int(n_nodes * (1 - 0.1 * g_split) / 2)
        train_mask[idx[val_num + test_num:]] = True
        valid_mask[idx[:val_num]] = True
        test_mask[idx[val_num:val_num + test_num]] = True
    elif g_data == 'ppi':
        pass
    else:
        train_mask = graph.ndata['train_mask']
        valid_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']
    # print('nodes: %d, train: %d, valid: %d, test: %d' % (
    #     n_nodes, train_mask.sum().item(),
    #     valid_mask.sum().item(), test_mask.sum().item()))
    train_labels = gpu(node_labels[train_mask])
    evaluate.start_run()
    if g_method in ('mlp', 'ann'):
        mlp = MLP(n_features, hid, n_labels)
        opt = optimize([*mlp.parameters()])
        arange = torch.arange(n_nodes)
        train_idx = arange[train_mask]
        for epoch in range(1, 1 + epochs):
            for perm in DataLoader(
                    train_idx, batch_size=batch_size, shuffle=True):
                opt.zero_grad()
                Y = mlp(node_features[perm])
                loss = F.cross_entropy(Y, node_labels[perm])
                loss.backward()
                opt.step()
                evaluate(Y)
    elif g_method in ('gcn', 'gcn-reuse'):
        is_reuse = 'reuse' in g_method
        X = node_features
        A = spnorm(gpu(graph.adj() + speye(n_nodes)), eps=0)
        in_feats = n_features
        if is_reuse:
            n_label_iters = 1
            in_feats = n_features + n_labels
            input_mask = train_mask & (torch.rand(n_nodes) < 0.5)
            train_mask = train_mask & (~input_mask)
            train_labels = gpu(node_labels[train_mask])
            trainY = gpu(F.one_hot(node_labels, n_labels).float())
            X = torch.cat((node_features, trainY), dim=-1)
            X[~input_mask, -n_labels:] = 0
        model = GCN(in_feats, hid, n_labels, A)
        opt = optimize([*model.parameters()])
        for epoch in range(1, 1 + epochs):
            opt.zero_grad()
            Y = model(X)
            if is_reuse:
                for _ in range(n_label_iters):
                    X[~input_mask, -n_labels:] = torch.softmax(
                        Y.detach()[~input_mask], dim=-1)
                    X[train_mask, -n_labels:] = trainY[train_mask]
                    Y = model(X)
            loss = F.cross_entropy(Y[train_mask], train_labels)
            loss.backward()
            opt.step()
            if is_reuse:
                with torch.no_grad():
                    X[train_mask, -n_labels:] = trainY[train_mask]
                    Y = model(X)
                    for _ in range(n_label_iters):
                        X[~input_mask, -n_labels:] = torch.softmax(
                            Y[~input_mask], dim=-1)
                        X[train_mask, -n_labels:] = trainY[train_mask]
                        Y = model(X)
                    X[~input_mask, -n_labels:] = 0
            evaluate(Y)
    elif g_method in ('tf', 'tf-reuse'):
        is_reuse = 'reuse' in g_method
        Y = gpu(torch.rand((n_nodes, n_labels)))
        if is_reuse:
            trainY = Y
        else:
            trainY = F.one_hot(node_labels, n_labels).float()
            trainY[~train_mask] = 0
        tf = TF(n_features, hid, n_labels)
        opt = optimize([*tf.parameters()])
        adj = graph.adj()
        if is_reuse:
            # Combine iterative predicted labels
            adj = adj + speye(n_nodes)
        arange = torch.arange(n_nodes)
        refs = [arange[adj[i].to_dense().bool()] for i in range(n_nodes)]
        for epoch in range(1, 11):
            for i in torch.randperm(n_nodes):
                opt.zero_grad()
                x = node_features[i].unsqueeze(0)
                ref = refs[i].unsqueeze(0)
                x0 = node_features[ref]
                y0 = trainY[ref]
                y = tf(x, x0, y0)
                if train_mask[i].item():
                    loss = F.cross_entropy(y, node_labels[i].unsqueeze(0))
                    loss.backward()
                    opt.step()
                Y[i] = y.detach().squeeze(0)
            evaluate(Y)
    elif g_method in ('tfs', 'tfs-reuse'):
        is_reuse = 'reuse' in g_method
        trainY = F.one_hot(node_labels, n_labels).float()
        trainY[~train_mask] = 0
        Y = trainY if is_reuse else gpu(torch.rand((n_nodes, n_labels)))
        tf = TF(n_features, hid, n_labels)
        opt = optimize([*tf.parameters()])
        k = 1 + int(n_edges / n_nodes)
        A = graph.adj().to_dense()
        # A[:, train_mask] *= 0.5
        topk = A.topk(k)
        refs = topk.indices
        mask = topk.values == 0
        # In case for some isolated nodes
        mask[:, 0] = False
        logw = gpu(torch.log(1 - mask.float()))
        arange = torch.arange(n_nodes)
        train_idx = arange[train_mask]
        eval_idx = arange[~train_mask]
        for epoch in range(1, 1 + epochs):
            for perm in DataLoader(
                    train_idx, batch_size=batch_size, shuffle=True):
                opt.zero_grad()
                x = node_features[perm]
                ref = refs[perm]
                x0 = node_features[ref]
                y0 = trainY[ref]
                y = tf(x, x0, y0, logw[perm])
                loss = F.cross_entropy(y, node_labels[perm])
                loss.backward()
                opt.step()
            with torch.no_grad():
                for perm in DataLoader(
                        eval_idx, batch_size=batch_size, shuffle=False):
                    x = node_features[perm]
                    ref = refs[perm]
                    x0 = node_features[ref]
                    y0 = trainY[ref]
                    y = tf(x, x0, y0, logw[perm])
                    Y[perm] = torch.softmax(y, dim=-1)
            evaluate(Y)
        # print(torch.softmax(tf.adapt.weight, dim=0).T.tolist())
        # exit()
    else:
        n_train = int(train_mask.sum())
        train_probs = gpu(F.one_hot(train_labels, n_labels)).float()
        X = node_features
        if g_method == 'cs':
            mlp = MLP(in_feats=n_features, hid_feats=hid, out_feats=n_labels)
            opt = optimize([*mlp.parameters()])
            best_acc = 0
            train_idx = torch.arange(n_nodes)[train_mask]
            for epoch in range(1 + epochs):
                for perm in DataLoader(
                        train_idx, batch_size=batch_size, shuffle=True):
                    opt.zero_grad()
                    logits = mlp(node_features[perm])
                    loss = F.cross_entropy(logits, node_labels[perm])
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    probs = torch.softmax(mlp(node_features), dim=-1)
                evaluate(probs)
                acc = evaluate.accs[-1][1]
                if acc > best_acc:
                    best_acc = acc
                    Y = probs
            delta_time = time.time()
            dy = train_probs - Y[train_mask]
            dY = gpu(torch.zeros(n_nodes, n_labels))
            alpha = 0.1
            A = spnorm(gpu(graph.adj()))
            for _ in range(50):
                dY[train_mask] = dy
                dY = (1 - alpha) * dY + alpha * (A @ dY)
            Y += dY
            Y[train_mask] = train_probs
            alpha = 0.1
            for _ in range(50):
                Y = (1 - alpha) * Y + alpha * (A @ Y)
            delta_time = time.time() - delta_time
            evaluate(Y)
            test_acc = evaluate.accs.pop()[-1]
            evaluate.times.pop()
            for accs in evaluate.accs:
                accs[-1] = test_acc
            evaluate.times = [t + delta_time for t in evaluate.times]
        elif g_method == 'fastreslpa':
            f = Res(n_features, hid, n_labels)
            opt = optimize([*f.parameters()])
            alpha = 0.9
            A = gpu(graph.adj())
            D = (torch.sparse.sum(A, dim=1) ** -1).to_dense().unsqueeze(-1)
            Y = gpu(torch.zeros(n_nodes, n_labels))
            for _ in range(50):
                opt.zero_grad()
                feats = f.dec(node_features)
                res = f(feats[src], feats[dst])
                Rs = torch.cat([
                    torch.sparse.sum(
                        torch.sparse_coo_tensor(A._indices(), res[:, c]),
                        dim=-1
                    ).to_dense().unsqueeze(-1)
                    for c in range(n_labels)
                ], dim=1)
                Y[train_mask] = train_probs
                Y = Y / (Y.norm(dim=1, keepdim=True) + 1e-5)
                Z = (1 - alpha) * Y + alpha * (A @ Y + Rs) * D
                loss = F.cross_entropy(Z[train_mask], train_labels)
                loss.backward()
                opt.step()
                Y = Z.detach()
                evaluate(Y)
        elif g_method == 'reslpa':
            f = Res(n_features, hid, n_labels)
            opt = optimize([*f.parameters()])
            alpha = 0.9
            A = gpu(graph.adj())
            D = (torch.sparse.sum(A, dim=1) ** -1).to_dense().unsqueeze(-1)
            for epoch in range(1, 1 + epochs):
                opt.zero_grad()
                feats = f.dec(node_features)
                res = f(feats[src], feats[dst])
                Rs = torch.cat([
                    torch.sparse.sum(
                        torch.sparse_coo_tensor(A._indices(), res[:, c]),
                        dim=-1
                    ).to_dense().unsqueeze(-1)
                    for c in range(n_labels)
                ], dim=1)
                Y = gpu(torch.zeros(n_nodes, n_labels))
                for _ in range(5):
                    Y[train_mask] = train_probs
                    Y = Y / (Y.norm(dim=1, keepdim=True) + 1e-5)
                    Y = (1 - alpha) * Y + alpha * (A @ Y + Rs) * D
                Y = Y / (Y.norm(dim=1, keepdim=True) + 1e-5)
                Y = (1 - alpha) * Y + alpha * (A @ Y + Rs) * D
                loss = F.cross_entropy(Y[train_mask], train_labels)
                loss.backward()
                opt.step()
                for _ in range(20):
                    Y[train_mask] = train_probs
                    Y = Y / (Y.norm(dim=1, keepdim=True) + 1e-5)
                    Y = (1 - alpha) * Y + alpha * (A @ Y + Rs) * D
                evaluate(Y)
        else:
            A = torch.sparse.softmax(gpu(graph.adj()), dim=1)
            alpha = 0.4
            Y = gpu(torch.zeros(n_nodes, n_labels))
            for _ in range(50):
                Y[train_mask] = train_probs
                Y = (1 - alpha) * Y + alpha * (A @ Y)
                evaluate(Y)
    evaluate.end_run()
evaluate.end_all()
