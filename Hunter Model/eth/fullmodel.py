import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_undirected
from torch_geometric.nn import SAGEConv, GCNConv, ChebConv, GATConv, ARMAConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn import SAGEConv, GCNConv, ChebConv, GATConv, ARMAConv
import numpy as np
class TimeAggregate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TimeAggregate, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, edge_index, time_encoding, num_nodes):
        row, col = edge_index
        out = torch.zeros((num_nodes, time_encoding.size(1)), device=time_encoding.device)
        out.index_add_(0, col, time_encoding)
        out = self.linear(out)
        return out

# Time Encoding
class TimeEncode(nn.Module):
    # Time Encoding proposed by TGAT
    def __init__(self, dimension):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(dim=1)
            # t has shape [batch_size, seq_len]
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        t = t.unsqueeze(dim=2)

        # output has shape [batch_size, seq_len, dimension]
        output = torch.cos(self.w(t))
        # Remove the extra dimension at the second position
        output = output.squeeze(dim=1)  # [num_edges, dimension]

        return output
# Custom dataset classes
class EthereumPonziDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_names = [f for f in os.listdir(root_dir) if f.endswith('_graph.json')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_names[idx])
        with open(file_path, 'r') as f:
            graph_data = json.load(f)

        G = nx.DiGraph()
        for edge in graph_data['edges']:
            from_node = edge['from']
            to_node = edge['to']
            value = float(edge['value'])

            from_balance = edge['from_balance']
            to_balance = edge['to_balance'] if edge['to_balance'] is not None else 0.0

            from_balance = from_balance if from_balance is not None and from_balance >= 0 else 0.0
            to_balance = to_balance if to_balance is not None and to_balance >= 0 else 0.0

            G.add_node(from_node, balance=from_balance)
            if to_node:
                G.add_node(to_node, balance=to_balance)

            if to_node:
                G.add_edge(from_node, to_node, weight=value)

        data = from_networkx(G)

        node_features = []
        for node, attributes in G.nodes(data=True):
            node_features.append([attributes['balance'] if attributes['balance'] is not None else 0.0])
        data.x = torch.tensor(node_features, dtype=torch.float)

        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        data.edge_attr = torch.tensor(edge_weights, dtype=torch.float)

        data.y = torch.tensor([1.0], dtype=torch.float)  # Positive sample label is 1.0

        return data


class EthereumNonPonziDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_names = [f for f in os.listdir(root_dir) if f.endswith('_graph.json')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_names[idx])
        with open(file_path, 'r') as f:
            graph_data = json.load(f)

        G = nx.DiGraph()
        for edge in graph_data['edges']:
            from_node = edge['from']
            to_node = edge['to']
            value = float(edge['value'])

            from_balance = edge['from_balance']
            to_balance = edge['to_balance'] if edge['to_balance'] is not None else 0.0

            from_balance = from_balance if from_balance is not None and from_balance >= 0 else 0.0
            to_balance = to_balance if to_balance is not None and to_balance >= 0 else 0.0

            G.add_node(from_node, balance=from_balance)
            if to_node:
                G.add_node(to_node, balance=to_balance)

            if to_node:
                G.add_edge(from_node, to_node, weight=value)

        data = from_networkx(G)

        node_features = []
        for node, attributes in G.nodes(data=True):
            node_features.append([attributes['balance'] if attributes['balance'] is not None else 0.0])
        data.x = torch.tensor(node_features, dtype=torch.float)

        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        data.edge_attr = torch.tensor(edge_weights, dtype=torch.float)

        data.y = torch.tensor([0.0], dtype=torch.float)  # Negative sample label is 0.0

        return data


def collate_fn(batch):
    data = batch[0]
    batch_size = len(batch)
    batch_data = Data(
        x=torch.cat([d.x for d in batch], dim=0),
        edge_index=torch.cat([d.edge_index for d in batch], dim=1),
        edge_attr=torch.cat([d.edge_attr for d in batch], dim=0),
        y=torch.cat([d.y for d in batch], dim=0),
        batch=torch.tensor([i for i in range(batch_size) for _ in range(batch[i].num_nodes)], dtype=torch.long)
    )
    return batch_data


# Define datasets and data loaders
ponzi_dataset = EthereumPonziDataset(root_dir='Balponzi')
non_ponzi_dataset = EthereumNonPonziDataset(root_dir='BalNponzi')

# Combine datasets
combined_dataset = ConcatDataset([ponzi_dataset, non_ponzi_dataset])

# Split into training and validation sets
train_size = int(0.8 * len(combined_dataset))
val_size = len(combined_dataset) - train_size
train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

# Use balanced sampling strategy
positive_samples = len([data for data in train_dataset if data.y.item() == 1])
negative_samples = len([data for data in train_dataset if data.y.item() == 0])

# Calculate class weights
targets = torch.tensor([data.y.item() for data in train_dataset])
class_sample_count = [negative_samples, positive_samples]
weights = 1. / torch.tensor(class_sample_count, dtype=torch.float)
samples_weight = torch.tensor([weights[int(data.y.item())] for data in train_dataset])

sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, collate_fn=collate_fn, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)


# Define the model
class filtering(nn.Module):
    def __init__(self, alpha, n_nodes):
        super(filtering, self).__init__()
        self.alpha = alpha
        self.n_nodes = n_nodes

    def forward(self, x, edge_index, K, is_lowpass=True):
        return x


class PyGNN(nn.Module):
    def __init__(self, dataset, args):
        super(PyGNN, self).__init__()

        self.args = args
        self.S_subnode = [_[0] for _ in dataset.py_subg]
        self.S_subedge = [_[1] for _ in dataset.py_subg]
        self.upsampl_ops = dataset.upsampl_ops
        #dataset, data = parse_dataset(args.dataname, dataset)

        self.n_bands = args.n_bands
        self.low_bands = args.low_bands
        self.aggregate = args.aggregate
        self.dropout = nn.Dropout(args.dropout)
        self.input_drop = nn.Dropout(args.input_drop)
        self.K = [args.K for _ in range(self.n_bands)]
        in_channels = dataset.n_feats
        if self.aggregate == "concat":
            last_in_channels = args.hidden * args.n_bands
        else:
            last_in_channels = args.hidden

        self.time_encoder = TimeEncode(args.hidden)
        self.time_aggregate = TimeAggregate(args.hidden, args.hidden)

        if self.args.aggregate == "gate":
            self.lin_low_t = nn.Linear(args.hidden * self.low_bands, args.hidden)
            if args.n_bands - self.low_bands > 0:
                self.lin_high_t = nn.Linear(args.hidden * (args.n_bands - self.low_bands), args.hidden)

        self.lin_low = nn.Linear(args.hidden, args.hidden)
        self.lin_high = nn.Linear(args.hidden, args.hidden)

        for b_idx in range(self.n_bands):
            if self.args.backbone == "SAGE":
                setattr(self, 'conv1_{}'.format(b_idx),
                        SAGEConv(in_channels + args.hidden, args.hidden, aggr="max"))
            elif self.args.backbone == "ChebNet":
                setattr(self, 'conv1_{}'.format(b_idx),
                        ChebConv(in_channels + args.hidden, args.hidden, K=args.order))
            elif self.args.backbone == "GAT":
                setattr(self, 'conv1_{}'.format(b_idx),
                        GATConv(in_channels + args.hidden, args.hidden, heads=args.heads, dropout=0))
            elif self.args.backbone == "ARMA":
                setattr(self, 'conv1_{}'.format(b_idx),
                ARMAConv(in_channels + args.hidden, args.hidden, num_stacks=1, num_layers=1, shared_weights=True,
                         dropout=0))
            elif self.args.backbone == "GCN":
                setattr(self, 'conv1_{}'.format(b_idx),
                        GCNConv(in_channels + args.hidden, args.hidden))
            else:
                assert 0, "backbone not implemented."

        self.filtering = filtering(alpha=args.alpha, n_nodes=dataset.n_nodes)
        self.lin_out = nn.Linear(last_in_channels, dataset.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature, edge_index, edge_attr):
        device = feature.device
        feature = self.input_drop(feature)

        time_encoding = self.time_encoder(edge_attr[:, 0].unsqueeze(dim=1)).to(device)
        time_aggregated = self.time_aggregate(edge_index, time_encoding, feature.size(0))
        feature = torch.cat([feature, time_aggregated], dim=1)

        pyramid_layers = []
        for blk_idx in range(self.n_bands):
            if blk_idx >= len(self.S_subedge):
                raise ValueError(f"blk_idx {blk_idx} is out of range for S_subedge with length {len(self.S_subedge)}")

            conv_layer = getattr(self, 'conv1_{}'.format(blk_idx))
            x = conv_layer(feature, self.S_subedge[blk_idx])
            x = F.relu(x)
            x = self.dropout(x)
            pyramid_layers.append(x)

        if self.aggregate == "concat":
            x = torch.cat(pyramid_layers, dim=1)
        elif self.aggregate == "sum":
            x = torch.sum(x, dim=1)
        elif self.aggregate == "gate":
            x_low = torch.cat(pyramid_layers[:self.low_bands], dim=1)
            x_high = torch.cat(pyramid_layers[self.low_bands:], dim=1)
            x_low_t = torch.tanh(self.lin_low_t(x_low))
            x_high_t = torch.relu(self.lin_high_t(x_high))
            gate = torch.sigmoid(self.lin_low(x_low_t) + self.lin_high(x_high_t))
            x = gate * x_low_t + (1 - gate) * x_high_t

        x = self.lin_out(x)
        return F.log_softmax(x, dim=1)


# def parse_dataset(dataname, dataset):
#     return dataset, Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels)
# 数据集和参数模拟
class Args:
    hidden = 64
    n_bands = 4
    low_bands = 2
    aggregate = "concat"
    backbone = "GCN"
    order = 2
    heads = 4
    alpha = 0.1
    input_drop = 0.5
    dropout = 0.5
    use_upsampl = True
    dataname = "custom"
    use_hp='False'
    K=3


dataset = Dataset()
args = Args()

model = PyGNN(dataset, args)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Training loop
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr)
        loss = F.nll_loss(output[data.y], data.y.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr)
            loss = F.nll_loss(output[data.y], data.y.long())
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
    return total_loss / len(val_loader), correct / len(val_loader.dataset)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training and evaluation
num_epochs = 20
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, device)
    print(
        f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
