import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


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

        data.y = torch.tensor([1.0], dtype=torch.float)  # 正样本标签为1.0

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

        data.y = torch.tensor([0.0], dtype=torch.float)  # 负样本标签为0.0

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


# 定义数据集和数据加载器
ponzi_dataset = EthereumPonziDataset(root_dir='Balponzi')
non_ponzi_dataset = EthereumNonPonziDataset(root_dir='BalNponzi')

# 合并数据集
combined_dataset = ConcatDataset([ponzi_dataset, non_ponzi_dataset])

# 使用平衡采样策略
positive_samples = len([data for data in combined_dataset if data.y.item() == 1])
negative_samples = len([data for data in combined_dataset if data.y.item() == 0])

# 计算正负样本的权重
targets = torch.tensor([data.y.item() for data in combined_dataset])
class_sample_count = [negative_samples, positive_samples]
weights = 1. / torch.tensor(class_sample_count, dtype=torch.float)
samples_weight = torch.tensor([weights[int(data.y.item())] for data in combined_dataset])

sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

data_loader = DataLoader(combined_dataset, batch_size=1, sampler=sampler, collate_fn=collate_fn)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, 1)  # 添加一个全连接层用于二分类
        self.leaky_relu = nn.LeakyReLU(0.2)  # 添加LeakyReLU激活函数
        self.sigmoid = nn.Sigmoid()  # 添加sigmoid激活函数

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.leaky_relu(x)  # 使用LeakyReLU激活函数
        x = self.conv2(x, edge_index)
        x = self.leaky_relu(x)  # 使用LeakyReLU激活函数

        # 对每个图进行全局平均池化
        x = global_mean_pool(x, data.batch)

        x = self.fc(x)  # 全连接层
        x = self.sigmoid(x)  # sigmoid激活函数
        return x.view(-1)  # 返回一维张量，用于二分类


# 创建模型
model = GCN(num_node_features=1, hidden_channels=64).to(device)  # 增加隐藏层的神经元数量

# 设置损失函数，不使用加权
loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

# 使用学习率调度器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 训练循环
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()  # 更新学习率
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(data_loader)}')

# 统计推理结果
total_samples = len(combined_dataset)
positive_samples = (targets == 1).sum().item()
negative_samples = (targets == 0).sum().item()

predicted_positive = 0
predicted_negative = 0
all_predictions = []
all_targets = []

# 推理
model.eval()
with torch.no_grad():
    for batch in data_loader:
        batch = batch.to(device)
        out = model(batch)
        # 尝试调整阈值
        threshold = 0.6  # 先用0.5试试，可以调整
        predicted_labels = (torch.sigmoid(out) > threshold).float()
        all_predictions.extend(predicted_labels.cpu().numpy())
        all_targets.extend(batch.y.cpu().numpy())
        predicted_positive += (predicted_labels == 1).sum().item()
        predicted_negative += (predicted_labels == 0).sum().item()

# 计算评价指标
precision = precision_score(all_targets, all_predictions, zero_division=0)
recall = recall_score(all_targets, all_predictions, zero_division=0)
f1 = f1_score(all_targets, all_predictions, zero_division=0)
accuracy = accuracy_score(all_targets, all_predictions)

# 打印结果
print(f"Dataset Positive Samples: {positive_samples} ({positive_samples / total_samples:.2%})")
print(f"Dataset Negative Samples: {negative_samples} ({negative_samples / total_samples:.2%})")
print(f"Predicted Positive Samples: {predicted_positive} ({predicted_positive / total_samples:.2%})")
print(f"Predicted Negative Samples: {predicted_negative} ({predicted_negative / total_samples:.2%})")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

import winsound

winsound.MessageBeep()  # 发出类似警告弹窗的声音，“当”
winsound.Beep(2000, 100)  # 发出“ju”的一声，持续0.1s
winsound.Beep(2000, 1000)  # 发出长达1s的滴声
