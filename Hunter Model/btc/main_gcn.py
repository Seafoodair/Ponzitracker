import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# 注释掉解压缩文件的相关代码，只保留路径设定
extracted_dir_path = 'dataset'

scaler = StandardScaler()


class BitcoinTransactionDataset(Dataset):
    def __init__(self, root_dir):
        self.file_names = []
        self.labels = []
        for sub_dir in ['ponzi', 'nonponzi']:
            full_sub_dir = os.path.join(root_dir, sub_dir)
            for wallet in os.listdir(full_sub_dir):
                wallet_path = os.path.join(full_sub_dir, wallet)
                if os.path.isdir(wallet_path):
                    for transaction_file in os.listdir(wallet_path):
                        if transaction_file.endswith('.json'):
                            self.file_names.append(os.path.join(wallet_path, transaction_file))
                            self.labels.append(1 if os.path.basename(full_sub_dir) == 'ponzi' else 0)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        with open(file_path, 'r') as f:
            transactions = json.load(f)

        G = nx.DiGraph()
        for transaction in transactions:
            inputs = transaction.get('inputs', [])
            outputs = transaction.get('out', [])

            for inp in inputs:
                from_node = inp['prev_out'].get('addr', 'unknown')
                to_node = transaction['hash']
                value = float(inp['prev_out'].get('value', 0))

                G.add_node(from_node, balance=inp['prev_out'].get('value', 0))
                G.add_node(to_node, balance=transaction.get('balance', 0))

                G.add_edge(from_node, to_node, weight=value)

            for out in outputs:
                from_node = transaction['hash']
                to_node = out.get('addr', 'unknown')
                value = float(out.get('value', 0))

                G.add_node(from_node, balance=transaction.get('balance', 0))
                G.add_node(to_node, balance=out.get('value', 0))

                G.add_edge(from_node, to_node, weight=value)

        data = from_networkx(G)

        node_features = []
        for node, attributes in G.nodes(data=True):
            node_features.append([attributes.get('balance', 0.0)])
        node_features = scaler.fit_transform(node_features)
        data.x = torch.tensor(node_features, dtype=torch.float)

        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        edge_weights = scaler.fit_transform(np.array(edge_weights).reshape(-1, 1)).flatten()
        data.edge_attr = torch.tensor(edge_weights, dtype=torch.float)

        data.y = torch.tensor([self.labels[idx]], dtype=torch.float)

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
dataset = BitcoinTransactionDataset(root_dir=extracted_dir_path)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)


# 检查数据加载器是否为空
if len(train_loader) == 0 or len(val_loader) == 0:
    raise ValueError("DataLoader is empty. Please check the dataset.")


# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, 1)  # 添加一个全连接层用于二分类
        self.leaky_relu = nn.LeakyReLU(0.2)  # 添加LeakyReLU激活函数

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, GCNConv):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.leaky_relu(x)  # 使用LeakyReLU激活函数
        x = self.conv2(x, edge_index)
        x = self.leaky_relu(x)

        # 对每个图进行全局平均池化
        x = global_mean_pool(x, data.batch)

        x = self.fc(x)  # 全连接层
        return x.view(-1)  # 返回一维张量，用于二分类


# 创建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_node_features=1, hidden_channels=64).to(device)  # 增加隐藏层的神经元数量

# 设置损失函数
loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

# 使用学习率调度器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 学习率
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# 定义早停机制
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

# 训练循环
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        # 使用梯度裁剪来防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 调整梯度裁剪值
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()  # 更新学习率

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = loss_fn(out, batch.y)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

    # 检查早停条件
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping")
        break

# 统计推理结果
total_samples = len(dataset)
targets = torch.tensor([data.y.item() for data in dataset])
positive_samples = (targets == 1).sum().item()
negative_samples = (targets == 0).sum().item()

predicted_positive = 0
predicted_negative = 0
all_predictions = []
all_targets = []

# 推理
model.eval()
with torch.no_grad():
    for batch in train_loader:
        batch = batch.to(device)
        out = model(batch)
        # 尝试调整阈值
        threshold = 0.5  # 先用0.5试试，可以调整
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
print(f"Predicted Positive Samples: {predicted_positive} ({predicted_positive / (predicted_positive+predicted_negative):.2%})")
print(f"Predicted Negative Samples: {predicted_negative} ({predicted_negative / (predicted_positive+predicted_negative):.2%})")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
