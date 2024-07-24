import networkx as nx
import matplotlib.pyplot as plt

# 假设的交易数据
transactions = [
    # (发送方, 接收方, 金额)
    ("wallet1", "addr4", 100),
    ("addr5", "wallet1", 50),
    ("wallet1", "addr6", 70),
    ("addr7", "wallet1", 30),
]

# 创建一个有向图
G = nx.DiGraph()

# 添加节点和边
for sender, receiver, amount in transactions:
    G.add_edge(sender, receiver, weight=amount)

# 绘图设置
pos = nx.spring_layout(G)  # 节点的布局方式
edge_labels = nx.get_edge_attributes(G, 'weight')  # 获取边的权重，这里是交易金额

# 画图
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# 显示图形
plt.show()
