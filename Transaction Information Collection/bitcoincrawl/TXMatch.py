#encoding=utf-8
"""
@author=gangwang
"""
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
# 假设的钱包数据：钱包名到地址的映射
wallets = {
    "wallet1": ["address11", "address12","address15"],
    "wallet2": ["address21", "address22"],
    "wallet3": ["address31", "address32"],
}

# 假设的交易记录：地址到涉及的其他地址的映射
transactions = {
    "address11": [("address21", 100)],
    "address12": [("address31", 50)],
    "address22": [("address11", 75), ("address32", 25),("address15", 25)],
    # 其他地址的交易...
}
# 将地址映射回钱包
address_to_wallet = {addr: wallet for wallet, addrs in wallets.items() for addr in addrs}

# # 构建钱包间的交易关系
# wallet_links = set()
# for addr, trans in transactions.items():
#     src_wallet = address_to_wallet.get(addr)
#     for t_addr in trans:
#         dst_wallet = address_to_wallet.get(t_addr)
#         if src_wallet and dst_wallet and src_wallet != dst_wallet:
#             wallet_links.add((src_wallet, dst_wallet))
# 构建钱包间的交易关系，聚合交易金额并统计次数
wallet_links = defaultdict(lambda: [0, 0])  # (源钱包, 目标钱包): [总交易金额, 交易次数]
for addr, trans in transactions.items():
    src_wallet = address_to_wallet.get(addr)
    for t_addr, amount in trans:
        dst_wallet = address_to_wallet.get(t_addr)
        if src_wallet and dst_wallet and src_wallet != dst_wallet:
            if src_wallet and dst_wallet and src_wallet != dst_wallet:  # 只处理同方向的交易
                wallet_links[(src_wallet, dst_wallet)][0] += amount
                wallet_links[(src_wallet, dst_wallet)][1] += 1
print(wallet_links)
# 使用networkx构建图
G = nx.DiGraph()
for (src_wallet, dst_wallet), (total_amount, count) in wallet_links.items():
    G.add_edge(src_wallet, dst_wallet, weight=total_amount, count=count)

# 可视化图
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)  # 定义节点位置
labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, arrowsize=20)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)  # 显示边的权重
plt.title('Wallet Relationship Graph with Aggregated Transactions')
plt.show()
