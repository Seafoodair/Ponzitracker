#encoding=utf-8
"""
@author=gang wang
"""
import requests
import json
import requests
import json
import os
import time
#API LIMIT
# 定义三个钱包，每个钱包包含若干地址.
# wallets = {
#     "wallet1": ["比特币地址1", "比特币地址2"],
#     "wallet2": ["比特币地址3", "比特币地址4"],
#     "wallet3": ["比特币地址5", "比特币地址6"],
# }
#wallets={}
import requests

# 获取私密代理IP API
api = "https://dps.kdlapi.com/api/getdps"

# 请求参数
params = {
    "secret_id": "oqfp8rntb2fy4ik1d5nq",
    "signature": "btcjbshckpewcm6vr0y2a6zu1q1o35xz",
    "num": 3,   # 提取数量
}
response = requests.get(api, params=params)
# 检查响应状态码
if response.status_code == 200:
    # 将响应内容转换为JSON格式
    data = response.json()
    # 假设返回结果在'data'字段中
    proxy_list = data.get('data', [])
    print("Proxy list:", proxy_list)
else:
    print(f"请求失败，状态码: {response.status_code}")

# 输出结果
print(proxy_list)
#proxy=list(response.text)

def get_addressofWallet(wallet_folder):
    wallets = {}
    # 遍历文件夹中的所有文件
    for filename in os.listdir(wallet_folder):
        # 检查文件是否为JSON文件
        if filename.endswith(".json"):
            file_path = os.path.join(wallet_folder, filename)

            # 尝试读取JSON文件并更新字典
            try:
                with open(file_path, 'r') as file:
                    addresses = json.load(file)
                    # 使用文件名作为键，地址列表作为值
                    wallets[filename] = addresses
            except FileNotFoundError:
                print(f"文件{filename}未找到。")
            except json.JSONDecodeError:
                print(f"解析{filename}时发生错误。")
    return wallets


def save_transactions(wallet_name, address, transactions):
    """将交易记录保存为JSON文件"""
    # 创建钱包对应的目录，如果不存在
    directory = os.path.join("transactionsdemo", wallet_name)
    if not os.path.exists(directory):
        os.makedirs(directory)


    # 构造文件名并保存JSON数据
    file_path = os.path.join(directory, f"{address}.json")
    with open(file_path, 'w') as file:
        json.dump(transactions, file, indent=4)



# 遍历钱包和地址
def fetch_and_save_transaction(wallets):
    for wallet_name, addresses in wallets.items():
        for address in addresses:
            #time.sleep(10)
            response = requests.get(f"https://blockchain.info/rawaddr/{address}",proxies={"http": "http://{}".format(proxy)})
            if response.ok:
                transactions = response.json()['txs']

                save_transactions(wallet_name, address, transactions)

                print(f"保存{address}的交易记录成功。")
            else:
                print(f"API请求失败，地址：{address}")
if __name__ == '__main__':
    wallet_folder = "Wallet/"
    #api_key ='3de5f727-c943-419a-84ad-6ede4bac172a'
    wallets=get_addressofWallet(wallet_folder)
    # 打印更新后的字典
    #print(wallets)
    fetch_and_save_transaction(wallets)

    print("finished")
#define wallet type

#define folder path
