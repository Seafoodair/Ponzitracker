import requests
import json
import os
import time

# 获取私密代理IP的API URL和请求参数
api = "https://dps.kdlapi.com/api/getdps"
params = {
    "secret_id": "oqfp8rntb2fy4ik1d5nq",
    "signature": "btcjbshckpewcm6vr0y2a6zu1q1o35xz",
    "num": 1,   # 提取数量
}

def get_proxies():
    """获取代理IP并解析为列表"""
    while True:
        response = requests.get(api, params=params)
        if response.status_code == 200:
            proxy_text = response.text
            proxy_list = proxy_text.strip().split('\n')
            if proxy_list:
                return proxy_list
            else:
                print("没有获取到代理IP，重试中...")
        else:
            print(f"请求代理IP失败，状态码: {response.status_code}")
        time.sleep(5)  # 重试前等待5秒

proxies = get_proxies()

def get_addressofWallet(wallet_folder):
    wallets = {}
    # 遍历文件夹中的所有文件
    for filename in os.listdir(wallet_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(wallet_folder, filename)
            try:
                with open(file_path, 'r') as file:
                    addresses = json.load(file)
                    wallets[filename] = addresses
            except FileNotFoundError:
                print(f"文件{filename}未找到。")
            except json.JSONDecodeError:
                print(f"解析{filename}时发生错误。")
    return wallets

def save_transactions(wallet_name, address, transactions):
    """将交易记录保存为JSON文件"""
    directory = os.path.join("transactionsdemo", wallet_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, f"{address}.json")
    with open(file_path, 'w') as file:
        json.dump(transactions, file, indent=4)

def fetch_and_save_transaction(wallets):
    global proxies  # 声明使用全局变量
    for wallet_name, addresses in wallets.items():
        for address in addresses:
            success = False
            while not success:
                try:
                    if not proxies:
                        proxies = get_proxies()
                    #proxy = {"http": f"http://{proxies[0]}"}
                    #print(proxy)
                    response = requests.get(f"https://blockchain.info/rawaddr/{address}", proxies={"http": "http://{}".format(proxies[0])})
                    if response.ok:
                        try:
                            data = response.json()
                            transactions = data.get('txs', [])
                            save_transactions(wallet_name, address, transactions)
                            print(f"保存{address}的交易记录成功。")
                            success = True
                        except json.JSONDecodeError as e:
                            print(f"解析交易记录响应时出错: {e}")
                    else:
                        print(f"API请求失败，地址：{address}")
                        proxies.pop(0)
                        if not proxies:
                            proxies = get_proxies()
                except Exception as e:
                    print(f"请求发生错误：{e}")
                    proxies.pop(0)
                    if not proxies:
                        proxies = get_proxies()

if __name__ == '__main__':
    wallet_folder = "Wallet/"
    wallets = get_addressofWallet(wallet_folder)
    fetch_and_save_transaction(wallets)
    print("finished")
