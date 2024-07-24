#encoding=utf-8
"""
@author=gang wang 5,14
"""
import requests
import os
import json
def get_proxy():
    return requests.get("http://127.0.0.1:5010/get/").json()

def delete_proxy(proxy):
    requests.get("http://127.0.0.1:5010/delete/?proxy={}".format(proxy))

# your spider code

def getHtml(wallets):
    # ....
    retry_count = 20 #try times
    for wallet_name, addresses in wallets.items():
        for address in addresses:
            while retry_count > 0:
                try:
                    proxy = get_proxy().get("proxy")
                    print(proxy)
                    html = requests.get(f"https://blockchain.info/rawaddr/{address}", proxies={"https": "http://{}".format(proxy)})
                    # html = requests.get('http://www.example.com', proxies={"http": "http://{}".format(proxy)})
                    # 使用代理访问
                    if html.ok:
                        print(html)

                        print("保存成功!")
                    else:
                        print(html.reason)
                except Exception:
                    print("保存失败")
                    retry_count -= 1
                finally:
                    delete_proxy(proxy)
    return None
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
if __name__ == '__main__':
    wallet_folder = "Wallet/"
    wallets = get_addressofWallet(wallet_folder)
    getHtml(wallets)

