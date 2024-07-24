#encoding=utf-8
"""
@author=wang gang
builder graph
"""
import os
import json
import requests
from Cryptio_btc_address_suggester.Bitcoin_address_suggester import get_possible_addr as sgstr
import pandas as pd
file_path = '../bitcoindata/Bitcoin_Ponzi_ml/datasets/final_aggregated_dataset.csv'
api_url="https://dps.kdlapi.com/api/getdps/?secret_id=oqfp8rntb2fy4ik1d5nq&signature=btcjbshckpewcm6vr0y2a6zu1q1o35xz&num=1&pt=1&format=text&sep=1"
proxy_ip = requests.get(api_url).text
proxies = {
    "http": "http://%(proxy)s/" % {"proxy": proxy_ip},
    "https": "http://%(proxy)s/" % {"proxy": proxy_ip}
}
def get_ponzi_address(file_path,label=1):
    # 使用pandas读取csv文件
    df = pd.read_csv(file_path)

    # 筛选class为1的数据
    filtered_df = df[df['class'] == label]

    # 将筛选后的数据保存到列表中
    # 假设我们只关心'address'列的数据
    filtered_list = filtered_df['address'].tolist()

    # 打印结果
    print(filtered_list)
    # output Ponzi address length
    print(len(filtered_list))  # 77
    return filtered_list
def get_nonponzi_address(file_path,label=0):
    # 使用pandas读取csv文件
    df = pd.read_csv(file_path)

    # 筛选class为1的数据
    filtered_df = df[df['class'] == label]

    # 将筛选后的数据保存到列表中
    # 假设我们只关心'address'列的数据
    filtered_list = filtered_df['address'].tolist()

    # 打印结果
    print(filtered_list)
    # output Ponzi address length
    print(len(filtered_list))  # 77
    return filtered_list
# 添加代理获取和删除函数
def get_proxy():
    try:
        return requests.get("http://127.0.0.1:5010/get/").json()['proxy']
    except Exception as e:
        print(f"Failed to get proxy: {e}")
        return None

def delete_proxy(proxy):
    try:
        requests.get(f"http://127.0.0.1:5010/delete/?proxy={proxy}")
    except Exception as e:
        print(f"Failed to delete proxy: {e}")

# 修改地址建议函数以使用代理
def get_samewallet_address(class_address, filename):
    i = 0
    for address in class_address:
        retry_count = 1
        while retry_count > 0:
            # proxy = get_proxy()
            # if not proxy:
            #     print("No proxy available, trying again...")
            #     continue
            try:
                suggested_addresses = sgstr.suggest(address, proxy={"http": f"http://{proxy}", "https": f"http://{proxy}"})
                #suggested_addresses=sgstr.suggest(address)
                print(len(suggested_addresses))

                file_path = os.path.join(filename, f'nonponzidict_{i + 1}.json')
                with open(file_path, 'w') as file:
                    json.dump(list(suggested_addresses.keys()), file, indent=4)

                i += 1
                break  # 成功获取数据后退出重试循环
            except Exception as e:
                print(f"Failed to fetch data using proxy {proxies}: {e}")
                #delete_proxy(proxy)
                retry_count -= 1

        if retry_count == 0:
            print(f"Failed to fetch data for address {address} after several attempts.")

if __name__ == '__main__':
    filename = '/home/wg/pythonProject1/POZISCHEME/bitcoincrawl/Wallet2'
    Nponzi_address = get_nonponzi_address(file_path)
    get_samewallet_address(Nponzi_address, filename)

