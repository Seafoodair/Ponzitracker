#encoding=utf-8
"""
crawltx :crawl all tx
Ourtest: get ponzi address wallet.
combine : address-wallet map
"""
import os.path

import pandas as pd
from Cryptio_btc_address_suggester.Bitcoin_address_suggester import get_possible_addr as sgstr
import json
# 替换为你的Excel文件路径
file_path = '../bitcoindata/Bitcoin_Ponzi_ml/datasets/final_aggregated_dataset.csv'


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
    return filtered_list[1278:]
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
#
def get_samewallet_address(class_address,filename):
    #i=0
    i=1278
    for address in class_address:
        suggested_addresses = sgstr.suggest(address).keys()
        print(len(suggested_addresses))
        # 构造每个字典的文件名
        #file_path =os.path.join(filename ,f'dict_{i + 1}.json')
        file_path = os.path.join(filename, f'nonponzidict_{i + 1}.json')
        i+=1
        # 打开文件并写入字典数据
        with open(file_path, 'w') as file:
            json.dump(list(suggested_addresses), file, indent=4)  # 使用indent参数让JSON数据格式化存储，便于阅读
    return suggested_addresses
if __name__ == '__main__':
    #get_nonponzi_address(file_path)
    filename = '/home/wg/pythonProject1/POZISCHEME/bitcoincrawl/Wallet2'
    # Ponzi_address=get_ponzi_address(file_path)
    # get_samewallet_address(Ponzi_address,filename)
    #get non-ponzi wallet possible address.
    Nponzi_address=get_nonponzi_address(file_path)
    get_samewallet_address(Nponzi_address,filename)