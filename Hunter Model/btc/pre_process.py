import os
import json
from datetime import datetime
import numpy as np


def timestamp_to_vector(timestamp):
    """
    将时间戳转换为向量形式
    """
    dt = datetime.fromtimestamp(timestamp)
    return [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second]


def process_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        transactions = [data]
    elif isinstance(data, list):
        transactions = data
    else:
        raise ValueError("Invalid JSON format")

    for transaction in transactions:
        if 'time' in transaction:
            timestamp = transaction['time']
            time_vector = timestamp_to_vector(timestamp)
            transaction['time_stamp'] = time_vector

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def process_all_files_in_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                process_json_file(file_path)
                print(f"Processed {file_path}")


# 示例使用
directory_path = 'dataset'  # 替换为你的JSON文件所在目录
process_all_files_in_directory(directory_path)
