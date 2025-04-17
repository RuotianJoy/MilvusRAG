#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from tqdm import tqdm

# 文件路径
input_file = "/DataOriginal/Data/US高校维基百科爬虫数据.json"
output_file = "/DataOriginal/Data/US高校维基百科爬虫数据_filtered.json"

print(f"正在处理文件: {input_file}")

# 检查文件大小
file_size = os.path.getsize(input_file) / (1024 * 1024)
print(f"文件大小: {file_size:.2f} MB")

# 读取原始JSON数据
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据记录数: {len(data)}")
    
    # 检查一条样本数据的字段
    if len(data) > 0:
        sample = data[0]
        print(f"样本数据字段: {list(sample.keys())}")
    
    # 过滤掉没有Name字段的记录 (注意大写)
    filtered_data = [item for item in data if 'Name' in item and item['Name']]
    
    print(f"过滤后的记录数: {len(filtered_data)}")
    print(f"已删除 {len(data) - len(filtered_data)} 条没有Name字段的记录")
    
    # 将字段名从Name转换为name (转为小写)
    normalized_data = []
    for item in filtered_data:
        normalized_item = {}
        for key, value in item.items():
            # 将键名转为小写
            normalized_key = key.lower()
            normalized_item[normalized_key] = value
        normalized_data.append(normalized_item)
    
    print(f"已将 {len(normalized_data)} 条记录的字段名标准化为小写")
    
    # 保存标准化后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=2)
    
    print(f"已将过滤和标准化后的数据保存到: {output_file}")
    
except json.JSONDecodeError as e:
    print(f"JSON解析错误: {e}")
    
    # 尝试逐行读取并处理
    print("尝试逐行读取并处理...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 判断是否为JSONL格式（每行一个JSON对象）
    if content.strip().startswith('[') and content.strip().endswith(']'):
        print("文件是标准JSON数组格式，但可能太大或格式有问题")
    else:
        print("尝试按JSONL格式处理（每行一个JSON对象）")
        filtered_data = []
        total_count = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    item = json.loads(line)
                    total_count += 1
                    
                    # 检查大写的Name字段
                    if 'Name' in item and item['Name']:
                        # 转换为小写字段名
                        normalized_item = {}
                        for key, value in item.items():
                            normalized_key = key.lower()
                            normalized_item[normalized_key] = value
                        filtered_data.append(normalized_item)
                except json.JSONDecodeError:
                    print(f"无法解析行: {line[:100]}...")
                    continue
        
        print(f"原始数据记录数: {total_count}")
        print(f"过滤后的记录数: {len(filtered_data)}")
        print(f"已删除 {total_count - len(filtered_data)} 条没有Name字段的记录")
        
        # 保存过滤后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        print(f"已将过滤后的数据保存到: {output_file}")
except Exception as e:
    print(f"处理过程中出错: {e}") 