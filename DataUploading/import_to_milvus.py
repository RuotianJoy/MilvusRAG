#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import time
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

# 配置参数
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "us_colleges"
DIMENSION = 384  # 修改为384维度
MODEL_NAME = "all-MiniLM-L6-v2"  # 此模型输出384维向量

# 连接到Milvus
def connect_to_milvus():
    print(f"连接到Milvus服务器 {MILVUS_HOST}:{MILVUS_PORT}")
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    print("成功连接到Milvus")

# 创建集合
def create_collection():
    if utility.has_collection(COLLECTION_NAME):
        print(f"集合 {COLLECTION_NAME} 已存在，将被删除并重新创建")
        utility.drop_collection(COLLECTION_NAME)
    
    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),  # 学校名称
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),  # 内容文本
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=255),  # 网址
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)  # 向量嵌入，使用384维
    ]
    
    # 创建集合架构和索引参数
    schema = CollectionSchema(fields=fields, description="美国大学维基百科数据")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    
    # 创建索引
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"成功创建集合 {COLLECTION_NAME} 和索引")
    
    return collection

# 加载模型生成向量嵌入
def generate_embeddings(texts, model):
    print(f"为 {len(texts)} 条文本生成向量嵌入")
    # 使用较小的批大小，避免内存问题
    batch_size = 8
    all_embeddings = []
    
    # 分批处理
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"处理批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch_texts)}条文本)")
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
    
    # 合并所有批次的嵌入
    embeddings = np.vstack(all_embeddings)
    print(f"嵌入向量维度: {embeddings.shape}")
    return embeddings

# 主函数
def main():
    # 文件路径
    input_file = "/DataOriginal/Data/US高校维基百科爬虫数据_filtered.json"
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：文件 {input_file} 不存在")
        return
    
    # 加载数据
    print(f"从 {input_file} 加载JSON数据")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"加载了 {len(data)} 条记录")
    
    # 设置设备为CPU，避免GPU相关问题
    device = "cpu"
    print(f"使用 {device} 设备进行嵌入计算")
    
    # 加载文本嵌入模型
    print(f"加载文本嵌入模型 {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    # 连接到Milvus
    connect_to_milvus()
    
    # 创建集合
    collection = create_collection()
    
    # 准备数据 - 使用较小的批处理大小
    print("准备数据插入")
    batch_size = 50  # 减小批处理大小
    
    for i in range(0, len(data), batch_size):
        end_idx = min(i + batch_size, len(data))
        batch = data[i:end_idx]
        
        # 准备数据字段
        names = []
        contents = []
        urls = []
        
        for item in batch:
            name = item.get('name', '')
            # 合并标题和正文为内容
            content = f"{name}. {item.get('content', '')}"
            url = item.get('url', '')
            
            names.append(name)
            contents.append(content[:65000])  # 截断以适应字段长度限制
            urls.append(url)
        
        # 生成嵌入
        try:
            embeddings = generate_embeddings(contents, model)
            
            # 准备插入数据
            insert_data = [
                names,
                contents,
                urls,
                embeddings.tolist()
            ]
            
            # 插入数据
            collection.insert(insert_data)
            print(f"已插入 {end_idx}/{len(data)} 条记录")
        
        except Exception as e:
            print(f"处理批次 {i} 到 {end_idx} 时出错: {e}")
            continue
    
    # 刷新集合确保数据可用于搜索
    collection.flush()
    print(f"成功将数据插入Milvus。总记录数: {collection.num_entities}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒") 