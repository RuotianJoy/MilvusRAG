#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ARWU排名数据导入脚本
将处理好的ARWU排名数据导入到Milvus向量数据库
包含三个集合：评分向量、增强向量和文本向量
"""

import os
import sys
import json
import time
import numpy as np
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

# 工作目录设置
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 输入输出文件路径
processed_file = os.path.join(project_root, "数据处理", "ARWU2024_processed.json")

# 连接参数
_HOST = 'localhost'
_PORT = '19530'
_PARTITION_NAME = "ARWU2024"

def connect_milvus():
    """连接到Milvus服务器"""
    print(f"连接到 Milvus 服务器 {_HOST}:{_PORT}")
    try:
        connections.connect(
            alias="default",
            host=_HOST,
            port=_PORT
        )
        print("连接成功")
        return True
    except Exception as e:
        print(f"连接失败: {str(e)}")
        return False

def create_score_collection(collection_name="arwu_score"):
    """创建评分向量集合"""
    # 删除已存在的同名集合
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"已删除现有集合: {collection_name}")
    
    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="university", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="university_en", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="country", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="country_en", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="continent", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="rank", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="rank_lower", dtype=DataType.INT64),
        FieldSchema(name="rank_upper", dtype=DataType.INT64),
        FieldSchema(name="rank_numeric", dtype=DataType.DOUBLE),
        FieldSchema(name="region_rank", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="total_score", dtype=DataType.DOUBLE),
        FieldSchema(name="alumni_award", dtype=DataType.DOUBLE),
        FieldSchema(name="prof_award", dtype=DataType.DOUBLE),
        FieldSchema(name="high_cited_scientist", dtype=DataType.DOUBLE),
        FieldSchema(name="ns_paper", dtype=DataType.DOUBLE),
        FieldSchema(name="inter_paper", dtype=DataType.DOUBLE),
        FieldSchema(name="avg_prof_performance", dtype=DataType.DOUBLE),
        FieldSchema(name="score_vector", dtype=DataType.FLOAT_VECTOR, dim=7)
    ]
    
    # 创建集合模式
    schema = CollectionSchema(fields=fields, description="ARWU2024 Score Vectors")
    
    # 创建集合
    collection = Collection(name=collection_name, schema=schema)
    print(f"已创建集合 {collection_name}")
    
    # 创建分区
    collection.create_partition(_PARTITION_NAME)
    print(f"已创建分区 {_PARTITION_NAME}")
    
    # 创建索引
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="score_vector", index_params=index_params)
    print("已创建索引")
    
    return collection

def create_enhanced_collection(collection_name="arwu_enhanced"):
    """创建增强向量集合"""
    # 删除已存在的同名集合
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"已删除现有集合: {collection_name}")
    
    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="university", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="university_en", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="country", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="country_en", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="continent", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="rank", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="rank_numeric", dtype=DataType.DOUBLE),
        FieldSchema(name="enhanced_vector", dtype=DataType.FLOAT_VECTOR, dim=10)
    ]
    
    # 创建集合模式
    schema = CollectionSchema(fields=fields, description="ARWU2024 Enhanced Vectors")
    
    # 创建集合
    collection = Collection(name=collection_name, schema=schema)
    print(f"已创建集合 {collection_name}")
    
    # 创建分区
    collection.create_partition(_PARTITION_NAME)
    print(f"已创建分区 {_PARTITION_NAME}")
    
    # 创建索引
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="enhanced_vector", index_params=index_params)
    print("已创建索引")
    
    return collection

def create_text_collection(collection_name="arwu_text"):
    """创建文本向量集合"""
    # 删除已存在的同名集合
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"已删除现有集合: {collection_name}")
    
    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="university", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="university_en", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="country", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="country_en", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="continent", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="rank", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="rank_numeric", dtype=DataType.DOUBLE),
        FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    
    # 创建集合模式
    schema = CollectionSchema(fields=fields, description="ARWU2024 Text Embedding Vectors")
    
    # 创建集合
    collection = Collection(name=collection_name, schema=schema)
    print(f"已创建集合 {collection_name}")
    
    # 创建分区
    collection.create_partition(_PARTITION_NAME)
    print(f"已创建分区 {_PARTITION_NAME}")
    
    # 创建索引
    index_params = {
        "metric_type": "IP",  # 内积距离，适合文本向量
        "index_type": "HNSW",  # 高维向量推荐使用HNSW
        "params": {
            "M": 16,
            "efConstruction": 200
        }
    }
    collection.create_index(field_name="text_vector", index_params=index_params)
    print("已创建索引")
    
    return collection

def import_score_data(collection, data_file):
    """导入评分向量数据"""
    if not collection:
        return False
    
    try:
        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"正在导入评分向量数据，共 {len(data)} 条记录")
        
        # 准备数据
        id_list = []
        university_list = []
        university_en_list = []
        country_list = []
        country_en_list = []
        continent_list = []
        rank_list = []
        rank_lower_list = []
        rank_upper_list = []
        rank_numeric_list = []
        region_rank_list = []
        total_score_list = []
        alumni_award_list = []
        prof_award_list = []
        high_cited_list = []
        ns_paper_list = []
        inter_paper_list = []
        avg_prof_list = []
        score_vector_list = []
        
        # 处理数据
        for i, item in enumerate(data):
            id_list.append(i)
            university_list.append(str(item.get("University", "")))
            university_en_list.append(str(item.get("University_English", "")))
            country_list.append(str(item.get("Country", "")))
            country_en_list.append(str(item.get("Country_English", "")))
            continent_list.append(str(item.get("Continent", "")))
            rank_list.append(str(item.get("Rank", "")))
            rank_lower_list.append(int(item.get("rank_lower", 0)))
            rank_upper_list.append(int(item.get("rank_upper", 0)))
            rank_numeric_list.append(float(item.get("rank_numeric", 0.0)))
            region_rank_list.append(str(item.get("Region_Rank", "")))
            total_score_list.append(float(item.get("Total_Score", 0.0)))
            alumni_award_list.append(float(item.get("Alumni_Award", 0.0)))
            prof_award_list.append(float(item.get("Prof_Award", 0.0)))
            high_cited_list.append(float(item.get("High_cited_Scientist", 0.0)))
            ns_paper_list.append(float(item.get("NS_Paper", 0.0)))
            inter_paper_list.append(float(item.get("Inter_Paper", 0.0)))
            avg_prof_list.append(float(item.get("Avg_Prof_Performance", 0.0)))
            
            # 处理向量
            vector = [float(x) for x in item.get("basic_score_vector", [0.0] * 7)]
            score_vector_list.append(vector)
        
        # 构造实体数据，使用列表的列表格式
        entities = [
            id_list,
            university_list,
            university_en_list,
            country_list,
            country_en_list,
            continent_list,
            rank_list,
            rank_lower_list,
            rank_upper_list,
            rank_numeric_list,
            region_rank_list,
            total_score_list,
            alumni_award_list,
            prof_award_list,
            high_cited_list,
            ns_paper_list,
            inter_paper_list,
            avg_prof_list,
            score_vector_list
        ]
        
        # 插入数据到分区
        insert_result = collection.insert(entities, partition_name=_PARTITION_NAME)
        print(f"成功插入评分向量数据: {insert_result.insert_count} 条记录")
        
        # 刷新集合
        collection.flush()
        print("评分向量集合已刷新")
        
        return True
    except Exception as e:
        print(f"导入评分向量数据时出错: {str(e)}")
        return False

def import_enhanced_data(collection, data_file):
    """导入增强向量数据"""
    if not collection:
        return False
    
    try:
        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"正在导入增强向量数据，共 {len(data)} 条记录")
        
        # 准备数据
        id_list = []
        university_list = []
        university_en_list = []
        country_list = []
        country_en_list = []
        continent_list = []
        rank_list = []
        rank_numeric_list = []
        enhanced_vector_list = []
        
        # 处理数据
        for i, item in enumerate(data):
            id_list.append(i)
            university_list.append(str(item.get("University", "")))
            university_en_list.append(str(item.get("University_English", "")))
            country_list.append(str(item.get("Country", "")))
            country_en_list.append(str(item.get("Country_English", "")))
            continent_list.append(str(item.get("Continent", "")))
            rank_list.append(str(item.get("Rank", "")))
            rank_numeric_list.append(float(item.get("rank_numeric", 0.0)))
            
            # 处理向量
            vector = [float(x) for x in item.get("enhanced_vector", [0.0] * 10)]
            enhanced_vector_list.append(vector)
        
        # 构造实体数据，使用列表的列表格式
        entities = [
            id_list,
            university_list,
            university_en_list,
            country_list,
            country_en_list,
            continent_list,
            rank_list,
            rank_numeric_list,
            enhanced_vector_list
        ]
        
        # 插入数据到分区
        insert_result = collection.insert(entities, partition_name=_PARTITION_NAME)
        print(f"成功插入增强向量数据: {insert_result.insert_count} 条记录")
        
        # 刷新集合
        collection.flush()
        print("增强向量集合已刷新")
        
        return True
    except Exception as e:
        print(f"导入增强向量数据时出错: {str(e)}")
        return False

def import_text_data(collection, data_file):
    """导入文本向量数据"""
    if not collection:
        return False
    
    try:
        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"正在导入文本向量数据，共 {len(data)} 条记录")
        
        # 准备数据
        id_list = []
        university_list = []
        university_en_list = []
        country_list = []
        country_en_list = []
        continent_list = []
        rank_list = []
        rank_numeric_list = []
        text_vector_list = []
        
        # 处理数据
        for i, item in enumerate(data):
            id_list.append(i)
            university_list.append(str(item.get("University", "")))
            university_en_list.append(str(item.get("University_English", "")))
            country_list.append(str(item.get("Country", "")))
            country_en_list.append(str(item.get("Country_English", "")))
            continent_list.append(str(item.get("Continent", "")))
            rank_list.append(str(item.get("Rank", "")))
            rank_numeric_list.append(float(item.get("rank_numeric", 0.0)))
            
            # 处理向量
            vector = [float(x) for x in item.get("text_embedding", [0.0] * 768)]
            text_vector_list.append(vector)
        
        # 构造实体数据，使用列表的列表格式
        entities = [
            id_list,
            university_list,
            university_en_list,
            country_list,
            country_en_list,
            continent_list,
            rank_list,
            rank_numeric_list,
            text_vector_list
        ]
        
        # 插入数据到分区
        insert_result = collection.insert(entities, partition_name=_PARTITION_NAME)
        print(f"成功插入文本向量数据: {insert_result.insert_count} 条记录")
        
        # 刷新集合
        collection.flush()
        print("文本向量集合已刷新")
        
        return True
    except Exception as e:
        print(f"导入文本向量数据时出错: {str(e)}")
        return False

def main():
    """主函数"""
    start_time = time.time()
    print("\n=== ARWU排名数据导入 ===\n")
    
    # 连接Milvus
    if not connect_milvus():
        print("无法连接到Milvus服务器，终止导入")
        return
    
    success = True
    collections = {}
    
    # 创建评分向量集合并导入数据
    print("\n--- 创建评分向量集合 ---")
    collections["score"] = create_score_collection()
    if collections["score"]:
        if not import_score_data(collections["score"], processed_file):
            success = False
    else:
        success = False
    
    # 创建增强向量集合并导入数据
    print("\n--- 创建增强向量集合 ---")
    collections["enhanced"] = create_enhanced_collection()
    if collections["enhanced"]:
        if not import_enhanced_data(collections["enhanced"], processed_file):
            success = False
    else:
        success = False
    
    # 创建文本向量集合并导入数据
    print("\n--- 创建文本向量集合 ---")
    collections["text"] = create_text_collection()
    if collections["text"]:
        if not import_text_data(collections["text"], processed_file):
            success = False
    else:
        success = False
    
    if success:
        print("\n=== 数据导入成功 ===")
    else:
        print("\n=== 数据导入部分失败 ===")
    
    # 计算耗时
    elapsed_time = time.time() - start_time
    print(f"\n总耗时: {elapsed_time:.2f} 秒\n")
    
    # 断开连接
    connections.disconnect("default")
    print("已断开Milvus连接")
    print("\n完成")

if __name__ == "__main__":
    main() 