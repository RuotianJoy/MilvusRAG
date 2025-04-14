#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ARWU排名数据完整导入脚本
将处理好的ARWU排名数据导入到Milvus向量数据库
包含三个集合：评分向量、增强向量和文本向量
"""

import os
import sys
import json
import time
import numpy as np
import configparser
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
processed_file = os.path.join(project_root, "DataProcessed", "ARWU2024_processed.json")
config_file = os.path.join(project_root, "Config", "Milvus.ini")

# 读取配置文件
def load_config():
    """读取配置文件"""
    config = configparser.ConfigParser()

    config.read(config_file, encoding='utf-8')
    return {
        'host': config.get('connection', 'host', fallback=''),
        'port': config.get('connection', 'port', fallback=''),
        'partition_name': config.get('Milvus', 'partition_name', fallback='ARWU2024')
    }


# 加载配置
milvus_config = load_config()
_HOST = milvus_config['host']
_PORT = milvus_config['port']
_PARTITION_NAME = milvus_config['partition_name']

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

def test_score_collection(collection, university="哈佛大学", top_k=3):
    """测试评分向量集合"""
    if not collection:
        return False
    
    try:
        # 加载集合
        collection.load()
        print(f"已加载评分向量集合，准备搜索相似于 {university} 的大学")
        
        # 查询参考大学的向量
        results = collection.query(
            expr=f'university == "{university}"',
            output_fields=["university", "university_en", "score_vector"]
        )
        
        if not results:
            print(f"未找到大学: {university}")
            return False
        
        reference_vector = results[0]["score_vector"]
        print(f"找到参考大学: {results[0]['university']} ({results[0]['university_en']})")
        
        # 搜索相似大学
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 16}
        }
        
        search_results = collection.search(
            data=[reference_vector],
            anns_field="score_vector",
            param=search_params,
            limit=top_k+1,  # 多搜索一个以排除自己
            output_fields=["university", "university_en", "country", "country_en", "continent", "rank", "total_score"]
        )
        
        # 输出结果
        print(f"\nTop {top_k} 相似大学 (基于评分向量):")
        count = 0
        for entity in search_results[0]:
            # 排除参考大学自己
            if entity.entity.get("university") != university:
                count += 1
                print(f"{count}. {entity.entity.get('university')} ({entity.entity.get('university_en')})")
                print(f"   国家: {entity.entity.get('country')} ({entity.entity.get('country_en')}), 大洲: {entity.entity.get('continent')}")
                print(f"   排名: {entity.entity.get('rank')}, 总分: {entity.entity.get('total_score'):.1f}")
                print(f"   相似度: {entity.distance:.2f}")
                if count >= top_k:
                    break
        
        # 释放集合
        collection.release()
        return True
    except Exception as e:
        print(f"测试评分向量集合时出错: {str(e)}")
        return False

def test_enhanced_collection(collection, university="清华大学", top_k=3):
    """测试增强向量集合"""
    if not collection:
        return False
    
    try:
        # 加载集合
        collection.load()
        print(f"已加载增强向量集合，准备搜索相似于 {university} 的大学")
        
        # 查询参考大学的向量
        results = collection.query(
            expr=f'university == "{university}"',
            output_fields=["university", "university_en", "enhanced_vector"]
        )
        
        if not results:
            print(f"未找到大学: {university}")
            return False
        
        reference_vector = results[0]["enhanced_vector"]
        print(f"找到参考大学: {results[0]['university']} ({results[0]['university_en']})")
        
        # 搜索相似大学
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 16}
        }
        
        search_results = collection.search(
            data=[reference_vector],
            anns_field="enhanced_vector",
            param=search_params,
            limit=top_k+1,  # 多搜索一个以排除自己
            output_fields=["university", "university_en", "country", "country_en", "rank"]
        )
        
        # 输出结果
        print(f"\nTop {top_k} 相似大学 (基于增强向量):")
        count = 0
        for entity in search_results[0]:
            # 排除参考大学自己
            if entity.entity.get("university") != university:
                count += 1
                print(f"{count}. {entity.entity.get('university')} ({entity.entity.get('university_en')})")
                print(f"   国家: {entity.entity.get('country')} ({entity.entity.get('country_en')})")
                print(f"   排名: {entity.entity.get('rank')}")
                print(f"   相似度: {entity.distance:.2f}")
                if count >= top_k:
                    break
        
        # 释放集合
        collection.release()
        return True
    except Exception as e:
        print(f"测试增强向量集合时出错: {str(e)}")
        return False

def test_text_collection(collection, query="著名的亚洲研究型大学", top_k=3):
    """测试文本向量集合 - 模拟查询"""
    if not collection:
        return False
    
    try:
        # 加载集合
        collection.load()
        print(f"已加载文本向量集合，准备模拟文本查询: '{query}'")
        print("(注: 实际查询应使用BERT模型生成查询向量，此处使用固定向量模拟)")
        
        # 为简单起见，这里使用一个固定的向量来模拟查询
        # 实际应用中应该使用BERT模型处理查询文本
        print("使用北京大学的文本向量作为查询向量模拟")
        
        # 查询参考大学的向量
        results = collection.query(
            expr=f'university == "北京大学"',
            output_fields=["university", "university_en", "text_vector"]
        )
        
        if not results:
            print("未找到模拟查询向量")
            return False
        
        query_vector = results[0]["text_vector"]
        
        # 搜索相似大学
        search_params = {
            "metric_type": "IP",  # 内积相似度
            "params": {"nprobe": 16}
        }
        
        search_results = collection.search(
            data=[query_vector],
            anns_field="text_vector",
            param=search_params,
            limit=top_k,
            output_fields=["university", "university_en", "country", "country_en", "continent", "rank"]
        )
        
        # 输出结果
        print(f"\nTop {top_k} 相似大学 (基于文本向量):")
        for i, entity in enumerate(search_results[0]):
            print(f"{i+1}. {entity.entity.get('university')} ({entity.entity.get('university_en')})")
            print(f"   国家: {entity.entity.get('country')} ({entity.entity.get('country_en')}), 大洲: {entity.entity.get('continent')}")
            print(f"   排名: {entity.entity.get('rank')}")
            print(f"   相似度: {entity.distance:.4f}")
        
        # 释放集合
        collection.release()
        return True
    except Exception as e:
        print(f"测试文本向量集合时出错: {str(e)}")
        return False

def main():
    """主函数"""
    start_time = time.time()
    print("\n=== ARWU排名数据完整导入 ===\n")
    
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
        
        # 测试集合功能
        print("\n--- 测试集合功能 ---")
        
        # 测试评分向量集合
        print("\n1. 测试评分向量集合")
        test_score_collection(collections["score"], "哈佛大学")
        
        # 测试增强向量集合
        print("\n2. 测试增强向量集合")
        test_enhanced_collection(collections["enhanced"], "清华大学")
        
        # 测试文本向量集合
        print("\n3. 测试文本向量集合 (模拟)")
        test_text_collection(collections["text"], "著名的亚洲研究型大学")
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
