#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ARWU排名数据测试脚本
测试已导入到Milvus向量数据库的ARWU排名数据
检查三个集合：评分向量、增强向量和文本向量
"""

import os
import sys
import time
from pymilvus import (
    connections,
    utility,
    Collection
)
import configparser

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 配置文件路径
config_file = os.path.join(project_root, "Config", "Milvus.ini")

# 读取配置文件
def load_config():
    """读取配置文件"""
    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')
    return {
        'host': config.get('connection', 'host', fallback='localhost'),
        'port': config.get('connection', 'port', fallback='19530')
    }


def connect_milvus():
    """连接到Milvus服务器"""
    # 加载配置
    milvus_config = load_config()
    host = milvus_config['host']
    port = milvus_config['port']
    print(f"连接到 Milvus 服务器 {host}:{port}")
    try:
        connections.connect(
            alias="default",
            host=_host,
            port=_port
        )
        print("连接成功")
        return True
    except Exception as e:
        print(f"连接失败: {str(e)}")
        return False

def test_score_collection(university="哈佛大学", top_k=3):
    """测试评分向量集合"""
    collection_name = "arwu_score"
    
    if not utility.has_collection(collection_name):
        print(f"未找到集合: {collection_name}")
        return False
    
    try:
        # 获取集合
        collection = Collection(name=collection_name)
        
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
            collection.release()
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

def test_enhanced_collection(university="清华大学", top_k=3):
    """测试增强向量集合"""
    collection_name = "arwu_enhanced"
    
    if not utility.has_collection(collection_name):
        print(f"未找到集合: {collection_name}")
        return False
    
    try:
        # 获取集合
        collection = Collection(name=collection_name)
        
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
            collection.release()
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

def test_text_collection(query="著名的亚洲研究型大学", top_k=3):
    """测试文本向量集合 - 模拟查询"""
    collection_name = "arwu_text"
    
    if not utility.has_collection(collection_name):
        print(f"未找到集合: {collection_name}")
        return False
    
    try:
        # 获取集合
        collection = Collection(name=collection_name)
        
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
            collection.release()
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

def check_collection_info():
    """检查集合信息"""
    try:
        print("\n=== 集合信息 ===")
        collections = utility.list_collections()
        print(f"可用集合数量: {len(collections)}")
        print(f"可用集合列表: {collections}")
        
        # 检查每个集合的数据量
        for collection_name in ["arwu_score", "arwu_enhanced", "arwu_text"]:
            if utility.has_collection(collection_name):
                collection = Collection(name=collection_name)
                count = collection.num_entities
                print(f"集合 {collection_name} 数据量: {count} 条记录")
            else:
                print(f"集合 {collection_name} 不存在")
        
        return True
    except Exception as e:
        print(f"检查集合信息时出错: {str(e)}")
        return False

def main():
    """主函数"""
    start_time = time.time()
    print("\n=== ARWU排名数据测试 ===\n")
    
    # 连接Milvus
    if not connect_milvus():
        print("无法连接到Milvus服务器，终止测试")
        return
    
    # 检查集合信息
    check_collection_info()
    
    success = True
    
    # 测试评分向量集合
    print("\n--- 测试评分向量集合 ---")
    if not test_score_collection("哈佛大学"):
        success = False
    
    # 测试增强向量集合
    print("\n--- 测试增强向量集合 ---")
    if not test_enhanced_collection("清华大学"):
        success = False
    
    # 测试文本向量集合
    print("\n--- 测试文本向量集合(模拟) ---")
    if not test_text_collection("著名的亚洲研究型大学"):
        success = False
    
    if success:
        print("\n=== 数据测试通过 ===")
    else:
        print("\n=== 数据测试部分失败 ===")
    
    # 计算耗时
    elapsed_time = time.time() - start_time
    print(f"\n总耗时: {elapsed_time:.2f} 秒\n")
    
    # 断开连接
    connections.disconnect("default")
    print("已断开Milvus连接")
    print("\n完成")

if __name__ == "__main__":
    main() 
