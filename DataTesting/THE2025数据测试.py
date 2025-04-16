#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
THE2025排名数据测试脚本
用于测试THE2025排名数据是否成功导入Milvus并能够进行查询
适配修改后的多集合设计（基本信息、学科和评分指标分别存储在三个集合中）
"""

from pymilvus import connections, Collection, utility
import numpy as np
import argparse
import os
import configparser

# 集合名称常量
BASIC_INFO_COLLECTION = "the2025_basic_info"
SUBJECTS_COLLECTION = "the2025_subjects"
METRICS_COLLECTION = "the2025_metrics"
META_COLLECTION = "the2025_meta"

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

def test_meta_collection():
    """测试元数据集合的基本查询功能"""
    # 加载配置
    milvus_config = load_config()
    host = milvus_config['host']
    port = milvus_config['port']
    
    # 连接到Milvus
    print("连接到Milvus...")
    connections.connect(host=host, port=port)
    
    # 检查集合是否存在
    if not utility.has_collection(META_COLLECTION):
        print(f"错误: 集合 {META_COLLECTION} 不存在!")
        connections.disconnect("default")
        return False
    
    # 获取集合
    print(f"加载集合 {META_COLLECTION}...")
    collection = Collection(META_COLLECTION)
    collection.load()
    
    try:
        # 获取集合信息
        print("\n元数据集合信息:")
        print(f"集合名称: {collection.name}")
        print(f"集合实体数量: {collection.num_entities}")
        print(f"集合分区: {collection.partitions}")
        
        # 获取字段信息
        schema = collection.schema
        print(f"集合字段: {[field.name for field in schema.fields]}")
        
        # 测试1: 简单查询前10所大学
        print("\n测试1: 查询前10所大学")
        results = collection.query(
            expr="id >= 0",
            output_fields=["name", "rank", "location", "overall_score"],
            limit=10
        )
        
        if results:
            print(f"找到 {len(results)} 所大学，显示前10所:")
            for i, r in enumerate(results[:10]):
                print(f"{i+1}. {r['name']} - 排名: {r['rank']}, 地区: {r['location']}, 总分: {r['overall_score']}")
        else:
            print("未找到任何记录")
        
        # 测试2: 按地区筛选
        print("\n测试2: 查询特定地区 (United Kingdom) 的前5所大学")
        results = collection.query(
            expr="location == 'United Kingdom'",
            output_fields=["name", "rank", "overall_score", "teaching_score", "research_score"],
            limit=5
        )
        
        if results:
            print(f"找到 {len(results)} 所英国大学，显示前5所:")
            for i, r in enumerate(results[:5]):
                print(f"{i+1}. {r['name']} - 排名: {r['rank']}")
                print(f"   总分: {r['overall_score']}, 教学: {r['teaching_score']}, 研究: {r['research_score']}")
        else:
            print("未找到相关记录")
            
        # 测试3: 按评分范围筛选
        print("\n测试3: 查询总评分在80分以上的前5所大学")
        results = collection.query(
            expr="overall_score >= 80",
            output_fields=["name", "rank", "location", "overall_score"],
            limit=5
        )
        
        if results:
            print(f"找到 {len(results)} 所高分大学，显示前5所:")
            for i, r in enumerate(results[:5]):
                print(f"{i+1}. {r['name']} - 排名: {r['rank']}, 地区: {r['location']}, 总分: {r['overall_score']}")
        else:
            print("未找到相关记录")
            
        return True
        
    except Exception as e:
        print(f"查询测试出错: {e}")
        return False
        
    finally:
        # 释放集合并断开连接
        collection.release()
        connections.disconnect("default")
        print("元数据集合测试完成，集合已释放，连接已断开")

def test_basic_info_vector():
    """测试基本信息向量搜索功能"""
    # 连接到Milvus
    print("\n连接到Milvus进行基本信息向量搜索测试...")
    connections.connect(host="localhost", port="19530")
    
    # 检查集合是否存在
    if not utility.has_collection(BASIC_INFO_COLLECTION):
        print(f"错误: 集合 {BASIC_INFO_COLLECTION} 不存在!")
        connections.disconnect("default")
        return False
    
    # 获取集合
    collection = Collection(BASIC_INFO_COLLECTION)
    collection.load()
    
    try:
        # 基于学校名称向量搜索
        print("\n基本信息向量搜索测试")
        # 获取一个已知大学作为查询样本
        first_univ = collection.query(
            expr="id >= 0",
            output_fields=["name", "basic_info_vector", "location"],
            limit=1
        )
        
        if first_univ and len(first_univ) > 0:
            sample_name = first_univ[0]["name"]
            sample_vector = first_univ[0]["basic_info_vector"]
            sample_location = first_univ[0]["location"]
            
            print(f"使用 '{sample_name}' ({sample_location}) 作为查询样本")
            
            # 执行向量搜索
            results = collection.search(
                data=[sample_vector],
                anns_field="basic_info_vector",
                param={"metric_type": "COSINE", "params": {"ef": 10}},
                limit=6,  # 第一个通常是查询向量本身
                output_fields=["name", "rank", "location", "overall_score"]
            )
            
            if results and len(results) > 0 and len(results[0]) > 0:
                print(f"找到 {len(results[0])} 所相似大学:")
                # 跳过第一个（通常是查询向量本身）
                for i, hit in enumerate(results[0][1:]):
                    print(f"{i+1}. {hit.entity.get('name')} - 相似度: {hit.distance:.4f}")
                    print(f"   排名: {hit.entity.get('rank')}, 地区: {hit.entity.get('location')}")
            else:
                print("未找到相似记录")
        else:
            print("无法获取样本大学")
        
        return True
            
    except Exception as e:
        print(f"基本信息向量搜索测试出错: {e}")
        return False
        
    finally:
        # 释放集合并断开连接
        collection.release()
        connections.disconnect("default")
        print("\n基本信息向量搜索测试完成，集合已释放，连接已断开")

def test_subjects_vector():
    """测试学科向量搜索功能"""
    # 连接到Milvus
    print("\n连接到Milvus进行学科向量搜索测试...")
    connections.connect(host="localhost", port="19530")
    
    # 检查集合是否存在
    if not utility.has_collection(SUBJECTS_COLLECTION):
        print(f"错误: 集合 {SUBJECTS_COLLECTION} 不存在!")
        connections.disconnect("default")
        return False
    
    # 获取集合
    collection = Collection(SUBJECTS_COLLECTION)
    collection.load()
    
    try:
        # 基于学科向量搜索
        print("\n学科向量搜索测试")
        # 获取一个大学的学科向量
        subject_sample = collection.query(
            expr="id >= 0",
            output_fields=["name", "subjects_vector"],
            limit=1
        )
        
        if subject_sample and len(subject_sample) > 0:
            sample_name = subject_sample[0]["name"]
            sample_vector = subject_sample[0]["subjects_vector"]
            
            print(f"使用 '{sample_name}' 的学科向量作为查询样本")
            
            # 执行向量搜索
            results = collection.search(
                data=[sample_vector],
                anns_field="subjects_vector",
                param={"metric_type": "COSINE", "params": {"ef": 10}},
                limit=6,
                output_fields=["name", "rank", "location"]
            )
            
            if results and len(results) > 0 and len(results[0]) > 0:
                print(f"找到 {len(results[0])} 所学科相似大学:")
                # 跳过第一个（通常是查询向量本身）
                for i, hit in enumerate(results[0][1:]):
                    print(f"{i+1}. {hit.entity.get('name')} - 相似度: {hit.distance:.4f}")
                    print(f"   排名: {hit.entity.get('rank')}, 地区: {hit.entity.get('location')}")
            else:
                print("未找到相似记录")
        else:
            print("无法获取样本大学")
            
        return True
            
    except Exception as e:
        print(f"学科向量搜索测试出错: {e}")
        return False
        
    finally:
        # 释放集合并断开连接
        collection.release()
        connections.disconnect("default")
        print("\n学科向量搜索测试完成，集合已释放，连接已断开")

def test_metrics_vector():
    """测试评分指标向量搜索功能"""
    # 连接到Milvus
    print("\n连接到Milvus进行评分指标向量搜索测试...")
    connections.connect(host="localhost", port="19530")
    
    # 检查集合是否存在
    if not utility.has_collection(METRICS_COLLECTION):
        print(f"错误: 集合 {METRICS_COLLECTION} 不存在!")
        connections.disconnect("default")
        return False
    
    # 获取集合
    collection = Collection(METRICS_COLLECTION)
    collection.load()
    
    try:
        # 基于评分指标向量搜索
        print("\n评分指标向量搜索测试")
        # 获取一个大学的评分向量
        metrics_sample = collection.query(
            expr="id >= 0",
            output_fields=["name", "metrics_vector", "overall_score"],
            limit=1
        )
        
        if metrics_sample and len(metrics_sample) > 0:
            sample_name = metrics_sample[0]["name"]
            sample_vector = metrics_sample[0]["metrics_vector"]
            sample_score = metrics_sample[0]["overall_score"]
            
            print(f"使用 '{sample_name}' (总分: {sample_score}) 的评分向量作为查询样本")
            
            # 执行向量搜索
            results = collection.search(
                data=[sample_vector],
                anns_field="metrics_vector",
                param={"metric_type": "L2", "params": {"ef": 10}},
                limit=6,
                output_fields=["name", "rank", "location", "overall_score"]
            )
            
            if results and len(results) > 0 and len(results[0]) > 0:
                print(f"找到 {len(results[0])} 所评分相似大学:")
                # 跳过第一个（通常是查询向量本身）
                for i, hit in enumerate(results[0][1:]):
                    print(f"{i+1}. {hit.entity.get('name')} - 距离: {hit.distance:.4f}")
                    print(f"   排名: {hit.entity.get('rank')}, 总分: {hit.entity.get('overall_score')}")
            else:
                print("未找到相似记录")
        else:
            print("无法获取样本大学")
        
        return True
            
    except Exception as e:
        print(f"评分指标向量搜索测试出错: {e}")
        return False
        
    finally:
        # 释放集合并断开连接
        collection.release()
        connections.disconnect("default")
        print("\n评分指标向量搜索测试完成，集合已释放，连接已断开")

def check_collections_exist():
    """检查所有需要的集合是否存在"""
    connections.connect(host="localhost", port="19530")
    
    collections = [
        BASIC_INFO_COLLECTION,
        SUBJECTS_COLLECTION,
        METRICS_COLLECTION,
        META_COLLECTION
    ]
    
    existing = []
    missing = []
    
    for coll in collections:
        if utility.has_collection(coll):
            existing.append(coll)
        else:
            missing.append(coll)
    
    connections.disconnect("default")
    
    return existing, missing

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试THE2025排名数据在Milvus中的查询")
    parser.add_argument("--test", choices=["all", "meta", "basic", "subjects", "metrics"], 
                        default="all", help="要测试的集合类型")
    args = parser.parse_args()
    
    # 首先检查所有集合是否存在
    existing, missing = check_collections_exist()
    
    if missing:
        print(f"警告: 以下集合不存在: {', '.join(missing)}")
    
    print(f"可用的集合: {', '.join(existing)}")
    
    success = True
    
    # 根据参数决定测试哪些集合
    if args.test == "all" or args.test == "meta":
        if META_COLLECTION in existing:
            print("\n=== 测试元数据集合 ===")
            if not test_meta_collection():
                success = False
                print("元数据集合测试失败!")
        else:
            print(f"跳过元数据集合测试，集合不存在")
    
    if args.test == "all" or args.test == "basic":
        if BASIC_INFO_COLLECTION in existing:
            print("\n=== 测试基本信息向量 ===")
            if not test_basic_info_vector():
                success = False
                print("基本信息向量测试失败!")
        else:
            print(f"跳过基本信息向量测试，集合不存在")
    
    if args.test == "all" or args.test == "subjects":
        if SUBJECTS_COLLECTION in existing:
            print("\n=== 测试学科向量 ===")
            if not test_subjects_vector():
                success = False
                print("学科向量测试失败!")
        else:
            print(f"跳过学科向量测试，集合不存在")
    
    if args.test == "all" or args.test == "metrics":
        if METRICS_COLLECTION in existing:
            print("\n=== 测试评分指标向量 ===")
            if not test_metrics_vector():
                success = False
                print("评分指标向量测试失败!")
        else:
            print(f"跳过评分指标向量测试，集合不存在")
    
    if success:
        print("\n所有测试成功完成!")
    else:
        print("\n部分测试失败，请检查日志获取详细信息")

if __name__ == "__main__":
    main() 
