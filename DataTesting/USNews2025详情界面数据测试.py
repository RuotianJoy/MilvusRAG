#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer
import os
import configparser
import numpy as np

# 禁用Metal性能着色器和GPU加速
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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

def connect_to_milvus():
    """连接到Milvus服务器"""
    # 加载配置
    milvus_config = load_config()
    host = milvus_config['host']
    port = milvus_config['port']
    
    try:
        connections.connect("default", host=host, port=port)
        print(f"✅ 已连接到Milvus服务器: {host}:{port}")
        return True
    except Exception as e:
        print(f"❌ 连接Milvus服务器失败: {str(e)}")
        return False

def init_model():
    """初始化并返回Sentence Transformer模型"""
    try:
        # 强制使用CPU设备
        device = "cpu"
        print(f"✅ 正在使用 {device} 设备加载模型...")
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print("✅ 成功加载SentenceTransformer模型")
        return model
    except Exception as e:
        print(f"❌ 加载模型失败: {str(e)}")
        raise

def get_embeddings(text, model, vector_dim=384):
    """获取文本的嵌入向量"""
    if not text:
        # 如果文本为空，返回全零向量
        return np.zeros(vector_dim).tolist()
    
    try:
        # 使用SentenceTransformer获取嵌入
        embedding = model.encode(text, show_progress_bar=False)
        return embedding.tolist()
    except Exception as e:
        print(f"❌ 获取嵌入向量失败: {str(e)}")
        # 返回全零向量作为备用
        return np.zeros(vector_dim).tolist()

def search_universities(query_text, limit=5):
    """使用向量搜索查询大学基础信息"""
    if not connect_to_milvus():
        return
    
    try:
        # 检查集合是否存在
        if not utility.has_collection("university_base"):
            print("❌ 集合 university_base 不存在，请先运行数据导入脚本")
            return
        
        # 加载集合
        collection = Collection("university_base")
        collection.load()
        print(f"✅ 已加载university_base集合，包含 {collection.num_entities} 条记录")
        
        # 初始化模型
        model = init_model()
        
        # 生成查询向量
        query_vector = get_embeddings(query_text, model)
        
        # 搜索参数
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }
        
        # 执行搜索
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["name", "rank", "display_rank"]
        )
        
        # 打印结果
        print(f"\n查询: '{query_text}'")
        print(f"搜索结果:")
        for i, hit in enumerate(results[0]):
            print(f"{i+1}. {hit.entity.get('name')}")
            print(f"   排名: {hit.entity.get('display_rank')}")
            print(f"   相似度得分: {hit.distance:.4f}")
        
        collection.release()
        print("\n✅ 搜索完成")
        return results[0]
        
    except Exception as e:
        print(f"❌ 搜索错误: {str(e)}")
        return None
    finally:
        try:
            # 确保释放集合
            if 'collection' in locals():
                collection.release()
            # 不要在每次查询后断开连接，保持连接以便后续查询
        except:
            pass

def search_university_summary(university_id=None, query_text=None, limit=3):
    """搜索大学概述信息"""
    if not connect_to_milvus():
        return
    
    try:
        # 检查集合是否存在
        if not utility.has_collection("university_summary"):
            print("❌ 集合 university_summary 不存在，请先运行数据导入脚本")
            return
        
        # 加载集合
        collection = Collection("university_summary")
        collection.load()
        print(f"✅ 已加载university_summary集合，包含 {collection.num_entities} 条记录")
        
        # 如果有大学ID，直接执行ID查询
        if university_id:
            expr = f"university_id == '{university_id}'"
            results = collection.query(
                expr=expr,
                output_fields=["university_id", "summary"]
            )
            
            if results:
                print(f"\n大学ID: {university_id}")
                print(f"概述信息:")
                print(f"{results[0].get('summary')}")
            else:
                print(f"❌ 未找到ID为 {university_id} 的大学概述信息")
                
            collection.release()
            return results
        
        # 如果有查询文本，执行向量搜索
        elif query_text:
            # 初始化模型
            model = init_model()
            
            # 生成查询向量
            query_vector = get_embeddings(query_text, model)
            
            # 搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }
            
            # 执行搜索
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["university_id", "summary"]
            )
            
            # 打印结果
            print(f"\n概述查询: '{query_text}'")
            print(f"搜索结果:")
            for i, hit in enumerate(results[0]):
                print(f"{i+1}. 大学ID: {hit.entity.get('university_id')}")
                print(f"   相似度得分: {hit.distance:.4f}")
                print(f"   摘要: {hit.entity.get('summary')[:200]}...")
            
            collection.release()
            return results[0]
        
        else:
            print("❌ 请提供大学ID或查询文本")
            return None
            
    except Exception as e:
        print(f"❌ 搜索错误: {str(e)}")
        return None
    finally:
        try:
            # 确保释放集合
            if 'collection' in locals():
                collection.release()
        except:
            pass

def search_university_subjects(university_id=None, subject_query=None, limit=3):
    """搜索大学学科排名信息"""
    if not connect_to_milvus():
        return
    
    try:
        # 检查集合是否存在
        if not utility.has_collection("university_subjects"):
            print("❌ 集合 university_subjects 不存在，请先运行数据导入脚本")
            return
        
        # 加载集合
        collection = Collection("university_subjects")
        collection.load()
        print(f"✅ 已加载university_subjects集合，包含 {collection.num_entities} 条记录")
        
        # 如果有大学ID，直接执行ID查询
        if university_id:
            expr = f"university_id == '{university_id}'"
            results = collection.query(
                expr=expr,
                output_fields=["university_id", "subject_data"]
            )
            
            if results:
                import json
                subject_data = json.loads(results[0].get('subject_data'))
                print(f"\n大学ID: {university_id}")
                print(f"学科排名信息:")
                for subject in subject_data:
                    print(f"- {subject.get('subject')}: #{subject.get('rank')}")
            else:
                print(f"❌ 未找到ID为 {university_id} 的大学学科排名信息")
                
            collection.release()
            return results
        
        # 如果有查询文本，执行向量搜索
        elif subject_query:
            # 初始化模型
            model = init_model()
            
            # 生成查询向量
            query_vector = get_embeddings(subject_query, model)
            
            # 搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }
            
            # 执行搜索
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["university_id", "subject_data"]
            )
            
            # 打印结果
            import json
            print(f"\n学科查询: '{subject_query}'")
            print(f"搜索结果:")
            for i, hit in enumerate(results[0]):
                print(f"{i+1}. 大学ID: {hit.entity.get('university_id')}")
                print(f"   相似度得分: {hit.distance:.4f}")
                subjects = json.loads(hit.entity.get('subject_data'))
                print(f"   学科排名 (前5):")
                for s in subjects[:5]:
                    print(f"   - {s.get('subject')}: #{s.get('rank')}")
            
            collection.release()
            return results[0]
        
        else:
            print("❌ 请提供大学ID或学科查询文本")
            return None
            
    except Exception as e:
        print(f"❌ 搜索错误: {str(e)}")
        return None
    finally:
        try:
            # 确保释放集合
            if 'collection' in locals():
                collection.release()
        except:
            pass

def list_available_collections():
    """列出所有可用的集合"""
    if not connect_to_milvus():
        return
    
    try:
        collections = utility.list_collections()
        print("\n可用的集合:")
        for i, col in enumerate(collections):
            print(f"{i+1}. {col}")
            if utility.has_collection(col):
                collection = Collection(col)
                print(f"   - 记录数: {collection.num_entities}")
                print(f"   - 字段: {[field.name for field in collection.schema.fields]}")
        return collections
    except Exception as e:
        print(f"❌ 列出集合错误: {str(e)}")
        return None

if __name__ == "__main__":
    print("=" * 50)
    print("USNews2025大学数据查询测试")
    print("=" * 50)
    
    # 列出可用集合
    print("\n【1. 列出可用集合】")
    collections = list_available_collections()
    
    if not collections:
        print("❌ 没有可用的集合，请先运行数据导入脚本")
        exit(0)
    
    # 测试基础查询
    print("\n【2. 大学基础信息查询】")
    base_queries = [
        "Harvard University",
        "top business school",
        "best medical university"
    ]
    
    # 执行查询
    print(f"\n执行第一个查询: '{base_queries[0]}'")
    result = search_universities(base_queries[0])
    
    # 如果有结果，获取第一个大学的ID
    university_id = None
    if result and len(result) > 0:
        university_id = result[0].id
    
    # 询问用户是否要继续测试
    print("\n是否要测试更多基础查询? (y/n)")
    response = input().strip().lower()
    if response == 'y':
        for query in base_queries[1:]:
            print("\n" + "-"*50)
            search_universities(query)
            print("-"*50)
    
    # 如果有大学ID，测试概述查询
    if university_id:
        print("\n【3. 大学概述查询】")
        print(f"使用大学ID: {university_id}")
        search_university_summary(university_id=university_id)
        
        print("\n用文本查询大学概述:")
        search_university_summary(query_text="research innovation technology")
    
        print("\n【4. 大学学科排名查询】")
        print(f"使用大学ID: {university_id}")
        search_university_subjects(university_id=university_id)
        
        print("\n用文本查询学科排名:")
        search_university_subjects(subject_query="computer science")
    
    # 在所有测试完成后断开连接
    try:
        connections.disconnect("default")
        print("✅ 已断开Milvus连接")
    except:
        pass
        
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50) 
