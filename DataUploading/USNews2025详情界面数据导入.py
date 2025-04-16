#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import configparser

class MilvusImporter:
    def __init__(self, processed_data_path=None):
        """
        初始化Milvus导入器
        """
        # 连接Milvus服务
        self.connect_to_milvus()
        
        # 设置向量维度
        self.vector_dim = 384  # Sentence-BERT模型维度
        
        # 加载处理后的数据
        self.processed_data_path = processed_data_path
        self.processed_data = self.load_processed_data()
        
        # 初始化嵌入模型
        self.init_embedding_model()

    # 读取配置文件
    def load_config(self):
        """读取配置文件"""
        # 配置文件路径
        config_file = os.path.join(project_root, "Config", "Milvus.ini")
        config = configparser.ConfigParser()
        config.read(config_file, encoding='utf-8')
        return {
            'host': config.get('connection', 'host', fallback='localhost'),
            'port': config.get('connection', 'port', fallback='19530')
        }
    
    def connect_to_milvus(self):
        """
        连接到Milvus服务
        """
        # 加载配置
        milvus_config = self.load_config()
        host = milvus_config['host']
        port = milvus_config['port']
        
        try:
            connections.connect("default", host=host, port=port)
            print(f"已成功连接到Milvus服务器: {host}:{port}")
        except Exception as e:
            print(f"连接Milvus服务器失败: {str(e)}")
            raise
    
    def load_processed_data(self):
        """
        加载处理后的数据
        """
        if not self.processed_data_path or not os.path.exists(self.processed_data_path):
            print("未指定处理后的数据文件路径或文件不存在")
            return []
        
        try:
            with open(self.processed_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"成功加载处理后的数据，共 {len(data)} 条记录")
            return data
        except Exception as e:
            print(f"加载数据文件失败: {str(e)}")
            return []
    
    def init_embedding_model(self):
        """
        初始化嵌入模型
        """
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("成功加载SentenceTransformer模型")
        except Exception as e:
            print(f"加载SentenceTransformer模型失败: {str(e)}")
            raise
    
    def get_embeddings(self, text):
        """
        获取文本的嵌入向量
        """
        if not text:
            # 如果文本为空，返回全零向量
            return np.zeros(self.vector_dim).tolist()
        
        try:
            # 使用SentenceTransformer获取嵌入
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"获取嵌入向量失败: {str(e)}")
            # 返回全零向量作为备用
            return np.zeros(self.vector_dim).tolist()
    
    def create_university_base_collection(self):
        """
        创建大学基础信息集合
        """
        collection_name = "usnews2025_university_base"
        
        # 检查集合是否已存在
        if utility.has_collection(collection_name):
            print(f"集合 {collection_name} 已存在，将删除并重新创建")
            utility.drop_collection(collection_name)
        
        # 定义集合字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="rank", dtype=DataType.INT64),
            FieldSchema(name="is_tied", dtype=DataType.BOOL),
            FieldSchema(name="display_rank", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]
        
        # 创建集合模式和集合
        schema = CollectionSchema(fields, "大学基础信息及向量")
        collection = Collection(collection_name, schema)
        
        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 500
            }
        }
        collection.create_index("embedding", index_params)
        
        print(f"成功创建集合: {collection_name}")
        return collection
    
    def create_university_summary_collection(self):
        """
        创建大学概述集合
        """
        collection_name = "usnews2025_university_summary"
        
        # 检查集合是否已存在
        if utility.has_collection(collection_name):
            print(f"集合 {collection_name} 已存在，将删除并重新创建")
            utility.drop_collection(collection_name)
        
        # 定义集合字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="university_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]
        
        # 创建集合模式和集合
        schema = CollectionSchema(fields, "大学概述信息及向量")
        collection = Collection(collection_name, schema)
        
        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 500
            }
        }
        collection.create_index("embedding", index_params)
        
        print(f"成功创建集合: {collection_name}")
        return collection
    
    def create_university_statistics_collection(self):
        """
        创建大学统计数据集合
        """
        collection_name = "usnews2025_university_statistics"
        
        # 检查集合是否已存在
        if utility.has_collection(collection_name):
            print(f"集合 {collection_name} 已存在，将删除并重新创建")
            utility.drop_collection(collection_name)
        
        # 定义集合字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="university_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="total_students", dtype=DataType.INT64),
            FieldSchema(name="international_students", dtype=DataType.INT64),
            FieldSchema(name="academic_staff", dtype=DataType.INT64),
            FieldSchema(name="international_staff", dtype=DataType.INT64),
            FieldSchema(name="undergraduate_degrees", dtype=DataType.INT64),
            FieldSchema(name="master_degrees", dtype=DataType.INT64),
            FieldSchema(name="doctoral_degrees", dtype=DataType.INT64),
            FieldSchema(name="research_staff", dtype=DataType.INT64),
            FieldSchema(name="new_undergraduate_students", dtype=DataType.INT64),
            FieldSchema(name="new_master_students", dtype=DataType.INT64),
            FieldSchema(name="new_doctoral_students", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]
        
        # 创建集合模式和集合
        schema = CollectionSchema(fields, "大学统计数据及向量")
        collection = Collection(collection_name, schema)
        
        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 500
            }
        }
        collection.create_index("embedding", index_params)
        
        print(f"成功创建集合: {collection_name}")
        return collection
    
    def create_university_subjects_collection(self):
        """
        创建大学学科排名集合
        """
        collection_name = "usnews2025_university_subjects"
        
        # 检查集合是否已存在
        if utility.has_collection(collection_name):
            print(f"集合 {collection_name} 已存在，将删除并重新创建")
            utility.drop_collection(collection_name)
        
        # 定义集合字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="university_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="subject_data", dtype=DataType.VARCHAR, max_length=65535),  # JSON字符串
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]
        
        # 创建集合模式和集合
        schema = CollectionSchema(fields, "大学学科排名及向量")
        collection = Collection(collection_name, schema)
        
        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 500
            }
        }
        collection.create_index("embedding", index_params)
        
        print(f"成功创建集合: {collection_name}")
        return collection
    
    def create_university_indicators_collection(self):
        """
        创建大学全球指标集合
        """
        collection_name = "usnews2025_university_indicators"
        
        # 检查集合是否已存在
        if utility.has_collection(collection_name):
            print(f"集合 {collection_name} 已存在，将删除并重新创建")
            utility.drop_collection(collection_name)
        
        # 定义集合字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="university_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="indicator_data", dtype=DataType.VARCHAR, max_length=65535),  # JSON字符串
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]
        
        # 创建集合模式和集合
        schema = CollectionSchema(fields, "大学全球指标及向量")
        collection = Collection(collection_name, schema)
        
        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 500
            }
        }
        collection.create_index("embedding", index_params)
        
        print(f"成功创建集合: {collection_name}")
        return collection
    
    def import_university_base_data(self, collection):
        """
        导入大学基础信息数据
        """
        if not self.processed_data:
            print("没有可导入的数据")
            return
        
        # 准备数据
        ids = []
        names = []
        ranks = []
        is_tieds = []
        display_ranks = []
        embeddings = []
        
        for university in self.processed_data:
            university_id = university.get("university_id")
            name = university.get("name", "")
            global_rank = university.get("global_rank", {})
            
            # 生成嵌入文本
            embed_text = f"{name} {global_rank.get('display_rank', '')}"
            embedding = self.get_embeddings(embed_text)
            
            ids.append(university_id)
            names.append(name)
            ranks.append(global_rank.get("numeric_rank", 0) or 0)
            is_tieds.append(global_rank.get("is_tied", False))
            display_ranks.append(global_rank.get("display_rank", ""))
            embeddings.append(embedding)
        
        # 执行插入
        collection.insert([
            ids,
            names,
            ranks,
            is_tieds,
            display_ranks,
            embeddings
        ])
        
        # 刷新集合确保数据被持久化
        collection.flush()
        
        print(f"成功导入 {len(ids)} 条基础信息数据")
    
    def import_university_summary_data(self, collection):
        """
        导入大学概述数据
        """
        if not self.processed_data:
            print("没有可导入的数据")
            return
        
        # 准备数据
        ids = []
        university_ids = []
        summaries = []
        embeddings = []
        
        for i, university in enumerate(self.processed_data):
            university_id = university.get("university_id")
            summary = university.get("summary", "")
            
            # 生成嵌入
            embedding = self.get_embeddings(summary)
            
            ids.append(f"summary_{i}_{university_id}")
            university_ids.append(university_id)
            summaries.append(summary)
            embeddings.append(embedding)
        
        # 执行插入
        collection.insert([
            ids,
            university_ids,
            summaries,
            embeddings
        ])
        
        # 刷新集合确保数据被持久化
        collection.flush()
        
        print(f"成功导入 {len(ids)} 条概述数据")
    
    def import_university_statistics_data(self, collection):
        """
        导入大学统计数据
        """
        if not self.processed_data:
            print("没有可导入的数据")
            return
        
        # 处理数字字符串的辅助函数
        def safe_to_int(value):
            if value is None:
                return np.int64(0)
            if isinstance(value, (int, float)):
                return np.int64(value)
            if isinstance(value, str):
                # 去除逗号并转换为浮点数，然后取整
                try:
                    cleaned = value.replace(',', '')
                    return np.int64(float(cleaned))
                except (ValueError, TypeError):
                    print(f"无法转换值 '{value}' 为整数，使用 0")
                    return np.int64(0)
            return np.int64(0)
        
        # 准备数据
        ids = []
        university_ids = []
        total_students_list = []
        international_students_list = []
        academic_staff_list = []
        international_staff_list = []
        undergraduate_degrees_list = []
        master_degrees_list = []
        doctoral_degrees_list = []
        research_staff_list = []
        new_undergraduate_students_list = []
        new_master_students_list = []
        new_doctoral_students_list = []
        embeddings = []
        
        for i, university in enumerate(self.processed_data):
            university_id = university.get("university_id")
            stats = university.get("university_data", {})
            
            # 将统计数据转换为文本以生成嵌入
            stats_text = " ".join([f"{k}: {v}" for k, v in stats.items()])
            embedding = self.get_embeddings(stats_text)
            
            ids.append(f"stats_{i}_{university_id}")
            university_ids.append(university_id)
            # 使用安全转换函数处理所有数值
            total_students_list.append(safe_to_int(stats.get("total_students", 0)))
            international_students_list.append(safe_to_int(stats.get("international_students", 0)))
            academic_staff_list.append(safe_to_int(stats.get("academic_staff", 0)))
            international_staff_list.append(safe_to_int(stats.get("international_staff", 0)))
            undergraduate_degrees_list.append(safe_to_int(stats.get("undergraduate_degrees", 0)))
            master_degrees_list.append(safe_to_int(stats.get("master_degrees", 0)))
            doctoral_degrees_list.append(safe_to_int(stats.get("doctoral_degrees", 0)))
            research_staff_list.append(safe_to_int(stats.get("research_staff", 0)))
            new_undergraduate_students_list.append(safe_to_int(stats.get("new_undergraduate_students", 0)))
            new_master_students_list.append(safe_to_int(stats.get("new_master_students", 0)))
            new_doctoral_students_list.append(safe_to_int(stats.get("new_doctoral_students", 0)))
            embeddings.append(embedding)
        
        # 执行插入
        collection.insert([
            ids,
            university_ids,
            total_students_list,
            international_students_list,
            academic_staff_list,
            international_staff_list,
            undergraduate_degrees_list,
            master_degrees_list,
            doctoral_degrees_list,
            research_staff_list,
            new_undergraduate_students_list,
            new_master_students_list,
            new_doctoral_students_list,
            embeddings
        ])
        
        # 刷新集合确保数据被持久化
        collection.flush()
        
        print(f"成功导入 {len(ids)} 条统计数据")
    
    def import_university_subjects_data(self, collection):
        """
        导入大学学科排名数据
        """
        if not self.processed_data:
            print("没有可导入的数据")
            return
        
        # 准备数据
        ids = []
        university_ids = []
        subject_data_list = []
        embeddings = []
        
        for i, university in enumerate(self.processed_data):
            university_id = university.get("university_id")
            subjects = university.get("subject_rankings", [])
            
            # 将学科数据转换为JSON字符串
            subject_data = json.dumps(subjects)
            
            # 生成用于嵌入的文本
            subject_text = " ".join([f"{s.get('subject')}: #{s.get('rank')}" for s in subjects])
            embedding = self.get_embeddings(subject_text)
            
            ids.append(f"subjects_{i}_{university_id}")
            university_ids.append(university_id)
            subject_data_list.append(subject_data)
            embeddings.append(embedding)
        
        # 执行插入
        collection.insert([
            ids,
            university_ids,
            subject_data_list,
            embeddings
        ])
        
        # 刷新集合确保数据被持久化
        collection.flush()
        
        print(f"成功导入 {len(ids)} 条学科排名数据")
    
    def import_university_indicators_data(self, collection):
        """
        导入大学全球指标数据
        """
        if not self.processed_data:
            print("没有可导入的数据")
            return
        
        # 准备数据
        ids = []
        university_ids = []
        indicator_data_list = []
        embeddings = []
        
        for i, university in enumerate(self.processed_data):
            university_id = university.get("university_id")
            indicators = university.get("global_indicators", {})
            
            # 将指标数据转换为JSON字符串
            indicator_data = json.dumps(indicators)
            
            # 生成用于嵌入的文本
            indicator_text = " ".join([f"{k}: {v}" for k, v in indicators.items()])
            embedding = self.get_embeddings(indicator_text)
            
            ids.append(f"indicators_{i}_{university_id}")
            university_ids.append(university_id)
            indicator_data_list.append(indicator_data)
            embeddings.append(embedding)
        
        # 执行插入
        collection.insert([
            ids,
            university_ids,
            indicator_data_list,
            embeddings
        ])
        
        # 刷新集合确保数据被持久化
        collection.flush()
        
        print(f"成功导入 {len(ids)} 条全球指标数据")
    
    def import_all_data(self):
        """
        导入所有数据到Milvus
        """
        if not self.processed_data:
            print("没有可导入的数据")
            return False
        
        try:
            # 创建集合
            base_collection = self.create_university_base_collection()
            summary_collection = self.create_university_summary_collection()
            statistics_collection = self.create_university_statistics_collection()
            subjects_collection = self.create_university_subjects_collection()
            indicators_collection = self.create_university_indicators_collection()
            
            # 导入数据
            self.import_university_base_data(base_collection)
            self.import_university_summary_data(summary_collection)
            self.import_university_statistics_data(statistics_collection)
            self.import_university_subjects_data(subjects_collection)
            self.import_university_indicators_data(indicators_collection)
            
            # 确保所有集合加载
            print("\n正在验证数据导入结果...")
            
            collections = {
                "usnews2025_university_base": base_collection,
                "usnews2025_university_summary": summary_collection,
                "usnews2025_university_statistics": statistics_collection,
                "usnews2025_university_subjects": subjects_collection,
                "usnews2025_university_indicators": indicators_collection
            }
            
            # 加载并验证所有集合
            for name, collection in collections.items():
                try:
                    # 重新加载集合
                    collection.load()
                    
                    # 获取记录数
                    count = collection.num_entities
                    print(f"集合 '{name}' 包含 {count} 条记录")
                    
                    # 如果记录数为0，可能有问题
                    if count == 0:
                        print(f"⚠️ 警告: 集合 '{name}' 中没有数据")
                except Exception as e:
                    print(f"⚠️ 验证集合 '{name}' 时出错: {str(e)}")
            
            print("\n所有数据导入完成")
            return True
        except Exception as e:
            print(f"导入数据时发生错误: {str(e)}")
            return False
    
    def test_vector_search(self, query_text="Harvard University", limit=3):
        """
        测试向量搜索功能
        """
        if not utility.has_collection("university_base"):
            print("基础信息集合不存在，请先导入数据")
            return
        
        try:
            # 加载集合
            collection = Collection("university_base")
            collection.load()
            
            # 生成查询向量
            query_vector = self.get_embeddings(query_text)
            
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
            
            # 显示结果
            print(f"\n向量搜索测试 - 查询: '{query_text}'")
            print("结果:")
            for i, hit in enumerate(results[0]):
                print(f"{i+1}. {hit.entity.get('name')} (排名: {hit.entity.get('display_rank')}, 相似度: {hit.distance:.4f})")
            
            return True
        except Exception as e:
            print(f"搜索测试失败: {str(e)}")
            return False


if __name__ == "__main__":
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 默认数据文件路径
    default_data_file = os.path.join(project_root, "DataProcessed", "USNews2025详情界面数据_processed.json")
    
    
    # 创建导入器并导入数据
    importer = MilvusImporter(processed_data_path=default_data_file)
    
    # 执行导入
    importer.import_all_data()
