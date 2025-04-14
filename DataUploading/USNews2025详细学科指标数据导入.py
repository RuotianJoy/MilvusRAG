#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import uuid
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer

class USNews2025SubjectDataImporter:
    def __init__(self, host="localhost", port="19530", processed_data_path=None):
        """
        初始化Milvus导入器
        
        Args:
            host: Milvus服务器主机
            port: Milvus服务器端口
            processed_data_path: 处理后的数据文件路径
        """
        self.host = host
        self.port = port
        
        # 连接Milvus服务
        self.connect_to_milvus()
        
        # 设置向量维度
        self.indicator_vector_dim = 15  # 指标向量维度，可能需要根据实际数据调整
        self.text_vector_dim = 384  # BERT向量维度
        
        # 初始化文本嵌入模型
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 加载处理后的数据
        self.processed_data_path = processed_data_path
        self.processed_data = self.load_processed_data()
    
    def connect_to_milvus(self):
        """
        连接到Milvus服务
        """
        try:
            connections.connect("default", host=self.host, port=self.port)
            print(f"已成功连接到Milvus服务器: {self.host}:{self.port}")
        except Exception as e:
            print(f"连接Milvus服务器失败: {str(e)}")
            raise
    
    def load_processed_data(self):
        """
        加载处理后的数据
        
        Returns:
            处理后的数据
        """
        if not self.processed_data_path or not os.path.exists(self.processed_data_path):
            print("未指定处理后的数据文件路径或文件不存在")
            return None
        
        try:
            with open(self.processed_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查数据结构
            if not all(key in data for key in ["schools", "subjects", "relations"]):
                print("数据格式不正确，缺少必要的字段")
                return None
                
            print(f"成功加载处理后的数据")
            print(f"学校数量: {len(data['schools'])}")
            print(f"学科数量: {len(data['subjects'])}")
            print(f"关系数量: {len(data['relations'])}")
            return data
        except Exception as e:
            print(f"加载数据文件失败: {str(e)}")
            return None
    
    def get_text_embedding(self, text):
        """
        获取文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            文本的嵌入向量
        """
        if not text:
            # 如果文本为空，返回全零向量
            return np.zeros(self.text_vector_dim).tolist()
        
        try:
            # 使用SentenceTransformer获取嵌入
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"获取嵌入向量失败: {str(e)}")
            # 返回全零向量作为备用
            return np.zeros(self.text_vector_dim).tolist()
    
    def create_schools_collection(self):
        """
        创建学校集合
        
        Returns:
            学校集合对象
        """
        collection_name = "usnews2025_schools"
        
        # 检查集合是否已存在
        if utility.has_collection(collection_name):
            print(f"集合 {collection_name} 已存在，将删除并重新创建")
            utility.drop_collection(collection_name)
        
        # 定义集合字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="school_name", dtype=DataType.VARCHAR, max_length=255),
            # 添加向量字段，Milvus要求每个集合至少有一个向量字段
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.text_vector_dim)
        ]
        
        # 创建集合模式和集合
        schema = CollectionSchema(fields, "USNews2025学校基本信息表")
        collection = Collection(collection_name, schema)
        
        # 为向量字段创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index("embedding", index_params)
        
        print(f"成功创建集合: {collection_name}")
        return collection
    
    def create_subjects_collection(self):
        """
        创建学科集合
        
        Returns:
            学科集合对象
        """
        collection_name = "usnews2025_subjects"
        
        # 检查集合是否已存在
        if utility.has_collection(collection_name):
            print(f"集合 {collection_name} 已存在，将删除并重新创建")
            utility.drop_collection(collection_name)
        
        # 定义集合字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="subject_name", dtype=DataType.VARCHAR, max_length=255),
            # 添加向量字段，Milvus要求每个集合至少有一个向量字段
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.text_vector_dim)
        ]
        
        # 创建集合模式和集合
        schema = CollectionSchema(fields, "USNews2025学科基本信息表")
        collection = Collection(collection_name, schema)
        
        # 为向量字段创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index("embedding", index_params)
        
        print(f"成功创建集合: {collection_name}")
        return collection
    
    def create_relations_collection(self, max_indicator_dim=15):
        """
        创建学校-学科关系集合，使用指标向量
        
        Args:
            max_indicator_dim: 指标向量的最大维度
            
        Returns:
            关系集合对象
        """
        collection_name = "usnews2025_school_subject_relations"
        
        # 检查集合是否已存在
        if utility.has_collection(collection_name):
            print(f"集合 {collection_name} 已存在，将删除并重新创建")
            utility.drop_collection(collection_name)
        
        # 定义集合字段 - 只保留一个向量字段 (indicator_vector)
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="school_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="subject_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="overall_score", dtype=DataType.FLOAT),
            FieldSchema(name="indicator_vector", dtype=DataType.FLOAT_VECTOR, dim=max_indicator_dim),
            FieldSchema(name="raw_data", dtype=DataType.JSON)
        ]
        
        # 创建集合模式和集合
        schema = CollectionSchema(fields, "USNews2025学校-学科关系表")
        collection = Collection(collection_name, schema)
        
        # 为向量字段创建索引
        index_params_l2 = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        
        # 创建向量索引
        collection.create_index("indicator_vector", index_params_l2)
        
        # 注释掉标量字段索引创建，或根据需要启用，但需要调整索引类型
        # 关键是Milvus的版本可能不支持对VARCHAR字段创建FLAT索引
        # 大多数情况下，主键会自动建立索引，不需要显式创建
        
        print(f"成功创建关系集合: {collection_name}")
        return collection
    
    def create_relations_text_collection(self):
        """
        创建学校-学科关系文本集合，使用文本向量
        
        Returns:
            关系文本集合对象
        """
        collection_name = "usnews2025_school_subject_relations_text"
        
        # 检查集合是否已存在
        if utility.has_collection(collection_name):
            print(f"集合 {collection_name} 已存在，将删除并重新创建")
            utility.drop_collection(collection_name)
        
        # 定义集合字段 - 只保留一个向量字段 (text_embedding)
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="relation_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="school_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="subject_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.text_vector_dim)
        ]
        
        # 创建集合模式和集合
        schema = CollectionSchema(fields, "USNews2025学校-学科关系文本表")
        collection = Collection(collection_name, schema)
        
        # 为向量字段创建索引
        index_params_cosine = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        
        # 创建向量索引
        collection.create_index("text_embedding", index_params_cosine)
        
        # 注释掉标量字段索引创建，主键会自动建立索引
        
        print(f"成功创建关系文本集合: {collection_name}")
        return collection
    
    def import_schools_data(self, collection):
        """
        导入学校数据
        
        Args:
            collection: 学校集合对象
            
        Returns:
            是否成功
        """
        if not self.processed_data or "schools" not in self.processed_data:
            print("没有可导入的学校数据")
            return False
        
        schools = self.processed_data["schools"]
        if not schools:
            print("学校数据为空")
            return False
        
        # 准备批量插入数据
        ids = []
        school_names = []
        embeddings = []
        
        for school in schools:
            ids.append(school["school_id"])
            school_name = school["school_name"]
            school_names.append(school_name)
            
            # 为学校生成向量表示，使用学校名称
            embedding = self.get_text_embedding(school_name)
            embeddings.append(embedding)
        
        insert_data = [
            ids,
            school_names,
            embeddings
        ]
        
        try:
            collection.insert(insert_data)
            print(f"成功导入 {len(ids)} 条学校数据")
            return True
        except Exception as e:
            print(f"导入学校数据失败: {str(e)}")
            return False
    
    def import_subjects_data(self, collection):
        """
        导入学科数据
        
        Args:
            collection: 学科集合对象
            
        Returns:
            是否成功
        """
        if not self.processed_data or "subjects" not in self.processed_data:
            print("没有可导入的学科数据")
            return False
        
        subjects = self.processed_data["subjects"]
        if not subjects:
            print("学科数据为空")
            return False
        
        # 准备批量插入数据
        ids = []
        subject_names = []
        embeddings = []
        
        for subject in subjects:
            ids.append(subject["subject_id"])
            subject_name = subject["subject_name"]
            subject_names.append(subject_name)
            
            # 为学科生成向量表示，使用学科名称
            embedding = self.get_text_embedding(subject_name)
            embeddings.append(embedding)
        
        insert_data = [
            ids,
            subject_names,
            embeddings
        ]
        
        try:
            collection.insert(insert_data)
            print(f"成功导入 {len(ids)} 条学科数据")
            return True
        except Exception as e:
            print(f"导入学科数据失败: {str(e)}")
            return False
    
    def import_relations_data(self, collection, max_indicator_dim=15):
        """
        导入学校-学科关系数据 (指标向量部分)
        
        Args:
            collection: 关系集合对象
            max_indicator_dim: 指标向量的最大维度
            
        Returns:
            是否成功
        """
        if not self.processed_data or "relations" not in self.processed_data:
            print("没有可导入的关系数据")
            return False
        
        relations = self.processed_data["relations"]
        if not relations:
            print("关系数据为空")
            return False
        
        # 准备批量插入数据
        ids = []
        school_ids = []
        subject_ids = []
        overall_scores = []
        indicator_vectors = []
        raw_data_list = []
        
        for relation in relations:
            # 对齐指标向量维度
            indicator_vector = relation["indicator_vector"]
            # 如果维度不足，填充-1
            if len(indicator_vector) < max_indicator_dim:
                indicator_vector = indicator_vector + [-1] * (max_indicator_dim - len(indicator_vector))
            # 如果维度超出，截断
            elif len(indicator_vector) > max_indicator_dim:
                indicator_vector = indicator_vector[:max_indicator_dim]
            
            ids.append(relation["relation_id"])
            school_ids.append(relation["school_id"])
            subject_ids.append(relation["subject_id"])
            overall_scores.append(relation["overall_score"])
            indicator_vectors.append(indicator_vector)
            raw_data_list.append(relation["raw_data"])
        
        insert_data = [
            ids,
            school_ids,
            subject_ids,
            overall_scores,
            indicator_vectors,
            raw_data_list
        ]
        
        try:
            collection.insert(insert_data)
            print(f"成功导入 {len(ids)} 条关系数据（指标向量部分）")
            return True
        except Exception as e:
            print(f"导入关系数据失败: {str(e)}")
            return False
    
    def import_relations_text_data(self, collection):
        """
        导入学校-学科关系数据 (文本向量部分)
        
        Args:
            collection: 关系文本集合对象
            
        Returns:
            是否成功
        """
        if not self.processed_data or "relations" not in self.processed_data:
            print("没有可导入的关系文本数据")
            return False
        
        relations = self.processed_data["relations"]
        if not relations:
            print("关系数据为空")
            return False
        
        # 准备批量插入数据
        ids = []  # 新的ID
        relation_ids = []  # 与主关系表的关联ID
        school_ids = []
        subject_ids = []
        text_embeddings = []
        
        for relation in relations:
            # 创建一个新的ID
            new_id = str(uuid.uuid4())
            
            ids.append(new_id)
            relation_ids.append(relation["relation_id"])
            school_ids.append(relation["school_id"])
            subject_ids.append(relation["subject_id"])
            text_embeddings.append(relation["text_embedding"])
        
        insert_data = [
            ids,
            relation_ids,
            school_ids,
            subject_ids,
            text_embeddings
        ]
        
        try:
            collection.insert(insert_data)
            print(f"成功导入 {len(ids)} 条关系数据（文本向量部分）")
            return True
        except Exception as e:
            print(f"导入关系文本数据失败: {str(e)}")
            return False
    
    def determine_max_indicator_dim(self):
        """
        确定指标向量的最大维度
        
        Returns:
            最大维度
        """
        if not self.processed_data or "relations" not in self.processed_data:
            return self.indicator_vector_dim
        
        max_dim = 0
        for relation in self.processed_data["relations"]:
            vector_len = len(relation.get("indicator_vector", []))
            max_dim = max(max_dim, vector_len)
        
        print(f"指标向量最大维度: {max_dim}")
        return max_dim
    
    def import_all_data(self):
        """
        导入所有数据
        
        Returns:
            是否成功
        """
        if not self.processed_data:
            print("没有可导入的数据")
            return False
        
        # 确定指标向量的最大维度
        max_indicator_dim = self.determine_max_indicator_dim()
        
        # 创建并导入学校数据
        print("\n===== 步骤1: 导入学校数据 =====")
        schools_collection = self.create_schools_collection()
        schools_success = self.import_schools_data(schools_collection)
        
        # 创建并导入学科数据
        print("\n===== 步骤2: 导入学科数据 =====")
        subjects_collection = self.create_subjects_collection()
        subjects_success = self.import_subjects_data(subjects_collection)
        
        # 创建并导入关系数据 (拆分为两个集合)
        print("\n===== 步骤3: 导入关系数据 (指标向量部分) =====")
        relations_collection = self.create_relations_collection(max_indicator_dim)
        relations_success = self.import_relations_data(relations_collection, max_indicator_dim)
        
        print("\n===== 步骤4: 导入关系数据 (文本向量部分) =====")
        relations_text_collection = self.create_relations_text_collection()
        relations_text_success = self.import_relations_text_data(relations_text_collection)
        
        # 最终判断是否全部成功
        if schools_success and subjects_success and relations_success and relations_text_success:
            print("\n所有数据导入成功")
            return True
        else:
            print("\n部分数据导入失败")
            return False


# 如果作为主程序运行
if __name__ == "__main__":
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理后的数据文件路径
    processed_data_path = os.path.join(current_dir, "USNews2025详细学科指标数据_processed.json")
    
    # 检查处理后的数据文件是否存在
    if not os.path.exists(processed_data_path):
        print(f"处理后的数据文件不存在: {processed_data_path}")
    else:
        importer = USNews2025SubjectDataImporter(
            host="localhost", 
            port="19530", 
            processed_data_path=processed_data_path
        )
        importer.import_all_data() 
