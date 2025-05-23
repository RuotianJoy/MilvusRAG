#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
THE2025排名数据导入脚本
将处理好的THE2025排名数据导入到Milvus向量数据库
由于Milvus不支持一个集合中存储多个向量字段，将拆分为三个集合：
1. the2025_basic_info - 基本信息向量
2. the2025_subjects - 学科向量
3. the2025_metrics - 评分指标向量
"""

import os
import sys
import json
import time
from pymilvus import (
    connections,
    utility,
    Collection,
    FieldSchema, 
    CollectionSchema, 
    DataType
)
import configparser

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 配置文件路径
config_file = os.path.join(project_root, "Config", "Milvus.ini")
# 默认数据文件路径
default_data_file = os.path.join(project_root, "DataProcessed", "THE2025_processed.json")

# 定义全局变量
_COLLECTION_PREFIX = "the2025"  # 集合名称前缀
_PARTITION_NAME = "main"  # 分区名称

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
            host=host,
            port=port
        )
        print("连接成功")
        return True
    except Exception as e:
        print(f"连接失败: {str(e)}")
        return False

def create_basic_info_collection():
    """创建基本信息向量集合"""
    collection_name = f"{_COLLECTION_PREFIX}_basic_info"
    
    # 检查集合是否已存在，如果存在则删除
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"已删除现有集合: {collection_name}")
    
    # 定义集合字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="rank", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="overall_score", dtype=DataType.FLOAT),
        FieldSchema(name="teaching_score", dtype=DataType.FLOAT),
        FieldSchema(name="research_score", dtype=DataType.FLOAT),
        FieldSchema(name="citations_score", dtype=DataType.FLOAT),
        FieldSchema(name="industry_income_score", dtype=DataType.FLOAT),
        FieldSchema(name="international_outlook_score", dtype=DataType.FLOAT),
        FieldSchema(name="basic_info_vector", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    
    # 创建集合模式
    schema = CollectionSchema(fields=fields, description="THE2025大学排名基本信息向量")
    
    # 创建集合
    collection = Collection(name=collection_name, schema=schema)
    print(f"已创建集合 {collection_name}")
    
    # 创建分区
    collection.create_partition(_PARTITION_NAME)
    print(f"已创建分区 {_PARTITION_NAME}")
    
    # 创建索引
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index(field_name="basic_info_vector", index_params=index_params)
    print(f"已创建 {collection_name} 索引")
    
    return collection

def create_subjects_collection():
    """创建学科向量集合，添加更丰富的学科相关字段"""
    collection_name = f"{_COLLECTION_PREFIX}_subjects"
    
    # 检查集合是否已存在，如果存在则删除
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"已删除现有集合: {collection_name}")
    
    # 定义集合字段 - 添加更丰富的学科相关字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="rank", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
        # 添加更丰富的学科相关字段
        FieldSchema(name="subjects", dtype=DataType.VARCHAR, max_length=4096),  # 存储学科列表JSON
        FieldSchema(name="subjects_count", dtype=DataType.INT32),  # 学科数量
        FieldSchema(name="top_subjects", dtype=DataType.VARCHAR, max_length=512),  # 前几个主要学科
        FieldSchema(name="has_computer_science", dtype=DataType.BOOL),  # 是否有计算机科学
        FieldSchema(name="has_engineering", dtype=DataType.BOOL),  # 是否有工程类学科
        FieldSchema(name="has_medicine", dtype=DataType.BOOL),  # 是否有医学类学科
        FieldSchema(name="subjects_vector", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    
    # 创建集合模式
    schema = CollectionSchema(fields=fields, description="THE2025大学排名学科向量")
    
    # 创建集合
    collection = Collection(name=collection_name, schema=schema)
    print(f"已创建集合 {collection_name}")
    
    # 创建分区
    collection.create_partition(_PARTITION_NAME)
    print(f"已创建分区 {_PARTITION_NAME}")
    
    # 创建索引
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 400}  # 提高efConstruction值，增强搜索质量
    }
    collection.create_index(field_name="subjects_vector", index_params=index_params)
    print(f"已创建 {collection_name} 索引")
    
    return collection

def create_metrics_collection():
    """创建评分指标向量集合"""
    collection_name = f"{_COLLECTION_PREFIX}_metrics"
    
    # 检查集合是否已存在，如果存在则删除
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"已删除现有集合: {collection_name}")
    
    # 定义集合字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="rank", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="overall_score", dtype=DataType.FLOAT),
        FieldSchema(name="teaching_score", dtype=DataType.FLOAT),
        FieldSchema(name="research_score", dtype=DataType.FLOAT),
        FieldSchema(name="citations_score", dtype=DataType.FLOAT),
        FieldSchema(name="industry_income_score", dtype=DataType.FLOAT),
        FieldSchema(name="international_outlook_score", dtype=DataType.FLOAT),
        FieldSchema(name="student_staff_ratio", dtype=DataType.FLOAT),
        FieldSchema(name="pc_intl_students", dtype=DataType.FLOAT),
        FieldSchema(name="number_students", dtype=DataType.INT32),  # 添加学生数量字段
        FieldSchema(name="metrics_vector", dtype=DataType.FLOAT_VECTOR, dim=10)
    ]
    
    # 创建集合模式
    schema = CollectionSchema(fields=fields, description="THE2025大学排名评分指标向量")
    
    # 创建集合
    collection = Collection(name=collection_name, schema=schema)
    print(f"已创建集合 {collection_name}")
    
    # 创建分区
    collection.create_partition(_PARTITION_NAME)
    print(f"已创建分区 {_PARTITION_NAME}")
    
    # 创建索引 - 优化索引参数
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 400}  # 提高efConstruction值
    }
    collection.create_index(field_name="metrics_vector", index_params=index_params)
    print(f"已创建 {collection_name} 索引")
    
    return collection

def create_meta_collection():
    """创建元数据集合（添加虚拟向量字段以满足Milvus的要求）"""
    collection_name = f"{_COLLECTION_PREFIX}_meta"
    
    # 检查集合是否已存在，如果存在则删除
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"已删除现有集合: {collection_name}")
    
    # 定义集合字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="rank", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="overall_score", dtype=DataType.FLOAT),
        FieldSchema(name="teaching_score", dtype=DataType.FLOAT),
        FieldSchema(name="research_score", dtype=DataType.FLOAT),
        FieldSchema(name="citations_score", dtype=DataType.FLOAT),
        FieldSchema(name="industry_income_score", dtype=DataType.FLOAT),
        FieldSchema(name="international_outlook_score", dtype=DataType.FLOAT),
        FieldSchema(name="student_staff_ratio", dtype=DataType.FLOAT),
        FieldSchema(name="pc_intl_students", dtype=DataType.FLOAT),
        FieldSchema(name="json_data", dtype=DataType.VARCHAR, max_length=65535),
        # 添加一个虚拟向量字段，以满足Milvus的要求
        FieldSchema(name="dummy_vector", dtype=DataType.FLOAT_VECTOR, dim=2)
    ]
    
    # 创建集合模式
    schema = CollectionSchema(fields=fields, description="THE2025大学排名元数据")
    
    # 创建集合
    collection = Collection(name=collection_name, schema=schema)
    print(f"已创建集合 {collection_name}")
    
    # 创建分区
    collection.create_partition(_PARTITION_NAME)
    print(f"已创建分区 {_PARTITION_NAME}")
    
    # 为虚拟向量创建索引
    index_params = {
        "metric_type": "L2",
        "index_type": "FLAT",
        "params": {}
    }
    collection.create_index(field_name="dummy_vector", index_params=index_params)
    print(f"已创建 {collection_name} 虚拟向量索引")
    
    return collection

def import_basic_info_data(collection, data):
    """导入基本信息向量数据"""
    if not collection:
        return False
    
    # 添加重试机制
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"正在导入基本信息向量数据，共 {len(data)} 条记录")
            
            # 准备待插入的数据
            ids = []
            names = []
            ranks = []
            locations = []
            overall_scores = []
            teaching_scores = []
            research_scores = []
            citations_scores = []
            industry_income_scores = []
            international_outlook_scores = []
            basic_info_vectors = []
            
            # 处理每条记录
            batch_size = 1000  # 每批处理的数量
            total_batches = (len(data) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(data))
                batch_data = data[start_idx:end_idx]
                
                # 清空批处理列表
                ids.clear()
                names.clear()
                ranks.clear()
                locations.clear()
                overall_scores.clear()
                teaching_scores.clear()
                research_scores.clear()
                citations_scores.clear()
                industry_income_scores.clear()
                international_outlook_scores.clear()
                basic_info_vectors.clear()
                
                # 填充数据
                for item in batch_data:
                    ids.append(int(item.get("id", 0)))
                    names.append(item.get("name", "Unknown"))
                    ranks.append(item.get("rank", "未知"))
                    locations.append(item.get("location", "Unknown"))
                    overall_scores.append(float(item.get("overall_score", 0)))
                    teaching_scores.append(float(item.get("teaching_score", 0)))
                    research_scores.append(float(item.get("research_score", 0)))
                    citations_scores.append(float(item.get("citations_score", 0)))
                    industry_income_scores.append(float(item.get("industry_income_score", 0)))
                    international_outlook_scores.append(float(item.get("international_outlook_score", 0)))
                    basic_info_vectors.append(item.get("basic_info_vector", [0] * 768))
                
                # 执行批量插入
                insert_data = [
                    ids,
                    names,
                    ranks,
                    locations,
                    overall_scores,
                    teaching_scores,
                    research_scores,
                    citations_scores,
                    industry_income_scores,
                    international_outlook_scores,
                    basic_info_vectors
                ]
                
                collection.insert(insert_data, partition_name=_PARTITION_NAME)
                print(f"已导入基本信息向量批次 {batch_idx + 1}/{total_batches}，记录 {start_idx + 1} - {end_idx}")
            
            # 刷新集合
            collection.flush()
            print(f"基本信息向量导入完成，共插入 {collection.num_entities} 条记录")
            
            return True
        except Exception as e:
            retry_count += 1
            print(f"导入基本信息向量数据失败，尝试重试 ({retry_count}/{max_retries}): {e}")
            import traceback
            traceback.print_exc()
            import time
            time.sleep(2)  # 延时后重试
    
    print("导入基本信息向量数据失败，已达到最大重试次数")
    return False

def import_metrics_data(collection, data):
    """导入评分指标向量数据"""
    if not collection:
        return False
    
    # 添加重试机制
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"正在导入评分指标向量数据，共 {len(data)} 条记录")
            
            # 准备待插入的数据
            ids = []
            names = []
            ranks = []
            locations = []
            overall_scores = []
            teaching_scores = []  # 添加教学评分
            research_scores = []  # 添加研究评分
            citations_scores = [] # 添加引用评分
            industry_income_scores = [] # 添加产业收入评分
            international_outlook_scores = [] # 添加国际化评分
            student_staff_ratios = []
            pc_intl_students = []
            number_students = []  # 添加学生数量
            metrics_vectors = []
            
            # 处理每条记录
            batch_size = 1000  # 每批处理的数量
            total_batches = (len(data) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(data))
                batch_data = data[start_idx:end_idx]
                
                # 清空批处理列表
                ids.clear()
                names.clear()
                ranks.clear()
                locations.clear()
                overall_scores.clear()
                teaching_scores.clear()
                research_scores.clear()
                citations_scores.clear()
                industry_income_scores.clear()
                international_outlook_scores.clear()
                student_staff_ratios.clear()
                pc_intl_students.clear()
                number_students.clear()
                metrics_vectors.clear()
                
                # 填充数据
                for item in batch_data:
                    ids.append(int(item.get("id", 0)))
                    names.append(item.get("name", "Unknown"))
                    ranks.append(item.get("rank", "未知"))
                    locations.append(item.get("location", "Unknown"))
                    overall_scores.append(float(item.get("overall_score", 0)))
                    teaching_scores.append(float(item.get("teaching_score", 0)))
                    research_scores.append(float(item.get("research_score", 0)))
                    citations_scores.append(float(item.get("citations_score", 0)))
                    industry_income_scores.append(float(item.get("industry_income_score", 0)))
                    international_outlook_scores.append(float(item.get("international_outlook_score", 0)))
                    student_staff_ratios.append(float(item.get("student_staff_ratio", 0)))
                    pc_intl_students.append(float(item.get("pc_intl_students", 0)))
                    # 处理学生数量，可能是字符串格式
                    try:
                        num_students = item.get("number_students", 0)
                        if isinstance(num_students, str):
                            # 移除非数字字符
                            import re
                            num_students = re.sub(r'[^\d]', '', num_students)
                            if num_students:
                                num_students = int(num_students)
                            else:
                                num_students = 0
                        number_students.append(int(num_students))
                    except:
                        number_students.append(0)
                    metrics_vectors.append(item.get("metrics_vector", [0] * 10))
                
                # 执行批量插入
                insert_data = [
                    ids,
                    names,
                    ranks,
                    locations,
                    overall_scores,
                    teaching_scores,
                    research_scores,
                    citations_scores,
                    industry_income_scores,
                    international_outlook_scores,
                    student_staff_ratios,
                    pc_intl_students,
                    number_students,
                    metrics_vectors
                ]
                
                collection.insert(insert_data, partition_name=_PARTITION_NAME)
                print(f"已导入评分指标向量批次 {batch_idx + 1}/{total_batches}，记录 {start_idx + 1} - {end_idx}")
            
            # 刷新集合
            collection.flush()
            print(f"评分指标向量导入完成，共插入 {collection.num_entities} 条记录")
            
            return True
        except Exception as e:
            retry_count += 1
            print(f"导入评分指标向量数据失败，尝试重试 ({retry_count}/{max_retries}): {e}")
            import traceback
            traceback.print_exc()
            import time
            time.sleep(2)  # 延时后重试
    
    print("导入评分指标向量数据失败，已达到最大重试次数")
    return False

def import_meta_data(collection, data):
    """导入元数据"""
    if not collection:
        return False
    
    # 添加重试机制
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"正在导入元数据，共 {len(data)} 条记录")
            
            # 准备待插入的数据
            ids = []
            names = []
            ranks = []
            locations = []
            overall_scores = []
            teaching_scores = []
            research_scores = []
            citations_scores = []
            industry_income_scores = []
            international_outlook_scores = []
            student_staff_ratios = []
            pc_intl_students = []
            json_data_list = []
            dummy_vectors = []  # 虚拟向量字段
            
            # 处理每条记录
            batch_size = 1000  # 每批处理的数量
            total_batches = (len(data) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(data))
                batch_data = data[start_idx:end_idx]
                
                # 清空批处理列表
                ids.clear()
                names.clear()
                ranks.clear()
                locations.clear()
                overall_scores.clear()
                teaching_scores.clear()
                research_scores.clear()
                citations_scores.clear()
                industry_income_scores.clear()
                international_outlook_scores.clear()
                student_staff_ratios.clear()
                pc_intl_students.clear()
                json_data_list.clear()
                dummy_vectors.clear()  # 清空虚拟向量列表
                
                # 填充数据
                for item in batch_data:
                    ids.append(int(item.get("id", 0)))
                    names.append(item.get("name", "Unknown"))
                    ranks.append(item.get("rank", "未知"))
                    locations.append(item.get("location", "Unknown"))
                    overall_scores.append(float(item.get("overall_score", 0)))
                    teaching_scores.append(float(item.get("teaching_score", 0)))
                    research_scores.append(float(item.get("research_score", 0)))
                    citations_scores.append(float(item.get("citations_score", 0)))
                    industry_income_scores.append(float(item.get("industry_income_score", 0)))
                    international_outlook_scores.append(float(item.get("international_outlook_score", 0)))
                    student_staff_ratios.append(float(item.get("student_staff_ratio", 0)))
                    pc_intl_students.append(float(item.get("pc_intl_students", 0)))
                    
                    # 处理JSON数据，确保它是有效的JSON字符串
                    try:
                        import json
                        # 如果json_data不存在或不是有效字符串，创建一个新的JSON对象
                        json_data = item.get("json_data", "{}")
                        if not isinstance(json_data, str):
                            # 尝试将对象转为JSON字符串
                            try:
                                json_data = json.dumps(json_data, ensure_ascii=False)
                            except:
                                json_data = "{}"
                        
                        # 验证JSON的有效性
                        try:
                            json.loads(json_data)
                        except:
                            json_data = "{}"
                            
                        json_data_list.append(json_data)
                    except:
                        json_data_list.append("{}")
                        
                    dummy_vectors.append([0.0, 0.0])  # 添加虚拟向量[0.0, 0.0]
                
                # 执行批量插入
                insert_data = [
                    ids,
                    names,
                    ranks,
                    locations,
                    overall_scores,
                    teaching_scores,
                    research_scores,
                    citations_scores,
                    industry_income_scores,
                    international_outlook_scores,
                    student_staff_ratios,
                    pc_intl_students,
                    json_data_list,
                    dummy_vectors  # 添加虚拟向量数据
                ]
                
                collection.insert(insert_data, partition_name=_PARTITION_NAME)
                print(f"已导入元数据批次 {batch_idx + 1}/{total_batches}，记录 {start_idx + 1} - {end_idx}")
            
            # 刷新集合
            collection.flush()
            print(f"元数据导入完成，共插入 {collection.num_entities} 条记录")
            
            return True
        except Exception as e:
            retry_count += 1
            print(f"导入元数据失败，尝试重试 ({retry_count}/{max_retries}): {e}")
            import traceback
            traceback.print_exc()
            import time
            time.sleep(2)  # 延时后重试
    
    print("导入元数据失败，已达到最大重试次数")
    return False

def import_subjects_data(collection, data):
    """导入学科向量数据，包含更丰富的学科信息"""
    if not collection:
        return False
    
    # 添加重试机制
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"正在导入学科向量数据，共 {len(data)} 条记录")
            
            # 准备待插入的数据
            ids = []
            names = []
            ranks = []
            locations = []
            subjects_list = []
            subjects_counts = []
            top_subjects_list = []
            has_computer_science_list = []
            has_engineering_list = []
            has_medicine_list = []
            subjects_vectors = []
            
            # 处理每条记录
            batch_size = 1000  # 每批处理的数量
            total_batches = (len(data) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(data))
                batch_data = data[start_idx:end_idx]
                
                # 清空批处理列表
                ids.clear()
                names.clear()
                ranks.clear()
                locations.clear()
                subjects_list.clear()
                subjects_counts.clear()
                top_subjects_list.clear()
                has_computer_science_list.clear()
                has_engineering_list.clear()
                has_medicine_list.clear()
                subjects_vectors.clear()
                
                # 填充数据
                for item in batch_data:
                    ids.append(int(item.get("id", 0)))
                    names.append(item.get("name", "Unknown"))
                    ranks.append(item.get("rank", "未知"))
                    locations.append(item.get("location", "Unknown"))
                    
                    # 处理学科数据
                    subjects = item.get("subjects", [])
                    
                    # 转换为JSON字符串
                    import json
                    subjects_json = json.dumps(subjects, ensure_ascii=False)
                    subjects_list.append(subjects_json)
                    
                    # 统计学科数量
                    subjects_count = len(subjects) if isinstance(subjects, list) else 0
                    subjects_counts.append(subjects_count)
                    
                    # 获取前3个主要学科
                    if isinstance(subjects, list) and subjects:
                        top_3_subjects = ", ".join(subjects[:3])
                        if len(subjects) > 3:
                            top_3_subjects += f" 等{len(subjects)}个"
                    else:
                        top_3_subjects = "无学科信息"
                    top_subjects_list.append(top_3_subjects)
                    
                    # 检查是否包含特定学科
                    has_cs = False
                    has_eng = False
                    has_med = False
                    
                    if isinstance(subjects, list):
                        # 计算机科学相关关键词
                        cs_keywords = ["计算机", "computer", "computing", "信息技术", "软件", "人工智能"]
                        # 工程学相关关键词
                        eng_keywords = ["工程", "engineering", "civil", "机械", "电子", "电气", "材料"]
                        # 医学相关关键词
                        med_keywords = ["医学", "medicine", "医疗", "临床", "健康", "药学", "生物医学"]
                        
                        for subject in subjects:
                            subject_lower = subject.lower() if isinstance(subject, str) else ""
                            # 检查计算机科学
                            if any(keyword in subject_lower for keyword in cs_keywords):
                                has_cs = True
                            # 检查工程学
                            if any(keyword in subject_lower for keyword in eng_keywords):
                                has_eng = True
                            # 检查医学
                            if any(keyword in subject_lower for keyword in med_keywords):
                                has_med = True
                    
                    has_computer_science_list.append(has_cs)
                    has_engineering_list.append(has_eng)
                    has_medicine_list.append(has_med)
                    
                    # 获取学科向量
                    subjects_vectors.append(item.get("subjects_vector", [0] * 768))
                
                # 执行批量插入
                insert_data = [
                    ids,
                    names,
                    ranks,
                    locations,
                    subjects_list,
                    subjects_counts,
                    top_subjects_list,
                    has_computer_science_list,
                    has_engineering_list,
                    has_medicine_list,
                    subjects_vectors
                ]
                
                collection.insert(insert_data, partition_name=_PARTITION_NAME)
                print(f"已导入学科向量批次 {batch_idx + 1}/{total_batches}，记录 {start_idx + 1} - {end_idx}")
            
            # 刷新集合
            collection.flush()
            print(f"学科向量导入完成，共插入 {collection.num_entities} 条记录")
            
            return True
        except Exception as e:
            retry_count += 1
            print(f"导入学科向量数据失败，尝试重试 ({retry_count}/{max_retries}): {e}")
            import traceback
            traceback.print_exc()
            import time
            time.sleep(2)  # 延时后重试
    
    print("导入学科向量数据失败，已达到最大重试次数")
    return False

def test_collections():
    """测试所有集合是否正常工作"""
    print("\n开始测试集合...")

    try:
        # 测试基本信息向量集合
        basic_info_collection = Collection(f"{_COLLECTION_PREFIX}_basic_info")
        basic_info_collection.load()
        
        results = basic_info_collection.query(
            expr="id >= 0",
            output_fields=["name", "rank", "location", "overall_score"],
            limit=3
        )
        print(f"\n基本信息向量集合测试: 找到 {len(results)} 条记录")
        for i, r in enumerate(results):
            print(f"{i+1}. {r['name']} - {r['rank']}, {r['location']}, 总分: {r['overall_score']}")
        
        # 测试学科向量集合
        subjects_collection = Collection(f"{_COLLECTION_PREFIX}_subjects")
        subjects_collection.load()
        
        results = subjects_collection.query(
            expr="id >= 0",
            output_fields=["name", "rank", "location"],
            limit=3
        )
        print(f"\n学科向量集合测试: 找到 {len(results)} 条记录")
        for i, r in enumerate(results):
            print(f"{i+1}. {r['name']} - {r['rank']}, {r['location']}")
        
        # 测试评分指标向量集合
        metrics_collection = Collection(f"{_COLLECTION_PREFIX}_metrics")
        metrics_collection.load()
        
        results = metrics_collection.query(
            expr="id >= 0",
            output_fields=["name", "rank", "location", "overall_score"],
            limit=3
        )
        print(f"\n评分指标向量集合测试: 找到 {len(results)} 条记录")
        for i, r in enumerate(results):
            print(f"{i+1}. {r['name']} - {r['rank']}, {r['location']}, 总分: {r['overall_score']}")
        
        # 测试元数据集合
        meta_collection = Collection(f"{_COLLECTION_PREFIX}_meta")
        meta_collection.load()
        
        results = meta_collection.query(
            expr="id >= 0",
            output_fields=["name", "rank", "location", "overall_score"],
            limit=3
        )
        print(f"\n元数据集合测试: 找到 {len(results)} 条记录")
        for i, r in enumerate(results):
            print(f"{i+1}. {r['name']} - {r['rank']}, {r['location']}, 总分: {r['overall_score']}")
        
        # 释放所有集合
        basic_info_collection.release()
        subjects_collection.release()
        metrics_collection.release()
        meta_collection.release()
        
        print("\n所有集合测试完成")
        return True
        
    except Exception as e:
        print(f"测试集合时出错: {e}")
        return False

def main():
    """主函数，协调整个导入过程"""
    start_time = time.time()
    
    # 连接到Milvus
    if not connect_milvus():
        print("无法连接到Milvus，退出程序")
        return
    
    try:
        # 加载处理好的数据
        print(f"加载处理后的数据: {default_data_file}")
        with open(default_data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"已加载数据，共 {len(data)} 条记录")
        
        # 创建所有集合
        print("\n创建集合...")
        basic_info_collection = create_basic_info_collection()
        subjects_collection = create_subjects_collection()
        metrics_collection = create_metrics_collection()
        meta_collection = create_meta_collection()
        
        # 导入数据到各个集合
        all_success = True
        
        if not import_basic_info_data(basic_info_collection, data):
            print("基本信息向量数据导入失败")
            all_success = False
        
        if not import_subjects_data(subjects_collection, data):
            print("学科向量数据导入失败")
            all_success = False
        
        if not import_metrics_data(metrics_collection, data):
            print("评分指标向量数据导入失败")
            all_success = False
        
        if not import_meta_data(meta_collection, data):
            print("元数据导入失败")
            all_success = False
        
        # 测试集合
        if all_success:
            print("\n所有数据导入成功")
            if test_collections():
                print("\n集合测试成功")
            else:
                print("\n集合测试失败")
        else:
            print("\n部分数据导入失败")
    
    except Exception as e:
        print(f"执行过程中出错: {e}")
    
    finally:
        # 断开连接
        try:
            connections.disconnect("default")
            print("\n已断开Milvus连接")
        except:
            pass
    
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main() 
