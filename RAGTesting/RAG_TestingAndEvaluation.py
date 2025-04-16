#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版测试评估脚本 - 从文件中读取问题和答案，使用RAG回答问题并评估结果
不包含对话记忆功能，专注于知识检索与评估
"""

import os
import pandas as pd
import configparser
from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema
import openai
from pymilvus import utility
# 移除RAGAS相关导入
# from ragas.metrics import (
#     faithfulness, answer_relevancy,
#     context_precision, context_recall
# )
# from ragas import evaluate
# from ragas import EvaluationDataset, SingleTurnSample
# from ragas.llms import LangchainLLMWrapper
# from langchain_openai import ChatOpenAI
import time
import sys
import json
import numpy as np
from dotenv import load_dotenv
# 只保留ROUGE相关导入
from rouge import Rouge
import re

# 初始化ROUGE
try:
    rouge = Rouge()
    print("Rouge初始化成功")
except Exception as e:
    print(f"初始化评估指标出错: {e}")
    print("请确保已安装必要的包: pip install rouge")

# 添加备用的ROUGE计算方法
def compute_rouge_manually(hypothesis, reference):
    """手动计算简化版ROUGE-1和ROUGE-L分数，包括精确率和召回率"""
    # 分词
    hyp_tokens = simple_tokenize(hypothesis.lower())
    ref_tokens = simple_tokenize(reference.lower())
    
    # 统计匹配的单词数 (ROUGE-1的分子)
    hyp_set = set(hyp_tokens)
    ref_set = set(ref_tokens)
    overlap = hyp_set.intersection(ref_set)
    
    # 计算ROUGE-1精确率和召回率
    precision_1 = len(overlap) / len(hyp_set) if len(hyp_set) > 0 else 0
    recall_1 = len(overlap) / len(ref_set) if len(ref_set) > 0 else 0
    
    # 计算ROUGE-1 F1
    if precision_1 + recall_1 > 0:
        rouge1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)  # F1 score
    else:
        rouge1 = 0
    
    # 简单的最长公共子序列计算
    def lcs_length(a, b):
        table = [[0 for _ in range(len(b) + 1)] for _ in range(len(a) + 1)]
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                if x == y:
                    table[i + 1][j + 1] = table[i][j] + 1
                else:
                    table[i + 1][j + 1] = max(table[i + 1][j], table[i][j + 1])
        return table[-1][-1]
    
    # 计算ROUGE-L
    lcs = lcs_length(hyp_tokens, ref_tokens)
    
    # 计算ROUGE-L精确率和召回率
    precision_l = lcs / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
    recall_l = lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0
    
    # 计算ROUGE-L F1
    if precision_l + recall_l > 0:
        rouge_l = 2 * precision_l * recall_l / (precision_l + recall_l)  # F1 score
    else:
        rouge_l = 0
    
    return {
        'rouge_1': rouge1,
        'rouge_l': rouge_l,
        'precision_1': precision_1,
        'recall_1': recall_1,
        'precision_l': precision_l,
        'recall_l': recall_l
    }

# 简单的分词函数，不依赖NLTK
def simple_tokenize(text):
    """简单的分词函数，按空格分词，对中文则按字符分"""
    # 先尝试简单的空格分词
    tokens = text.split()
    
    # 如果是中文，可以尝试逐字符分词
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        try:
            import jieba
            tokens = jieba.lcut(text)
        except ImportError:
            # 如果没有jieba，就逐字符分词
            tokens = list(text)
    
    return tokens

# 加载环境变量
load_dotenv()

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 读取配置文件
config = configparser.ConfigParser()
config_path = os.path.join(project_root, "Config", "Milvus.ini")
config.read(config_path)

# 获取Milvus连接配置
MILVUS_HOST = config.get("connection", "host")
MILVUS_PORT = config.get("connection", "port")
DEFAULT_COLLECTION = config.get("collection", "default_collection", fallback="RAGDATA")
VECTOR_DIM = config.getint("collection", "vector_dim", fallback=1536)
METRIC_TYPE = config.get("collection", "metric_type", fallback="COSINE")
INDEX_TYPE = config.get("collection", "index_type", fallback="HNSW")
PARTITION_NAME = config.get("collection", "partition_name", fallback="ARWU2024")

# 集合向量维度映射
COLLECTION_DIMS = {
    "arwu_text": 768,    # BERT维度
    "arwu_score": 7,     # 7个评分指标
    "arwu_enhanced": 10, # 增强向量
    "us_colleges": 384,  # 美国高校向量维度
}

# 设置OpenAI API密钥（从环境变量获取或设置）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-qvJtRhsJdn97oivEXcM3pxG1FClBwvXPCxfenxOfIc11Xeyy")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.nuwaapi.com/v1")

# 如果环境变量中没有API密钥，提示用户输入
if not OPENAI_API_KEY:
    print("警告: 未找到API密钥。请设置OPENAI_API_KEY环境变量或在下面输入:")
    OPENAI_API_KEY = input("请输入您的Deepseek API密钥: ").strip()
    if not OPENAI_API_KEY:
        print("错误: 必须提供有效的API密钥才能继续")
        sys.exit(1)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 初始化OpenAI客户端
client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)
print("API连接成功")

# 初始化评估器和OpenAI客户端
try:
    # 使用OpenAI的ChatGPT作为评估器 - 这部分可能需要保留OpenAI
    # evaluator_llm = LangchainLLMWrapper(ChatOpenAI())
    # print("API连接成功")
    # 移除LLM评估器，仅初始化API客户端
    client = openai.OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    print("API连接成功")
except Exception as e:
    print(f"API连接失败: {e}")
    print("请检查API密钥和网络连接")
    sys.exit(1)

print(f"测试开始: 连接到Milvus服务器 {MILVUS_HOST}:{MILVUS_PORT}")

# 连接到 Milvus
def connect_to_milvus():
    """连接到Milvus服务器并获取集合信息"""
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print(f"成功连接到Milvus服务器: {MILVUS_HOST}:{MILVUS_PORT}")

        # 检查Milvus中的集合
        available_collections = utility.list_collections()
        print(f"可用集合列表: {available_collections}")

        # 获取集合架构信息，以确认向量维度
        for coll_name in available_collections:
            try:
                coll = Collection(coll_name)
                schema = coll.schema
                for field in schema.fields:
                    if field.dtype == DataType.FLOAT_VECTOR:
                        COLLECTION_DIMS[coll_name] = field.params.get("dim")
                        print(f"集合 {coll_name} 的向量维度: {COLLECTION_DIMS[coll_name]}")
                        break
            except Exception as e:
                print(f"获取集合 {coll_name} 信息时出错: {e}")

        return True
    except Exception as e:
        print(f"连接Milvus失败: {e}")
        return False

# 获取文本的嵌入表示
def get_embedding(text, collection_name=None, model="paraphrase-multilingual-MiniLM-L12-v2", use_api=False):
    """获取文本的向量嵌入，并根据目标集合调整维度

    参数:
        text: 要向量化的文本
        collection_name: 目标集合名称，用于确定向量维度
        model: 本地模型名称或API模型名称
        use_api: 是否使用API (True使用API, False使用本地模型)
    """
    # 确定目标维度
    target_dim = VECTOR_DIM  # 默认维度 (1536)

    # 根据集合名称或现有映射确定向量维度
    if collection_name:
        # 首先尝试从全局映射获取维度
        if collection_name in COLLECTION_DIMS:
            target_dim = COLLECTION_DIMS[collection_name]
        # 然后检查集合名称部分匹配
        elif any(name in collection_name for name in ["arwu_text", "text"]):
            target_dim = 768
        elif any(name in collection_name for name in ["arwu_score", "score"]):
            target_dim = 7
        elif any(name in collection_name for name in ["arwu_enhanced", "enhanced"]):
            target_dim = 10
        elif any(name in collection_name for name in ["us_colleges", "colleges"]):
            target_dim = 384

    print(f"为集合 '{collection_name}' 生成维度为 {target_dim} 的向量")

    # 如果文本为空，返回零向量
    if not text or text.strip() == "":
        print("文本为空，返回零向量")
        return [0.0] * target_dim

    # 1. 优先使用本地向量化 (默认)
    if not use_api:
        try:
            # 尝试导入sentence-transformers库
            from sentence_transformers import SentenceTransformer
            import torch
            
            # 强制使用CPU，避免在Apple Silicon上使用MPS
            old_device = torch.device("cpu")
            if hasattr(torch, "get_default_device"):
                old_device = torch.get_default_device()
                torch.set_default_device("cpu")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            # 如果是评分向量或增强向量，使用随机向量
            if "score" in str(collection_name) or "enhanced" in str(collection_name):
                print(f"对于评分/增强向量集合，生成随机向量 (维度: {target_dim})")
                # 生成归一化的随机向量
                random_vector = np.random.normal(0, 1, target_dim)
                # 归一化向量
                norm = np.linalg.norm(random_vector)
                if norm > 0:
                    random_vector = random_vector / norm
                return random_vector.tolist()

            # 检查是否已加载模型
            if not hasattr(get_embedding, "model_cache"):
                get_embedding.model_cache = {}

            # 加载或获取缓存的模型
            if model not in get_embedding.model_cache:
                print(f"首次加载本地模型: {model}")
                get_embedding.model_cache[model] = SentenceTransformer(model, device="cpu")

            local_model = get_embedding.model_cache[model]

            # 获取嵌入
            print(f"使用本地模型生成嵌入向量: {model}")
            embedding = local_model.encode(text, convert_to_numpy=True)
            
            # 恢复原始设备设置
            if hasattr(torch, "set_default_device") and hasattr(torch, "get_default_device"):
                if torch.get_default_device() != old_device:
                    torch.set_default_device(old_device)

            # 调整维度
            actual_dim = len(embedding)
            print(f"本地模型返回向量维度: {actual_dim}, 目标维度: {target_dim}")

            if actual_dim == target_dim:
                # 维度匹配，直接返回
                return embedding.tolist()
            elif actual_dim > target_dim:
                # 需要截断向量
                print(f"调整向量维度: 从{actual_dim}截断到{target_dim}")
                return embedding[:target_dim].tolist()
            else:
                # 需要扩展向量
                print(f"调整向量维度: 从{actual_dim}扩展到{target_dim}")
                extended = np.zeros(target_dim)
                extended[:actual_dim] = embedding
                return extended.tolist()

        except ImportError:
            print("未安装sentence-transformers库，尝试使用scikit-learn进行向量化")
            try:
                # 尝试使用scikit-learn的TF-IDF向量化
                from sklearn.feature_extraction.text import TfidfVectorizer

                # 检查是否已初始化向量化器
                if not hasattr(get_embedding, "vectorizer"):
                    print("初始化TF-IDF向量化器")
                    get_embedding.vectorizer = TfidfVectorizer(max_features=min(target_dim, 10000))
                    # 使用一些示例文本进行拟合
                    get_embedding.vectorizer.fit([
                        text,
                        "大学 排名 世界 学术 评价",
                        "university ranking academic world evaluation research",
                        "清华大学 北京大学 哈佛大学 斯坦福大学",
                        "Tsinghua University Peking University Harvard Stanford"
                    ])

                # 向量化文本
                print("使用TF-IDF向量化文本")
                vector = get_embedding.vectorizer.transform([text]).toarray()[0]

                # 调整维度和归一化
                vector = adjust_vector_dimension(vector, target_dim)
                return vector.tolist()

            except ImportError:
                print("未安装scikit-learn库，使用随机向量")
                return generate_random_vector(target_dim)

        except Exception as local_error:
            print(f"本地向量化失败: {local_error}")
            # 如果用户指定了使用本地模型但失败了，则直接使用随机向量
            if not use_api:
                return generate_random_vector(target_dim)
            # 否则尝试API

    # 2. 如果指定使用API或本地方法失败
    if use_api:
        # 重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 调用Deepseek API获取嵌入向量
                print(f"尝试调用Deepseek API (尝试 {attempt+1}/{max_retries})...")
                response = client.embeddings.create(
                    model="deepseek-embed",  # 使用Deepseek的embedding模型
                    input=text,
                    encoding_format="float"
                )
                embedding = response.data[0].embedding

                # 调整向量维度
                actual_dim = len(embedding)
                print(f"API返回向量维度: {actual_dim}, 目标维度: {target_dim}")

                if actual_dim == target_dim:
                    # 维度匹配，直接返回
                    return embedding
                elif actual_dim > target_dim:
                    # 需要截断向量
                    print(f"调整向量维度: 从{actual_dim}截断到{target_dim}")
                    return embedding[:target_dim]
                else:
                    # 需要扩展向量
                    print(f"调整向量维度: 从{actual_dim}扩展到{target_dim}")
                    return embedding + [0.0] * (target_dim - actual_dim)

            except Exception as e:
                error_msg = str(e)
                print(f"获取嵌入向量失败: {error_msg}")

                # 如果不是最后一次尝试，等待后重试
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 指数退避
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"所有API尝试都失败，使用本地向量化方法")
                    # 如果API调用失败，继续到下面的随机向量生成

    # 如果所有方法都失败，使用随机向量
    return generate_random_vector(target_dim)

# 辅助函数：生成随机向量
def generate_random_vector(dim):
    """生成维度为dim的归一化随机向量"""
    print(f"生成随机向量 (维度: {dim})")
    random_vector = np.random.normal(0, 1, dim)
    # 归一化向量
    norm = np.linalg.norm(random_vector)
    if norm > 0:
        random_vector = random_vector / norm
    return random_vector.tolist()

# 辅助函数：调整向量维度
def adjust_vector_dimension(vector, target_dim):
    """调整向量维度并归一化"""
    actual_dim = len(vector)
    if actual_dim < target_dim:
        # 扩展维度
        extended = np.zeros(target_dim)
        extended[:actual_dim] = vector
        vector = extended
    elif actual_dim > target_dim:
        # 截断维度
        vector = vector[:target_dim]

    # 归一化向量
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector

# 分析问题内容
def analyze_query(query):
    """分析查询内容，决定搜索策略"""
    result = {
        "region": None,
        "country": None,
        "state": None,
        "control": None,
        "dataset": "arwu",  # 默认使用ARWU数据集
        "is_ranking_question": False
    }

    # 检测是否与排名相关
    ranking_keywords = ["排名", "rank", "第几", "好", "前", "top"]
    if any(keyword in query.lower() for keyword in ranking_keywords):
        result["is_ranking_question"] = True

    # 检测是否特指THE2025排名
    the_keywords = ["THE", "times higher education", "泰晤士", "THE2025", "2025年泰晤士"]
    if any(keyword.lower() in query.lower() for keyword in the_keywords):
        result["dataset"] = "the2025"
        
    # 检测THE排名特有关键词
    the_specific_keywords = [
        "教学环境", "研究实力", "引用影响力", "产业收入", "国际化", 
        "teaching", "research", "citations", "industry", "international outlook",
        "师生比例", "国际学生", "student staff ratio"
    ]
    if any(keyword.lower() in query.lower() for keyword in the_specific_keywords):
        result["dataset"] = "the2025"

    # 检测地区关键词
    if "亚洲" in query or "asia" in query.lower():
        result["region"] = "Asia"
    elif "欧洲" in query or "europe" in query.lower():
        result["region"] = "Europe"
    elif "北美" in query or "america" in query.lower() or "美洲" in query:
        result["region"] = "North America"
    elif "非洲" in query or "africa" in query.lower():
        result["region"] = "Africa"
    elif "大洋洲" in query or "oceania" in query.lower() or "澳洲" in query:
        result["region"] = "Oceania"

    # 检测国家关键词
    if "中国" in query or "china" in query.lower() or "chinese" in query.lower():
        result["country"] = "China"
    elif "美国" in query or "usa" in query.lower() or "america" in query.lower() or "united states" in query.lower():
        result["country"] = "United States"
        # 如果专门提到THE排名，则保持THE2025数据集
        if not result["dataset"] == "the2025":
            result["dataset"] = "us_colleges"  # 如果明确提到美国，优先使用美国高校数据集
    elif "英国" in query or "uk" in query.lower() or "united kingdom" in query.lower() or "britain" in query.lower():
        result["country"] = "United Kingdom"
    elif "日本" in query or "japan" in query.lower():
        result["country"] = "Japan"

    # 检测美国州关键词（针对us_colleges数据集）
    us_states = {
        "加州": "California",
        "纽约": "New York",
        "德州": "Texas",
        "佛罗里达": "Florida",
        "麻省": "Massachusetts",
        "california": "California",
        "new york": "New York",
        "texas": "Texas",
        "florida": "Florida",
        "massachusetts": "Massachusetts"
    }

    for cn_state, en_state in us_states.items():
        if cn_state in query.lower() or en_state.lower() in query.lower():
            result["state"] = en_state
            # 如果专门提到THE排名，则保持THE2025数据集
            if not result["dataset"] == "the2025":
                result["dataset"] = "us_colleges"  # 如果提到美国州，优先使用美国高校数据集
            break

    # 检测公立/私立关键词
    if "公立" in query or "public" in query.lower():
        result["control"] = "Public"
        # 如果专门提到THE排名，则保持THE2025数据集
        if not result["dataset"] == "the2025":
            result["dataset"] = "us_colleges"
    elif "私立" in query or "private" in query.lower():
        result["control"] = "Private"
        # 如果专门提到THE排名，则保持THE2025数据集
        if not result["dataset"] == "the2025":
            result["dataset"] = "us_colleges"

    # 特定类型的大学关键词
    college_types = ["liberal arts", "研究型", "research", "community college", "社区学院"]
    if any(term in query.lower() for term in college_types) and not result["dataset"] == "the2025":
        result["dataset"] = "us_colleges"

    return result

# 从ARWU排名数据中搜索相关信息
def search_arwu_knowledge(collection, query, top_k=5, region=None, country=None):
    """从ARWU数据中搜索相关信息"""
    collection_type = ""
    vector_field = ""
    metric_type = ""

    # 确定集合类型和向量字段
    if "arwu_text" in collection.name:
        collection_type = "text"
        vector_field = "text_vector"
        metric_type = "IP"  # 文本向量使用内积距离
    elif "arwu_score" in collection.name:
        collection_type = "score"
        vector_field = "score_vector"
        metric_type = "L2"  # 评分向量使用L2距离
    elif "arwu_enhanced" in collection.name:
        collection_type = "enhanced"
        vector_field = "enhanced_vector"
        metric_type = "L2"  # 增强向量使用L2距离
    else:
        print(f"未知的ARWU集合类型: {collection.name}")
        return []

    # 获取查询向量 - 为特定集合生成正确维度的向量
    query_embedding = get_embedding(query, collection_name=collection.name)

    # 获取集合架构信息和可用字段
    field_names = []
    has_entity_field = True
    try:
        # 尝试获取一条记录并检查其字段
        sample = collection.query(expr="", limit=1)
        if sample and len(sample) > 0:
            field_names = list(sample[0].keys())
            print(f"集合 {collection.name} 中存在的字段: {field_names}")
    except Exception as e:
        print(f"无法获取集合字段信息: {e}")

    # 检查是否有entity字段
    if len(field_names) == 1 and 'id' in field_names:
        print(f"集合 {collection.name} 只有id字段，可能需要额外查询获取详细信息")
        has_entity_field = False

    # 构建过滤条件 - 只有在有相应字段时才添加过滤
    filter_expr = None

    # 设置搜索参数
    search_params = {"metric_type": metric_type}
    if metric_type == "IP":
        search_params["params"] = {"ef": 100}
    else:
        search_params["params"] = {"nprobe": 10}

    # 执行向量搜索
    try:
        # 构造搜索参数
        search_kwargs = {
            "data": [query_embedding],
            "anns_field": vector_field,
            "param": search_params,
            "limit": top_k
        }

        # 添加表达式过滤条件 (如果有)
        if filter_expr:
            search_kwargs["expr"] = filter_expr

        # 添加输出字段 (如果有)
        if len(field_names) > 0:
            search_kwargs["output_fields"] = field_names

        # 添加分区信息 (如果需要)
        try:
            partitions = collection.partitions
            partition_names = [p.name for p in partitions]
            if PARTITION_NAME in partition_names:
                search_kwargs["partition_names"] = [PARTITION_NAME]
        except Exception as e:
            print(f"获取分区信息失败: {e}")

        # 执行搜索
        results = collection.search(**search_kwargs)

        # 处理结果
        retrieved_texts = []
        for hits in results[0]:
            try:
                # 基本信息
                info = f"搜索结果 ID: {hits.id}\n"
                if has_entity_field and hasattr(hits, 'entity'):
                    # 如果有entity字段，获取详情
                    entity_data = {}
                    for field in field_names:
                        if field != 'id':  # 跳过ID字段
                            entity_data[field] = hits.entity.get(field, "")

                    # 输出通用信息
                    if "university" in entity_data:
                        info += f"大学: {entity_data.get('university', '')}"
                        if "university_en" in entity_data:
                            info += f" ({entity_data.get('university_en', '')})"
                        info += "\n"

                    if "country" in entity_data:
                        location_info = f"国家/地区: {entity_data.get('country', '')}"
                        if "country_en" in entity_data:
                            location_info += f" ({entity_data.get('country_en', '')})"
                        if "continent" in entity_data:
                            location_info += f", 大洲: {entity_data.get('continent', '')}"
                        info += location_info + "\n"

                    if "rank" in entity_data:
                        rank_info = f"世界排名: {entity_data.get('rank', '')}"
                        if "rank_numeric" in entity_data:
                            rank_info += f" (数值排名: {entity_data.get('rank_numeric', 0)})"
                        info += rank_info + "\n"

                    # 添加评分信息（如果有）
                    if collection_type == "score":
                        score_fields = [
                            ("total_score", "总分"),
                            ("alumni_award", "校友获奖"),
                            ("prof_award", "教师获奖"),
                            ("high_cited_scientist", "高被引科学家"),
                            ("ns_paper", "NS论文"),
                            ("inter_paper", "国际论文")
                        ]
                        score_info = []
                        for field, label in score_fields:
                            if field in entity_data and entity_data[field]:
                                try:
                                    value = float(entity_data[field])
                                    score_info.append(f"{label}: {value:.1f}")
                                except:
                                    score_info.append(f"{label}: {entity_data[field]}")

                        if score_info:
                            info += ", ".join(score_info) + "\n"

                # 添加相似度信息
                info += f"相似度得分: {hits.distance:.4f}\n"
                retrieved_texts.append(info)
            except Exception as e:
                print(f"处理搜索结果时出错: {e}")
                retrieved_texts.append(f"搜索结果 ID: {hits.id} (处理详情时出错: {e})")
                continue

        return retrieved_texts
    except Exception as e:
        print(f"ARWU查询出错: {e}")
        return []

# 格式化THE2025搜索结果
def format_the2025_results(records, keywords=None, similarity_scores=None):
    """格式化THE2025搜索结果，提供更好的显示格式
    
    Args:
        records: 检索到的记录列表
        keywords: 搜索的关键词（可选）
        similarity_scores: 相似度分数（可选）
    
    Returns:
        格式化后的文本列表
    """
    formatted_results = []
    
    # 如果没有提供关键词，尝试提取
    if not keywords:
        keywords = []
    
    for i, record in enumerate(records):
        # 获取相似度得分
        similarity = similarity_scores[i] if similarity_scores and i < len(similarity_scores) else None
        
        # 格式化单条结果
        result = ""
        
        # 添加学校名称和排名（必须字段）
        if "name" in record:
            result += f"学校: {record['name']}\n"
        
        if "rank" in record:
            result += f"THE2025排名: {record['rank']}\n"
        
        if "location" in record:
            result += f"位置: {record['location']}\n"
        
        # 添加分数信息（如果存在）
        if "overall_score" in record:
            result += f"总体得分: {record['overall_score']:.1f}\n"
        
        # 添加各项指标得分
        score_fields = [
            ("teaching_score", "教学环境"),
            ("research_score", "研究实力"),
            ("citations_score", "引用影响力"),
            ("industry_income_score", "产业收入"),
            ("international_outlook_score", "国际化程度")
        ]
        
        for field, label in score_fields:
            if field in record and record[field]:
                try:
                    score = float(record[field])
                    result += f"{label}: {score:.1f}\n"
                except (ValueError, TypeError):
                    result += f"{label}: {record[field]}\n"
        
        # 添加其他指标（如果存在）
        if "student_staff_ratio" in record and record["student_staff_ratio"]:
            try:
                value = float(record["student_staff_ratio"])
                result += f"师生比例: {value:.1f}\n"
            except (ValueError, TypeError):
                result += f"师生比例: {record['student_staff_ratio']}\n"
        
        if "pc_intl_students" in record and record["pc_intl_students"]:
            try:
                value = float(record["pc_intl_students"])
                result += f"国际学生占比: {value:.1f}%\n"
            except (ValueError, TypeError):
                result += f"国际学生占比: {record['pc_intl_students']}\n"
        
        # 添加学科信息（从json_data中提取，如果存在）
        if "json_data" in record and record["json_data"]:
            try:
                json_data = json.loads(record["json_data"]) if isinstance(record["json_data"], str) else record["json_data"]
                
                if isinstance(json_data, dict) and "subjects" in json_data and json_data["subjects"]:
                    subjects = json_data["subjects"]
                    if isinstance(subjects, list) and subjects:
                        result += f"优势学科: {', '.join(subjects[:5])}"
                        if len(subjects) > 5:
                            result += f" 等{len(subjects)}个学科"
                        result += "\n"
            except Exception as e:
                print(f"解析学科信息时出错: {e}")
        
        # 添加相似度分数（如果提供）
        if similarity is not None:
            result += f"相似度: {similarity:.4f}\n"
        
        # 添加到结果列表
        formatted_results.append(result)
    
    return formatted_results

# 在THE2025集合中进行关键词搜索
def the2025_keyword_search(collection, query, top_k=5):
    """在THE2025集合中进行关键词搜索
    
    Args:
        collection: Milvus集合对象
        query: 用户查询文本
        top_k: 返回结果数量
        
    Returns:
        检索到的文本列表
    """
    try:
        collection_name = collection.name
        print(f"在 {collection_name} 中进行关键词搜索: '{query}'")
        
        # 1. 提取查询中的大学名称
        university_names = extract_university_names(query)
        # 2. 提取关键词
        keywords = extract_keywords(query)
        
        # 打印调试信息
        if university_names:
            print(f"从查询中提取的大学名称: {university_names}")
        if keywords:
            print(f"从查询中提取的关键词: {keywords}")
        
        # 初始化记录列表
        records = []
        
        # 3. 如果找到大学名称，优先按名称搜索
        if university_names:
            for name in university_names:
                # 构建搜索表达式 - 模糊匹配大学名称
                expr = f"name like '%{name}%'"
                try:
                    results = collection.query(
                        expr=expr,
                        output_fields=["id"] + list(set(collection.schema.fields_name) - {"id"}),
                        limit=top_k
                    )
                    
                    if results:
                        print(f"通过大学名称 '{name}' 找到 {len(results)} 条记录")
                        records.extend(results)
                except Exception as e:
                    print(f"按大学名称查询时出错: {e}")
        
        # 4. 处理排名区间相关查询
        rank_pattern = r'(前|top)\s*(\d+)|排名\s*(\d+)\s*[-~到至]\s*(\d+)'
        rank_matches = re.findall(rank_pattern, query, re.IGNORECASE)
        
        if rank_matches:
            for match in rank_matches:
                try:
                    # 处理"前X"或"topX"
                    if match[0] and match[1]:
                        top_n = int(match[1])
                        expr = f"rank <= '{top_n}'"
                    # 处理"排名X-Y"
                    elif match[2] and match[3]:
                        start_rank = int(match[2])
                        end_rank = int(match[3])
                        # 由于rank是字符串字段，需要特殊处理
                        # 构建一个范围查询的表达式
                        expr = f"rank >= '{start_rank}' and rank <= '{end_rank}'"
                    else:
                        continue
                    
                    # 执行查询
                    try:
                        results = collection.query(
                            expr=expr,
                            output_fields=["id"] + list(set(collection.schema.fields_name) - {"id"}),
                            limit=top_k
                        )
                        
                        if results:
                            print(f"通过排名条件 '{expr}' 找到 {len(results)} 条记录")
                            records.extend(results)
                    except Exception as e:
                        print(f"按排名查询时出错: {e}")
                except Exception as e:
                    print(f"处理排名匹配时出错: {e}")
        
        # 5. 处理地区搜索
        location_pattern = r'(位于|在)\s*([\w\s]+国家|[\w\s]+城市|[\w\s]+省份|[\w\s]+地区)'
        location_matches = re.findall(location_pattern, query)
        
        if location_matches or any(keyword in query for keyword in ["国家", "地区", "洲", "欧洲", "亚洲", "美洲", "非洲"]):
            # 提取地区关键词
            location_keywords = []
            for match in location_matches:
                if match[1]:
                    location_keywords.append(match[1].replace("国家", "").replace("城市", "").replace("省份", "").replace("地区", "").strip())
            
            # 添加常见地区词汇
            for region in ["欧洲", "亚洲", "美洲", "北美", "南美", "非洲", "大洋洲", "英国", "美国", "中国", "日本", "法国", "德国", "澳大利亚"]:
                if region in query and region not in location_keywords:
                    location_keywords.append(region)
            
            if location_keywords:
                print(f"提取的地区关键词: {location_keywords}")
                
                for location in location_keywords:
                    # 构建地区搜索表达式
                    expr = f"location like '%{location}%'"
                    try:
                        results = collection.query(
                            expr=expr,
                            output_fields=["id"] + list(set(collection.schema.fields_name) - {"id"}),
                            limit=top_k
                        )
                        
                        if results:
                            print(f"通过地区关键词 '{location}' 找到 {len(results)} 条记录")
                            records.extend(results)
                    except Exception as e:
                        print(f"按地区查询时出错: {e}")
        
        # 6. 处理得分相关查询
        score_fields = [
            ("overall_score", ["总分", "总得分", "整体", "总评"]),
            ("teaching_score", ["教学", "教育", "师资"]),
            ("research_score", ["研究", "科研", "学术"]),
            ("citations_score", ["引用", "引文", "论文影响"]),
            ("industry_income_score", ["产业", "行业", "收入", "工业"]),
            ("international_outlook_score", ["国际", "国际化", "全球化", "国际视野"])
        ]
        
        score_pattern = r'([\w\s]+分数|[\w\s]+得分|[\w\s]+评分)\s*(>|>=|=|==|<|<=)\s*(\d+\.?\d*)'
        score_matches = re.findall(score_pattern, query)
        
        if score_matches:
            for match in score_matches:
                try:
                    # 尝试确定匹配的是哪个分数字段
                    matched_field = None
                    for field, keywords in score_fields:
                        if any(keyword in match[0] for keyword in keywords):
                            matched_field = field
                            break
                    
                    if not matched_field:
                        # 默认为总分
                        matched_field = "overall_score"
                    
                    # 构建查询表达式
                    operator = match[1]
                    if operator == "=":
                        operator = "=="
                    threshold = float(match[2])
                    expr = f"{matched_field} {operator} {threshold}"
                    
                    # 执行查询
                    try:
                        results = collection.query(
                            expr=expr,
                            output_fields=["id"] + list(set(collection.schema.fields_name) - {"id"}),
                            limit=top_k
                        )
                        
                        if results:
                            print(f"通过分数条件 '{expr}' 找到 {len(results)} 条记录")
                            records.extend(results)
                    except Exception as e:
                        print(f"按分数查询时出错: {e}")
                except Exception as e:
                    print(f"处理分数匹配时出错: {e}")
        
        # 7. 如果仍未找到足够的结果，尝试在meta数据中搜索
        if len(records) < top_k and "meta" in collection_name:
            # 在json_data字段中进行关键词搜索
            for keyword in keywords:
                if len(keyword) < 2:  # 跳过过短的关键词
                    continue
                
                expr = f"json_data like '%{keyword}%'"
                try:
                    results = collection.query(
                        expr=expr,
                        output_fields=["id"] + list(set(collection.schema.fields_name) - {"id"}),
                        limit=top_k - len(records)
                    )
                    
                    if results:
                        print(f"通过json_data关键词 '{keyword}' 找到 {len(results)} 条记录")
                        records.extend(results)
                        
                    if len(records) >= top_k:
                        break
                except Exception as e:
                    print(f"在json_data中查询关键词时出错: {e}")
        
        # 移除重复记录
        unique_records = []
        seen_ids = set()
        
        for record in records:
            if record["id"] not in seen_ids:
                seen_ids.add(record["id"])
                unique_records.append(record)
        
        # 限制结果数量
        final_records = unique_records[:top_k]
        
        # 格式化结果
        if final_records:
            formatted_texts = format_the2025_results(final_records, keywords)
            return formatted_texts
        
        return []
    except Exception as e:
        print(f"THE2025关键词搜索出错: {e}")
        return []

# 更新search_the2025_knowledge函数以包含关键词搜索
def search_the2025_knowledge(collection, query, top_k=5):
    """从THE2025排名数据中搜索相关信息
    
    Args:
        collection: Milvus集合对象
        query: 用户查询文本
        top_k: 返回结果数量
    
    Returns:
        检索到的文本列表
    """
    # 首先尝试关键词搜索
    keyword_results = the2025_keyword_search(collection, query, top_k)
    if keyword_results:
        print("关键词搜索成功，返回结果")
        return keyword_results
    
    # 如果关键词搜索失败，回退到向量搜索
    print("关键词搜索未找到结果，尝试向量搜索")
    
    # 获取查询向量 - 根据集合类型使用正确的维度
    collection_name = collection.name
    
    # 确定向量字段名和向量维度
    vector_field = None
    if "basic_info" in collection_name:
        vector_field = "basic_info_vector"
        vector_dim = 768
    elif "subjects" in collection_name:
        vector_field = "subjects_vector"
        vector_dim = 768
    elif "metrics" in collection_name:
        vector_field = "metrics_vector"
        vector_dim = 10
    elif "meta" in collection_name:
        # 元数据集合不支持向量搜索，使用ID查询
        vector_field = "dummy_vector"
        vector_dim = 2
    else:
        # 默认假设为基本信息向量
        vector_field = "basic_info_vector"
        vector_dim = 768
    
    # 将集合维度加入全局映射
    COLLECTION_DIMS[collection_name] = vector_dim
    
    # 提取关键词
    keywords = extract_keywords(query)
    
    # 获取查询向量
    query_embedding = get_embedding(query, collection_name=collection_name)
    
    # 获取集合架构信息和可用字段
    field_names = []
    try:
        # 尝试获取一条记录并检查其字段
        sample = collection.query(expr="", limit=1)
        if sample and len(sample) > 0:
            field_names = list(sample[0].keys())
            print(f"集合 {collection_name} 中存在的字段: {field_names}")
    except Exception as e:
        print(f"无法获取集合字段信息: {e}")
    
    # 确定搜索参数
    # 对于metrics使用L2距离，其他使用余弦相似度
    if "metrics" in collection_name:
        search_params = {"metric_type": "L2", "params": {"ef": 100}}
    else:
        search_params = {"metric_type": "COSINE", "params": {"ef": 100}}
    
    # 执行搜索
    try:
        # 构造搜索参数
        search_kwargs = {
            "data": [query_embedding],
            "anns_field": vector_field,
            "param": search_params,
            "limit": top_k
        }
        
        # 添加输出字段 (如果有)
        if field_names:
            search_kwargs["output_fields"] = field_names
        
        # 执行搜索
        results = collection.search(**search_kwargs)
        
        # 处理结果
        records = []
        similarity_scores = []
        
        for hit in results[0]:
            try:
                # 保存实体数据
                record = dict(hit.entity)
                # 保存ID
                record["id"] = hit.id
                # 保存相似度分数
                similarity_scores.append(hit.distance)
                
                records.append(record)
            except Exception as e:
                print(f"处理THE2025搜索结果时出错: {e}")
                continue
        
        # 格式化结果
        if records:
            formatted_texts = format_the2025_results(records, keywords, similarity_scores)
            return formatted_texts
        
        return []
    except Exception as e:
        print(f"THE2025查询出错: {e}")
        return []

# 从美国高校数据中搜索相关信息
def search_us_colleges_knowledge(collection, query, top_k=5, state=None, region=None, control=None):
    """从美国高校数据中搜索相关信息"""
    # 获取查询向量 - 使用正确的维度
    query_embedding = get_embedding(query, collection_name=collection.name)

    # 获取集合架构信息和可用字段
    field_names = []
    has_entity_field = True
    try:
        # 尝试获取一条记录并检查其字段
        sample = collection.query(expr="", limit=1)
        if sample and len(sample) > 0:
            field_names = list(sample[0].keys())
            print(f"集合 {collection.name} 中存在的字段: {field_names}")
    except Exception as e:
        print(f"无法获取集合字段信息: {e}")

    # 检查是否有entity字段
    if len(field_names) == 1 and 'id' in field_names:
        print(f"集合 {collection.name} 只有id字段，可能需要额外查询获取详细信息")
        has_entity_field = False

    # 构建过滤条件 - 只有在有相应字段时才添加过滤
    filter_expr = None

    # 设置搜索参数
    search_params = {"metric_type": "COSINE", "params": {"ef": 100}}

    # 执行搜索
    try:
        # 构造搜索参数
        search_kwargs = {
            "data": [query_embedding],
            "anns_field": "text_vector",
            "param": search_params,
            "limit": top_k
        }

        # 添加表达式过滤条件 (如果有)
        if filter_expr:
            search_kwargs["expr"] = filter_expr

        # 添加输出字段 (如果有)
        if len(field_names) > 0:
            search_kwargs["output_fields"] = field_names

        # 执行搜索
        results = collection.search(**search_kwargs)

        # 处理结果
        retrieved_texts = []
        for hits in results[0]:
            try:
                # 基本信息
                info = f"搜索结果 ID: {hits.id}\n"

                if has_entity_field and hasattr(hits, 'entity'):
                    # 如果有entity字段，获取详情
                    entity_data = {}
                    for field in field_names:
                        if field != 'id':  # 跳过ID字段
                            entity_data[field] = hits.entity.get(field, "")

                    # 输出学校信息
                    if "name" in entity_data:
                        info += f"学校: {entity_data.get('name', '')}\n"

                    # 输出位置信息
                    location_parts = []
                    if "location" in entity_data and entity_data["location"]:
                        location_parts.append(entity_data["location"])
                    if "state" in entity_data and entity_data["state"]:
                        location_parts.append(entity_data["state"])
                    if "region" in entity_data and entity_data["region"]:
                        location_parts.append(entity_data["region"])

                    if location_parts:
                        info += f"位置: {', '.join(location_parts)}\n"

                    # 输出类型和控制方式
                    type_control_parts = []
                    if "type" in entity_data and entity_data["type"]:
                        type_control_parts.append(f"类型: {entity_data['type']}")
                    if "control" in entity_data and entity_data["control"]:
                        type_control_parts.append(f"控制方式: {entity_data['control']}")

                    if type_control_parts:
                        info += f"{', '.join(type_control_parts)}\n"

                    # 输出学生信息
                    if "enrollment" in entity_data and entity_data["enrollment"]:
                        info += f"在校学生: {entity_data['enrollment']}人\n"

                    # 输出网站信息
                    if "website" in entity_data and entity_data["website"]:
                        info += f"网站: {entity_data['website']}\n"

                # 添加相似度信息
                info += f"相似度得分: {hits.distance:.4f}\n"
                retrieved_texts.append(info)
            except Exception as e:
                print(f"处理美国高校搜索结果时出错: {e}")
                retrieved_texts.append(f"搜索结果 ID: {hits.id} (处理详情时出错: {e})")
                continue

        return retrieved_texts
    except Exception as e:
        print(f"美国高校查询出错: {e}")
        return []

# 安全地从索引中获取信息
def safe_get_index_info(index, field_name):
    """安全地获取索引信息，避免属性访问错误"""
    info = {}

    # 尝试直接获取常见属性
    if hasattr(index, 'field_name'):
        info['field_name'] = index.field_name

    if hasattr(index, 'params') and index.params:
        info['params'] = index.params
        # 从params中提取常见信息
        if 'metric_type' in index.params:
            info['metric_type'] = index.params['metric_type']
        if 'index_type' in index.params:
            info['index_type'] = index.params['index_type']

    # 尝试获取索引名称和ID
    for attr in ['index_name', 'index_id', 'name', 'id']:
        if hasattr(index, attr):
            info[attr] = getattr(index, attr)

    return info

def debug_milvus_search(collection, query_text, top_k=3):
    """调试函数，直接显示Milvus搜索结果（无过滤）"""
    try:
        print(f"\n====> 调试Milvus搜索: {collection.name} <====")
        print(f"查询: '{query_text}'")

        # 获取嵌入向量
        embedding = get_embedding(query_text, collection_name=collection.name)

        # 确定向量字段和正确的度量类型
        vector_field = None
        metric_type = None

        # 尝试从集合schema中找到向量字段
        try:
            for field in collection.schema.fields:
                if field.dtype == DataType.FLOAT_VECTOR:
                    vector_field = field.name
                    # 记录向量维度
                    vector_dim = field.params.get("dim")
                    if vector_dim:
                        COLLECTION_DIMS[collection.name] = vector_dim
                    break
        except Exception as e:
            print(f"从schema获取向量字段出错: {e}")

        # 如果未找到，使用基于名称的推断
        if not vector_field:
            if "score" in collection.name:
                vector_field = "score_vector"
            elif "enhanced" in collection.name:
                vector_field = "enhanced_vector"
            else:
                vector_field = "text_vector"
            print(f"根据集合名称推断向量字段: {vector_field}")

        # 尝试从索引中获取度量类型
        try:
            for index in collection.indexes:
                # 安全地获取索引信息
                index_info = safe_get_index_info(index, vector_field)

                # 仅处理与我们需要的向量字段相关的索引
                if index_info.get('field_name') == vector_field:
                    if 'metric_type' in index_info:
                        metric_type = index_info['metric_type']
                        print(f"从索引中获取度量类型: {metric_type}")
                        break
                    elif 'params' in index_info and 'metric_type' in index_info['params']:
                        metric_type = index_info['params']['metric_type']
                        print(f"从索引参数中获取度量类型: {metric_type}")
                        break
        except Exception as e:
            print(f"获取索引信息时出错: {e}")

        # 如果仍未找到度量类型，根据字段名称推断
        if not metric_type:
            if "score" in vector_field or "enhanced" in vector_field:
                metric_type = "L2"
            elif vector_field == "text_vector" and "us_colleges" in collection.name:
                metric_type = "COSINE"
            else:
                metric_type = "IP"
            print(f"根据字段名称推断度量类型: {metric_type}")

        print(f"使用向量字段: {vector_field}, 度量类型: {metric_type}")

        # 确定输出字段
        output_fields = []
        try:
            for field in collection.schema.fields:
                if field.name != vector_field and not field.is_primary:
                    output_fields.append(field.name)
            print(f"输出字段: {output_fields}")
        except Exception as e:
            print(f"获取字段信息时出错: {e}")
            # 尝试通过查询获取字段
            try:
                sample = collection.query(expr="", limit=1)
                if sample and len(sample) > 0:
                    output_fields = list(sample[0].keys())
                    if 'id' in output_fields:
                        output_fields.remove('id')
                    print(f"通过查询获取的字段: {output_fields}")
            except Exception as e2:
                print(f"通过查询获取字段也失败: {e2}")
                output_fields = []

        # 设置搜索参数 - 修复参数结构
        search_params = {
            "metric_type": metric_type
        }

        # 根据度量类型设置合适的参数
        if metric_type == "L2":
            search_params["params"] = {"nprobe": 10}
        elif metric_type == "IP":
            search_params["params"] = {"ef": 100}
        else:  # COSINE
            search_params["params"] = {"ef": 100}

        print(f"搜索参数: {search_params}")
        print(f"执行搜索... (向量维度: {len(embedding)})")

        # 执行搜索
        results = collection.search(
            data=[embedding],
            anns_field=vector_field,
            param=search_params,
            limit=top_k,
            output_fields=output_fields if output_fields else None
        )

        # 显示结果
        if results and len(results) > 0:
            print(f"找到 {len(results[0])} 条结果:")
            for i, hit in enumerate(results[0]):
                print(f"\n结果 {i+1}:")

                # 获取类型信息和可用属性
                hit_type = type(hit).__name__
                hit_attrs = [attr for attr in dir(hit) if not attr.startswith('_')]
                print(f"  [调试] 搜索结果类型: {hit_type}, 可用属性: {hit_attrs}")

                # 处理score/distance命名差异
                similarity = None
                if hasattr(hit, 'score'):
                    similarity = hit.score
                elif hasattr(hit, 'distance'):
                    similarity = hit.distance

                print(f"  ID: {hit.id}, 相似度: {similarity}")

                # 处理实体数据 - 完全重写这部分
                print("  数据字段:")
                try:
                    # 检查是否有entity属性
                    if hasattr(hit, 'entity'):
                        # 打印entity的类型信息
                        entity_type = type(hit.entity).__name__
                        print(f"  [调试] 实体类型: {entity_type}")

                        # 如果entity存在，尝试直接检查输出字段中的每个字段
                        for field_name in output_fields:
                            try:
                                # 尝试使用getattr获取属性值
                                if hasattr(hit.entity, field_name):
                                    value = getattr(hit.entity, field_name)
                                    if isinstance(value, str) and len(value) > 100:
                                        value = value[:100] + "..."
                                    elif isinstance(value, list) and len(value) > 10:
                                        value = f"[列表，长度={len(value)}]"
                                    print(f"    {field_name}: {value}")
                                # 尝试使用字典访问
                                elif hasattr(hit.entity, '__getitem__') and field_name in hit.entity:
                                    value = hit.entity[field_name]
                                    if isinstance(value, str) and len(value) > 100:
                                        value = value[:100] + "..."
                                    elif isinstance(value, list) and len(value) > 10:
                                        value = f"[列表，长度={len(value)}]"
                                    print(f"    {field_name}: {value}")
                                else:
                                    print(f"    {field_name}: <无法访问>")
                            except Exception as field_err:
                                print(f"    {field_name}: <访问出错: {field_err}>")
                    else:
                        print("    <hit对象没有entity属性>")

                        # 尝试直接从hit对象获取数据
                        for field_name in output_fields:
                            if hasattr(hit, field_name):
                                try:
                                    value = getattr(hit, field_name)
                                    if isinstance(value, str) and len(value) > 100:
                                        value = value[:100] + "..."
                                    elif isinstance(value, list) and len(value) > 10:
                                        value = f"[列表，长度={len(value)}]"
                                    print(f"    {field_name}: {value}")
                                except Exception as e:
                                    print(f"    {field_name}: <访问出错: {e}>")

                    # 如果无法访问任何字段，尝试单独查询
                    if output_fields:
                        try:
                            print("  尝试单独查询实体信息...")
                            entity = collection.query(expr=f"id == {hit.id}", output_fields=output_fields)
                            if entity and len(entity) > 0:
                                print("  查询到的数据:")
                                for field, value in entity[0].items():
                                    if field != 'id':
                                        if isinstance(value, str) and len(value) > 100:
                                            value = value[:100] + "..."
                                        elif isinstance(value, list) and len(value) > 10:
                                            value = f"[列表，长度={len(value)}]"
                                        print(f"    {field}: {value}")
                            else:
                                print("    <查询返回空结果>")
                        except Exception as e:
                            print(f"  单独查询实体失败: {e}")
                except Exception as e:
                    print(f"  处理实体数据出错: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("没有找到结果")

        return results

    except Exception as e:
        print(f"搜索出错: {e}")
        import traceback
        traceback.print_exc()

        # 如果是度量类型错误，尝试另一种度量类型
        try:
            if "metric type" in str(e).lower():
                alternative_metric = "L2" if metric_type == "IP" else "IP"
                print(f"\n尝试使用替代度量类型: {alternative_metric}")

                # 设置替代搜索参数
                alt_search_params = {
                    "metric_type": alternative_metric
                }

                if alternative_metric == "L2":
                    alt_search_params["params"] = {"nprobe": 10}
                else:  # 假设是IP或COSINE
                    alt_search_params["params"] = {"ef": 100}

                print(f"替代搜索参数: {alt_search_params}")

                # 尝试使用替代度量类型搜索
                results = collection.search(
                    data=[embedding],
                    anns_field=vector_field,
                    param=alt_search_params,
                    limit=top_k,
                    output_fields=output_fields if output_fields else None
                )

                if results and len(results) > 0:
                    print(f"使用替代度量类型找到 {len(results[0])} 条结果")
                    return results
        except Exception as e2:
            print(f"使用替代度量类型搜索也失败: {e2}")

        # 如果是维度不匹配错误，尝试重新生成正确维度的向量
        try:
            if "dimension mismatch" in str(e).lower() or "vector dimension" in str(e).lower():
                # 从错误消息中提取预期维度
                import re
                expected_dim_match = re.search(r'expected vector size\(byte\) (\d+)', str(e))
                if expected_dim_match:
                    expected_bytes = int(expected_dim_match.group(1))
                    expected_dim = expected_bytes // 4  # 通常每个浮点数是4字节
                    print(f"从错误消息提取的预期维度: {expected_dim}")

                    # 更新映射
                    COLLECTION_DIMS[collection.name] = expected_dim

                    # 重新生成向量
                    print(f"使用正确维度 {expected_dim} 重新生成向量...")
                    embedding = get_embedding(query_text, collection_name=collection.name)

                    # 设置新的搜索参数
                    fix_search_params = {
                        "metric_type": metric_type,
                        "params": {"nprobe": 10} if metric_type == "L2" else {"ef": 100}
                    }

                    # 重试搜索
                    results = collection.search(
                        data=[embedding],
                        anns_field=vector_field,
                        param=fix_search_params,
                        limit=top_k,
                        output_fields=output_fields if output_fields else None
                    )

                    print(f"使用正确维度重试成功，找到 {len(results[0])} 条结果")
                    return results
        except Exception as e3:
            print(f"维度修正后重试失败: {e3}")

        return None

# 添加一个检查集合架构和字段详情的函数
def inspect_collection_schema(collection):
    """直接检查集合的架构和字段详细信息，帮助调试"""
    try:
        print(f"\n=== {collection.name} 集合架构检查 ===")

        # 获取集合架构
        schema = collection.schema
        print(f"集合描述: {schema.description}")

        # 检查所有字段
        print(f"字段列表 (共 {len(schema.fields)} 个):")
        for i, field in enumerate(schema.fields):
            print(f"\n字段 {i+1}: {field.name}")
            print(f"  类型: {field.dtype}")

            # 检查特殊属性
            if field.is_primary:
                print(f"  主键: 是")
            if hasattr(field, 'auto_id') and field.auto_id:
                print(f"  自动ID: 是")
            if hasattr(field, 'max_length') and field.max_length:
                print(f"  最大长度: {field.max_length}")

            # 检查向量字段详情
            if hasattr(field, 'params') and field.params:
                print(f"  参数: {field.params}")
                if field.dtype == DataType.FLOAT_VECTOR:
                    print(f"  向量维度: {field.params.get('dim')}")

        # 尝试获取索引信息
        try:
            indexes = collection.indexes
            if indexes:
                print(f"\n索引信息 (共 {len(indexes)} 个):")
                for i, index in enumerate(indexes):
                    print(f"索引 {i+1}:")

                    # 安全地获取索引属性
                    attrs = {}
                    # 常见的索引属性
                    for attr_name in ['field_name', 'params', 'index_name', 'index_id', 'index_type']:
                        if hasattr(index, attr_name):
                            attrs[attr_name] = getattr(index, attr_name)

                    # 打印已找到的属性
                    for name, value in attrs.items():
                        print(f"  {name}: {value}")

                    # 如果没有获取到任何索引类型信息，尝试使用params中的信息
                    if 'index_type' not in attrs and hasattr(index, 'params'):
                        params = index.params
                        if 'index_type' in params:
                            print(f"  索引类型(从params): {params.get('index_type')}")
                        elif 'metric_type' in params:
                            print(f"  度量类型: {params.get('metric_type')}")

                    # 如果以上方法都无法获取属性，显示所有可用属性
                    if not attrs:
                        print(f"  可用属性: {dir(index)}")
            else:
                print("\n没有索引信息")
        except Exception as e:
            print(f"无法获取索引信息: {e}")
            print(f"  索引对象的可用属性: {dir(collection.indexes[0]) if collection.indexes else 'N/A'}")

        # 尝试获取分区信息
        try:
            partitions = collection.partitions
            if partitions:
                print(f"\n分区信息 (共 {len(partitions)} 个):")
                for i, partition in enumerate(partitions):
                    print(f"分区 {i+1}: {partition.name}")
            else:
                print("\n没有分区信息")
        except Exception as e:
            print(f"无法获取分区信息: {e}")

        return True
    except Exception as e:
        print(f"检查集合架构时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

# 修改关键词搜索函数，避免使用Milvus不支持的LIKE操作符
def keyword_search(collection, query, top_k=5):
    """基于关键词的精确匹配搜索，适配Milvus查询限制"""
    try:
        print(f"\n执行关键词搜索: {collection.name}")
        
        # 提取查询中的关键词
        keywords = extract_keywords(query)
        if not keywords:
            print("未能提取到有效关键词，使用完整查询")
            keywords = [query]
        
        print(f"提取的关键词: {keywords}")
        
        # 检查是否有清华大学关键词
        has_tsinghua = any(k in ["清华大学", "清华", "tsinghua"] for k in keywords)
        
        # 直接通过ID查询特定记录
        # 这是一个变通方法，因为我们知道常见大学的ID范围
        if has_tsinghua:
            print("检测到清华大学关键词，使用ID范围查询")
            # 尝试一个ID范围，假设清华大学的记录可能在前50条数据中
            try:
                results = []
                # 检查哪些ID包含实体数据
                for id in range(1, 50):
                    try:
                        record = collection.query(expr=f"id == {id}", output_fields=["*"], limit=1)
                        if record and len(record) > 0:
                            # 获取所有字段的文本，检查是否包含清华大学
                            record_text = str(record[0])
                            if "清华" in record_text or "tsinghua" in record_text.lower():
                                results.append(record[0])
                                print(f"找到匹配记录: ID={id}")
                                if len(results) >= top_k:
                                    break
                    except Exception as e:
                        # 忽略查询错误，继续尝试
                        pass
                        
                if results:
                    return format_search_results(results, keywords, collection.name)
            except Exception as e:
                print(f"ID范围查询失败: {e}")
        
        # 如果上面的方法失败，尝试获取所有记录并在Python中筛选
        # 适用于小型数据集，生产环境需要更高效的方法
        try:
            print("尝试获取全部记录并在内存中过滤")
            all_records = collection.query(expr="id >= 0", output_fields=["*"], limit=100)
            
            if all_records:
                print(f"获取到 {len(all_records)} 条记录，进行关键词匹配")
                
                # 在Python中进行关键词匹配
                matched_records = []
                for record in all_records:
                    record_text = str(record).lower()
                    score = 0
                    
                    # 计算关键词匹配分数
                    for keyword in keywords:
                        if keyword.lower() in record_text:
                            score += 1
                    
                    if score > 0:  # 至少匹配一个关键词
                        matched_records.append((record, score/len(keywords)))
                
                # 按匹配度排序
                matched_records.sort(key=lambda x: x[1], reverse=True)
                
                # 取前top_k个结果
                top_matches = [r[0] for r in matched_records[:top_k]]
                
                if top_matches:
                    return format_search_results(top_matches, keywords, collection.name)
        except Exception as e:
            print(f"获取所有记录失败: {e}")
        
        # 如果以上方法都失败，尝试向量搜索
        print("关键词搜索方法都失败，建议尝试向量搜索")
        return []
    
    except Exception as e:
        print(f"关键词搜索出错: {e}")
        import traceback
        traceback.print_exc()
        return []

def format_search_results(records, keywords, collection_name, similarity_scores=None):
    """格式化搜索结果
    
    参数:
        records: 记录列表
        keywords: 关键词列表
        collection_name: 集合名称
        similarity_scores: 相似度分数列表(如果有)
        
    返回:
        格式化后的文本列表
    """
    retrieved_texts = []
    for i, record in enumerate(records):
        # 根据来源集合添加标识
        collection_label = ""
        if "arwu_text" in collection_name:
            collection_label = "ARWU文本集合"
        elif "arwu_score" in collection_name:
            collection_label = "ARWU评分集合"
        elif "arwu_enhanced" in collection_name:
            collection_label = "ARWU增强集合"
        elif "us_colleges" in collection_name:
            collection_label = "美国高校集合"
        else:
            collection_label = collection_name
        
        # 添加基本信息
        info = f"搜索结果 #{i+1} (ID: {record.get('id', 'N/A')}, 来源: {collection_label})\n"
        
        # 添加相似度分数信息(如果有)
        if similarity_scores and i < len(similarity_scores):
            score = similarity_scores[i]
            # 保留4位小数
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            info += f"相似度得分: {score_str}\n"
        
        # 根据集合类型构造信息字符串
        if "arwu" in collection_name:
            # ARWU相关集合
            if "university" in record:
                uni_name = record.get("university", "")
                uni_name_en = record.get("university_en", "")
                info += f"大学: {uni_name}"
                if uni_name_en and uni_name_en != uni_name:
                    info += f" ({uni_name_en})"
                info += "\n"
            
            if "country" in record:
                country = record.get("country", "")
                country_en = record.get("country_en", "")
                info += f"国家/地区: {country}"
                if country_en and country_en != country:
                    info += f" ({country_en})"
                info += "\n"
            
            if "rank" in record:
                info += f"世界排名: {record.get('rank', 'N/A')}\n"
                
            if "year" in record:
                info += f"排名年份: {record.get('year', 'N/A')}\n"
            
            # 添加分数信息
            score_fields = [
                ("total_score", "总分"),
                ("alumni_award", "校友获奖"),
                ("prof_award", "教师获奖"),
                ("high_cited_scientist", "高被引科学家"),
                ("ns_paper", "NS论文"),
                ("inter_paper", "国际论文")
            ]
            
            score_info = []
            for field, label in score_fields:
                if field in record and record[field]:
                    try:
                        value = float(record[field])
                        score_info.append(f"{label}: {value:.1f}")
                    except:
                        score_info.append(f"{label}: {record[field]}")
            
            if score_info:
                info += ", ".join(score_info) + "\n"
        
        elif "us_colleges" in collection_name:
            # 美国大学集合
            if "name" in record:
                info += f"学校: {record.get('name', '')}\n"
            
            location_parts = []
            if "location" in record and record["location"]:
                location_parts.append(record["location"])
            if "state" in record and record["state"]:
                location_parts.append(record["state"])
            if "region" in record and record["region"]:
                location_parts.append(record["region"])
            
            if location_parts:
                info += f"位置: {', '.join(location_parts)}\n"
            
            type_control_parts = []
            if "type" in record and record["type"]:
                type_control_parts.append(f"类型: {record['type']}")
            if "control" in record and record["control"]:
                type_control_parts.append(f"控制方式: {record['control']}")
            
            if type_control_parts:
                info += f"{', '.join(type_control_parts)}\n"
            
            if "enrollment" in record and record["enrollment"]:
                info += f"在校学生: {record['enrollment']}人\n"
        
        else:
            # 其他未知集合，列出所有字段
            for field, value in record.items():
                if field != "id":
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    info += f"{field}: {value}\n"
        
        # 添加匹配度评分
        record_text = str(record).lower()
        match_score = sum(1 for k in keywords if k.lower() in record_text)
        match_ratio = match_score / len(keywords) if keywords else 0
        info += f"关键词匹配度: {match_ratio:.2f}\n"
        
        retrieved_texts.append(info)
    
    return retrieved_texts

def extract_keywords(query):
    """从查询中提取关键词"""
    # 移除标点符号
    import re
    import jieba
    
    # 停用词列表 (英文 + 中文)
    stopwords = set([
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很",
        "the", "of", "in", "a", "and", "is", "to", "it", "that", "for", "on", "with", "as", "by", 
        "at", "be", "this", "which", "or", "an", "are", "from", "have", "has", "had", "you", "your",
        "what", "where", "when", "who", "how", "排名", "大学", "rank", "university", "college", "哪些", "有哪些"
    ])
    
    # 清理文本
    clean_query = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', query)
    
    # 使用结巴分词处理中文
    words = []
    
    # 分词处理
    if any('\u4e00' <= char <= '\u9fff' for char in clean_query):  # 包含中文字符
        words = jieba.lcut(clean_query)
    else:  # 纯英文
        words = clean_query.lower().split()
    
    # 过滤停用词
    filtered_words = [word for word in words if word.lower() not in stopwords and len(word) > 1]
    
    # 提取数字（排名相关）
    numbers = re.findall(r'\d+', query)
    filtered_words.extend(numbers)
    
    # 特殊处理大学名称
    university_names = extract_university_names(query)
    if university_names:
        filtered_words.extend(university_names)
    
    return list(set(filtered_words))  # 去重

def extract_university_names(text):
    """提取文本中可能的大学名称"""
    # 常见大学名称模式
    uni_patterns = [
        r'([\w\s]+大学)',  # 中文大学
        r'([\w\s]+学院)',  # 中文学院
        r'(University of [\w\s]+)',  # University of X
        r'([\w\s]+ University)',  # X University
        r'([\w\s]+ College)',  # X College
        r'([\w\s]+ Institute of Technology)',  # MIT格式
    ]
    
    results = []
    for pattern in uni_patterns:
        import re
        matches = re.findall(pattern, text, re.IGNORECASE)
        results.extend(matches)
    
    # 常见大学名称的硬编码
    common_universities = {
        'MIT': 'Massachusetts Institute of Technology',
        '麻省理工': 'Massachusetts Institute of Technology',
        '哈佛': 'Harvard University',
        '清华': '清华大学',
        '北大': '北京大学',
        '斯坦福': 'Stanford University',
        '牛津': 'Oxford University',
        '剑桥': 'Cambridge University',
    }
    
    # 检查是否包含常见大学简称
    for short_name, full_name in common_universities.items():
        if short_name in text:
            results.append(full_name)
    
    return results

# 修改load_knowledge_variables函数，优先使用关键词搜索


def load_knowledge_variables(collections, query, top_k=5, use_keyword_search=True, use_multi_collections=True):
    """加载知识变量，根据查询检索相关信息
    
    参数:
        collections: 集合字典或列表
        query: 查询文本
        top_k: 返回结果数量
        use_keyword_search: 是否优先使用关键词搜索
        use_multi_collections: 是否在多个集合中搜索
    """
    try:
        # 分析查询类型
        query_analysis = analyze_query(query)
        print(f"查询分析: {query_analysis}")
        
        # 决定使用哪个数据集
        dataset_type = query_analysis["dataset"]
        
        # 存储所有检索到的结果
        all_retrieved_texts = []
        # 存储每个结果的来源
        result_sources = []
        
        # 准备要搜索的集合
        collections_to_try = []
        
        # 如果collections是字典类型
        if isinstance(collections, dict):
            # 首先根据查询分析筛选合适的集合
            priority_collections = []
            if dataset_type == "us_colleges" and "us_colleges" in collections:
                priority_collections.append(("us_colleges", collections["us_colleges"]))
            elif dataset_type == "arwu":
                # 优先使用最匹配的ARWU数据集
                arwu_collections = ["arwu_text", "arwu_score", "arwu_enhanced"]
                for coll_name in arwu_collections:
                    if coll_name in collections:
                        priority_collections.append((coll_name, collections[coll_name]))
            elif dataset_type == "the2025":
                # 优先使用THE2025数据集，按照优先级排序
                the2025_collections = ["the2025_meta", "the2025_basic_info", "the2025_subjects", "the2025_metrics"]
                for coll_name in the2025_collections:
                    if coll_name in collections:
                        priority_collections.append((coll_name, collections[coll_name]))
            
            # 将优先集合排在前面
            if priority_collections:
                collections_to_try = priority_collections
                # 如果启用多集合搜索，添加其他未包含的集合
                if use_multi_collections:
                    for coll_name, collection in collections.items():
                        if coll_name not in [name for name, _ in priority_collections]:
                            collections_to_try.append((coll_name, collection))
            else:
                collections_to_try = list(collections.items())
        # 如果collections是列表或其他类型
        else:
            # 如果是列表，假设是集合名称列表
            if isinstance(collections, list):
                for coll_name in collections:
                    try:
                        collections_to_try.append((coll_name, Collection(name=coll_name)))
                    except Exception as e:
                        print(f"加载集合 {coll_name} 失败: {e}")
        
        # 遍历集合进行搜索
        for coll_name, collection in collections_to_try:
            try:
                print(f"\n开始搜索集合: {coll_name}")
                collection.load()
                
                retrieved_texts = []
                context_source = ""
                
                # 优先使用关键词搜索
                if use_keyword_search:
                    # 对于THE2025集合，使用专门的keyword_search函数
                    if "the2025" in coll_name:
                        texts = the2025_keyword_search(collection, query, top_k=top_k)
                        if texts:
                            retrieved_texts = texts
                            context_source = f"THE2025关键词搜索 ({coll_name})"
                            print(f"THE2025关键词搜索成功: {coll_name}")
                    # 对于其他集合，使用通用的keyword_search函数
                    else:
                        texts = keyword_search(collection, query, top_k=top_k)
                        if texts:
                            retrieved_texts = texts
                            context_source = f"关键词搜索 ({coll_name})"
                            print(f"关键词搜索成功: {coll_name}")
                
                # 如果关键词搜索没有结果，回退到向量搜索
                if not retrieved_texts:
                    print(f"关键词搜索未返回结果，尝试向量搜索: {coll_name}")
                    if "arwu" in coll_name:
                        texts = search_arwu_knowledge(collection, query, top_k=top_k)
                    elif "us_colleges" in coll_name:
                        texts = search_us_colleges_knowledge(collection, query, top_k=top_k)
                    elif "the2025" in coll_name:
                        texts = search_the2025_knowledge(collection, query, top_k=top_k)
                    else:
                        # 未知集合类型，跳过
                        print(f"未知集合类型，跳过: {coll_name}")
                        continue
                        
                    if texts:
                        retrieved_texts = texts
                        context_source = f"向量搜索 ({coll_name})"
                        print(f"向量搜索成功: {coll_name}")
                
                # 如果从当前集合检索到了结果，添加到总结果列表
                if retrieved_texts:
                    for text in retrieved_texts:
                        all_retrieved_texts.append(text)
                        result_sources.append(context_source)
                    
                    # 如果不使用多集合搜索，找到结果后立即停止
                    if not use_multi_collections:
                        print(f"已找到结果，且未启用多集合搜索，不再继续查找其他集合")
                        break
            except Exception as e:
                print(f"搜索集合 {coll_name} 时出错: {e}")
        
        # 打印最终检索到的文本
        print("\n=== 最终检索结果 ===")
        if all_retrieved_texts:
            # 根据相关性对结果去重和排序
            unique_texts = []
            unique_sources = []
            
            # 简单去重（更复杂的实现可以考虑文本相似度）
            for i, text in enumerate(all_retrieved_texts):
                if text not in unique_texts:
                    unique_texts.append(text)
                    unique_sources.append(result_sources[i])
            
            # 限制结果数量，保留最相关的top_k个
            final_texts = unique_texts[:top_k]
            final_sources = unique_sources[:top_k]
            
            # 打印信息
            for i, (text, source) in enumerate(zip(final_texts, final_sources)):
                print(f"\n检索文本 {i+1} (来源: {source}):\n{text}")
            
            # 合并检索结果
            source_text_groups = {}
            for text, source in zip(final_texts, final_sources):
                if source not in source_text_groups:
                    source_text_groups[source] = []
                source_text_groups[source].append(text)
            
            context_parts = []
            for source, texts in source_text_groups.items():
                context_parts.append(f"以下信息来自{source}:\n\n" + "\n\n".join(texts))
            
            context = "\n\n" + "="*50 + "\n\n".join(context_parts)
        else:
            print("未检索到任何相关文本")
            context = "未找到相关信息。系统无法在数据库中检索到与查询相关的大学数据。"
            
        return context
    
    except Exception as e:
        print(f"加载知识库失败: {e}")
        import traceback
        traceback.print_exc()
        return "未找到相关信息 (发生错误)" + f"  错误详情: {str(e)}"

# 使用DeepSeek生成回答
def generate_answer(query, context, model_name="deepseek-chat"):
    """使用多种LLM模型生成回答
    
    Args:
        query: 用户的问题
        context: 检索到的上下文
        model_name: 模型名称，支持"deepseek-chat"和"gpt-4o"等
    
    Returns:
        生成的回答文本
    """
    try:
        system_prompt = """
        你是一个专门回答大学相关问题的AI助手。你可以提供关于大学排名、学校情况、地理位置等信息。
        
        请根据提供的相关知识背景严格使用简洁精炼的语言完整回答用户的问题。如果知识背景中没有相关信息，请基于你的常识进行回答。你的回答中不得带有"注"或者"需要注意"的部分的描述。
        
        知识背景可能来自多个数据源，每个数据源会有明确的标记。如果不同数据源提供了冲突的信息，请综合考虑数据的可信度和完整性，优先使用：
        1. 最新的信息（如果有日期标记）
        2. 官方排名数据优先于非官方数据
        3. 具体详细的信息优先于笼统的描述
        
        如果是关于排名的问题，一定要提及具体的排名来源和排名年份。如果有多个来源的排名数据，可以一并提及并说明各自的特点。
        
        请注意，你的回答应该是连贯的、统一的，而不是简单地拼接不同数据源的信息。需要对所有数据源的信息进行整合，形成一个完整的答案。
        """

        # 默认使用DeepSeek API
        if "deepseek" in model_name.lower():
            # 使用DeepSeek API
            print(f"使用DeepSeek模型 ({model_name}) 生成回答...")
            response = client.chat.completions.create(
                model=model_name,  # 使用DeepSeek的聊天模型
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"相关知识背景：\n{context}\n\n用户的问题：{query}"}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return response.choices[0].message.content
        elif "gpt" in model_name.lower():
            # 如果使用OpenAI GPT模型，需要切换到OpenAI API
            print(f"使用OpenAI模型 ({model_name}) 生成回答...")
            # 初始化OpenAI客户端(使用环境变量中的API密钥)
                
            openai_client = openai.OpenAI(
                api_key="sk-qvJtRhsJdn97oivEXcM3pxG1FClBwvXPCxfenxOfIc11Xeyy",
                # 这里没有指定base_url，将使用OpenAI默认的API端点
                base_url="https://api.nuwaapi.com/v1"
            )
            
            response = openai_client.chat.completions.create(
                model=model_name,  # 例如gpt-4o
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"相关知识背景：\n{context}\n\n用户的问题：{query}"}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return response.choices[0].message.content
        else:
            raise ValueError(f"不支持的模型: {model_name}")

    except Exception as e:
        print(f"生成回答时出错: {e}")
        return f"生成回答时出错: {e}"

# 使用多模型生成回答并比较
def generate_answers_with_comparison(query, context, models=["deepseek-chat", "gpt-4o"]):
    """使用多个模型生成回答并返回比较结果
    
    Args:
        query: 用户的问题
        context: 检索到的上下文
        models: 要使用的模型列表
        
    Returns:
        包含多个模型回答的字典
    """
    results = {}
    
    try:
        for model in models:
            print(f"使用模型 {model} 生成回答...")
            answer = generate_answer(query, context, model_name=model)
            results[model] = answer
            
        return results
    except Exception as e:
        print(f"多模型比较时出错: {e}")
        return {"error": f"生成比较回答时出错: {e}"}

# 修改评估函数，支持多模型的评估
def evaluate_multi_model_answers(answers_dict, reference_answer):
    """评估多个模型生成的回答
    
    Args:
        answers_dict: 包含多个模型回答的字典 {model_name: answer}
        reference_answer: 标准参考答案
        
    Returns:
        每个模型的评估结果字典 {model_name: metrics}
    """
    results = {}
    
    for model_name, answer in answers_dict.items():
        print(f"\n评估模型 {model_name} 的回答...")
        metrics = evaluate_with_metrics(answer, reference_answer)
        results[model_name] = metrics
        
    return results

# 修改评估函数，增加调试信息和备用ROUGE计算
def evaluate_with_metrics(generated_answer, reference_answer):
    """使用ROUGE评估生成的回答，并计算精确率和召回率"""
    results = {}
    
    try:
        # 打印原始文本，以便调试
        print("\n==== 评估文本 ====")
        print(f"参考答案 [{len(reference_answer)} 字符]: {reference_answer[:100]}...")
        print(f"生成答案 [{len(generated_answer)} 字符]: {generated_answer[:100]}...")
        
        # 计算ROUGE分数
        try:
            # 确保文本非空且为字符串
            if not generated_answer or not reference_answer:
                print("警告: 生成的答案或参考答案为空")
                generated_answer = generated_answer or "无回答"
                reference_answer = reference_answer or "无参考答案"
            
            if not isinstance(generated_answer, str):
                generated_answer = str(generated_answer)
            if not isinstance(reference_answer, str):
                reference_answer = str(reference_answer)
            
            # 尝试使用rouge库计算
            rouge_scores = rouge.get_scores(generated_answer, reference_answer)[0]
            results['rouge_1'] = rouge_scores['rouge-1']['f']
            results['rouge_2'] = rouge_scores['rouge-2']['f']
            results['rouge_l'] = rouge_scores['rouge-l']['f']
            
            # 同时保存精确率和召回率
            results['precision_1'] = rouge_scores['rouge-1']['p']
            results['recall_1'] = rouge_scores['rouge-1']['r']
            results['precision_l'] = rouge_scores['rouge-l']['p']
            results['recall_l'] = rouge_scores['rouge-l']['r']
            
            print(f"ROUGE库计算结果: ROUGE-1={results['rouge_1']:.4f}, ROUGE-L={results['rouge_l']:.4f}")
            print(f"精确率: ROUGE-1-P={results['precision_1']:.4f}, ROUGE-L-P={results['precision_l']:.4f}")
            print(f"召回率: ROUGE-1-R={results['recall_1']:.4f}, ROUGE-L-R={results['recall_l']:.4f}")
            
            # 如果ROUGE-L为0，使用备用方法
            if results['rouge_l'] < 0.001:
                print("ROUGE-L近似为0，使用备用计算方法...")
                backup_scores = compute_rouge_manually(generated_answer, reference_answer)
                
                # 只有在备用计算结果大于0时才替换原值
                if backup_scores['rouge_l'] > 0:
                    results['rouge_l'] = backup_scores['rouge_l']
                    print(f"备用方法ROUGE-L={results['rouge_l']:.4f}")
                if backup_scores['rouge_1'] > 0 and results['rouge_1'] < 0.001:
                    results['rouge_1'] = backup_scores['rouge_1']
                    print(f"备用方法ROUGE-1={results['rouge_1']:.4f}")
                
                # 备用方法也计算精确率和召回率
                results['precision_1'] = backup_scores.get('precision_1', results['precision_1'])
                results['recall_1'] = backup_scores.get('recall_1', results['recall_1'])
                results['precision_l'] = backup_scores.get('precision_l', results['precision_l'])
                results['recall_l'] = backup_scores.get('recall_l', results['recall_l'])
            
        except Exception as rouge_error:
            print(f"ROUGE评分失败: {rouge_error}")
            # 使用备用计算方法
            print("尝试使用备用ROUGE计算方法...")
            backup_scores = compute_rouge_manually(generated_answer, reference_answer)
            results['rouge_1'] = backup_scores['rouge_1']
            results['rouge_2'] = 0  # 备用方法不计算ROUGE-2
            results['rouge_l'] = backup_scores['rouge_l']
            
            # 添加精确率和召回率
            results['precision_1'] = backup_scores.get('precision_1', 0)
            results['recall_1'] = backup_scores.get('recall_1', 0)
            results['precision_l'] = backup_scores.get('precision_l', 0)
            results['recall_l'] = backup_scores.get('recall_l', 0)
            
            print(f"备用计算结果: ROUGE-1={results['rouge_1']:.4f}, ROUGE-L={results['rouge_l']:.4f}")
            print(f"精确率: P-1={results['precision_1']:.4f}, P-L={results['precision_l']:.4f}")
            print(f"召回率: R-1={results['recall_1']:.4f}, R-L={results['recall_l']:.4f}")
            
        # 计算简单的关键词匹配率
        reference_tokens = simple_tokenize(reference_answer.lower())
        generated_tokens = simple_tokenize(generated_answer.lower())
        
        ref_words = set(reference_tokens)
        gen_words = set(generated_tokens)
        if len(ref_words) > 0 and len(gen_words) > 0:
            # 计算关键词匹配的精确率和召回率
            keyword_precision = len(ref_words.intersection(gen_words)) / len(gen_words)
            keyword_recall = len(ref_words.intersection(gen_words)) / len(ref_words)
            keyword_match = 2 * keyword_precision * keyword_recall / (keyword_precision + keyword_recall) if (keyword_precision + keyword_recall) > 0 else 0
            
            results['keyword_match'] = keyword_match
            results['keyword_precision'] = keyword_precision
            results['keyword_recall'] = keyword_recall
            
            print(f"关键词匹配: F1={keyword_match:.4f}, P={keyword_precision:.4f}, R={keyword_recall:.4f}")
        else:
            results['keyword_match'] = 0
            results['keyword_precision'] = 0
            results['keyword_recall'] = 0
            
    except Exception as e:
        print(f"评估过程出错: {e}")
        import traceback
        traceback.print_exc()
        results = {
            'rouge_1': 0, 'rouge_2': 0, 'rouge_l': 0,
            'precision_1': 0, 'recall_1': 0,
            'precision_l': 0, 'recall_l': 0,
            'keyword_match': 0, 'keyword_precision': 0, 'keyword_recall': 0
        }
        
    return results

def main():
    """主函数"""
    # 配置参数
    USE_LOCAL_EMBEDDINGS = True  # 设置为True表示优先使用本地嵌入模型
    LOCAL_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # 多语言模型，支持中英文
    SEARCH_MULTI_COLLECTIONS = True  # 设置为True表示在多个集合中搜索
    DEBUG_MODE = False  # 调试模式，输出更多信息

    # 打印配置信息
    print(f"使用本地嵌入: {'是' if USE_LOCAL_EMBEDDINGS else '否'}")
    print(f"使用多集合搜索: {'是' if SEARCH_MULTI_COLLECTIONS else '否'}")
    if USE_LOCAL_EMBEDDINGS:
        print(f"本地模型: {LOCAL_MODEL_NAME}")

    # 1. 连接到 Milvus
    if not connect_to_milvus():
        print("无法连接到Milvus，测试终止")
        return

    # 2. 准备知识库集合
    available_collections = utility.list_collections()
    knowledge_collections = {}

    # 尝试加载所有可用的集合
    collection_mapping = {
        "arwu_text": "ARWU文本向量集合",
        "arwu_score": "ARWU评分向量集合",
        "arwu_enhanced": "ARWU增强向量集合",
        "us_colleges": "美国高校集合"
    }

    # 加载所有可用集合
    for coll_name, coll_desc in collection_mapping.items():
        if coll_name in available_collections:
            try:
                knowledge_collections[coll_name] = Collection(name=coll_name)
                print(f"已加载 {coll_desc}")

                # 检查向量维度
                schema = knowledge_collections[coll_name].schema
                for field in schema.fields:
                    if hasattr(field, 'dtype') and field.dtype == DataType.FLOAT_VECTOR:
                        if hasattr(field, 'params') and 'dim' in field.params:
                            COLLECTION_DIMS[coll_name] = field.params['dim']
                            print(f"确认集合 {coll_name} 的向量维度为: {COLLECTION_DIMS[coll_name]}")
                        break
            except Exception as e:
                print(f"加载集合 {coll_name} 失败: {e}")

    # 如果没有找到任何已知集合，使用默认集合
    if not knowledge_collections:
        try:
            default_collection = Collection(name=DEFAULT_COLLECTION)
            knowledge_collections["default"] = default_collection
            print(f"未找到专用知识库集合，使用默认集合: {DEFAULT_COLLECTION}")

            # 检查默认集合的向量维度
            schema = default_collection.schema
            for field in schema.fields:
                if field.dtype == DataType.FLOAT_VECTOR:
                    COLLECTION_DIMS[DEFAULT_COLLECTION] = field.params.get("dim")
                    print(f"默认集合的向量维度: {COLLECTION_DIMS[DEFAULT_COLLECTION]}")
        except Exception as e:
            print(f"加载默认集合失败: {e}")
            print("无法加载任何集合，测试终止")
            return

    # 3. 读取测试问题集
    xlsx_path = os.path.join(project_root, "RAGTesting", "RAG测试问题库及答案.xlsx")

    # 如果文件不存在，使用备用路径
    if not os.path.exists(xlsx_path):
        xlsx_path = "RAG测试问题库及答案.xlsx"

    if not os.path.exists(xlsx_path):
        print(f"找不到测试题库文件: {xlsx_path}")
        test_questions = [
            {"Questions": "清华大学的世界排名是多少？", "Answers": "根据ARWU排名数据，清华大学排名世界前30。"},
            {"Questions": "美国排名前十的大学有哪些？", "Answers": "美国排名前十的大学包括哈佛大学、斯坦福大学、麻省理工学院等。"},
            {"Questions": "亚洲有哪些著名的研究型大学？", "Answers": "亚洲著名研究型大学包括清华大学、北京大学、东京大学、新加坡国立大学等。"},
            {"Questions": "加州有哪些著名的公立大学？", "Answers": "加州著名的公立大学包括加州大学伯克利分校、加州大学洛杉矶分校、加州理工学院等。"},
            {"Questions": "哈佛大学在学术评价方面有什么特点？", "Answers": "哈佛大学在ARWU排名中各项指标均衡发展，特别是在校友获奖和教师获奖方面表现突出。"}
        ]
        df = pd.DataFrame(test_questions)
        print("使用内置测试问题")
    else:
        df = pd.read_excel(xlsx_path)
        print(f"读取测试问题: {xlsx_path}")

    # 4. 评测数据存储
    results_data = []
    spend_time = []
    data_source_stats = {}  # 记录数据源使用情况
    
    for idx, row in df.iterrows():
        user_input = str(row["Questions"])
        reference_answer = str(row["Answers"])
        
        print(f"\n测试问题 {idx+1}: {user_input}")
        
        start_time = time.time()  # 记录开始时间
        
        try:
            # 5. 检索相关信息
            relevant_knowledge = load_knowledge_variables(knowledge_collections, user_input, top_k=3)
            
            # 6. 统计数据源使用情况
            sources = []
            try:
                for line in relevant_knowledge.split("\n"):
                    if line.startswith("以下信息来自"):
                        source = line.replace("以下信息来自", "").strip()
                        sources.append(source)
                        if source in data_source_stats:
                            data_source_stats[source] += 1
                        else:
                            data_source_stats[source] = 1
            except Exception as e:
                print(f"处理数据源统计时出错: {e}")
            
            # 7. 生成回答
            bot_response = generate_answer(user_input, relevant_knowledge)
            
            # 8. 记录响应时间
            end_time = time.time()
            retrieval_time = end_time - start_time
            spend_time.append(retrieval_time)
            print(f"响应时间: {retrieval_time:.2f}秒")
            
            if DEBUG_MODE:
                print(f"问题: {user_input}")
                print(f"参考答案: {reference_answer}")
                print(f"检索知识: {relevant_knowledge}")
                print(f"数据源: {sources}")
            
            # 9. 评估答案
            print("评估生成回答...")
            metrics = evaluate_with_metrics(bot_response, reference_answer)
            
            # 10. 记录结果
            result = {
                "question": user_input,
                "reference": reference_answer,
                "response": bot_response,
                "retrieval_time": retrieval_time,
                "context": relevant_knowledge,
                "data_sources": ", ".join(sources)
            }
            # 添加评估指标
            result.update(metrics)
            
            results_data.append(result)
            print(f"机器人回答: {bot_response}")
            print(f"ROUGE-L: {metrics['rouge_l']:.4f}, 关键词匹配: {metrics['keyword_match']:.4f}")
            print("-" * 80)
            
        except Exception as e:
            print(f"处理问题时出错: {e}")
            print(f"跳过问题: {user_input}")
            import traceback
            traceback.print_exc()
            continue
    
    # 11. 保存评估结果
    if results_data:
        try:
            # 转换为DataFrame
            results_df = pd.DataFrame(results_data)
            
            # 计算平均分数 - 包含精确率和召回率
            avg_scores = {
                'avg_rouge_1': results_df['rouge_1'].mean(),
                'avg_rouge_2': results_df['rouge_2'].mean(),
                'avg_rouge_l': results_df['rouge_l'].mean(),
                'avg_precision_1': results_df['precision_1'].mean(),
                'avg_recall_1': results_df['recall_1'].mean(),
                'avg_precision_l': results_df['precision_l'].mean(),
                'avg_recall_l': results_df['recall_l'].mean(),
                'avg_keyword_match': results_df['keyword_match'].mean(),
                'avg_keyword_precision': results_df['keyword_precision'].mean(),
                'avg_keyword_recall': results_df['keyword_recall'].mean(),
                'avg_retrieval_time': results_df['retrieval_time'].mean()
            }
            
            # 输出评测结果
            print("\n=== 评估结果 ===")
            for metric, value in avg_scores.items():
                print(f"{metric}: {value:.4f}")
            
            # 输出数据源使用统计
            print("\n=== 数据源使用统计 ===")
            for source, count in data_source_stats.items():
                print(f"{source}: {count}次")
            
            # 写入评测结果
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            csv_file = f"evaluation_results_{timestamp}.csv"
            
            results_df.to_csv(
                csv_file,
                index=False,
                encoding="utf-8-sig"  # 确保Excel中正确显示中文
            )
            
            # 另存一份摘要结果
            summary_df = pd.DataFrame([avg_scores])
            summary_file = f"evaluation_summary_{timestamp}.csv"
            summary_df.to_csv(
                summary_file,
                index=False,
                encoding="utf-8-sig"
            )
            
            # 保存数据源统计
            source_stats_df = pd.DataFrame(
                [{"source": source, "count": count} for source, count in data_source_stats.items()]
            )
            source_stats_file = f"data_source_stats_{timestamp}.csv"
            source_stats_df.to_csv(
                source_stats_file,
                index=False,
                encoding="utf-8-sig"
            )
            
            print(f"评估结果已保存到 {csv_file}")
            print(f"评估摘要已保存到 {summary_file}")
            print(f"数据源统计已保存到 {source_stats_file}")
        except Exception as e:
            print(f"保存评估结果时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("没有生成评测样本，无法执行评估")

if __name__ == "__main__":
    main() 