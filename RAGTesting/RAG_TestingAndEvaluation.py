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
    "us_colleges": 384,
    "university_subjects": 384,
    "usnews2025_school_subject_relations_text": 384,
    "usnews2025_school_subject_relations": 14,
    "usnews2025_subjects": 384,
    "university_base": 384,
    "university_summary": 384,
    "the2025_subjects": 768,
    "university_statistics": 384,
    "the2025_basic_info": 768,
    "the2025_metrics": 10,
    "the2025_meta": 2,
    "university_indicators": 384,
    "usnews2025_schools": 384,
    "us_colleges_WIKI": 384,
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
def get_embedding(text, collection_name=None, model="all-MiniLM-L6-v2", use_api=False):
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
    """分析查询，返回查询类型和参数
    
    参数:
        query: 查询文本
        
    返回:
        包含查询类型和参数的字典
    """
    # 转换为小写以便更好地匹配
    query_lower = query.lower()
    
    # 初始化返回对象
    result = {
        "dataset": "unknown",  # 数据集类型
        "is_ranking_question": False,  # 是否是排名问题
        "is_subject_question": False,  # 是否是学科问题
        "is_metric_question": False,  # 是否是指标问题
        "rank_range": None,  # 排名范围
        "region": None,  # 地区
        "country": None,  # 国家
        "state": None,  # 州（针对美国大学）
        "control": None  # 控制类型（公立/私立）
    }
    
    # 检测数据集类型
    if any(term in query_lower for term in ["us college", "美国大学", "美国学院", "美国高校", "美国"]):
        result["dataset"] = "us_colleges"
    elif any(term in query_lower for term in ["arwu", "世界大学学术排名", "软科", "软科排名"]):
        result["dataset"] = "arwu"
    elif any(term in query_lower for term in ["the", "times higher education", "泰晤士", "the2025"]):
        result["dataset"] = "the2025"
    
    # 检测排名问题
    ranking_terms = ["排名", "rank", "名次", "位列", "列第", "位置", "多少名", "第几", "top", "榜单", "位次", "前"]
    if any(term in query_lower for term in ranking_terms):
        result["is_ranking_question"] = True
        
        # 提取排名范围
        rank_numbers = extract_ranking_numbers(query)
        if rank_numbers:
            # 如果有多个排名数字，假设可能是一个范围查询
            if len(rank_numbers) >= 2:
                rank_low = min(int(n) for n in rank_numbers)
                rank_high = max(int(n) for n in rank_numbers)
                result["rank_range"] = (rank_low, rank_high)
            else:
                # 单个排名数字，可能是"排名第X的学校"或"前X名学校"这样的查询
                rank = int(rank_numbers[0])
                
                # 检查是否是"前X名"类型的查询
                if any(pattern in query_lower for pattern in ["前" + str(rank), "top" + str(rank), "前" + ranking_number_to_chinese(rank)]):
                    # 对于"前X名"类型的查询，设置范围为1到X
                    result["rank_range"] = (1, rank)
                else:
                    # 对于其他类型，设置为具体排名
                    result["rank_range"] = (rank, rank)
    
    # 检测学科问题
    subject_terms = ["专业", "学科", "系", "科目", "faculty", "subject", "major", "department", "school of", "college of"]
    if any(term in query_lower for term in subject_terms):
        result["is_subject_question"] = True
    
    # 检测评价指标问题
    metric_terms = ["指标", "分数", "评分", "得分", "分项", "指数", "权重", "标准", "score", "metric"]
    if any(term in query_lower for term in metric_terms):
        result["is_metric_question"] = True
        
    # 检测特定地区或国家
    region_country_mapping = {
        "亚洲": "亚洲", "欧洲": "欧洲", "北美": "北美", "南美": "南美", 
        "非洲": "非洲", "大洋洲": "大洋洲", "asia": "亚洲", "europe": "欧洲", 
        "north america": "北美", "south america": "南美", "africa": "非洲", 
        "oceania": "大洋洲", "australia": "澳大利亚", "澳洲": "澳大利亚"
    }
    
    country_keywords = {
        "中国": "中国", "美国": "美国", "英国": "英国", "加拿大": "加拿大", 
        "日本": "日本", "澳大利亚": "澳大利亚", "德国": "德国", "法国": "法国", 
        "西班牙": "西班牙", "意大利": "意大利", "新加坡": "新加坡", "韩国": "韩国",
        "china": "中国", "usa": "美国", "united states": "美国", "uk": "英国", 
        "united kingdom": "英国", "canada": "加拿大", "japan": "日本", 
        "australia": "澳大利亚", "germany": "德国", "france": "法国", 
        "spain": "西班牙", "italy": "意大利", "singapore": "新加坡"
    }
    
    # 检测地区
    for keyword, region in region_country_mapping.items():
        if keyword in query_lower:
            result["region"] = region
            break
    
    # 检测国家
    for keyword, country in country_keywords.items():
        if keyword in query_lower:
            result["country"] = country
            break
    
    # 对于美国高校，检测州和控制类型
    if result["dataset"] == "us_colleges":
        # 州名检测 - 简单示例，实际应用中需要更全面的州名列表
        us_states = {
            "california": "California", "new york": "New York", "texas": "Texas",
            "florida": "Florida", "illinois": "Illinois", "pennsylvania": "Pennsylvania",
            "ohio": "Ohio", "massachusetts": "Massachusetts", "washington": "Washington",
            "密歇根": "Michigan", "加州": "California", "纽约": "New York", "德克萨斯": "Texas"
        }
        
        for keyword, state in us_states.items():
            if keyword in query_lower:
                result["state"] = state
                break
        
        # 控制类型检测
        if any(term in query_lower for term in ["public", "公立"]):
            result["control"] = "Public"
        elif any(term in query_lower for term in ["private", "私立"]):
            result["control"] = "Private"
    
    return result

def extract_ranking_numbers(text):
    """提取文本中的排名数字，包括阿拉伯数字和中文数字
    
    参数:
        text: 输入文本
    
    返回:
        提取到的排名数字列表（转换为阿拉伯数字字符串）
    """
    import re
    
    # 定义中文数字映射
    cn_num = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, 
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5, 
        '陆': 6, '柒': 7, '捌': 8, '玖': 9, '拾': 10
    }
    
    # 查找排名相关的表达
    rank_patterns = [
        # 阿拉伯数字模式
        r'排名[第为是在]?\s*(\d+)',
        r'第\s*(\d+)\s*名',
        r'名列第\s*(\d+)',
        r'位列第\s*(\d+)',
        r'排在第\s*(\d+)',
        r'名次[为是]?\s*(\d+)',
        r'全球[第位排]?\s*(\d+)',
        r'世界[第位排]?\s*(\d+)',
        r'[第位]\s*(\d+)\s*[位]',
        r'[位于在]?\s*(\d+)\s*[位名]',
        
        # 中文数字模式
        r'排名[第为是在]?\s*([零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+)',
        r'第\s*([零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+)\s*名',
        r'名列第\s*([零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+)',
        r'位列第\s*([零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+)',
        r'排在第\s*([零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+)',
        r'名次[为是]?\s*([零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+)',
        r'全球[第位排]?\s*([零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+)',
        r'世界[第位排]?\s*([零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+)',
        r'[第位]\s*([零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+)\s*[位]',
        r'[位于在]?\s*([零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+)\s*[位名]',
        
        # 添加"前N名"、"前N位"等模式
        r'前\s*(\d+)\s*名',
        r'前\s*(\d+)\s*位', 
        r'前\s*(\d+)\s*所',
        r'前\s*(\d+)\s*个',
        r'前\s*([零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+)\s*名',
        r'前\s*([零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+)\s*位',
        r'前\s*([零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+)\s*所',
        r'前\s*([零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+)\s*个',
        
        # 添加英文的"top N"、"first N"等模式
        r'top\s*(\d+)',
        r'first\s*(\d+)',
        r'best\s*(\d+)'
    ]
    
    # 独立的排名数字，如"THE2025排名1"
    standalone_patterns = [
        r'排名[^\d]*?(\d+)',
        r'rank[^\d]*?(\d+)',
        r'全球榜[^\d]*?(\d+)',
        r'世界榜[^\d]*?(\d+)'
    ]
    
    # 特殊词语转换为数字
    special_words = {
        "首位": "1", "冠军": "1", "榜首": "1", "第一": "1", 
        "亚军": "2", "季军": "3", "前五": "5", "前十": "10", 
        "前三": "3", "top10": "10", "top5": "5", "top3": "3", "top1": "1",
        "前二十": "20", "前三十": "30", "前五十": "50", "前一百": "100",
        "前20": "20", "前30": "30", "前50": "50", "前100": "100",
        "top20": "20", "top30": "30", "top50": "50", "top100": "100"
    }
    
    results = []
    
    # 查找排名表达式
    for pattern in rank_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            num_str = match.group(1)
            
            # 如果是中文数字，转换为阿拉伯数字
            if any(c in cn_num for c in num_str):
                # 处理简单的中文数字（目前仅支持个位数和"十"）
                if num_str == "十":
                    results.append("10")
                elif len(num_str) == 1 and num_str in cn_num:
                    results.append(str(cn_num[num_str]))
                elif len(num_str) == 2 and num_str[0] == "十":
                    # 处理"十X"的情况
                    if num_str[1] in cn_num:
                        results.append(str(10 + cn_num[num_str[1]]))
                elif len(num_str) == 2 and num_str[1] == "十":
                    # 处理"X十"的情况
                    if num_str[0] in cn_num:
                        results.append(str(cn_num[num_str[0]] * 10))
                else:
                    # 其他情况，假设是单个数字
                    for c in num_str:
                        if c in cn_num:
                            results.append(str(cn_num[c]))
            else:
                # 阿拉伯数字
                results.append(num_str)
    
    # 查找独立排名数字
    for pattern in standalone_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            results.append(match.group(1))
    
    # 查找特殊词语
    for word, num in special_words.items():
        if word.lower() in text.lower():
            results.append(num)
    
    # 最后尝试直接匹配1-100的数字作为可能的排名
    # 但仅在文本中提到"排名"、"第"、"名次"、"位"等关键词时
    if any(word in text for word in ["排名", "第", "名次", "位", "rank", "榜", "top", "前"]):
        # 查找独立的1-100数字
        for match in re.finditer(r'\b([1-9]|[1-9][0-9]|100)\b', text):
            results.append(match.group(1))
    
    # 去重并排序
    unique_results = sorted(list(set(results)), key=lambda x: int(x))
    
    print(f"提取到的排名数字: {unique_results}")
    return unique_results

def extract_keywords(query):
    """从查询中提取关键词
    
    参数:
        query: 查询文本
        
    返回:
        关键词列表
    """
    import re
    from collections import Counter
    import jieba
    
    # 停用词
    stopwords = {
        "的", "了", "和", "是", "在", "我", "有", "一个", "什么", "，", "。", "？", "！", 
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in", "for", "with", 
        "as", "at", "by", "about", "like", "that", "this", "it", "be", "or", "on", "from", 
        "你", "我们", "他们", "她们", "它们", "请问", "告诉", "我想", "知道", "查询", "想了解",
        "can", "could", "would", "should", "will", "shall", "may", "might", "must", 
        "have", "has", "had", "do", "does", "did", "which", "who", "whom", "whose", 
        "what", "where", "when", "why", "how", "请", "求"
    }
    
    # 提取关键词
    keywords = []
    
    # 首先提取排名关键词（这很重要）
    rank_numbers = extract_ranking_numbers(query)
    if rank_numbers:
        keywords.extend(rank_numbers)
    
    # 然后提取大学名称（这也很重要）
    university_names = extract_university_names(query)
    if university_names:
        keywords.extend(university_names)
    
    # 提取学科名称
    subject_names = extract_subject_names(query)
    if subject_names:
        keywords.extend(subject_names)
    
    # 对于英文，使用简单的分词
    english_words = re.findall(r'\b[a-zA-Z]+\b', query)
    english_words = [word.lower() for word in english_words if word.lower() not in stopwords and len(word) > 2]
    
    # 对于中文，使用jieba分词
    try:
        chinese_text = re.sub(r'[^\u4e00-\u9fff]+', ' ', query)  # 只保留中文字符
        if chinese_text.strip():
            chinese_words = list(jieba.cut(chinese_text))
            chinese_words = [word for word in chinese_words if word not in stopwords and len(word) > 1]
            # 提取2-3个字的词语，这些更可能是有意义的关键词
            for word in chinese_words:
                if 2 <= len(word) <= 3 and word not in keywords:
                    keywords.append(word)
    except:
        # 如果jieba分词失败，使用简单的按字符分词
        print("Jieba分词失败，使用简单分词")
        chinese_chars = list(chinese_text.replace(" ", ""))
        chinese_words = [''.join(chinese_chars[i:i+2]) for i in range(0, len(chinese_chars), 2) if i+1 < len(chinese_chars)]
    
    # 查找特殊关键词
    special_keywords = {
        "THE2025": ["THE2025", "THE", "2025", "泰晤士", "泰晤士高等教育"],
        "ARWU": ["ARWU", "世界大学学术排名", "软科"],
        "USNews": ["USNews", "US News", "美国新闻"],
        "排名": ["排名", "rank", "ranking", "位列", "名次"],
        "QS": ["QS", "QS世界大学排名"],
        "指标": ["指标", "评分", "分数", "score", "metric", "分项"]
    }
    
    for keyword_group, variations in special_keywords.items():
        for variation in variations:
            if variation.lower() in query.lower() and keyword_group not in keywords:
                keywords.append(keyword_group)
                break
    
    # 添加英文单词（限制数量，避免噪音）
    common_word_counter = Counter(english_words)
    important_english = [word for word, count in common_word_counter.most_common(5) if count >= 1 and word not in keywords]
    keywords.extend(important_english)
    
    # 附加学校相关关键词
    if "大学" in query or "university" in query.lower() or "college" in query.lower() or "校" in query:
        if "大学" not in keywords and "university" not in keywords:
            keywords.append("大学")
    
    # 去除重复，保留唯一关键词
    return list(dict.fromkeys(keywords))

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

                    # 添加位置信息
                    location_parts = []
                    if "country" in entity_data and entity_data["country"]:
                        location_parts.append(entity_data["country"])
                    if "country_en" in entity_data and entity_data["country_en"]:
                        location_parts.append(f"({entity_data['country_en']})")
                    if "continent" in entity_data and entity_data["continent"]:
                        location_parts.append(entity_data["continent"])

                    if location_parts:
                        info += f"位置: {', '.join(location_parts)}\n"

                    # 添加排名信息
                    if "rank" in entity_data:
                        try:
                            rank_value = int(entity_data["rank"])
                            info += f"排名: {rank_value}\n"
                        except (ValueError, TypeError):
                            info += f"排名: {entity_data['rank']}\n"

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


def normalize_similarity_score(score, metric_type):
    """归一化相似度评分到0-100范围

    参数:
        score: 原始相似度评分
        metric_type: 度量类型 (COSINE或L2)

    返回:
        0-100范围的归一化评分
    """
    if metric_type == "COSINE":
        # COSINE评分范围通常为[-1,1]，转换到[0,100]
        return min(100, max(0, (score + 1) * 50))
    elif metric_type == "L2":
        # L2距离越小越好，假设最大距离为10
        return min(100, max(0, 100 - score * 10))
    else:
        return min(100, max(0, score * 100))

# 添加中英文大学名称映射
UNIVERSITY_NAME_MAP = {
    '哈佛大学': 'Harvard University',
    '斯坦福大学': 'Stanford University',
    '麻省理工学院': 'Massachusetts Institute of Technology (MIT)',
    '剑桥大学': 'University of Cambridge',
    '加州大学': 'University of California',
    '新加坡国立大学': 'National University of Singapore (NUS)',
    '清华大学': 'Tsinghua University',
    '北京大学': 'Peking University',
    '复旦大学': 'Fudan University',
    '牛津大学': 'University of Oxford',
    '巴黎萨克雷大学': 'Paris-Saclay University',
    '巴黎第六大学': 'University of Paris VI'
}

def search_the2025_knowledge(collection, query, top_k=5, region=None, country=None):
    """从THE 2025排名数据中搜索相关信息

    参数:
        collection: Milvus集合对象
        query: 查询文本
        top_k: 返回的结果数量
        region: 可选的地区过滤条件
        country: 可选的国家过滤条件

    返回:
        包含检索结果的文本列表
    """
    # 确定集合类型和向量字段
    collection_type = ""
    vector_field = ""
    metric_type = ""

    # 根据集合名称确定类型、向量字段和度量方式
    if "the2025_subjects" in collection.name:
        collection_type = "subjects"
        vector_field = "subjects_vector"
        metric_type = "COSINE"  # 文本向量使用余弦相似度
    elif "the2025_basic_info" in collection.name:
        collection_type = "basic_info"
        vector_field = "basic_info_vector"
        metric_type = "COSINE"  # 文本向量使用余弦相似度
    elif "the2025_metrics" in collection.name:
        collection_type = "metrics"
        vector_field = "metrics_vector"
        metric_type = "L2"  # 量化指标向量使用欧氏距离
    elif "the2025_meta" in collection.name:
        collection_type = "meta"
        vector_field = "dummy_vector"
        metric_type = "COSINE"  # 默认使用余弦相似度
    else:
        print(f"未知的THE2025集合类型: {collection.name}")
        vector_field = "subjects_vector"  # 默认向量字段
        metric_type = "COSINE"  # 默认使用余弦相似度

    # 获取查询向量 - 为特定集合生成正确维度的向量
    query_embedding = get_embedding(query, collection_name=collection.name)

    # 获取集合架构信息和可用字段
    field_names = []
    has_entity_field = True

    # 尝试获取所有字段名
    try:
        for field in collection.schema.fields:
            if field.name != vector_field and field.name != "id":  # 跳过向量字段和ID字段
                field_names.append(field.name)
        print(f"获取到字段: {field_names}")
    except Exception as e:
        print(f"获取字段名失败: {e}")
        # 尝试通过查询获取一条记录来推断字段名
        try:
            sample = collection.query(expr="id >= 0", limit=1)
            if sample and len(sample) > 0:
                field_names = [field for field in sample[0].keys() if field != "id" and field != vector_field]
                print(f"通过查询样本获取到字段: {field_names}")
            else:
                print("查询样本返回空结果")
        except Exception as e2:
            print(f"获取样本查询失败: {e2}")
            # 手动指定常见字段
            if "the2025_basic_info" in collection.name:
                field_names = ["name", "rank", "location", "overall_score", "teaching_score",
                              "research_score", "citations_score", "industry_income_score",
                              "international_outlook_score"]
            elif "the2025_subjects" in collection.name:
                field_names = ["name", "rank", "location", "subjects", "subjects_count",
                              "top_subjects", "has_computer_science", "has_engineering", "has_medicine"]
            elif "the2025_metrics" in collection.name:
                field_names = ["name", "rank", "location", "overall_score", "teaching_score",
                              "research_score", "citations_score", "industry_income_score",
                              "international_outlook_score", "student_staff_ratio", "pc_intl_students",
                              "number_students"]
            elif "the2025_meta" in collection.name:
                field_names = ["name", "rank", "location", "overall_score", "json_data"]
            print(f"使用预定义字段列表: {field_names}")

    # 构建过滤表达式
    filter_expr = ""

    # 如果指定了地区过滤
    if region:
        if isinstance(region, list):
            region_conditions = [f"location like '%{r}%'" for r in region]
            filter_expr = " || ".join(region_conditions)
        else:
            filter_expr = f"location like '%{region}%'"

    # 如果指定了国家过滤
    if country:
        country_expr = ""
        if isinstance(country, list):
            country_conditions = [f"location like '%{c}%'" for c in country]
            country_expr = " || ".join(country_conditions)
        else:
            country_expr = f"location like '%{country}%'"

        if filter_expr:
            filter_expr = f"({filter_expr}) && ({country_expr})"
        else:
            filter_expr = country_expr

    # 设置搜索参数
    search_params = {
        "metric_type": metric_type
    }

    if metric_type == "COSINE":
        search_params["params"] = {"ef": 100}
    else:  # L2
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
            if "main" in partition_names:  # THE2025数据的主分区名称
                search_kwargs["partition_names"] = ["main"]
        except Exception as e:
            print(f"获取分区信息失败: {e}")

        # 执行搜索
        print(f"执行搜索，参数: {search_kwargs}")
        results = collection.search(**search_kwargs)

        # 处理结果
        retrieved_texts = []
        for hits in results[0]:
            try:
                # 基本信息
                info = f"搜索结果 ID: {hits.id}\n"
                print(f"处理结果 ID: {hits.id}")

                # 检查hit对象中可用的属性
                hit_attrs = [attr for attr in dir(hits) if not attr.startswith('_')]
                entity_data = {}

                # 首先尝试从entity获取数据
                if hasattr(hits, 'entity') and hits.entity:
                    print(f"结果有entity属性")
                    try:
                        # 尝试查看entity属性
                        entity_attrs = [attr for attr in dir(hits.entity) if not attr.startswith('_')]
                        print(f"Entity属性: {entity_attrs}")

                        # 获取数据的方法有多种，尝试不同的方式
                        for field in field_names:
                            if field != 'id' and not field.endswith('_vector'):  # 跳过ID和向量字段
                                try:
                                    # 方法1: 使用get方法
                                    if hasattr(hits.entity, 'get'):
                                        entity_data[field] = hits.entity.get(field, "")
                                    # 方法2: 直接属性访问
                                    elif hasattr(hits.entity, field):
                                        entity_data[field] = getattr(hits.entity, field, "")
                                    # 方法3: 字典访问
                                    elif isinstance(hits.entity, dict) and field in hits.entity:
                                        entity_data[field] = hits.entity[field]
                                except Exception as field_error:
                                    print(f"获取字段 {field} 失败: {field_error}")
                                    entity_data[field] = ""
                    except Exception as entity_error:
                        print(f"处理entity数据时出错: {entity_error}")

                # 如果entity获取失败，尝试直接从hits获取数据
                if not entity_data and hasattr(hits, 'get'):
                    print(f"尝试直接从hits获取数据")
                    for field in field_names:
                        if field != 'id' and not field.endswith('_vector'):
                            try:
                                entity_data[field] = hits.get(field, "")
                            except:
                                pass

                # 如果无法从entity获取，尝试直接从搜索结果对象获取
                if not entity_data:
                    print(f"尝试从hits直接获取属性")
                    for field in field_names:
                        if hasattr(hits, field):
                            try:
                                entity_data[field] = getattr(hits, field)
                            except:
                                pass

                # 如果仍然无法获取数据，尝试单独查询
                if not entity_data:
                    print(f"尝试单独查询ID={hits.id}的数据")
                    try:
                        query_results = collection.query(
                            expr=f"id == {hits.id}",
                            output_fields=field_names
                        )
                        if query_results and len(query_results) > 0:
                            entity_data = query_results[0]
                            print(f"单独查询成功，找到数据: {list(entity_data.keys())}")
                    except Exception as query_error:
                        print(f"单独查询失败: {query_error}")

                # 现在处理获取到的数据
                print(f"获取到字段: {list(entity_data.keys())}")

                # 输出大学名称
                if "name" in entity_data and entity_data["name"]:
                    university_name = entity_data.get('name', '')
                    info += f"大学: {university_name}"
                    # 添加英文名称映射（如果存在）
                    if university_name in UNIVERSITY_NAME_MAP:
                        info += f" ({UNIVERSITY_NAME_MAP[university_name]})"
                    info += "\n"

                # 输出位置信息
                if "location" in entity_data and entity_data["location"]:
                    info += f"位置: {entity_data.get('location', '')}\n"

                # 输出排名信息
                if "rank" in entity_data and entity_data["rank"]:
                    try:
                        rank_value = int(entity_data["rank"])
                        info += f"THE 2025排名: {rank_value}\n"
                    except (ValueError, TypeError):
                        info += f"THE 2025排名: {entity_data['rank']}\n"

                # 根据集合类型添加特定信息
                if collection_type == "basic_info":
                    score_fields = [
                        ("overall_score", "总分"),
                        ("teaching_score", "教学得分"),
                        ("research_score", "研究得分"),
                        ("citations_score", "引用得分"),
                        ("industry_income_score", "产业收入得分"),
                        ("international_outlook_score", "国际视野得分")
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
                        info += "评分数据:\n"
                        for score in score_info:
                            info += f"  - {score}\n"

                elif collection_type == "metrics":
                    # 显示详细指标信息
                    metric_fields = [
                        ("overall_score", "总分"),
                        ("teaching_score", "教学得分"),
                        ("research_score", "研究得分"),
                        ("citations_score", "引用得分"),
                        ("industry_income_score", "产业收入得分"),
                        ("international_outlook_score", "国际视野得分"),
                        ("student_staff_ratio", "师生比"),
                        ("pc_intl_students", "国际学生百分比")
                    ]

                    metrics_info = []
                    for field, label in metric_fields:
                        if field in entity_data and entity_data[field]:
                            try:
                                value = float(entity_data[field])
                                metrics_info.append(f"{label}: {value:.1f}")
                            except:
                                metrics_info.append(f"{label}: {entity_data[field]}")

                    if metrics_info:
                        info += "指标数据:\n"
                        for metric in metrics_info:
                            info += f"  - {metric}\n"

                    # 添加学生数量信息
                    if "number_students" in entity_data and entity_data["number_students"]:
                        try:
                            num = int(entity_data["number_students"])
                            info += f"在校学生: {num:,}人\n"
                        except:
                            info += f"在校学生: {entity_data['number_students']}\n"

                elif collection_type == "subjects":
                    # 显示学科信息
                    if "subjects" in entity_data:
                        if isinstance(entity_data["subjects"], list):
                            subjects = entity_data["subjects"]
                            if subjects:
                                info += f"提供学科: {', '.join(subjects[:5])}"
                                if len(subjects) > 5:
                                    info += f" 等 {len(subjects)} 个学科"
                                info += "\n"
                        elif isinstance(entity_data["subjects"], str):
                            try:
                                import json
                                subjects = json.loads(entity_data["subjects"])
                                if isinstance(subjects, list) and subjects:
                                    info += f"提供学科: {', '.join(subjects[:5])}"
                                    if len(subjects) > 5:
                                        info += f" 等 {len(subjects)} 个学科"
                                    info += "\n"
                            except:
                                # 如果无法解析JSON，直接使用原始字符串
                                info += f"提供学科: {entity_data['subjects'][:200]}\n"

                    # 添加top_subjects字段的处理
                    if "top_subjects" in entity_data and entity_data["top_subjects"]:
                        info += f"主要学科: {entity_data['top_subjects']}\n"

                    # 处理subjects_count字段（如果存在）
                    if "subjects_count" in entity_data and entity_data["subjects_count"]:
                        info += f"学科数量: {entity_data['subjects_count']}\n"

                    # 添加是否有特定学科领域的信息
                    special_fields = [
                        ("has_computer_science", "是否有计算机科学"),
                        ("has_engineering", "是否有工程学"),
                        ("has_medicine", "是否有医学")
                    ]
                    for field, label in special_fields:
                        if field in entity_data:
                            value = entity_data[field]
                            if isinstance(value, bool):
                                info += f"{label}: {'是' if value else '否'}\n"
                            else:
                                info += f"{label}: {value}\n"

                elif collection_type == "meta":
                    # 处理元数据集合的特殊字段
                    if "json_data" in entity_data and entity_data["json_data"]:
                        try:
                            import json
                            data_json = json.loads(entity_data["json_data"])
                            info += "元数据:\n"

                            # 显示前几个键值对
                            keys = list(data_json.keys())[:5]
                            for key in keys:
                                info += f"  - {key}: {data_json[key]}\n"
                            if len(data_json) > 5:
                                info += f"  - ... 共有 {len(data_json)} 个数据项\n"
                        except:
                            # 如果无法解析JSON
                            data_text = entity_data["json_data"]
                            if len(data_text) > 200:
                                data_text = data_text[:200] + "..."
                            info += f"元数据: {data_text}\n"

                    # 添加各项评分信息
                    score_fields = [
                        ("overall_score", "总分"),
                        ("teaching_score", "教学得分"),
                        ("research_score", "研究得分"),
                        ("citations_score", "引用得分"),
                        ("industry_income_score", "产业收入得分"),
                        ("international_outlook_score", "国际视野得分"),
                        ("student_staff_ratio", "学生/教师比例"),
                        ("pc_intl_students", "国际学生百分比")
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
                        info += "THE 2025评分数据:\n"
                        for score in score_info:
                            info += f"  - {score}\n"

                # 添加归一化相似度信息
                normalized_score = normalize_similarity_score(hits.distance, metric_type)
                info += f"相似度得分: {normalized_score:.1f}/100 (原始得分: {hits.distance:.4f})\n"
                retrieved_texts.append(info)
            except Exception as e:
                print(f"处理THE2025搜索结果时出错: {e}")
                import traceback
                traceback.print_exc()
                # 即使出错也尝试添加一些基本信息
                retrieved_texts.append(f"搜索结果 ID: {hits.id}\n相似度得分: {normalize_similarity_score(hits.distance, metric_type):.1f}/100 (原始得分: {hits.distance:.4f})\n处理详情时出错: {e}")
                continue

        return retrieved_texts
    except Exception as e:
        print(f"THE2025查询出错: {e}")
        import traceback
        traceback.print_exc()
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
        print(f"执行关键词搜索: {collection.name}")

        # 提取关键词 - 改进关键词提取逻辑
        # 先尝试提取大学名称
        university_names = extract_university_names(query)

        # 再提取一般关键词
        general_keywords = extract_keywords(query)

        # 合并关键词并去重
        keywords = []
        if university_names:
            keywords.extend(university_names)
        if general_keywords:
            for keyword in general_keywords:
                if keyword not in keywords and len(keyword) > 0:  # 忽略过短的关键词
                    keywords.append(keyword)

        # 如果最终没有有效关键词，使用原始查询中最长的词
        if not keywords:
            # 基础分词，按空格和标点符号分割
            import re
            simple_tokens = re.findall(r'\w+', query)
            # 选择长度大于2的词
            simple_tokens = [t for t in simple_tokens if len(t) > 2]
            if simple_tokens:
                # 按长度排序并取最长的几个词
                simple_tokens.sort(key=len, reverse=True)
                keywords = simple_tokens[:3]  # 最多取3个最长的词

        print(f"提取的关键词: {keywords}")

        # 如果没有提取到有效关键词，直接返回空结果
        if not keywords:
            print("未提取到有效关键词，无法执行关键词搜索")
            return []

        # 确定要搜索的字段 - 针对不同集合类型优化
        search_fields = []

        # 根据集合名推断可能的字段
        if "the2025" in collection.name:
            # THE2025集合优先搜索这些字段
            primary_fields = ["name", "rank"]
            secondary_fields = ["location", "rank"]
        elif "arwu" in collection.name:
            # ARWU集合优先搜索这些字段
            primary_fields = ["university", "university_en"]
            secondary_fields = ["country", "country_en"]
        elif "us_" in collection.name or "us_colleges" in collection.name:
            # 美国高校集合优先搜索这些字段 - 更新为匹配导入文件中的字段
            primary_fields = ["name", "url"]
            secondary_fields = ["url"]
        else:
            # 默认字段列表
            primary_fields = ["name", "university"]
            secondary_fields = ["location", "country", "description"]

        # 尝试获取集合中实际存在的字段
        actual_fields = []
        try:
            # 通过schema获取字段
            for field in collection.schema.fields:
                if field.dtype == DataType.VARCHAR:
                    actual_fields.append(field.name)
        except Exception as schema_error:
            print(f"获取schema字段失败: {schema_error}")
            # 尝试通过查询样本获取字段
            try:
                sample = collection.query(expr="id >= 0", limit=1)
                if sample and len(sample) > 0:
                    actual_fields = [field for field in sample[0].keys() if field != "id"]
            except Exception as query_error:
                print(f"获取样本字段失败: {query_error}")

        # 按优先级确定搜索字段
        for field in primary_fields:
            if field in actual_fields:
                search_fields.append(field)

        # 如果主要字段没有找到，添加次要字段
        if not search_fields:
            for field in secondary_fields:
                if field in actual_fields:
                    search_fields.append(field)

        # 如果仍然没有找到适合的字段，尝试使用非数值型字段
        if not search_fields and actual_fields:
            search_fields = [f for f in actual_fields if f != "id" and not f.endswith("_vector")]

        print(f"将在以下字段中搜索关键词: {search_fields}")

        # 如果没有有效的搜索字段，返回空结果
        if not search_fields:
            print("未找到适合的搜索字段")
            return []

        # 特殊处理清华大学关键词 - 因为可能只能通过ID搜索到
        if any(k.lower() in ["清华", "tsinghua", "清华大学"] for k in keywords):
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

        # 通过生成LIKE表达式进行查询 - 针对Milvus的查询语法优化
        all_results = []

        # 处理大学名称关键词 - 通常是最重要的
        uni_results = []
        if university_names:
            # 首先尝试完整的大学名称匹配
            for uni_name in university_names:
                for field in search_fields:
                    try:
                        # 完全匹配
                        expr = f"{field} == '{uni_name}'"
                        results = collection.query(expr=expr, output_fields=["*"], limit=top_k)
                        if results:
                            print(f"完全匹配找到 {len(results)} 条记录，关键词: '{uni_name}', 字段: {field}")
                            uni_results.extend(results)

                        # 部分匹配
                        expr = f"{field} like '%{uni_name}%'"
                        results = collection.query(expr=expr, output_fields=["*"], limit=top_k)
                        if results:
                            print(f"部分匹配找到 {len(results)} 条记录，关键词: '{uni_name}', 字段: {field}")
                            for res in results:
                                if res not in uni_results:  # 避免重复
                                    uni_results.append(res)
                    except Exception as e:
                        print(f"大学名称查询失败: {e}")

            # 如果找到结果，添加到总结果中
            if uni_results:
                all_results.extend(uni_results)

        # 如果大学名称搜索未找到结果，尝试使用一般关键词
        if not all_results:
            for keyword in keywords:
                if len(keyword) < 3:
                    continue  # 跳过过短的关键词

                for field in search_fields:
                    try:
                        # 尝试LIKE查询
                        expr = f"{field} like '%{keyword}%'"
                        results = collection.query(expr=expr, output_fields=["*"], limit=top_k)

                        if results:
                            print(f"关键词查询找到 {len(results)} 条记录，关键词: '{keyword}', 字段: {field}")
                            for res in results:
                                if res not in all_results:  # 避免重复
                                    all_results.append(res)
                                    if len(all_results) >= top_k:
                                        break

                            if len(all_results) >= top_k:
                                break
                    except Exception as e:
                        print(f"关键词查询失败: {e}")

                if len(all_results) >= top_k:
                    break

        # 如果找到结果，返回格式化后的结果
        if all_results:
            # 截取前top_k个结果
            final_results = all_results[:top_k]
            return format_search_results(final_results, keywords, collection.name)

        # 如果上面的方法失败，尝试获取所有记录并在Python中筛选
        # 适用于小型数据集，生产环境需要更高效的方法
        try:
            print("尝试获取有限数量的记录并在内存中过滤")
            # 修正查询表达式，使用有效的条件
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
        elif "the2025" in collection_name:
            collection_label = "THE2025集合"
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
            
            # 处理内容字段 - 截取内容摘要
            if "content" in record and record["content"]:
                content = record.get("content", "")
                # 截取前200个字符作为摘要
                summary = content[:200] + "..." if len(content) > 200 else content
                info += f"摘要: {summary}\n"
            
            # 显示来源URL
            if "url" in record and record["url"]:
                info += f"来源: {record.get('url', '')}\n"
            
            # 以下字段在当前导入中不存在，已移除处理逻辑
            # location_parts = []
            # if "location" in record and record["location"]:
            #     location_parts.append(record["location"])
            # if "state" in record and record["state"]:
            #     location_parts.append(record["state"])
            # if "region" in record and record["region"]:
            #     location_parts.append(record["region"])
            # 
            # if location_parts:
            #     info += f"位置: {', '.join(location_parts)}\n"
            # 
            # type_control_parts = []
            # if "type" in record and record["type"]:
            #     type_control_parts.append(f"类型: {record['type']}")
            # if "control" in record and record["control"]:
            #     type_control_parts.append(f"控制方式: {record['control']}")
            # 
            # if type_control_parts:
            #     info += f"{', '.join(type_control_parts)}\n"
            # 
            # if "enrollment" in record and record["enrollment"]:
            #     info += f"在校学生: {record['enrollment']}人\n"

        elif "the2025" in collection_name:
            # THE2025相关集合
            if "name" in record:
                uni_name = record.get("name", "")
                info += f"大学: {uni_name}"
                # 添加英文名称映射（如果存在）
                if uni_name in UNIVERSITY_NAME_MAP:
                    info += f" ({UNIVERSITY_NAME_MAP[uni_name]})"
                info += "\n"
            
            if "location" in record:
                info += f"位置: {record.get('location', '')}\n"
            
            if "rank" in record:
                info += f"THE2025排名: {record.get('rank', 'N/A')}\n"
            
            # 添加分数信息
            the_score_fields = [
                ("overall_score", "总分"),
                ("teaching_score", "教学得分"),
                ("research_score", "研究得分"),
                ("citations_score", "引用得分"),
                ("industry_income_score", "产业收入得分"),
                ("international_outlook_score", "国际视野得分")
            ]
            
            score_info = []
            for field, label in the_score_fields:
                if field in record and record[field]:
                    try:
                        value = float(record[field])
                        score_info.append(f"{label}: {value:.1f}")
                    except:
                        score_info.append(f"{label}: {record[field]}")
            
            if score_info:
                info += "评分数据:\n" + ", ".join(score_info) + "\n"

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

def extract_subject_names(text):
    """从文本中提取学科名称

    参数:
        text: 要分析的文本

    返回:
        提取到的学科名称列表
    """
    common_subjects = [
        "计算机科学", "计算机", "人工智能", "机器学习", "数据科学",
        "物理", "物理学", "化学", "生物", "生物学", "医学",
        "数学", "统计学", "工程", "电子工程", "机械工程",
        "经济学", "金融", "商业", "管理", "法学", "法律",
        "文学", "历史", "哲学", "艺术", "音乐", "教育学",
        "社会科学", "心理学", "信息技术", "通信工程", "土木工程",
        "环境科学", "材料科学", "航空航天", "能源科学", "地球科学"
    ]

    found_subjects = []
    for subject in common_subjects:
        if subject in text:
            found_subjects.append(subject)

    return found_subjects

# 修改load_knowledge_variables函数，优先使用关键词搜索


def search_us_colleges_WIKI_knowledge(collection, query, top_k=5, state=None, region=None, control=None):
    """从美国大学维基百科数据中搜索相关信息
    
    参数:
        collection: Milvus集合对象
        query: 查询文本
        top_k: 返回的结果数量
        state: 可选的州过滤条件
        region: 可选的地区过滤条件
        control: 可选的学校控制类型过滤条件(如公立、私立等)
    
    返回:
        包含检索结果的文本列表
    """
    # 确定向量字段和度量方式
    vector_field = "embedding"
    metric_type = "L2"  # 使用L2距离度量，与导入设置一致
    
    # 获取查询向量
    query_embedding = get_embedding(query, collection_name=collection.name)
    
    # 获取集合架构信息和可用字段
    field_names = ["name", "content", "url"]  # 根据导入脚本中定义的字段
    
    # 构建过滤表达式
    filter_expr = ""
    
    # 如果指定了州过滤
    if state:
        if isinstance(state, list):
            state_conditions = [f"content like '%{s}%'" for s in state]
            filter_expr = " || ".join(state_conditions)
        else:
            filter_expr = f"content like '%{state}%'"
    
    # 如果指定了地区过滤
    if region:
        region_expr = ""
        if isinstance(region, list):
            region_conditions = [f"content like '%{r}%'" for r in region]
            region_expr = " || ".join(region_conditions)
        else:
            region_expr = f"content like '%{region}%'"
        
        if filter_expr:
            filter_expr = f"({filter_expr}) && ({region_expr})"
        else:
            filter_expr = region_expr
    
    # 如果指定了控制类型过滤
    if control:
        control_expr = ""
        if isinstance(control, list):
            control_conditions = [f"content like '%{c}%'" for c in control]
            control_expr = " || ".join(control_conditions)
        else:
            control_expr = f"content like '%{control}%'"
        
        if filter_expr:
            filter_expr = f"({filter_expr}) && ({control_expr})"
        else:
            filter_expr = control_expr
    
    # 设置搜索参数
    search_params = {
        "metric_type": metric_type,
        "params": {"M": 8, "efSearch": 64}  # 与导入脚本中的HNSW索引参数相匹配
    }
    
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
        
        # 添加输出字段
        search_kwargs["output_fields"] = field_names
        
        # 执行搜索
        print(f"执行US Colleges维基百科知识搜索，参数: {search_kwargs}")
        results = collection.search(**search_kwargs)
        
        # 处理结果
        retrieved_texts = []
        for hits in results[0]:
            try:
                # 基本信息
                info = f"搜索结果 ID: {hits.id}\n"
                
                # 从实体中获取数据
                if hasattr(hits, 'entity'):
                    # 学校名称
                    if hasattr(hits.entity, 'name'):
                        school_name = hits.entity.name
                        info += f"学校: {school_name}\n"
                    else:
                        school_name = "未知学校"
                    
                    # 内容摘要 - 提取更智能的摘要
                    if hasattr(hits.entity, 'content'):
                        content = hits.entity.content
                        
                        # 从内容中提取关键信息
                        location_info = extract_location_from_content(content)
                        if location_info:
                            info += f"位置: {location_info}\n"
                        
                        # 提取学校类型信息
                        type_info = extract_school_type_from_content(content)
                        if type_info:
                            info += f"类型: {type_info}\n"
                        
                        # 提取创建年份
                        founded_year = extract_founded_year_from_content(content)
                        if founded_year:
                            info += f"创立年份: {founded_year}\n"
                        
                        # 提取在校学生人数
                        enrollment = extract_enrollment_from_content(content)
                        if enrollment:
                            info += f"在校学生: {enrollment}\n"
                        
                        # 智能摘要 - 尝试找到最相关的段落而不是简单截取
                        # 1. 先按句子分割
                        import re
                        sentences = re.split(r'(?<=[.!?])\s+', content)
                        
                        # 2. 选择相关度最高的几个句子
                        selected_sentences = []
                        
                        # 优先选择包含学校名称的句子
                        for sentence in sentences[:10]:  # 只考虑前10个句子，通常包含最重要信息
                            if school_name in sentence:
                                selected_sentences.append(sentence)
                                if len(selected_sentences) >= 2:  # 最多选2个句子
                                    break
                        
                        # 如果没找到包含学校名称的句子，取前几句
                        if not selected_sentences and sentences:
                            selected_sentences = sentences[:2]
                        
                        summary = " ".join(selected_sentences)
                        
                        # 如果摘要仍然太长，截断
                        if len(summary) > 300:
                            summary = summary[:300] + "..."
                        
                        info += f"摘要: {summary}\n"
                    
                    # 网址
                    if hasattr(hits.entity, 'url'):
                        info += f"来源: {hits.entity.url}\n"
                
                # 添加相似度信息
                normalized_score = normalize_similarity_score(hits.distance, metric_type)
                info += f"相似度: {normalized_score:.2f}% (原始距离: {hits.distance:.4f})\n"
                
                retrieved_texts.append(info)
            except Exception as e:
                print(f"处理搜索结果时出错: {e}")
                retrieved_texts.append(f"搜索结果 ID: {hits.id} (处理详情时出错: {e})")
                continue
        
        return retrieved_texts
    except Exception as e:
        print(f"US Colleges维基百科知识查询出错: {e}")
        return []

# 辅助函数：从内容中提取位置信息
def extract_location_from_content(content):
    """从内容中提取位置信息"""
    import re
    
    # 首先尝试匹配常见的位置表达模式
    location_patterns = [
        r'located in ([^\.]+)',
        r'campus(es)? (is|are) in ([^\.]+)',
        r'based in ([^\.]+)',
        r'university in ([^\.]+)',
        r'college in ([^\.]+)',
    ]
    
    for pattern in location_patterns:
        matches = re.search(pattern, content, re.IGNORECASE)
        if matches:
            return matches.group(1).strip()
    
    # 尝试匹配州名
    us_states = [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", 
        "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", 
        "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", 
        "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", 
        "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", 
        "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", 
        "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", 
        "Wisconsin", "Wyoming"
    ]
    
    for state in us_states:
        if state in content:
            # 尝试找到包含州名的上下文
            sentences = re.split(r'(?<=[.!?])\s+', content)
            for sentence in sentences:
                if state in sentence:
                    # 截取包含州名的短语
                    words = sentence.split()
                    for i, word in enumerate(words):
                        if state in word:
                            # 尝试获取州名周围的上下文
                            start = max(0, i-3)
                            end = min(len(words), i+4)
                            return " ".join(words[start:end])
            
            return state
    
    return None

# 辅助函数：从内容中提取学校类型信息
def extract_school_type_from_content(content):
    """从内容中提取学校类型信息"""
    import re
    
    # 检查是否私立或公立
    private_match = re.search(r'(private|私立)(\s+(university|college|institution))?', content, re.IGNORECASE)
    public_match = re.search(r'(public|公立)(\s+(university|college|institution))?', content, re.IGNORECASE)
    
    if private_match:
        return "私立"
    elif public_match:
        return "公立"
    
    return None

# 辅助函数：从内容中提取创办年份
def extract_founded_year_from_content(content):
    """从内容中提取创办年份"""
    import re
    
    # 匹配常见的创立年份表达
    founded_patterns = [
        r'founded in (\d{4})',
        r'established in (\d{4})',
        r'创立于(\d{4})',
        r'始建于(\d{4})',
    ]
    
    for pattern in founded_patterns:
        matches = re.search(pattern, content, re.IGNORECASE)
        if matches:
            return matches.group(1)
    
    # 尝试匹配任何四位数年份（1600-2023年间）
    year_matches = re.findall(r'\b(1[6-9]\d{2}|20[0-2]\d)\b', content)
    if year_matches:
        # 简单假设：如果找到了年份，第一个年份很可能是创建年份
        return year_matches[0]
    
    return None

# 辅助函数：从内容中提取在校学生人数
def extract_enrollment_from_content(content):
    """从内容中提取在校学生人数"""
    import re
    
    # 匹配常见的学生人数表达
    enrollment_patterns = [
        r'enrollment of ([\d,]+)',
        r'approximately ([\d,]+) students',
        r'about ([\d,]+) students',
        r'student body of ([\d,]+)',
        r'student population of ([\d,]+)',
        r'学生人数(约|为)?([\d,]+)',
        r'在校生(约|为)?([\d,]+)',
    ]
    
    for pattern in enrollment_patterns:
        matches = re.search(pattern, content, re.IGNORECASE)
        if matches:
            try:
                # 移除逗号并尝试转换为整数
                enrollment_str = matches.group(1).replace(',', '')
                return f"{int(enrollment_str):,}人"
            except:
                return matches.group(1)
    
    return None

def load_knowledge_variables(collections, query, top_k=5, use_keyword_search=True, use_multi_collections=True, max_collections=3):
    """加载知识变量，根据查询检索相关信息

    参数:
        collections: 集合字典或列表
        query: 查询文本
        top_k: 返回结果数量
        use_keyword_search: 是否优先使用关键词搜索
        use_multi_collections: 是否在多个集合中搜索
        max_collections: 最多查询的集合数量，默认为3
    """
    try:
        # 分析查询类型
        query_analysis = analyze_query(query)
        print(f"查询分析: {query_analysis}")
        
        # 调整top_k值，根据rank_range处理"前N名"类型的查询
        adjusted_top_k = top_k
        if query_analysis["is_ranking_question"] and query_analysis["rank_range"]:
            # 如果是类似"前N名学校有哪些"的查询
            if query_analysis["rank_range"][0] == 1:  # 从1开始的范围
                # 使用范围的上限作为top_k值，例如"前10名"会返回10个结果
                adjusted_top_k = max(top_k, query_analysis["rank_range"][1])
                print(f"检测到'前{query_analysis['rank_range'][1]}名'类型的查询，调整返回结果数量为: {adjusted_top_k}")
            # 如果是具体排名区间查询
            elif query_analysis["rank_range"][0] != query_analysis["rank_range"][1]:
                # 计算区间大小，确保返回足够的结果
                range_size = query_analysis["rank_range"][1] - query_analysis["rank_range"][0] + 1
                adjusted_top_k = max(top_k, range_size)
                print(f"检测到排名区间查询 {query_analysis['rank_range']}，调整返回结果数量为: {adjusted_top_k}")

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

            # 提取查询中提到的大学名称
            university_names = extract_university_names(query)
            print(f"提取到的大学名称: {university_names}")

            # 提取学科名称 (使用新增的函数)
            subject_names = extract_subject_names(query)
            if subject_names:
                print(f"提取到的学科名称: {subject_names}")

            # 优先处理与查询类型匹配的集合
            if dataset_type == "us_colleges":
                us_collections = ["us_colleges"]
                for coll_name in us_collections:
                    if coll_name in collections:
                        priority_collections.append((coll_name, collections[coll_name]))
            elif dataset_type == "arwu":
                # 优先使用ARWU数据集
                arwu_collections = ["arwu_text", "arwu_score", "arwu_enhanced"]
                for coll_name in arwu_collections:
                    if coll_name in collections:
                        priority_collections.append((coll_name, collections[coll_name]))
            elif dataset_type == "the2025":
                # 优先使用THE2025数据集
                the_collections = []

                # 根据查询特征和提取的信息精确选择THE2025集合

                # 针对学科查询，优先使用学科集合
                if query_analysis["is_subject_question"]:
                    if subject_names:
                        # 如果提取到了具体学科名称，优先使用学科集合
                        the_collections.insert(0, "the2025_subjects")
                    else:
                        the_collections.append("the2025_subjects")

                # 针对指标相关查询，优先使用metrics集合
                if query_analysis["is_metric_question"]:
                    # 检查是否包含特定指标关键词
                    metric_terms = ["教学", "研究", "引用", "产业", "国际化", "师生比", "男女比"]
                    if any(term in query for term in metric_terms):
                        the_collections.insert(0, "the2025_metrics")  # 优先级高
                    else:
                        the_collections.append("the2025_metrics")

                # 对于排名问题，使用基本信息集合
                if query_analysis["is_ranking_question"]:
                    # 检查是否有排名区间
                    if query_analysis["rank_range"]:
                        the_collections.insert(0, "the2025_basic_info")  # 排名区间查询优先级最高
                    else:
                        the_collections.append("the2025_basic_info")

                # 如果没有特定条件但是一般性的排名查询，使用基本信息集合
                if not query_analysis["is_subject_question"] and not query_analysis["is_metric_question"] and not query_analysis["is_ranking_question"]:
                    the_collections.insert(0, "the2025_basic_info")

                # 如果学校查询，添加元数据集合
                if university_names:
                    if "the2025_meta" not in the_collections:
                        the_collections.append("the2025_meta")

                # 确保集合列表中没有重复
                the_collections = list(dict.fromkeys(the_collections))

                # 将THE集合添加到优先级集合中
                for coll_name in the_collections:
                    if coll_name in collections:
                        priority_collections.append((coll_name, collections[coll_name]))

            # 如果查询涉及特定大学，优先使用大学详情集合
            if university_names:
                university_collections = [
                    # 优先使用USNews2025大学集合
                    "usnews2025_university_base", "usnews2025_university_summary",
                    "usnews2025_university_statistics", "usnews2025_university_indicators",
                    "usnews2025_university_subjects",
                    # 其次使用其他大学集合
                    "university_summary", "university_statistics",
                    "university_indicators", "university_subjects",
                    "the2025_basic_info", "arwu_text"
                ]
                for coll_name in university_collections:
                    if coll_name in collections and not any(name == coll_name for name, _ in priority_collections):
                        priority_collections.append((coll_name, collections[coll_name]))

            # 将优先集合排在前面
            if priority_collections:
                collections_to_try = priority_collections[:max_collections]  # 只使用前max_collections个高优先级集合
                print(f"使用优先级最高的前{len(collections_to_try)}个集合，共有{len(priority_collections)}个匹配集合")

                # 如果启用多集合搜索且优先集合不足max_collections个，添加其他未包含的集合
                if use_multi_collections and len(collections_to_try) < max_collections:
                    remaining_slots = max_collections - len(collections_to_try)
                    additional_collections = []

                    for coll_name, collection in collections.items():
                        if coll_name not in [name for name, _ in priority_collections]:
                            additional_collections.append((coll_name, collection))
                            if len(additional_collections) >= remaining_slots:
                                break

                    collections_to_try.extend(additional_collections)
                    print(f"添加了{len(additional_collections)}个额外集合以补充到{max_collections}个")
            else:
                # 如果没有找到优先集合，使用前max_collections个集合
                collections_to_try = list(collections.items())[:max_collections]
                print(f"未找到优先集合，使用前{len(collections_to_try)}个常规集合")
        # 如果collections是列表或其他类型
        else:
            # 如果是列表，假设是集合名称列表
            if isinstance(collections, list):
                loaded_collections = []
                for coll_name in collections[:max_collections]:  # 只处理前max_collections个
                    try:
                        loaded_collections.append((coll_name, Collection(name=coll_name)))
                    except Exception as e:
                        print(f"加载集合 {coll_name} 失败: {e}")
                collections_to_try = loaded_collections
        
        # 输出最终选择的集合列表，按优先级排序
        print(f"\n最终选择的集合(按优先级):")
        for i, (coll_name, _) in enumerate(collections_to_try):
            print(f"{i+1}. {coll_name}")
        
        # 记录找到结果的集合数量
        collections_with_results = 0
        
        # 遍历集合进行搜索
        for coll_name, collection in collections_to_try:
            try:
                print(f"\n开始搜索集合: {coll_name}")
                collection.load()

                retrieved_texts = []
                context_source = ""

                # 优先使用关键词搜索
                if use_keyword_search:
                    texts = keyword_search(collection, query, top_k=adjusted_top_k)
                    if texts:
                        retrieved_texts = texts
                        context_source = f"关键词搜索 ({coll_name})"
                        print(f"关键词搜索成功: {coll_name}")

                # 如果关键词搜索没有结果，回退到向量搜索
                if not retrieved_texts:
                    print(f"关键词搜索未返回结果，尝试向量搜索: {coll_name}")

                    # 根据集合名称调用相应的搜索函数
                    if "arwu" in coll_name:
                        texts = search_arwu_knowledge(collection, query, top_k=adjusted_top_k,
                                                     region=query_analysis.get("region"),
                                                     country=query_analysis.get("country"))
                    elif "the2025" in coll_name:
                        texts = search_the2025_knowledge(collection, query, top_k=adjusted_top_k,
                                                        region=query_analysis.get("region"),
                                                        country=query_analysis.get("country"))
                    elif "us_colleges" in coll_name:
                        texts = search_us_colleges_WIKI_knowledge(collection, query, top_k=adjusted_top_k,
                                                                   state=query_analysis.get("state"),
                                                                   region=query_analysis.get("region"),
                                                                   control=query_analysis.get("control"))
                    else:
                        # 未知集合类型，跳过
                        print(f"未知集合类型，跳过: {coll_name}")
                        continue

                    if texts:
                        retrieved_texts = texts
                        context_source = f"向量搜索 ({coll_name})"
                        print(f"向量搜索成功: {coll_name}")
                
                # 对于us_colleges集合，即使关键词搜索有结果，也执行向量搜索并合并结果
                elif "us_colleges" in coll_name:
                    print(f"对于美国大学集合，同时执行向量搜索以补充关键词搜索结果")
                    vector_texts = search_us_colleges_WIKI_knowledge(collection, query, top_k=adjusted_top_k,
                                                              state=query_analysis.get("state"),
                                                              region=query_analysis.get("region"),
                                                              control=query_analysis.get("control"))
                    
                    if vector_texts:
                        # 为向量搜索结果添加标识
                        vector_texts = [f"[向量搜索] {text}" for text in vector_texts]
                        # 为关键词搜索结果添加标识
                        retrieved_texts = [f"[关键词搜索] {text}" for text in retrieved_texts]
                        # 合并结果
                        retrieved_texts.extend(vector_texts)
                        context_source = f"混合搜索 (关键词+向量) ({coll_name})"
                        print(f"成功合并关键词搜索和向量搜索结果，共 {len(retrieved_texts)} 条")

                # 如果从当前集合检索到了结果，添加到总结果列表
                if retrieved_texts:
                    collections_with_results += 1

                    # 评估结果相关性
                    relevance_score = 0.0

                    # 简单评估结果相关性：检查结果文本中是否包含大学名称或学科名称
                    if university_names or subject_names:
                        for text in retrieved_texts:
                            text_lower = text.lower()
                            # 检查大学名称
                            for uni_name in university_names:
                                if uni_name.lower() in text_lower:
                                    relevance_score += 1.0

                            # 检查学科名称
                            for subj_name in subject_names:
                                if subj_name in text_lower:
                                    relevance_score += 0.5

                    # 赋予排名高的结果更高相关性（如果是THE2025或ARWU集合）
                    if "the2025" in coll_name or "arwu" in coll_name:
                        for i, text in enumerate(retrieved_texts):
                            # 前三个结果权重更高
                            if i < 3:
                                relevance_score += 0.5 * (3 - i)

                    print(f"结果相关性评分: {relevance_score:.2f}")

                    for text in retrieved_texts:
                        all_retrieved_texts.append(text)
                        result_sources.append(context_source)

                    # 如果找到足够的结果（至少top_k条）并已检索了至少2个集合，可以提前停止
                    if len(all_retrieved_texts) >= top_k * 2 and collections_with_results >= 2:
                        print(f"已找到足够的结果({len(all_retrieved_texts)}条)并检索了{collections_with_results}个集合，停止继续搜索")
                        break

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
            
            # 限制结果数量，始终保留5个最相关的结果（无论传入的top_k是多少）
            max_results = 5
            final_texts = unique_texts[:max_results]
            final_sources = unique_sources[:max_results]
            
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
        return "加载知识库时出错: " + str(e)

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
        请根据提供的相关知识背景使用简洁精炼的语言完整回答用户的问题，
        如果知识背景中没有相关信息或者缺失信息，一定要基于你的常识进行回答补充缺失信息。你的回答中不得带有"注"或者"需要注意"的部分的描述。
        如果是关于排名的问题，一定要提及具体的排名来源和排名年份。如果有多个来源的排名数据，可以一并提及并说明各自的特点。
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
    LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"  # 多语言模型，支持中英文
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
            relevant_knowledge = load_knowledge_variables(knowledge_collections, user_input, top_k=5)
            
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

# 辅助函数：将数字转换为中文表示
def ranking_number_to_chinese(num):
    """将数字转换为中文表示，用于辅助识别"前X名"类型的查询"""
    if not isinstance(num, int) or num <= 0:
        return ""
    
    if num <= 10:
        chinese_nums = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        return chinese_nums[num]
    elif num < 20:
        return "十" + ranking_number_to_chinese(num - 10)
    elif num < 100:
        tens = num // 10
        remainder = num % 10
        if remainder == 0:
            return ranking_number_to_chinese(tens) + "十"
        else:
            return ranking_number_to_chinese(tens) + "十" + ranking_number_to_chinese(remainder)
    else:
        return str(num)  # 对于大于等于100的数字，直接返回字符串

def search_knowledge_from_multi_collections(
    collections, query, top_k=5, 
    use_keyword_search=True, use_multi_collections=True, max_collections=3
):
    """从多个集合中搜索相关知识
    
    参数:
        collections: 集合列表或字典
        query: 查询字符串
        query_analysis: 查询分析结果
        top_k: 默认检索条目数量
        use_keyword_search: 是否使用关键词搜索
        use_multi_collections: 是否使用多集合搜索
        max_collections: 最多使用的集合数量
        
    返回:
        检索到的文本列表
    """
    try:
        # 分析查询，获取查询类型和参数
        query_analysis = analyze_query(query)
        
        # 准备选项
        options = []
        
        # 调整top_k值，根据rank_range处理"前N名"类型的查询
        adjusted_top_k = top_k
        if query_analysis["is_ranking_question"] and query_analysis["rank_range"]:
            # 如果是类似"前N名学校有哪些"的查询
            if query_analysis["rank_range"][0] == 1:  # 从1开始的范围
                # 使用范围的上限作为top_k值，例如"前10名"会返回10个结果
                adjusted_top_k = max(top_k, query_analysis["rank_range"][1])
                print(f"检测到'前{query_analysis['rank_range'][1]}名'类型的查询，调整返回结果数量为: {adjusted_top_k}")
            # 如果是具体排名区间查询
            elif query_analysis["rank_range"][0] != query_analysis["rank_range"][1]:
                # 计算区间大小，确保返回足够的结果
                range_size = query_analysis["rank_range"][1] - query_analysis["rank_range"][0] + 1
                adjusted_top_k = max(top_k, range_size)
                print(f"检测到排名区间查询 {query_analysis['rank_range']}，调整返回结果数量为: {adjusted_top_k}")
            
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

            # 提取查询中提到的大学名称
            university_names = extract_university_names(query)
            print(f"提取到的大学名称: {university_names}")

            # 提取学科名称 (使用新增的函数)
            subject_names = extract_subject_names(query)
            if subject_names:
                print(f"提取到的学科名称: {subject_names}")

            # 优先处理与查询类型匹配的集合
            if dataset_type == "us_colleges":
                us_collections = ["us_colleges"]
                for coll_name in us_collections:
                    if coll_name in collections:
                        priority_collections.append((coll_name, collections[coll_name]))
            elif dataset_type == "arwu":
                # 优先使用ARWU数据集
                arwu_collections = ["arwu_text", "arwu_score", "arwu_enhanced"]
                for coll_name in arwu_collections:
                    if coll_name in collections:
                        priority_collections.append((coll_name, collections[coll_name]))
            elif dataset_type == "the2025":
                # 优先使用THE2025数据集
                the_collections = []

                # 根据查询特征和提取的信息精确选择THE2025集合

                # 针对学科查询，优先使用学科集合
                if query_analysis["is_subject_question"]:
                    if subject_names:
                        # 如果提取到了具体学科名称，优先使用学科集合
                        the_collections.insert(0, "the2025_subjects")
                    else:
                        the_collections.append("the2025_subjects")

                # 针对指标相关查询，优先使用metrics集合
                if query_analysis["is_metric_question"]:
                    # 检查是否包含特定指标关键词
                    metric_terms = ["教学", "研究", "引用", "产业", "国际化", "师生比", "男女比"]
                    if any(term in query for term in metric_terms):
                        the_collections.insert(0, "the2025_metrics")  # 优先级高
                    else:
                        the_collections.append("the2025_metrics")

                # 对于排名问题，使用基本信息集合
                if query_analysis["is_ranking_question"]:
                    # 检查是否有排名区间
                    if query_analysis["rank_range"]:
                        the_collections.insert(0, "the2025_basic_info")  # 排名区间查询优先级最高
                    else:
                        the_collections.append("the2025_basic_info")

                # 如果没有特定条件但是一般性的排名查询，使用基本信息集合
                if not query_analysis["is_subject_question"] and not query_analysis["is_metric_question"] and not query_analysis["is_ranking_question"]:
                    the_collections.insert(0, "the2025_basic_info")

                # 如果学校查询，添加元数据集合
                if university_names:
                    if "the2025_meta" not in the_collections:
                        the_collections.append("the2025_meta")

                # 确保集合列表中没有重复
                the_collections = list(dict.fromkeys(the_collections))

                # 将THE集合添加到优先级集合中
                for coll_name in the_collections:
                    if coll_name in collections:
                        priority_collections.append((coll_name, collections[coll_name]))

            # 如果查询涉及特定大学，优先使用大学详情集合
            if university_names:
                university_collections = [
                    # 优先使用USNews2025大学集合
                    "usnews2025_university_base", "usnews2025_university_summary",
                    "usnews2025_university_statistics", "usnews2025_university_indicators",
                    "usnews2025_university_subjects",
                    # 其次使用其他大学集合
                    "university_summary", "university_statistics",
                    "university_indicators", "university_subjects",
                    "the2025_basic_info", "arwu_text"
                ]
                for coll_name in university_collections:
                    if coll_name in collections and not any(name == coll_name for name, _ in priority_collections):
                        priority_collections.append((coll_name, collections[coll_name]))

            # 将优先集合排在前面
            if priority_collections:
                collections_to_try = priority_collections[:max_collections]  # 只使用前max_collections个高优先级集合
                print(f"使用优先级最高的前{len(collections_to_try)}个集合，共有{len(priority_collections)}个匹配集合")

                # 如果启用多集合搜索且优先集合不足max_collections个，添加其他未包含的集合
                if use_multi_collections and len(collections_to_try) < max_collections:
                    remaining_slots = max_collections - len(collections_to_try)
                    additional_collections = []

                    for coll_name, collection in collections.items():
                        if coll_name not in [name for name, _ in priority_collections]:
                            additional_collections.append((coll_name, collection))
                            if len(additional_collections) >= remaining_slots:
                                break

                    collections_to_try.extend(additional_collections)
                    print(f"添加了{len(additional_collections)}个额外集合以补充到{max_collections}个")
            else:
                # 如果没有找到优先集合，使用前max_collections个集合
                collections_to_try = list(collections.items())[:max_collections]
                print(f"未找到优先集合，使用前{len(collections_to_try)}个常规集合")
        # 如果collections是列表或其他类型
        else:
            # 如果是列表，假设是集合名称列表
            if isinstance(collections, list):
                loaded_collections = []
                for coll_name in collections[:max_collections]:  # 只处理前max_collections个
                    try:
                        loaded_collections.append((coll_name, Collection(name=coll_name)))
                    except Exception as e:
                        print(f"加载集合 {coll_name} 失败: {e}")
                collections_to_try = loaded_collections
        
        # 输出最终选择的集合列表，按优先级排序
        print(f"\n最终选择的集合(按优先级):")
        for i, (coll_name, _) in enumerate(collections_to_try):
            print(f"{i+1}. {coll_name}")
        
        # 记录找到结果的集合数量
        collections_with_results = 0
        
        # 遍历集合进行搜索
        for coll_name, collection in collections_to_try:
            try:
                print(f"\n开始搜索集合: {coll_name}")
                collection.load()

                retrieved_texts = []
                context_source = ""

                # 优先使用关键词搜索
                if use_keyword_search:
                    texts = keyword_search(collection, query, top_k=adjusted_top_k)
                    if texts:
                        retrieved_texts = texts
                        context_source = f"关键词搜索 ({coll_name})"
                        print(f"关键词搜索成功: {coll_name}")

                # 如果关键词搜索没有结果，回退到向量搜索
                if not retrieved_texts:
                    print(f"关键词搜索未返回结果，尝试向量搜索: {coll_name}")

                    # 根据集合名称调用相应的搜索函数
                    if "arwu" in coll_name:
                        texts = search_arwu_knowledge(collection, query, top_k=adjusted_top_k,
                                                     region=query_analysis.get("region"),
                                                     country=query_analysis.get("country"))
                    elif "the2025" in coll_name:
                        texts = search_the2025_knowledge(collection, query, top_k=adjusted_top_k,
                                                        region=query_analysis.get("region"),
                                                        country=query_analysis.get("country"))
                    elif "us_colleges" in coll_name:
                        texts = search_us_colleges_WIKI_knowledge(collection, query, top_k=adjusted_top_k,
                                                                   state=query_analysis.get("state"),
                                                                   region=query_analysis.get("region"),
                                                                   control=query_analysis.get("control"))
                    else:
                        # 未知集合类型，跳过
                        print(f"未知集合类型，跳过: {coll_name}")
                        continue

                    if texts:
                        retrieved_texts = texts
                        context_source = f"向量搜索 ({coll_name})"
                        print(f"向量搜索成功: {coll_name}")
                
                # 对于us_colleges集合，即使关键词搜索有结果，也执行向量搜索并合并结果
                elif "us_colleges" in coll_name:
                    print(f"对于美国大学集合，同时执行向量搜索以补充关键词搜索结果")
                    vector_texts = search_us_colleges_WIKI_knowledge(collection, query, top_k=adjusted_top_k,
                                                              state=query_analysis.get("state"),
                                                              region=query_analysis.get("region"),
                                                              control=query_analysis.get("control"))
                    
                    if vector_texts:
                        # 为向量搜索结果添加标识
                        vector_texts = [f"[向量搜索] {text}" for text in vector_texts]
                        # 为关键词搜索结果添加标识
                        retrieved_texts = [f"[关键词搜索] {text}" for text in retrieved_texts]
                        # 合并结果
                        retrieved_texts.extend(vector_texts)
                        context_source = f"混合搜索 (关键词+向量) ({coll_name})"
                        print(f"成功合并关键词搜索和向量搜索结果，共 {len(retrieved_texts)} 条")

                # 如果从当前集合检索到了结果，添加到总结果列表
                if retrieved_texts:
                    collections_with_results += 1

                    # 评估结果相关性
                    relevance_score = 0.0

                    # 简单评估结果相关性：检查结果文本中是否包含大学名称或学科名称
                    if university_names or subject_names:
                        for text in retrieved_texts:
                            text_lower = text.lower()
                            # 检查大学名称
                            for uni_name in university_names:
                                if uni_name.lower() in text_lower:
                                    relevance_score += 1.0

                            # 检查学科名称
                            for subj_name in subject_names:
                                if subj_name in text_lower:
                                    relevance_score += 0.5

                    # 赋予排名高的结果更高相关性（如果是THE2025或ARWU集合）
                    if "the2025" in coll_name or "arwu" in coll_name:
                        for i, text in enumerate(retrieved_texts):
                            # 前三个结果权重更高
                            if i < 3:
                                relevance_score += 0.5 * (3 - i)

                    print(f"结果相关性评分: {relevance_score:.2f}")

                    for text in retrieved_texts:
                        all_retrieved_texts.append(text)
                        result_sources.append(context_source)

                    # 如果找到足够的结果（至少top_k条）并已检索了至少2个集合，可以提前停止
                    if len(all_retrieved_texts) >= top_k * 2 and collections_with_results >= 2:
                        print(f"已找到足够的结果({len(all_retrieved_texts)}条)并检索了{collections_with_results}个集合，停止继续搜索")
                        break

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
            
            # 限制结果数量，始终保留5个最相关的结果（无论传入的top_k是多少）
            max_results = 5
            final_texts = unique_texts[:max_results]
            final_sources = unique_sources[:max_results]
            
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
        return "加载知识库时出错: " + str(e)

if __name__ == "__main__":
    main() 