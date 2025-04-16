import json
import logging
import numpy as np
import os
from pymilvus import (
    connections, 
    Collection,
    utility
)
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import configparser

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载环境变量
load_dotenv()

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

COLLECTION_NAME = "us_colleges_1744732028"

# 全局模型
MODEL = None
VECTOR_DIM = 384  # 默认值，会根据模型自动调整

def load_model():
    """加载SentenceTransformer模型"""
    global MODEL, VECTOR_DIM
    if MODEL is None:
        logging.info("加载SentenceTransformer模型...")
        try:
            # 使用轻量级模型
            MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            VECTOR_DIM = MODEL.get_sentence_embedding_dimension()
            logging.info(f"SentenceTransformer模型加载成功，向量维度: {VECTOR_DIM}")
        except Exception as e:
            logging.error(f"加载模型时出错: {e}")
            raise

def connect_to_milvus():
    """连接到Milvus服务器"""
    # 加载配置
    milvus_config = load_config()
    host = milvus_config['host']
    port = milvus_config['port']
    try:
        connections.connect(
            alias="default", 
            host=host, 
            port=port
        )
        logging.info(f"成功连接到Milvus服务器: {host}:{port}")
        return True
    except Exception as e:
        logging.error(f"连接Milvus失败: {e}")
        return False

def check_collection():
    """检查集合是否存在和加载状态"""
    if not utility.has_collection(COLLECTION_NAME):
        logging.error(f"集合 '{COLLECTION_NAME}' 不存在")
        return None
    
    collection = Collection(COLLECTION_NAME)
    
    # 检查集合加载状态
    if not utility.load_state(COLLECTION_NAME):
        logging.info(f"集合 '{COLLECTION_NAME}' 未加载到内存，正在加载...")
        collection.load()
    
    logging.info(f"集合 '{COLLECTION_NAME}' 已加载，包含 {collection.num_entities} 条记录")
    return collection

def get_embedding(text):
    """使用SentenceTransformer获取文本嵌入向量"""
    global MODEL
    
    # 确保模型已加载
    if MODEL is None:
        load_model()
    
    try:
        # 预处理文本，截断过长的文本
        max_seq_length = 512
        if len(text.split()) > max_seq_length * 0.8:  # 按词数粗略估计
            # 截断过长的文本
            words = text.split()
            text = " ".join(words[:int(max_seq_length * 0.8)])
            logging.warning(f"文本被截断，原长度：{len(words)} 词，截断后：{len(text.split())} 词")
        
        # 使用模型生成嵌入
        embedding = MODEL.encode(text)
        
        return embedding
    
    except Exception as e:
        logging.error(f"获取向量嵌入时出错: {e}")
        return None

def search_by_keyword(collection, field, keyword):
    """通过关键字搜索"""
    # Milvus不支持前缀通配符模式(%keyword)，修改为仅后缀通配符(keyword%)或完全匹配
    # 使用三种方式：完全匹配、前缀匹配和分词匹配
    results = []
    
    try:
        # 1. 完全匹配
        expr = f"{field} == '{keyword}'"
        exact_results = collection.query(
            expr=expr,
            output_fields=["id", "name", "state", "control", "type", "vector_type"]
        )
        if exact_results:
            logging.info(f"完全匹配查询找到 {len(exact_results)} 条结果")
            results.extend(exact_results)
        
        # 2. 前缀匹配 (keyword%)
        expr = f"{field} like '{keyword}%'"
        prefix_results = collection.query(
            expr=expr,
            output_fields=["id", "name", "state", "control", "type", "vector_type"]
        )
        if prefix_results:
            # 排除已匹配的结果
            prefix_results = [r for r in prefix_results if r not in results]
            logging.info(f"前缀匹配查询找到 {len(prefix_results)} 条结果")
            results.extend(prefix_results)
        
        # 3. 将关键词分解为多个部分，分别查询（对于大学名称中可能包含关键词的情况）
        keywords = keyword.split()
        for kw in keywords:
            if len(kw) > 3:  # 只对长度大于3的关键词进行搜索，减少噪声
                expr = f"{field} like '{kw}%'"
                kw_results = collection.query(
                    expr=expr,
                    output_fields=["id", "name", "state", "control", "type", "vector_type"]
                )
                if kw_results:
                    # 排除已匹配的结果
                    kw_results = [r for r in kw_results if r not in results]
                    logging.info(f"关键词 '{kw}' 匹配查询找到 {len(kw_results)} 条结果")
                    results.extend(kw_results)
        
        logging.info(f"通过关键词 '{keyword}' 在 '{field}' 字段中搜索，总计找到 {len(results)} 条结果")
        return results
    except Exception as e:
        logging.error(f"关键词搜索出错: {e}")
        return []

def search_by_vector(collection, query, vector_type="detail", top_k=5):
    """通过向量相似度搜索"""
    # 获取查询文本的向量嵌入
    query_vector = get_embedding(query)
    if query_vector is None:
        return []
    
    # 执行向量搜索，添加向量类型过滤
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": 100}
    }
    
    expr = f"vector_type == '{vector_type}'"
    
    results = collection.search(
        data=[query_vector.tolist()], 
        anns_field="vector", 
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=["id", "name", "state", "control", "type", "vector_type", "json_data"]
    )
    
    logging.info(f"通过查询 '{query}' 在 '{vector_type}' 类型向量中搜索，找到 {len(results[0])} 条结果")
    return results[0]

def search_by_hybrid(collection, query, filter_expr, vector_type="detail", top_k=5):
    """混合查询：向量相似度 + 标量过滤"""
    # 获取查询文本的向量嵌入
    query_vector = get_embedding(query)
    if query_vector is None:
        return []
    
    # 执行混合查询，添加向量类型过滤
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": 100}
    }
    
    # 组合过滤条件
    combined_expr = f"({filter_expr}) && vector_type == '{vector_type}'"
    
    results = collection.search(
        data=[query_vector.tolist()], 
        anns_field="vector", 
        param=search_params,
        limit=top_k, 
        expr=combined_expr,
        output_fields=["id", "name", "state", "control", "type", "vector_type", "json_data"]
    )
    
    logging.info(f"通过查询 '{query}' 和过滤条件 '{combined_expr}' 搜索，找到 {len(results[0])} 条结果")
    return results[0]

def display_results(results, show_details=False):
    """显示搜索结果"""
    if not results:
        print("未找到结果")
        return
    
    print("\n搜索结果:")
    print("-" * 80)
    
    for i, hit in enumerate(results):
        try:
            # 检查结果类型并相应处理
            if hasattr(hit, 'distance'):  # 向量搜索结果的Hit对象
                # 直接访问属性而不是使用get方法
                entity = hit.entity
                distance = hit.distance
                
                # 尝试直接访问字段属性
                try:
                    name = str(entity.name) if hasattr(entity, 'name') else "N/A"
                except:
                    name = "N/A"
                    
                try:
                    school_type = str(entity.type) if hasattr(entity, 'type') else "N/A"
                except:
                    school_type = "N/A"
                    
                try:
                    control = str(entity.control) if hasattr(entity, 'control') else "N/A"
                except:
                    control = "N/A"
                    
                try:
                    state = str(entity.state) if hasattr(entity, 'state') else "N/A"
                except:
                    state = "N/A"
                    
                try:
                    vector_type = str(entity.vector_type) if hasattr(entity, 'vector_type') else "N/A"
                except:
                    vector_type = "N/A"
                
                print(f"{i+1}. {name} (相似度: {1-distance:.4f})")
                print(f"   类型: {school_type}")
                print(f"   控制类型: {control}")
                print(f"   州: {state}")
                print(f"   向量类型: {vector_type}")
                
                if show_details:
                    try:
                        # 尝试直接访问json_data属性
                        if hasattr(entity, 'json_data'):
                            json_data_str = str(entity.json_data)
                            json_data = json.loads(json_data_str)
                            
                            if "location" in json_data:
                                loc = json_data["location"]
                                print(f"   位置: {loc.get('city', '')}, {loc.get('state', '')}, {loc.get('country', '')}")
                            
                            if "founded" in json_data:
                                print(f"   创立年份: {json_data.get('founded', 'N/A')}")
                            
                            if "website" in json_data:
                                print(f"   网站: {json_data.get('website', 'N/A')}")
                    except Exception as e:
                        logging.error(f"解析JSON数据时出错: {e}")
            else:  # 标量查询结果 (dict对象)
                print(f"{i+1}. {hit.get('name', 'N/A')}")
                print(f"   类型: {hit.get('type', 'N/A')}")
                print(f"   控制类型: {hit.get('control', 'N/A')}")
                print(f"   州: {hit.get('state', 'N/A')}")
                print(f"   向量类型: {hit.get('vector_type', 'N/A')}")
        except Exception as e:
            logging.error(f"显示结果时出错: {e}")
            logging.error(f"错误详情: {str(e)}")
            print(f"{i+1}. [显示错误] - {type(hit)}")
        
        print("-" * 80)

def run_test_queries(collection):
    """运行一些测试查询"""
    print("\n=== 测试1: 按学校名称关键字搜索 ===")
    name_results = search_by_keyword(collection, "name", "Harvard")
    display_results(name_results)
    
    print("\n=== 测试2: 按州搜索 ===")
    state_results = search_by_keyword(collection, "state", "California")
    # 只显示前5条结果
    display_results(state_results[:5])
    
    print("\n=== 测试3: 基本向量搜索 - 顶尖私立研究型大学 ===")
    vector_results = search_by_vector(
        collection, 
        "prestigious private research university in Northeast", 
        "basic", 
        5
    )
    display_results(vector_results, show_details=True)
    
    print("\n=== 测试4: 详细向量搜索 - 有商学院的大学 ===")
    detail_results = search_by_vector(
        collection, 
        "university with strong business school and entrepreneurship programs", 
        "detail", 
        5
    )
    display_results(detail_results, show_details=True)
    
    print("\n=== 测试5: 特色向量搜索 - 有特色体育文化的学校 ===")
    feature_results = search_by_vector(
        collection, 
        "university with strong athletic tradition and unique mascot", 
        "feature", 
        5
    )
    display_results(feature_results, show_details=True)
    
    print("\n=== 测试6: 混合查询 - 加州的艺术学校 ===")
    hybrid_results = search_by_hybrid(
        collection,
        "art school or university with strong creative programs",
        "state like 'California%'",
        "detail",
        5
    )
    display_results(hybrid_results, show_details=True)

def main():
    """主函数"""
    # 连接到Milvus
    if not connect_to_milvus():
        return
    
    # 检查集合
    collection = check_collection()
    if not collection:
        return
    
    # 运行测试查询
    run_test_queries(collection)

if __name__ == "__main__":
    main() 
