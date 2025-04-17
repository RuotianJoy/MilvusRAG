#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import time

# 配置参数
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "us_colleges"
MODEL_NAME = "all-MiniLM-L6-v2"  # 384维度的模型

# 连接到Milvus
def connect_to_milvus():
    print(f"连接到Milvus服务器 {MILVUS_HOST}:{MILVUS_PORT}")
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    print("成功连接到Milvus")

# 获取集合
def get_collection():
    collection = Collection(COLLECTION_NAME)
    collection.load()  # 加载集合到内存以加速搜索
    return collection

# 关键词搜索（使用Milvus的混合搜索功能）
def keyword_search(collection, keyword, limit=10):
    print(f"执行关键词搜索: '{keyword}'")
    
    # 构建查询表达式
    query_expr = f'name like "%{keyword}%" or content like "%{keyword}%"'
    
    # 开始搜索
    start_time = time.time()
    results = collection.query(
        expr=query_expr,
        output_fields=["name", "content", "url"],
        limit=limit
    )
    end_time = time.time()
    
    print(f"找到 {len(results)} 条结果 (耗时: {end_time - start_time:.2f} 秒)")
    return results

# 向量搜索
def vector_search(collection, query_text, limit=10):
    print(f"执行向量搜索: '{query_text}'")
    
    # 加载模型和生成查询向量
    model = SentenceTransformer(MODEL_NAME, device="cpu")  # 使用CPU设备
    query_embedding = model.encode([query_text])[0].tolist()
    
    # 开始搜索
    start_time = time.time()
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"ef": 64}},
        limit=limit,
        output_fields=["name", "content", "url"]
    )
    end_time = time.time()
    
    # 处理结果
    search_results = []
    for hits in results:
        for hit in hits:
            result = hit.entity._row_data
            result["distance"] = hit.distance
            search_results.append(result)
    
    print(f"找到 {len(search_results)} 条结果 (耗时: {end_time - start_time:.2f} 秒)")
    return search_results

# 混合搜索（结合关键词和向量搜索）
def hybrid_search(collection, query, limit=10):
    print(f"执行混合搜索: '{query}'")
    
    # 首先执行向量搜索
    vector_results = vector_search(collection, query, limit=limit*2)
    
    # 然后执行关键词搜索
    keyword_results = keyword_search(collection, query, limit=limit*2)
    
    # 合并结果（简单合并去重）
    seen_ids = set()
    hybrid_results = []
    
    # 先添加向量搜索结果（通常相关性更好）
    for result in vector_results:
        if len(hybrid_results) >= limit:
            break
        if result["id"] not in seen_ids:
            seen_ids.add(result["id"])
            hybrid_results.append(result)
    
    # 再添加关键词搜索结果
    for result in keyword_results:
        if len(hybrid_results) >= limit:
            break
        if result["id"] not in seen_ids:
            seen_ids.add(result["id"])
            hybrid_results.append(result)
    
    print(f"混合搜索找到 {len(hybrid_results)} 条结果")
    return hybrid_results

# 显示结果
def display_results(results, show_content=False):
    if not results:
        print("没有找到结果")
        return
    
    print("\n===== 搜索结果 =====")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.get('name', '未知学校')}")
        if 'distance' in result:
            print(f"   相似度得分: {1.0 / (1.0 + result['distance']):.4f}")
        print(f"   URL: {result.get('url', '无链接')}")
        
        if show_content and 'content' in result:
            # 只显示内容的前200个字符
            content = result['content']
            print(f"   内容摘要: {content[:200]}..." if len(content) > 200 else content)
    print("\n===================")

# 主函数
def main():
    parser = argparse.ArgumentParser(description="Milvus 向量数据库搜索工具")
    parser.add_argument("--query", "-q", type=str, required=True, help="搜索查询")
    parser.add_argument("--mode", "-m", type=str, default="hybrid", 
                        choices=["keyword", "vector", "hybrid"], 
                        help="搜索模式: keyword(关键词), vector(向量), hybrid(混合)")
    parser.add_argument("--limit", "-l", type=int, default=10, help="结果数量限制")
    parser.add_argument("--show-content", "-s", action="store_true", help="显示内容摘要")
    
    args = parser.parse_args()
    
    # 连接到Milvus
    connect_to_milvus()
    
    # 获取集合
    collection = get_collection()
    
    # 根据模式执行搜索
    if args.mode == "keyword":
        results = keyword_search(collection, args.query, args.limit)
    elif args.mode == "vector":
        results = vector_search(collection, args.query, args.limit)
    else:  # hybrid
        results = hybrid_search(collection, args.query, args.limit)
    
    # 显示结果
    display_results(results, args.show_content)
    
    # 释放集合
    collection.release()

if __name__ == "__main__":
    main() 