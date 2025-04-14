#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

def search_universities(query_text, limit=5):
    """使用向量搜索查询大学"""
    try:
        # 连接到Milvus服务器
        connections.connect("default", host="localhost", port="19530")
        print(f"✅ 已连接到Milvus服务器")
        
        # 加载集合
        collection = Collection("university_base")
        collection.load()
        print(f"✅ 已加载university_base集合，包含 {collection.num_entities} 条记录")
        
        # 初始化模型
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"✅ 已初始化SentenceTransformer模型")
        
        # 生成查询向量
        query_vector = model.encode(query_text).tolist()
        
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
        
        # 断开连接
        connections.disconnect("default")
        print("\n✅ 搜索完成")
        
    except Exception as e:
        print(f"❌ 搜索错误: {str(e)}")

if __name__ == "__main__":
    # 测试查询
    queries = [
        "Harvard University",
        "top school",
        "best university",
        "computer science university"
    ]
    
    # 执行第一个查询
    search_universities(queries[0])
    
    # 询问用户是否要继续测试
    print("\n是否要测试更多查询? (y/n)")
    response = input().strip().lower()
    if response == 'y':
        for query in queries[1:]:
            print("\n" + "-"*50)
            search_universities(query)
            print("-"*50) 