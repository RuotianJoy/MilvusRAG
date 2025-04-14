from pymilvus import connections, Collection
import numpy as np
import argparse

def test_query(collection_name="us_colleges"):
    # 连接到Milvus
    print("连接到Milvus...")
    connections.connect(host="localhost", port="19530")
    
    # 获取集合
    print(f"加载集合 {collection_name}...")
    collection = Collection(collection_name)
    collection.load()
    
    try:
        # 获取集合信息
        print("\n集合信息:")
        print(f"集合名称: {collection.name}")
        print(f"集合实体数量: {collection.num_entities}")
        print(f"集合维度: {collection.schema.fields[-1].params['dim']}")
        
        # 测试1: 按地区和学校类型查询
        print("\n测试1: 按地区和学校类型查询 (Northeast地区的Masters university)")
        results = collection.query(
            expr="region == 'Northeast' and type == 'Masters university'",
            output_fields=["name", "location", "control", "enrollment", "website"],
            limit=5
        )
        if results:
            print(f"找到 {len(results)} 个结果，显示前5个:")
            for i, r in enumerate(results[:5]):
                print(f"{i+1}. {r['name']} - {r['location']}, {r['control']}, 学生数: {r['enrollment']}")
                if 'website' in r and r['website']:
                    print(f"   网站: {r['website']}")
        else:
            print("未找到匹配的记录")
        
        # 测试2: 按州和学校控制类型查询
        print("\n测试2: 按州和控制类型查询 (California的Public学校)")
        results = collection.query(
            expr="state == 'California' and control == 'Public'",
            output_fields=["name", "location", "type", "enrollment"],
            limit=5
        )
        if results:
            print(f"找到 {len(results)} 个结果，显示前5个:")
            for i, r in enumerate(results[:5]):
                print(f"{i+1}. {r['name']} - {r['location']}, {r['type']}, 学生数: {r['enrollment']}")
        else:
            print("未找到匹配的记录")
        
        # 测试3: 向量相似度查询示例
        print("\n测试3: 向量相似度查询示例")
        try:
            # 查询集合中的第一个向量作为目标
            first_vector = collection.query(
                expr="id >= 0", 
                output_fields=["text_vector", "name"],
                limit=1
            )
            if first_vector and "text_vector" in first_vector[0]:
                target_vector = first_vector[0]["text_vector"]
                target_name = first_vector[0]["name"]
                print(f"使用学校 '{target_name}' 作为查询目标")
                
                results = collection.search(
                    data=[target_vector],
                    anns_field="text_vector",
                    param={"metric_type": "COSINE", "params": {"ef": 10}},
                    limit=6,  # 多取一个，因为第一个通常是查询向量本身
                    output_fields=["name", "location", "type", "control", "state"]
                )
                if results and results[0]:
                    print(f"找到 {len(results[0])} 个相似结果 (排除第一个自身结果):")
                    for i, r in enumerate(results[0][1:6]):  # 跳过第一个（通常是自身）
                        print(f"{i+1}. {r.entity.get('name')} - {r.entity.get('location')}, {r.entity.get('state')}")
                        print(f"   类型: {r.entity.get('type')}, {r.entity.get('control')}")
                        print(f"   相似度: {r.distance:.4f}")
                        print("-------------------")
                else:
                    print("未找到相似的记录")
            else:
                print("无法获取向量进行测试")
                
            # 测试4: 随机向量查询
            print("\n测试4: 随机向量查询示例")
            # 生成随机向量
            random_vector = np.random.random(384).tolist()
            results = collection.search(
                data=[random_vector],
                anns_field="text_vector",
                param={"metric_type": "COSINE", "params": {"ef": 10}},
                limit=5,
                output_fields=["name", "location", "type", "control", "state"]
            )
            if results and results[0]:
                print(f"随机向量查询结果，找到 {len(results[0])} 个相似结果:")
                for i, r in enumerate(results[0]):
                    print(f"{i+1}. {r.entity.get('name')} - {r.entity.get('location')}, {r.entity.get('state')}")
                    print(f"   类型: {r.entity.get('type')}, {r.entity.get('control')}")
                    print(f"   相似度: {r.distance:.4f}")
                    print("-------------------")
            else:
                print("未找到相似的记录")
                
        except Exception as e:
            print(f"向量搜索测试失败: {e}")
    
    except Exception as e:
        print(f"执行查询测试时出错: {e}")
    
    finally:
        # 释放并断开连接
        try:
            collection.release()
            connections.disconnect("default")
            print("\n集合已释放，连接已断开")
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="测试Milvus中的Wiki美国高校数据")
    parser.add_argument("--collection", default="us_colleges", help="Milvus集合名称")
    args = parser.parse_args()
    
    test_query(args.collection)

if __name__ == "__main__":
    main() 