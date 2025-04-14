import json
import os
import argparse
from tqdm import tqdm
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# 连接到Milvus
def connect_to_milvus(host="localhost", port="19530"):
    try:
        connections.connect(
            alias="default", 
            host=host,
            port=port
        )
        print("成功连接到Milvus")
        return True
    except Exception as e:
        print(f"连接Milvus失败: {e}")
        return False

# 创建Milvus集合
def create_collection(collection_name="us_colleges", drop_exists=False):
    # 检查集合是否已存在
    if utility.has_collection(collection_name):
        if drop_exists:
            utility.drop_collection(collection_name)
            print(f"已删除现有的{collection_name}集合")
        else:
            print(f"集合{collection_name}已存在，直接使用")
            return Collection(collection_name)
    
    # 定义字段 - 设置足够大的最大长度限制
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=500),  # 增加到500
        FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=200),  # 增加到200
        FieldSchema(name="control", dtype=DataType.VARCHAR, max_length=100),  # 增加到100
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=200),  # 增加到200
        FieldSchema(name="enrollment", dtype=DataType.INT64),
        FieldSchema(name="state", dtype=DataType.VARCHAR, max_length=200),  # 增加到200
        FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=200),  # 增加到200
        FieldSchema(name="website", dtype=DataType.VARCHAR, max_length=500),  # 增加到500
        FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=384)  # BERT模型维度为384
    ]
    
    # 定义集合模式
    schema = CollectionSchema(fields, description="美国高校数据集")
    collection = Collection(collection_name, schema)
    print(f"成功创建{collection_name}集合")
    
    return collection

# 创建索引
def create_index(collection, index_type="HNSW"):
    # 创建索引
    try:
        if index_type == "HNSW":
            index_params = {
                "metric_type": "COSINE",  # 余弦相似度
                "index_type": "HNSW",     # HNSW索引
                "params": {"M": 8, "efConstruction": 200}
            }
        else:
            # 默认使用IVF_FLAT
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
        
        collection.create_index("text_vector", index_params)
        print(f"成功为集合创建{index_type}索引")
    except Exception as e:
        print(f"创建索引失败，将尝试继续导入数据: {e}")

# 从JSON文件加载处理后的数据
def load_processed_data(input_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        print(f"成功加载处理后的数据，共有 {len(processed_data)} 所学校")
        return processed_data
    except Exception as e:
        print(f"加载处理后的数据失败: {e}")
        return None

# 将数据插入到Milvus
def insert_to_milvus(collection, data, batch_size=50):  # 减小批处理大小
    total_count = len(data)
    inserted_count = 0
    errors_count = 0
    
    # 按批次插入数据
    for i in tqdm(range(0, total_count, batch_size)):
        batch_data = data[i:i+batch_size]
        try:
            collection.insert(batch_data)
            inserted_count += len(batch_data)
        except Exception as e:
            print(f"插入数据批次 {i//batch_size + 1} 失败: {e}")
            errors_count += 1
            
            # 如果批量插入失败，尝试逐条插入
            if errors_count <= 5:  # 最多尝试修复5批
                print("尝试逐条插入数据...")
                for j, item in enumerate(batch_data):
                    try:
                        collection.insert([item])
                        inserted_count += 1
                    except Exception as e_single:
                        print(f"  单条插入失败 (批次 {i//batch_size + 1}, 项 {j}): {e_single}")
    
    # 执行FLUSH操作
    collection.flush()
    print(f"成功插入 {inserted_count} 条记录到Milvus (总共 {total_count} 条)")

# 执行简单的查询测试
def test_query(collection):
    try:
        collection.load()
        
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
                output_fields=["text_vector"],
                limit=1
            )
            if first_vector and "text_vector" in first_vector[0]:
                target_vector = first_vector[0]["text_vector"]
                results = collection.search(
                    data=[target_vector],
                    anns_field="text_vector",
                    param={"metric_type": "COSINE", "params": {"ef": 10}},
                    limit=5,
                    output_fields=["name", "location", "type", "control", "state"]
                )
                if results and results[0]:
                    print(f"找到 {len(results[0])} 个相似结果:")
                    for i, r in enumerate(results[0]):
                        print(f"{i+1}. {r.entity.get('name')} - {r.entity.get('location')}, {r.entity.get('state')}, 相似度: {r.distance:.4f}")
                else:
                    print("未找到相似的记录")
            else:
                print("无法获取向量进行测试")
        except Exception as e:
            print(f"向量搜索测试失败: {e}")
    
    except Exception as e:
        print(f"执行查询测试时出错: {e}")
    
    finally:
        # 释放集合
        try:
            collection.release()
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="将处理后的Wiki美国高校数据导入到Milvus")
    parser.add_argument("--input", default="Wiki美国高校初步数据_processed.json", help="处理后的JSON数据文件路径")
    parser.add_argument("--host", default="localhost", help="Milvus服务器主机名")
    parser.add_argument("--port", default="19530", help="Milvus服务器端口")
    parser.add_argument("--collection", default="us_colleges", help="Milvus集合名称")
    parser.add_argument("--recreate", action="store_true", help="是否重新创建集合（如果已存在）")
    parser.add_argument("--index-type", default="HNSW", choices=["HNSW", "IVF_FLAT"], help="索引类型")
    parser.add_argument("--batch-size", type=int, default=50, help="批量插入的大小")
    parser.add_argument("--test", action="store_true", help="是否进行查询测试")
    args = parser.parse_args()
    
    # 如果文件不存在且没有提供完整路径，尝试在当前目录中查找
    input_file = args.input
    if not os.path.exists(input_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(script_dir, args.input)
        if not os.path.exists(input_file):
            print(f"无法找到输入文件: {args.input}")
            return
    
    # 连接到Milvus
    if not connect_to_milvus(args.host, args.port):
        return
    
    # 创建集合 - 强制重新创建集合以应用新的字段长度限制
    collection = create_collection(args.collection, drop_exists=True)
    if not collection:
        return
    
    # 加载处理后的数据
    data = load_processed_data(input_file)
    if not data:
        return
    
    # 创建索引
    create_index(collection, args.index_type)
    
    # 插入数据到Milvus
    insert_to_milvus(collection, data, args.batch_size)
    
    # 进行查询测试
    if args.test:
        test_query(collection)
    
    print("\n导入完成！数据已成功导入到Milvus数据库。")
    print("可以使用以下命令进行向量相似度搜索：")
    print(f"""
from pymilvus import connections, Collection
import numpy as np

# 连接到Milvus
connections.connect(host="{args.host}", port="{args.port}")

# 加载集合
collection = Collection("{args.collection}")
collection.load()

# 生成一个随机向量作为目标向量（实际使用时应该使用真实的查询向量）
target_vector = np.random.random(384).tolist()

# 向量相似度搜索（寻找相似学校）
results = collection.search(
    data=[target_vector],  # 目标向量
    anns_field="text_vector",
    param={{"metric_type": "COSINE", "params": {{"ef": 10}}}},
    limit=5,
    expr="state == 'California'",  # 可选过滤条件
    output_fields=["name", "location", "type", "control", "enrollment"]
)

# 显示结果
for i, hits in enumerate(results):
    for hit in hits:
        print(f"ID: {{hit.id}}, 距离: {{hit.distance}}")
        print(f"名称: {{hit.entity.get('name')}}")
        print(f"位置: {{hit.entity.get('location')}}, {{hit.entity.get('state')}}")
        print(f"类型: {{hit.entity.get('type')}}, {{hit.entity.get('control')}}")
        print(f"学生数: {{hit.entity.get('enrollment')}}")
        print("-------------------")

# 释放集合
collection.release()
""")

if __name__ == "__main__":
    main() 