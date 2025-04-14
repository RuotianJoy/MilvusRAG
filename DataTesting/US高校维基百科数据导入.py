import json
import logging
import os
import time
import numpy as np
from pymilvus import (
    connections, 
    FieldSchema, 
    CollectionSchema, 
    DataType, 
    Collection, 
    utility
)
from dotenv import load_dotenv

# 设置日志格式，只保留必要的信息
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 加载环境变量
load_dotenv()

# 基本配置
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "us_colleges_wiki"
VECTOR_DIM = 768  # BERT向量维度

class MilvusImporter:
    def __init__(self, input_file_path):
        self.input_file_path = input_file_path
        self.data = None
        self.collection = None
    
    def run(self):
        """执行完整的导入流程"""
        try:
            start_time = time.time()
            
            # 步骤1: 加载JSON数据
            if not self._load_data():
                return False
                
            # 步骤2: 连接Milvus
            if not self._connect_to_milvus():
                return False
                
            # 步骤3: 创建集合
            if not self._create_collection():
                return False
                
            # 步骤4: 导入数据
            success = self._import_data()
            
            end_time = time.time()
            if success:
                logging.info(f"导入完成，耗时: {end_time - start_time:.2f}秒")
            
            return success
        except Exception as e:
            logging.error(f"导入过程中发生错误: {e}")
            return False
    
    def _load_data(self):
        """加载JSON数据"""
        logging.info(f"正在加载数据文件: {self.input_file_path}")
        
        try:
            if not os.path.exists(self.input_file_path):
                logging.error(f"文件不存在: {self.input_file_path}")
                return False
                
            with open(self.input_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
                
            if not isinstance(self.data, list) or len(self.data) == 0:
                logging.error("数据格式错误或为空")
                return False
                
            logging.info(f"成功加载 {len(self.data)} 条数据")
            return True
        except Exception as e:
            logging.error(f"加载数据失败: {e}")
            return False
    
    def _connect_to_milvus(self):
        """连接到Milvus服务器"""
        try:
            connections.connect(
                alias="default",
                host=MILVUS_HOST,
                port=MILVUS_PORT
            )
            logging.info(f"已连接到Milvus服务器: {MILVUS_HOST}:{MILVUS_PORT}")
            return True
        except Exception as e:
            logging.error(f"连接Milvus失败: {e}")
            return False
    
    def _create_collection(self):
        """创建Milvus集合"""
        try:
            # 删除已存在的同名集合
            if utility.has_collection(COLLECTION_NAME):
                utility.drop_collection(COLLECTION_NAME)
                logging.info(f"已删除现有集合: {COLLECTION_NAME}")
            
            # 创建集合字段
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, description="学校ID", is_primary=True, max_length=100),
                FieldSchema(name="name", dtype=DataType.VARCHAR, description="学校名称", max_length=500),
                FieldSchema(name="state", dtype=DataType.VARCHAR, description="州", max_length=100),
                FieldSchema(name="control", dtype=DataType.VARCHAR, description="公立/私立", max_length=50),
                FieldSchema(name="type", dtype=DataType.VARCHAR, description="学校类型", max_length=100),
                FieldSchema(name="enrollment", dtype=DataType.INT64, description="入学人数"),
                FieldSchema(name="founded", dtype=DataType.INT64, description="成立年份"),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, description="向量", dim=VECTOR_DIM),
                FieldSchema(name="vector_type", dtype=DataType.VARCHAR, description="向量类型", max_length=20),
                FieldSchema(name="json_data", dtype=DataType.VARCHAR, description="完整JSON数据", max_length=65535)
            ]
            
            schema = CollectionSchema(fields=fields, description="美国高校维基百科数据")
            self.collection = Collection(name=COLLECTION_NAME, schema=schema)
            
            # 创建向量索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            }
            self.collection.create_index("vector", index_params)
            
            # 创建分区
            self._create_partitions()
            
            # 加载集合
            self.collection.load()
            
            logging.info(f"已创建集合并加载: {COLLECTION_NAME}")
            return True
        except Exception as e:
            logging.error(f"创建集合失败: {e}")
            return False
    
    def _create_partitions(self):
        """创建集合分区"""
        regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West", "Other"]
        for region in regions:
            self.collection.create_partition(region)
    
    def _import_data(self):
        """导入数据到Milvus"""
        logging.info("开始导入数据...")
        
        # 向量类型
        vector_types = ["basic", "detail", "feature"]
        total_inserted = 0
        
        for vector_type in vector_types:
            inserted = self._import_vector_type(vector_type)
            total_inserted += inserted
            logging.info(f"已导入 {vector_type} 类型向量: {inserted} 条")
        
        # 确保所有数据已写入
        self.collection.flush()
        
        # 验证导入结果
        actual_count = self.collection.num_entities
        logging.info(f"导入完成，实际导入: {actual_count} 条")
        
        return actual_count > 0
    
    def _import_vector_type(self, vector_type):
        """导入特定类型的向量数据"""
        # 按区域组织数据
        region_data = self._organize_by_region()
        
        total_inserted = 0
        
        for region, colleges in region_data.items():
            logging.info(f"处理 {region} 区域的 {len(colleges)} 条记录，类型: {vector_type}")
            for college in colleges:
                # 逐条处理数据，而不是批量处理
                try:
                    # 获取向量
                    vector = self._get_vector(college, vector_type)
                    if vector is None:
                        continue
                    
                    # 准备单条记录数据
                    college_id = college.get('id', '')
                    if not college_id:
                        continue
                    
                    # 确保ID是字符串
                    if not isinstance(college_id, str):
                        college_id = str(college_id)
                    
                    # ID加上向量类型后缀确保唯一性
                    college_id = f"{college_id}_{vector_type}"
                    
                    # 处理location字段
                    location = college.get("location", {})
                    state = ""
                    if isinstance(location, dict):
                        state = str(location.get("state", "") or "")
                    
                    # 处理数值字段
                    try:
                        enrollment = int(college.get("enrollment", 0) or 0)
                    except (ValueError, TypeError):
                        enrollment = 0
                    
                    try:
                        founded = int(college.get("founded", 0) or 0)
                    except (ValueError, TypeError):
                        founded = 0
                    
                    # 准备JSON数据
                    json_data = json.dumps(college, ensure_ascii=False)
                    
                    # 构建单条记录
                    single_data = {
                        "id": college_id,
                        "name": str(college.get("name", "") or ""),
                        "state": state,
                        "control": str(college.get("control", "") or ""),
                        "type": str(college.get("type", "") or ""),
                        "enrollment": enrollment,
                        "founded": founded,
                        "vector": vector,
                        "vector_type": vector_type,
                        "json_data": json_data
                    }
                    
                    # 插入单条记录
                    self.collection.insert([single_data], partition_name=region)
                    total_inserted += 1
                    
                    # 每100条记录输出一次进度
                    if total_inserted % 100 == 0:
                        logging.info(f"已成功插入 {total_inserted} 条 {vector_type} 类型记录")
                        
                except Exception as e:
                    logging.error(f"插入记录 {college.get('id', 'unknown')} 失败: {e}")
                    continue
        
        # 完成后刷新数据
        self.collection.flush()
        logging.info(f"已完成 {vector_type} 类型数据插入，共 {total_inserted} 条记录")
        return total_inserted
    
    def _organize_by_region(self):
        """将数据按区域组织"""
        region_data = {
            "Northeast": [],
            "Southeast": [],
            "Midwest": [],
            "Southwest": [],
            "West": [],
            "Other": []
        }
        
        for college in self.data:
            region = college.get("region", "").strip() if college.get("region") else ""
            region = self._normalize_region(region)
            region_data[region].append(college)
        
        return region_data
    
    def _normalize_region(self, region):
        """标准化区域名称"""
        if not region:
            return "Other"
            
        region = region.lower()
        region_mapping = {
            "northeast": "Northeast",
            "southeast": "Southeast",
            "midwest": "Midwest",
            "southwest": "Southwest",
            "west": "West"
        }
        
        return region_mapping.get(region, "Other")
    
    def _get_vector(self, college, vector_type):
        """从学校数据中获取特定类型的向量"""
        if "vectors" not in college:
            return None
            
        vectors = college["vectors"]
        
        # 尝试可能的字段名
        field_names = [
            f"{vector_type}_vector",  # basic_vector
            vector_type,              # basic
            f"{vector_type}_embedding" # basic_embedding
        ]
        
        for field in field_names:
            if field in vectors and isinstance(vectors[field], list):
                vector = vectors[field]
                # 检查向量维度
                if len(vector) == VECTOR_DIM:
                    return vector
        
        return None


def main():
    """主函数"""
    # 获取脚本路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建输入文件路径
    input_file = os.path.join(script_dir, "US高校维基百科数据_processed.json")
    
    # 添加调试日志
    logging.info(f"脚本目录: {script_dir}")
    logging.info(f"输入文件路径: {input_file}")
    logging.info(f"文件是否存在: {os.path.exists(input_file)}")
    
    # 创建导入器并运行
    importer = MilvusImporter(input_file)
    success = importer.run()
    
    if success:
        logging.info("数据导入成功")
    else:
        logging.error("数据导入失败")


if __name__ == "__main__":
    main()