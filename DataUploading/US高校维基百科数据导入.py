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
import configparser
import math
from tqdm import tqdm
import ijson  # 用于流式解析JSON
import concurrent.futures  # 用于并行处理
import queue
import threading
from sentence_transformers import SentenceTransformer

# 设置日志格式，只保留必要的信息
logging.basicConfig(
    level=logging.DEBUG,  # 更改为DEBUG级别以获取更多信息
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 加载环境变量
load_dotenv()

# 定义常量
COLLECTION_NAME = f"us_colleges_{int(time.time())}"  # 使用时间戳创建唯一名称
VECTOR_DIM = 128  # 初始设置，将根据实际数据调整
NUM_CHUNKS = 100  # 数据分块数量
MAX_WORKERS = os.cpu_count() or 2  # 设置并行处理的最大工作线程数
BATCH_SIZE = 500  # 增加批处理大小以提高效率

# 全局变量
MODEL = None

def load_model():
    global MODEL, VECTOR_DIM
    # 使用仅128维的轻量级模型
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # 384维
    # 或者更小的
    # MODEL = SentenceTransformer('paraphrase-albert-small-v2')  # 768维但更快
    # MODEL = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # 384维
    VECTOR_DIM = MODEL.get_sentence_embedding_dimension()  # 自动获取维度

class MilvusImporter:
    def __init__(self, input_file_path):
        self.input_file_path = input_file_path
        self.collection = None
        self.vector_dim_detected = False
        self.insert_lock = threading.Lock()  # 添加锁以确保线程安全的插入操作

    def run(self):
        """执行完整的导入流程"""
        try:
            start_time = time.time()

            # 步骤1: 检查文件是否存在
            if not os.path.exists(self.input_file_path):
                logging.error(f"文件不存在: {self.input_file_path}")
                return False

            # 步骤1.5: 检测向量维度（从文件中读取第一条记录）
            global VECTOR_DIM
            detected_dim = self._detect_vector_dimension_from_file()
            if detected_dim:
                VECTOR_DIM = detected_dim
                logging.info(f"已检测到向量维度: {VECTOR_DIM}")
                self.vector_dim_detected = True
            else:
                logging.warning("无法检测到向量维度，使用默认值")

            # 步骤2: 连接Milvus
            if not self._connect_to_milvus():
                return False

            # 步骤3: 创建集合
            if not self._create_collection():
                return False

            # 步骤4: 分块处理数据并导入
            success = self._process_file_in_chunks()

            end_time = time.time()
            if success:
                logging.info(f"导入完成，耗时: {end_time - start_time:.2f}秒")
                logging.info(f"集合名称: {COLLECTION_NAME} (请记录此名称用于后续查询)")

            return success
        except Exception as e:
            import traceback
            logging.error(f"导入过程中发生错误: {str(e)}")
            logging.error(f"错误详情: {traceback.format_exc()}")
            return False

    def _detect_vector_dimension_from_file(self):
        """从文件中读取第一条记录来检测向量维度"""
        try:
            with open(self.input_file_path, 'r', encoding='utf-8') as f:
                # 读取文件的开始部分，只解析第一个对象
                parser = ijson.parse(f)
                
                # 跳过文件开头，找到第一个数组元素
                for prefix, event, value in parser:
                    if prefix == 'item' and event == 'start_map':
                        break
                
                # 构建第一个对象
                college = {}
                current_path = []
                current_obj = college
                
                # 解析单个对象
                for prefix, event, value in parser:
                    if prefix == 'item' and event == 'end_map':
                        break
                    
                    # 处理字段
                    if '.' in prefix:
                        parts = prefix.split('.')
                        if parts[0] == 'item':
                            parts = parts[1:]
                        
                        # 构建嵌套结构
                        obj = college
                        for i, part in enumerate(parts[:-1]):
                            if part not in obj:
                                obj[part] = {} if i < len(parts) - 2 else []
                            obj = obj[part]
                        
                        # 添加值
                        if event == 'string' or event == 'number' or event == 'boolean' or event == 'null':
                            obj[parts[-1]] = value
                
                # 检查向量
                if "vectors" in college:
                    vectors = college.get("vectors", {})
                    
                    # 检查向量类型
                    for vector_type in ["basic", "detail", "feature"]:
                        field_names = [
                            f"{vector_type}_vector",
                            vector_type,
                            f"{vector_type}_embedding"
                        ]
                        
                        for field in field_names:
                            if isinstance(vectors, dict) and field in vectors:
                                vector_data = vectors[field]
                                if isinstance(vector_data, list):
                                    return len(vector_data)
            
            # 如果无法从第一条记录中检测向量维度，尝试更直接的方式读取JSON
            logging.info("尝试直接读取JSON文件的前几条记录来检测向量维度...")
            with open(self.input_file_path, 'r', encoding='utf-8') as f:
                # 读取前几行
                content = ""
                for _ in range(10):  # 读取前10行
                    line = f.readline()
                    if not line:
                        break
                    content += line
                
                # 尝试解析为JSON
                try:
                    if content.strip().startswith('['):
                        # 是一个数组，尝试解析第一个元素
                        end_idx = content.find('}]')
                        if end_idx > 0:
                            first_obj_str = content[:end_idx+1]
                            college_data = json.loads(first_obj_str + ']')[0]
                            
                            # 检查向量
                            if "vectors" in college_data:
                                vectors = college_data["vectors"]
                                for vector_type in ["basic", "detail", "feature"]:
                                    for field in [f"{vector_type}_vector", vector_type, f"{vector_type}_embedding"]:
                                        if field in vectors and isinstance(vectors[field], list):
                                            return len(vectors[field])
                    else:
                        # 尝试作为单个对象解析
                        end_idx = content.find('}')
                        if end_idx > 0:
                            obj_str = content[:end_idx+1]
                            college_data = json.loads(obj_str)
                            
                            # 检查向量
                            if "vectors" in college_data:
                                vectors = college_data["vectors"]
                                for vector_type in ["basic", "detail", "feature"]:
                                    for field in [f"{vector_type}_vector", vector_type, f"{vector_type}_embedding"]:
                                        if field in vectors and isinstance(vectors[field], list):
                                            return len(vectors[field])
                except json.JSONDecodeError:
                    logging.warning("直接解析JSON片段失败")
            
            # 再尝试使用正则表达式查找向量模式
            with open(self.input_file_path, 'r', encoding='utf-8') as f:
                # 读取前50000个字符，尝试找到向量维度
                sample = f.read(50000)
                # 查找向量模式 [...数字, 数字...]
                import re
                vector_patterns = r'\[(?:-?\d+\.?\d*(?:e[+-]?\d+)?,\s*){10,}(?:-?\d+\.?\d*(?:e[+-]?\d+)?)\]'
                matches = re.findall(vector_patterns, sample)
                
                if matches:
                    # 计算第一个匹配到的向量的维度
                    vector_str = matches[0]
                    vector = json.loads(vector_str)
                    logging.info(f"通过正则表达式找到向量，维度: {len(vector)}")
                    return len(vector)
                    
            return None
        except Exception as e:
            logging.error(f"检测向量维度时出错: {str(e)}")
            import traceback
            logging.debug(f"错误详情: {traceback.format_exc()}")
            return None

    @staticmethod
    def load_config(project_root):
        """读取配置文件"""
        # 配置文件路径
        config_file = os.path.join(project_root, "Config", "Milvus.ini")
        config = configparser.ConfigParser()
        config.read(config_file, encoding='utf-8')
        return {
            'host': config.get('connection', 'host', fallback='localhost'),
            'port': config.get('connection', 'port', fallback='19530')
        }

    def _connect_to_milvus(self):
        """连接到Milvus服务器"""
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 加载配置
        milvus_config = self.load_config(project_root)
        host = milvus_config['host']
        port = milvus_config['port']

        try:
            connections.connect(
                alias="default",
                host=host,
                port=port
            )
            logging.info(f"已连接到Milvus服务器: {host}:{port}")
            return True
        except Exception as e:
            logging.error(f"连接Milvus失败: {e}")
            return False

    def _create_collection(self):
        """创建Milvus集合"""
        try:
            logging.info(f"创建新集合: {COLLECTION_NAME}")

            # 创建集合字段
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, description="学校ID", is_primary=True, max_length=500),
                FieldSchema(name="name", dtype=DataType.VARCHAR, description="学校名称", max_length=500),
                FieldSchema(name="state", dtype=DataType.VARCHAR, description="州", max_length=200),
                FieldSchema(name="control", dtype=DataType.VARCHAR, description="公立/私立", max_length=200),
                FieldSchema(name="type", dtype=DataType.VARCHAR, description="学校类型", max_length=200),
                FieldSchema(name="enrollment", dtype=DataType.INT64, description="入学人数"),
                FieldSchema(name="founded", dtype=DataType.INT64, description="成立年份"),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, description="向量", dim=VECTOR_DIM),
                FieldSchema(name="vector_type", dtype=DataType.VARCHAR, description="向量类型", max_length=200),
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

            # 加载集合（添加重试逻辑）
            self._load_collection_with_retry()

            logging.info(f"已创建集合并加载: {COLLECTION_NAME}")
            return True
        except Exception as e:
            logging.error(f"创建集合失败: {e}")
            return False

    def _load_collection_with_retry(self, max_retries=3, initial_delay=2):
        """带重试逻辑的集合加载"""
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                logging.info(f"尝试加载集合 (尝试 {attempt + 1}/{max_retries})...")
                self.collection.load()
                logging.info("集合加载成功")
                return True
            except Exception as e:
                logging.warning(f"加载集合失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logging.info(f"等待 {delay} 秒后重试...")
                    time.sleep(delay)
                    # 每次重试增加延迟（指数退避）
                    delay *= 2

        logging.error(f"在 {max_retries} 次尝试后仍无法加载集合")
        return False

    def _create_partitions(self):
        """创建集合分区"""
        regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West", "Other"]
        for region in regions:
            self.collection.create_partition(region)

    def _process_file_in_chunks(self):
        """使用并行处理分块处理JSON文件"""
        logging.info(f"开始分块并行处理JSON文件...")
        
        # 获取文件大小
        file_size = os.path.getsize(self.input_file_path)
        logging.info(f"文件大小: {file_size / (1024 * 1024):.2f} MB")
        
        # 解析文件得到JSON对象列表
        all_colleges = self._parse_json_file()
        total_records = len(all_colleges)
        logging.info(f"共解析出 {total_records} 条记录")
        
        # 计算每个块的大小
        chunk_size = max(100, total_records // NUM_CHUNKS)
        num_chunks = (total_records + chunk_size - 1) // chunk_size
        logging.info(f"分割为 {num_chunks} 块，每块约 {chunk_size} 条记录")
        
        # 分割数据为块
        chunks = [all_colleges[i:i+chunk_size] for i in range(0, total_records, chunk_size)]
        
        # 创建一个线程池来并行处理数据块
        total_inserted = 0
        vector_types = ["basic", "detail", "feature"]
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 创建任务列表
            future_to_chunk = {}
            
            # 为每个块和向量类型提交任务
            for chunk_idx, chunk in enumerate(chunks):
                for vector_type in vector_types:
                    future = executor.submit(
                        self._import_data_chunk,
                        chunk,
                        vector_type,
                        f"Chunk {chunk_idx+1}/{len(chunks)}, Type: {vector_type}"
                    )
                    future_to_chunk[future] = (chunk_idx, vector_type)
            
            # 处理完成的任务
            with tqdm(total=len(future_to_chunk), desc="处理数据块") as pbar:
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_idx, vector_type = future_to_chunk[future]
                    try:
                        inserted = future.result()
                        total_inserted += inserted
                        logging.info(f"块 {chunk_idx+1}, 类型 {vector_type}: 已导入 {inserted} 条记录")
                    except Exception as e:
                        logging.error(f"处理块 {chunk_idx+1}, 类型 {vector_type} 时出错: {str(e)}")
                    pbar.update(1)
        
        # 确保所有数据已写入
        self.collection.flush()
        
        # 验证导入结果
        actual_count = self.collection.num_entities
        logging.info(f"并行导入完成，实际导入: {actual_count} 条，预期至少: {total_inserted} 条")
        
        return actual_count > 0

    def _parse_json_file(self):
        """优化的JSON文件解析方法"""
        logging.info("开始解析JSON文件...")
        
        try:
            # 直接读取整个文件并解析为JSON
            with open(self.input_file_path, 'r', encoding='utf-8') as f:
                # 尝试直接加载整个JSON
                try:
                    content = f.read()
                    # 确保内容是一个JSON数组
                    if content.strip().startswith('[') and content.strip().endswith(']'):
                        all_colleges = json.loads(content)
                        return all_colleges
                    else:
                        # 如果不是数组格式，尝试包装成数组
                        all_colleges = json.loads(f'[{content}]')
                        return all_colleges
                except json.JSONDecodeError:
                    # 如果整个文件解析失败，回退到逐行解析
                    logging.warning("直接解析JSON失败，尝试逐行解析...")
        except Exception as e:
            logging.error(f"读取文件时出错: {str(e)}")
            
        # 回退方法：流式解析
        all_colleges = []
        try:
            with open(self.input_file_path, 'r', encoding='utf-8') as f:
                # 使用ijson流式解析器
                objects = ijson.items(f, 'item')
                for obj in tqdm(objects, desc="解析JSON对象"):
                    all_colleges.append(obj)
        except Exception as e:
            logging.error(f"流式解析JSON时出错: {str(e)}")
            
            # 最后的回退：尝试逐行读取并手动解析
            try:
                all_colleges = []
                with open(self.input_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 尝试分割内容为单独的JSON对象
                    if '},{' in content:
                        # 分割并修复各个对象
                        parts = content.replace('[', '').replace(']', '').split('},{')
                        for i, part in enumerate(parts):
                            # 修复每个对象
                            if not part.startswith('{'):
                                part = '{' + part
                            if not part.endswith('}'):
                                part = part + '}'
                            try:
                                obj = json.loads(part)
                                all_colleges.append(obj)
                            except json.JSONDecodeError:
                                logging.warning(f"无法解析对象 {i+1}")
            except Exception as e:
                logging.error(f"手动解析JSON时出错: {str(e)}")
                
        return all_colleges

    def _import_data_chunk(self, chunk_data, vector_type, chunk_desc=""):
        """导入特定类型的向量数据块 - 线程安全版本"""
        # 按区域组织数据
        region_data = self._organize_by_region(chunk_data)

        total_inserted = 0

        for region, colleges in region_data.items():
            if not colleges:
                continue
                
            logging.info(f"{chunk_desc} - 处理 {region} 区域的 {len(colleges)} 条记录")

            # 批量收集记录
            batch_entities = []

            for college in colleges:
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
                    entity = {
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

                    batch_entities.append(entity)

                    # 达到批处理大小时插入
                    if len(batch_entities) >= BATCH_SIZE:
                        inserted = self._safe_insert_batch(batch_entities, region)
                        total_inserted += inserted
                        # 清空批处理
                        batch_entities = []

                except Exception as e:
                    logging.error(f"{chunk_desc} - 处理记录 {college.get('id', 'unknown')} 失败: {e}")
                    continue

            # 插入剩余的记录
            if batch_entities:
                inserted = self._safe_insert_batch(batch_entities, region)
                total_inserted += inserted

        return total_inserted
        
    def _safe_insert_batch(self, batch_entities, partition_name):
        """线程安全的批量插入方法"""
        if not batch_entities:
            return 0
            
        try:
            # 记录当前批次信息
            logging.debug(f"准备插入 {len(batch_entities)} 条记录到分区 {partition_name}")
            
            # 检查批次数据
            for i, entity in enumerate(batch_entities):
                # 确保向量不为空且维度正确
                if "vector" not in entity or entity["vector"] is None:
                    logging.debug(f"记录 {i} (ID: {entity.get('id', 'unknown')}) 向量为空，移除")
                    batch_entities[i] = None
                elif len(entity["vector"]) != VECTOR_DIM:
                    logging.debug(f"记录 {i} (ID: {entity.get('id', 'unknown')}) 向量维度错误: {len(entity['vector'])} != {VECTOR_DIM}，移除")
                    batch_entities[i] = None
            
            # 移除无效记录
            batch_entities = [e for e in batch_entities if e is not None]
            if not batch_entities:
                logging.warning(f"批次中没有有效记录，跳过插入")
                return 0
                
            # 记录有效批次信息
            logging.debug(f"过滤后准备插入 {len(batch_entities)} 条有效记录到分区 {partition_name}")
            
            # 使用锁确保线程安全
            with self.insert_lock:
                self.collection.insert(batch_entities, partition_name=partition_name)
                inserted_count = len(batch_entities)
                logging.debug(f"成功插入 {inserted_count} 条记录到分区 {partition_name}")
                return inserted_count
        except Exception as e:
            logging.error(f"批量插入失败 ({partition_name}): {e}")
            import traceback
            logging.debug(f"插入错误详情: {traceback.format_exc()}")
            
            # 尝试记录第一条记录以便调试
            if batch_entities:
                first_entity = batch_entities[0]
                logging.debug(f"第一条记录示例: ID={first_entity.get('id', 'unknown')}, "
                             f"向量维度={(len(first_entity.get('vector', [])) if first_entity.get('vector') else 'None')}")
            
            # 如果批量插入失败，尝试分成更小的批次
            if len(batch_entities) > 10:
                logging.info(f"尝试使用更小的批次重新插入 {len(batch_entities)} 条记录...")
                mid = len(batch_entities) // 2
                count1 = self._safe_insert_batch(batch_entities[:mid], partition_name)
                count2 = self._safe_insert_batch(batch_entities[mid:], partition_name)
                return count1 + count2
            elif len(batch_entities) > 1:
                # 如果少于10条但大于1条，尝试逐条插入
                logging.info(f"尝试逐条插入 {len(batch_entities)} 条记录...")
                total_inserted = 0
                for entity in batch_entities:
                    try:
                        result = self._safe_insert_batch([entity], partition_name)
                        total_inserted += result
                    except Exception as e:
                        logging.error(f"插入单条记录失败: {e}")
                return total_inserted
                
            return 0

    def _organize_by_region(self, data):
        """将数据按区域组织"""
        region_data = {
            "Northeast": [],
            "Southeast": [],
            "Midwest": [],
            "Southwest": [],
            "West": [],
            "Other": []
        }

        for college in data:
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
        # 检查是否有向量字段
        if "vectors" not in college:
            return None

        vectors = college["vectors"]

        # 如果vectors不是字典类型，尝试直接作为向量使用
        if not isinstance(vectors, dict):
            if isinstance(vectors, list) and len(vectors) == VECTOR_DIM:
                return vectors
            else:
                return None

        # 尝试可能的字段名
        field_names = [
            f"{vector_type}_vector",  # basic_vector
            vector_type,  # basic
            f"{vector_type}_embedding",  # basic_embedding
            "vector",  # 通用向量字段
            "embedding"  # 通用嵌入字段
        ]

        # 查找指定向量类型
        for field in field_names:
            if field in vectors:
                value = vectors[field]
                # 检查是否为列表且维度匹配
                if isinstance(value, list):
                    # 检查向量维度
                    if len(value) == VECTOR_DIM:
                        return value
                    else:
                        logging.debug(f"向量维度不匹配: {len(value)} vs 预期 {VECTOR_DIM}, 字段: {field}")
                # 检查是否为嵌套结构
                elif isinstance(value, dict) and "vector" in value and isinstance(value["vector"], list):
                    if len(value["vector"]) == VECTOR_DIM:
                        return value["vector"]
                    else:
                        logging.debug(f"嵌套向量维度不匹配: {len(value['vector'])} vs 预期 {VECTOR_DIM}, 字段: {field}")
                # 对于字符串类型的向量，尝试解析
                elif isinstance(value, str):
                    try:
                        # 尝试将字符串解析为向量
                        if value.startswith('[') and value.endswith(']'):
                            vector_list = json.loads(value)
                            if isinstance(vector_list, list) and len(vector_list) == VECTOR_DIM:
                                return vector_list
                    except:
                        pass

        # 如果找不到匹配的向量，尝试加载模型生成向量
        global MODEL
        if MODEL is None:
            load_model()
        
        # 获取学校名称
        school_name = college.get('name', '')
        if school_name and MODEL is not None:
            try:
                # 使用学校名称生成向量
                logging.debug(f"尝试为学校 '{school_name}' 生成向量")
                vector = MODEL.encode(school_name).tolist()
                return vector
            except Exception as e:
                logging.debug(f"为学校 '{school_name}' 生成向量失败: {str(e)}")

        # 如果只找到一个向量字段，无论类型，都尝试使用它
        if len(vectors) == 1:
            only_vector = list(vectors.values())[0]
            if isinstance(only_vector, list) and len(only_vector) == VECTOR_DIM:
                return only_vector

        # 查找任何维度匹配的向量
        for key, value in vectors.items():
            if isinstance(value, list) and len(value) == VECTOR_DIM:
                return value

        # 记录找不到向量信息
        logging.debug(f"无法为学校 {college.get('id', 'unknown')} 找到类型为 {vector_type} 的向量")
        return None


def main():
    """主函数"""
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 默认数据文件路径
    default_data_file = os.path.join(project_root, "DataProcessed", "US高校维基百科数据_processed.json")

    # 添加调试日志
    logging.info(f"脚本目录: {script_dir}")
    logging.info(f"输入文件路径: {default_data_file}")
    logging.info(f"文件是否存在: {os.path.exists(default_data_file)}")
    logging.info(f"使用 {MAX_WORKERS} 个工作线程进行并行处理")

    # 检查文件格式
    try:
        logging.info("检查数据文件格式...")
        with open(default_data_file, 'r', encoding='utf-8') as f:
            # 读取文件前100个字符
            sample = f.read(100)
            logging.info(f"文件开头样本: {sample}...")
            
            # 获取文件大小
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            logging.info(f"文件大小: {file_size/1024/1024:.2f} MB")
            
            # 尝试读取第一条记录
            f.seek(0)
            try:
                first_line = f.readline().strip()
                if first_line.startswith('['):
                    # 文件以 [ 开头，是JSON数组
                    logging.info("文件格式: JSON数组")
                    # 尝试解析
                    f.seek(0)
                    import json
                    try:
                        first_record = next(iter(json.load(f)))
                        logging.info(f"成功读取第一条记录，包含字段: {list(first_record.keys())}")
                        if 'vectors' in first_record:
                            logging.info(f"向量字段内容: {first_record['vectors']}")
                    except json.JSONDecodeError:
                        logging.warning("无法解析JSON数组")
                else:
                    # 可能是JSON Lines格式
                    logging.info("文件格式: 可能是JSON Lines")
                    try:
                        first_record = json.loads(first_line)
                        logging.info(f"成功读取第一条记录，包含字段: {list(first_record.keys())}")
                        if 'vectors' in first_record:
                            logging.info(f"向量字段内容: {first_record['vectors']}")
                    except json.JSONDecodeError:
                        logging.warning("无法解析第一行为JSON")
            except Exception as e:
                logging.error(f"读取第一条记录时出错: {str(e)}")
    except Exception as e:
        logging.error(f"检查文件格式时出错: {str(e)}")
    
    # 加载模型
    try:
        load_model()
        logging.info(f"已加载模型，向量维度: {VECTOR_DIM}")
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")

    # 创建导入器并运行
    importer = MilvusImporter(default_data_file)
    success = importer.run()

    if success:
        logging.info(f"数据导入成功，集合名称: {COLLECTION_NAME}")
    else:
        logging.error("数据导入失败")


if __name__ == "__main__":
    main()