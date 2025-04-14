import json
import os
import re
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, connections

# 定义输入和输出文件路径
# 使用更可靠的路径构建方式
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
input_file = os.path.join(project_root, '第三方排名网站数据爬取', 'THE2025.json')
output_file = os.path.join(project_root, '数据处理', 'THE2025_processed.json')

# 使用BERT模型进行文本嵌入
def create_text_embedding(text, model_name="bert-base-multilingual-cased"):
    """为文本创建嵌入向量"""
    # 加载模型和分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # 生成嵌入
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # 使用[CLS]标记的输出作为文本嵌入
    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
    return embedding.tolist()

def convert_to_float(value, default=0.0):
    """将字符串值转换为浮点数，处理缺失值和范围值"""
    if not value or value == "null" or value == "None":
        return default
    
    # 处理范围值，例如"25.2–30.6"
    if "–" in value or "-" in value or "鈥?" in value:
        try:
            # 处理不同类型的破折号
            value = value.replace("鈥?", "-")
            parts = re.split(r'[–-]', value)
            # 计算范围的平均值
            return sum(float(part) for part in parts if part.strip()) / len([part for part in parts if part.strip()])
        except (ValueError, ZeroDivisionError):
            return default
            
    try:
        return float(value)
    except ValueError:
        # 尝试清除非数字字符
        cleaned = re.sub(r'[^\d.]', '', value)
        try:
            if cleaned:
                return float(cleaned)
        except ValueError:
            pass
        return default

def convert_to_int(value, default=0):
    """将字符串值转换为整数，处理缺失值和特殊格式"""
    if not value or value == "null" or value == "None":
        return default
    
    # 处理范围值，如"1201-1500"
    if "-" in value or "–" in value or "鈥?" in value:
        try:
            value = value.replace("鈥?", "-")
            parts = re.split(r'[–-]', value)
            # 计算范围的平均值
            return int(sum(int(re.sub(r'[^\d]', '', part)) for part in parts if re.sub(r'[^\d]', '', part)) / 
                       len([part for part in parts if re.sub(r'[^\d]', '', part)]))
        except (ValueError, ZeroDivisionError):
            return default
    
    try:
        # 尝试直接转换
        return int(value)
    except ValueError:
        # 尝试清除非数字字符
        cleaned = re.sub(r'[^\d]', '', value)
        try:
            if cleaned:
                return int(cleaned)
        except ValueError:
            pass
        return default

def process_student_ratio(ratio_str):
    """处理学生比例字符串，例如'53 : 47'或类似格式"""
    if not ratio_str or ratio_str == "null":
        return {"female": 0.5, "male": 0.5}  # 默认平均值
    
    # 分割比例
    parts = ratio_str.split(":")
    if len(parts) != 2:
        parts = ratio_str.split(" : ")
    
    if len(parts) == 2:
        try:
            female = float(parts[0].strip())
            male = float(parts[1].strip())
            total = female + male
            if total > 0:  # 防止除零错误
                return {"female": female/total, "male": male/total}
        except ValueError:
            pass
    
    return {"female": 0.5, "male": 0.5}  # 默认值

def process_subjects(subjects_str):
    """处理学科列表字符串，返回标准化的学科数组"""
    if not subjects_str or subjects_str == "null":
        return []
    
    # 分割学科
    if "," in subjects_str:
        subjects = [s.strip() for s in subjects_str.split(",")]
    else:
        subjects = [subjects_str.strip()]
    
    return [s for s in subjects if s]  # 过滤空字符串

def preprocess_university_data(raw_data):
    """预处理大学数据，处理缺失值和标准化格式"""
    try:
        # 提取基本信息
        processed_data = {
            "id": raw_data.get("nid", 0),
            "name": raw_data.get("name", "Unknown"),
            "rank": raw_data.get("rank", "未知"),
            "rank_order": convert_to_int(raw_data.get("rank_order", "0")),
            "location": raw_data.get("location", "Unknown"),
            
            # 评分指标 - 转换为浮点数
            "overall_score": convert_to_float(raw_data.get("scores_overall", "0")),
            "teaching_score": convert_to_float(raw_data.get("scores_teaching", "0")),
            "research_score": convert_to_float(raw_data.get("scores_research", "0")),
            "citations_score": convert_to_float(raw_data.get("scores_citations", "0")),
            "industry_income_score": convert_to_float(raw_data.get("scores_industry_income", "0")),
            "international_outlook_score": convert_to_float(raw_data.get("scores_international_outlook", "0")),
            
            # 统计数据 - 处理特殊格式
            "number_students": raw_data.get("stats_number_students", "0").replace(",", ""),
            "student_staff_ratio": convert_to_float(raw_data.get("stats_student_staff_ratio", "0")),
            "pc_intl_students": convert_to_float(raw_data.get("stats_pc_intl_students", "0%").replace("%", "")) / 100,
            
            # 性别比例 - 处理为结构化数据
            "female_male_ratio": process_student_ratio(raw_data.get("stats_female_male_ratio", "")),
            
            # 学科信息 - 处理为数组
            "subjects": process_subjects(raw_data.get("subjects_offered", "")),
            
            # 其他信息
            "url": raw_data.get("url", ""),
            "aliases": raw_data.get("aliases", ""),
            "record_type": raw_data.get("record_type", ""),
            
            # 原始JSON数据
            "json_data": json.dumps(raw_data)
        }
        
        return processed_data
    except Exception as e:
        print(f"处理数据时出错: {e}")
        # 返回基本信息和原始JSON，确保数据不丢失
        return {
            "id": raw_data.get("nid", 0),
            "name": raw_data.get("name", "Unknown"),
            "json_data": json.dumps(raw_data),
            "error": str(e)
        }

def generate_vectors(university_data):
    """为大学数据生成各类向量"""
    try:
        # 1. 基本信息向量 - 使用BERT模型
        basic_info_text = f"{university_data['name']} {university_data['location']} {university_data.get('aliases', '')}"
        basic_info_vector = create_text_embedding(basic_info_text)
        
        # 2. 学科信息向量 - 使用BERT模型
        subjects_text = ", ".join(university_data.get("subjects", []))
        subjects_vector = create_text_embedding(subjects_text) if subjects_text else [0] * 768
        
        # 3. 指标向量 - 直接使用评分数据，需要归一化
        metrics = [
            university_data.get("overall_score", 0), 
            university_data.get("teaching_score", 0),
            university_data.get("research_score", 0),
            university_data.get("citations_score", 0),
            university_data.get("industry_income_score", 0),
            university_data.get("international_outlook_score", 0),
            university_data.get("student_staff_ratio", 0),
            university_data.get("pc_intl_students", 0),
            university_data.get("female_male_ratio", {}).get("female", 0.5),
            convert_to_float(university_data.get("number_students", "0")) / 10000  # 缩放学生数量
        ]
        
        # 使用Min-Max缩放进行归一化 (这里假设分数范围是0-100)
        metrics_vector = [min(1.0, max(0.0, m/100)) if i < 6 else m for i, m in enumerate(metrics)]
        
        return {
            "basic_info_vector": basic_info_vector,
            "subjects_vector": subjects_vector,
            "metrics_vector": metrics_vector
        }
    except Exception as e:
        print(f"生成向量时出错: {e}")
        # 返回默认向量
        return {
            "basic_info_vector": [0] * 768,
            "subjects_vector": [0] * 768,
            "metrics_vector": [0] * 10
        }

def process_data():
    """处理THE2025数据并保存到输出文件"""
    print(f"开始处理 THE2025 数据...")
    
    try:
        # 加载原始数据
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        print(f"已加载原始数据，共 {len(raw_data)} 条记录")
        
        # 处理数据
        processed_data = []
        for i, item in enumerate(raw_data):
            # 预处理基本数据
            university_data = preprocess_university_data(item)
            
            # 生成向量
            vectors = generate_vectors(university_data)
            
            # 合并数据
            university_data.update(vectors)
            
            processed_data.append(university_data)
            
            # 显示进度
            if (i+1) % 100 == 0 or i+1 == len(raw_data):
                print(f"已处理 {i+1}/{len(raw_data)} 条记录")
        
        # 保存处理后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成，数据已保存到 {output_file}")
        return processed_data
        
    except Exception as e:
        print(f"处理数据时出错: {e}")
        return None

def setup_milvus_collection():
    """设置Milvus集合"""
    try:
        # 连接到Milvus
        connections.connect(host="localhost", port="19530")
        
        # 定义集合字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="rank", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="overall_score", dtype=DataType.FLOAT),
            FieldSchema(name="basic_info_vector", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="subjects_vector", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="metrics_vector", dtype=DataType.FLOAT_VECTOR, dim=10),
            FieldSchema(name="json_data", dtype=DataType.VARCHAR, max_length=65535)
        ]
        
        # 创建集合模式
        schema = CollectionSchema(fields, "THE2025大学排名数据集")
        
        # 创建集合
        collection_name = "the2025_universities"
        collection = Collection(collection_name, schema)
        
        # 为向量字段创建索引
        # 为基本信息向量创建索引
        index_params = {
            "metric_type": "COSINE",  # 余弦相似度
            "index_type": "HNSW",     # 层次可导航小世界图索引
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index("basic_info_vector", index_params)
        
        # 为学科向量创建索引
        collection.create_index("subjects_vector", index_params)
        
        # 为指标向量创建索引
        collection.create_index("metrics_vector", {
            "metric_type": "L2",      # 欧几里得距离
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        })
        
        print(f"已创建集合和索引: {collection_name}")
        return collection
        
    except Exception as e:
        print(f"设置Milvus集合时出错: {e}")
        return None

if __name__ == "__main__":
    # 处理数据
    processed_data = process_data()
    print("数据处理完成，请使用导入脚本将数据导入到Milvus") 