import json
import os
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, connections

# 定义输入和输出文件路径
# 使用更可靠的路径构建方式
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
input_file = os.path.join(project_root, 'DataOriginal\Data', 'ARWU2024.json')
output_file = os.path.join(project_root, 'DataProcessed', 'ARWU2024_processed.json')

# 大学名称中英文映射表
university_map = {
    "哈佛大学": "Harvard University",
    "斯坦福大学": "Stanford University",
    "麻省理工学院": "Massachusetts Institute of Technology",
    "剑桥大学": "University of Cambridge",
    "加州大学-伯克利": "University of California, Berkeley",
    "牛津大学": "University of Oxford",
    "普林斯顿大学": "Princeton University",
    "哥伦比亚大学": "Columbia University",
    "芝加哥大学": "University of Chicago",
    "耶鲁大学": "Yale University",
    "巴黎萨克雷大学": "Université Paris-Saclay",
    "宾夕法尼亚大学": "University of Pennsylvania",
    "加州大学-洛杉矶": "University of California, Los Angeles",
    "伦敦大学学院": "University College London",
    "约翰斯·霍普金斯大学": "Johns Hopkins University",
    "华盛顿大学": "University of Washington",
    "加州大学-旧金山": "University of California, San Francisco",
    "苏黎世联邦理工学院": "ETH Zurich",
    "清华大学": "Tsinghua University",
    "华盛顿大学-圣路易斯": "Washington University in St. Louis",
    "北京大学": "Peking University",
    "帝国理工学院": "Imperial College London",
    "多伦多大学": "University of Toronto",
    "浙江大学": "Zhejiang University",
    "东京大学": "University of Tokyo",
    "洛克菲勒大学": "Rockefeller University",
    "密歇根大学-安娜堡": "University of Michigan, Ann Arbor",
    "纽约大学": "New York University",
    "哥本哈根大学": "University of Copenhagen",
    "巴黎文理研究大学": "Paris Sciences et Lettres University",
    "北卡罗来纳大学-教堂山": "University of North Carolina at Chapel Hill",
    "威斯康星大学-麦迪逊": "University of Wisconsin-Madison",
    "墨尔本大学": "University of Melbourne",
    "上海交通大学": "Shanghai Jiao Tong University",
    "杜克大学": "Duke University",
    "爱丁堡大学": "University of Edinburgh",
    "索邦大学": "Sorbonne University",
    "中国科学技术大学": "University of Science and Technology of China",
    "慕尼黑大学": "Ludwig Maximilian University of Munich",
    "德克萨斯州大学-奥斯汀": "University of Texas at Austin",
    "明尼苏达大学-双城": "University of Minnesota, Twin Cities",
    "海德堡大学": "Heidelberg University",
    "曼彻斯特大学": "University of Manchester",
    "伦敦国王学院": "King's College London",
    "德克萨斯大学西南医学中心": "University of Texas Southwestern Medical Center",
    "乌得勒支大学": "Utrecht University",
    "马里兰大学-大学城": "University of Maryland, College Park",
    "巴黎西岱大学": "Université Paris Cité",
    "波恩大学": "University of Bonn",
    "南加州大学": "University of Southern California",
    "昆士兰大学": "University of Queensland",
    "加州大学-圣塔芭芭拉": "University of California, Santa Barbara",
    "科罗拉多大学-博尔德": "University of Colorado Boulder",
    "范德堡大学": "Vanderbilt University",
    "苏黎世大学": "University of Zurich",
    "新加坡国立大学": "National University of Singapore",
    "魏茨曼科学研究学院": "Weizmann Institute of Science",
    "奥斯陆大学": "University of Oslo",
    "悉尼大学": "University of Sydney",
    "加州大学-欧文": "University of California, Irvine",
    "新南威尔士大学": "University of New South Wales",
    "鲁汶大学（佛兰德语）": "KU Leuven",
    "华中科技大学": "Huazhong University of Science and Technology",
    "奥胡斯大学": "Aarhus University",
    "耶路撒冷希伯来大学": "Hebrew University of Jerusalem",
    "俄亥俄州立大学-哥伦布": "Ohio State University, Columbus",
    "以色列理工学院": "Technion – Israel Institute of Technology",
    "德克萨斯大学安德森肿瘤中心": "University of Texas MD Anderson Cancer Center",
    "乌普萨拉大学": "Uppsala University",
    "武汉大学": "Wuhan University",
    "匹兹堡大学": "University of Pittsburgh",
    "中南大学": "Central South University",
    "西安交通大学": "Xi'an Jiaotong University",
    "布里斯托尔大学": "University of Bristol",
    "四川大学": "Sichuan University",
    "赫尔辛基大学": "University of Helsinki",
    "普渡大学-西拉法叶": "Purdue University, West Lafayette",
    "华威大学": "University of Warwick",
    "郑州大学": "Zhengzhou University",
    "延世大学": "Yonsei University",
    "浙江工业大学": "Zhejiang University of Technology",
    "云南大学": "Yunnan University",
    "浙江理工大学": "Zhejiang Sci-Tech University",
    "浙江中医药大学": "Zhejiang Chinese Medical University",
    "浙江师范大学": "Zhejiang Normal University",
    "中南财经政法大学": "Zhongnan University of Economics and Law",
    "浙江农林大学": "Zhejiang A&F University"
}

# 国家名称中英文映射表
country_map = {
    "美国": "United States",
    "英国": "United Kingdom",
    "中国": "China",
    "法国": "France",
    "德国": "Germany",
    "瑞士": "Switzerland",
    "日本": "Japan",
    "丹麦": "Denmark",
    "加拿大": "Canada",
    "澳大利亚": "Australia",
    "以色列": "Israel",
    "挪威": "Norway",
    "比利时": "Belgium",
    "韩国": "South Korea",
    "新加坡": "Singapore",
    "瑞典": "Sweden",
    "荷兰": "Netherlands"
}

# 国家到大洲的映射
continent_map = {
    "美国": "North America",
    "加拿大": "North America",
    "英国": "Europe",
    "法国": "Europe",
    "德国": "Europe",
    "瑞士": "Europe",
    "丹麦": "Europe",
    "挪威": "Europe",
    "比利时": "Europe",
    "瑞典": "Europe",
    "荷兰": "Europe",
    "中国": "Asia",
    "日本": "Asia",
    "韩国": "Asia",
    "新加坡": "Asia",
    "澳大利亚": "Oceania",
    "以色列": "Asia"
}

def create_basic_score_vector(item):
    """创建基于原始评分的7维向量"""
    return [
        float(item["Total_Score"] or 0),
        float(item["Alumni_Award"] or 0),
        float(item["Prof_Award"] or 0),
        float(item["High_cited_Scientist"] or 0),
        float(item["NS_Paper"] or 0),
        float(item["Inter_Paper"] or 0),
        float(item["Avg_Prof_Performance"] or 0)
    ]

def create_enhanced_vector(item):
    """创建增强特征向量，包含基本评分和计算的特征"""
    base_vector = create_basic_score_vector(item)
    # 计算衍生特征
    research_strength = (float(item["NS_Paper"] or 0) + float(item["Inter_Paper"] or 0)) / 2
    talent_quality = (float(item["Prof_Award"] or 0) + float(item["High_cited_Scientist"] or 0)) / 2
    rank_factor = 1.0 / (float(item["rank_numeric"]) / 100 + 1)  # 排名因子
    # 添加到向量
    enhanced_vector = base_vector + [research_strength, talent_quality, rank_factor]
    return enhanced_vector

def create_text_embedding(text, model_name="bert-base-multilingual-cased"):
    """为大学创建文本嵌入向量"""
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

def process_rank_range(rank_str):
    """处理排名范围，支持例如"801-900"这样的格式"""
    if '-' in rank_str:
        # 处理类似"801-900"的情况
        rank_parts = rank_str.split('-')
        rank_lower = int(rank_parts[0])
        rank_upper = int(rank_parts[1])
        rank_numeric = (rank_lower + rank_upper) // 2
    else:
        # 处理单个数字的情况
        try:
            rank_numeric = int(rank_str)
            rank_lower = rank_numeric
            rank_upper = rank_numeric
        except ValueError:
            # 处理无法转换为整数的情况
            rank_numeric = 0
            rank_lower = 0
            rank_upper = 0
    
    return {
        "rank_lower": rank_lower,
        "rank_upper": rank_upper,
        "rank_numeric": rank_numeric
    }

def handle_missing_values(item):
    """处理缺失值，将所有数值字段的空值填充为0"""
    numeric_fields = [
        "Total_Score", 
        "Alumni_Award", 
        "Prof_Award", 
        "High_cited_Scientist", 
        "NS_Paper", 
        "Inter_Paper", 
        "Avg_Prof_Performance"
    ]
    
    for field in numeric_fields:
        value = item.get(field, "")
        if value is None or value == "":
            item[field] = 0.0
        else:
            try:
                item[field] = float(value)
            except (ValueError, TypeError):
                item[field] = 0.0
    
    return item

# 数据处理部分
def process_data():
    """处理ARWU排名数据并进行向量化，但不导入到Milvus"""
    # 检查输入文件存在否
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在!")
        # 尝试查找替代路径
        possible_paths = [
            os.path.join(project_root, 'ARWU2024.json'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), '第三方排名网站数据爬取', 'ARWU2024.json'),
            os.path.join(os.path.dirname(__file__), 'ARWU2024.json')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"找到替代输入文件: {path}")
                input_file_to_use = path
                break
        else:
            print("无法找到有效的输入文件，终止处理")
            return None
    else:
        input_file_to_use = input_file
        
    print(f"处理输入文件: {input_file_to_use}")
    
    # 读取输入文件
    with open(input_file_to_use, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 定义数值字段列表
    numerical_fields = [
        "Total_Score", 
        "Alumni_Award", 
        "Prof_Award", 
        "High_cited_Scientist", 
        "NS_Paper", 
        "Inter_Paper", 
        "Avg_Prof_Performance"
    ]
    
    # 初始化field_min_max字典，用于记录每个字段的最大最小值
    field_min_max = {}
    for field in numerical_fields:
        field_min_max[field] = {"min": float('inf'), "max": float('-inf')}
    
    # 首先收集所有有效值以计算最大最小值
    for item in data:
        for field in numerical_fields:
            if field in item and item[field] not in [None, "", "N/A"]:
                try:
                    value = float(item[field])
                    if value < field_min_max[field]["min"]:
                        field_min_max[field]["min"] = value
                    if value > field_min_max[field]["max"]:
                        field_min_max[field]["max"] = value
                except (ValueError, TypeError):
                    continue
    
    # 处理每所大学的数据
    processed_data = []
    
    for item in data:
        processed_item = {}
        
        # 复制原始数据
        for key, value in item.items():
            processed_item[key] = value
        
        # 处理Rank字段
        rank_info = process_rank_range(item.get("Rank", "N/A"))
        processed_item.update(rank_info)
        
        # 处理Region_Rank字段
        region_rank_info = process_rank_range(item.get("Region_Rank", "N/A"))
        processed_item["region_rank_lower"] = region_rank_info["rank_lower"]
        processed_item["region_rank_upper"] = region_rank_info["rank_upper"]
        processed_item["region_rank_numeric"] = region_rank_info["rank_numeric"]
        
        # 添加大学和国家的英文名称
        processed_item["University_English"] = university_map.get(item["University"], "")
        processed_item["Country_English"] = country_map.get(item["Country"], "")
        
        # 添加大陆分类
        processed_item["Continent"] = continent_map.get(item["Country"], "Unknown")
        
        # 处理评分指标(转换为数值并处理空值)
        for field in numerical_fields:
            try:
                if field in item and item[field] not in [None, "", "N/A"]:
                    processed_item[field] = float(item[field])
                else:
                    processed_item[field] = 0.0
            except (ValueError, TypeError):
                processed_item[field] = 0.0
        
        # 创建特征向量
        processed_item["basic_score_vector"] = create_basic_score_vector(processed_item)
        processed_item["enhanced_vector"] = create_enhanced_vector(processed_item)
        
        try:
            # 创建文本嵌入向量
            text = f"{item['University']} {university_map.get(item['University'], '')} {item['Country']} {country_map.get(item['Country'], '')}"
            processed_item["text_embedding"] = create_text_embedding(text)
        except Exception as e:
            print(f"处理大学 {item['University']} 的文本向量时出错: {str(e)}")
            processed_item["text_embedding"] = [0] * 768  # BERT默认维度
        
        # 添加归一化字段
        for field in numerical_fields:
            if field_min_max[field]["max"] != field_min_max[field]["min"]:
                try:
                    if processed_item[field] is not None:
                        normalized_value = (float(processed_item[field]) - field_min_max[field]["min"]) / (field_min_max[field]["max"] - field_min_max[field]["min"])
                        processed_item[f"{field}_normalized"] = normalized_value
                    else:
                        processed_item[f"{field}_normalized"] = None
                except (ValueError, TypeError):
                    processed_item[f"{field}_normalized"] = None
            else:
                processed_item[f"{field}_normalized"] = None
        
        processed_data.append(processed_item)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据处理完成，结果已保存至: {output_file}")
    return processed_data


# 5. Milvus集合设计
def setup_milvus_collection():
    """设置Milvus集合"""
    # 连接到Milvus服务器
    # 尝试不同的连接方式
    connection_success = False
    
    # 尝试方法1: 默认localhost
    try:
        connections.connect("default", host="localhost", port="19530", timeout=10.0)
        print("成功连接到Milvus (localhost:19530)")
        connection_success = True
    except Exception as e:
        print(f"连接到localhost:19530失败: {str(e)}")
    
    # 尝试方法2: 使用Docker主机IP
    if not connection_success:
        try:
            # 如果是Docker桥接网络，通常使用host.docker.internal或宿主机IP
            connections.connect("default", host="host.docker.internal", port="19530", timeout=10.0)
            print("成功连接到Milvus (host.docker.internal:19530)")
            connection_success = True
        except Exception as e:
            print(f"连接到host.docker.internal:19530失败: {str(e)}")
    
    # 尝试方法3: 直接使用容器名称
    if not connection_success:
        try:
            # 如果在同一Docker网络中，可以直接使用容器名称
            connections.connect("default", host="milvus", port="19530", timeout=10.0)
            print("成功连接到Milvus (milvus:19530)")
            connection_success = True
        except Exception as e:
            print(f"连接到milvus:19530失败: {str(e)}")
    
    if not connection_success:
        print("无法连接到Milvus服务器。请检查Docker设置和网络连接。")
        print("您可以通过以下方法修复此问题:")
        print("1. 确保Milvus容器正在运行: docker ps | grep milvus")
        print("2. 检查端口映射: docker ps --format '{{.Names}} {{.Ports}}' | grep milvus")
        print("3. 手动指定Milvus服务器IP和端口: 修改代码中的host和port参数")
        return None
    
    collections = {}
    
    # 创建基本评分向量集合
    try:
        # 5.1 基本评分向量集合字段定义
        score_fields = [
            # ID字段
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            # 基础信息字段
            FieldSchema(name="university", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="university_en", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="country", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="country_en", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="continent", dtype=DataType.VARCHAR, max_length=50),
            # 排名字段
            FieldSchema(name="rank", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="rank_lower", dtype=DataType.INT64),
            FieldSchema(name="rank_upper", dtype=DataType.INT64),
            FieldSchema(name="rank_numeric", dtype=DataType.DOUBLE),
            FieldSchema(name="region_rank", dtype=DataType.VARCHAR, max_length=20),
            # 评分字段
            FieldSchema(name="total_score", dtype=DataType.DOUBLE),
            FieldSchema(name="alumni_award", dtype=DataType.DOUBLE),
            FieldSchema(name="prof_award", dtype=DataType.DOUBLE),
            FieldSchema(name="high_cited_scientist", dtype=DataType.DOUBLE),
            FieldSchema(name="ns_paper", dtype=DataType.DOUBLE),
            FieldSchema(name="inter_paper", dtype=DataType.DOUBLE),
            FieldSchema(name="avg_prof_performance", dtype=DataType.DOUBLE),
            # 向量字段 - 只保留一个向量字段
            FieldSchema(name="score_vector", dtype=DataType.FLOAT_VECTOR, dim=7)
        ]
        
        # 创建集合模式
        score_schema = CollectionSchema(fields=score_fields, description="ARWU2024 University Rankings - Score Vectors")
        
        # 创建集合
        score_collection_name = "arwu_universities_2024_score"
        try:
            # 如果集合已存在，获取它
            score_collection = Collection(name=score_collection_name)
            print(f"集合 {score_collection_name} 已存在")
        except Exception:
            # 如果集合不存在，创建它
            score_collection = Collection(name=score_collection_name, schema=score_schema)
            print(f"已创建新集合 {score_collection_name}")
        
        collections["score"] = score_collection
        
        # 创建分区
        partition_name = "ARWU2024"
        try:
            partitions = score_collection.partitions
            partition_names = [p.name for p in partitions]
            if partition_name not in partition_names:
                score_collection.create_partition(partition_name)
                print(f"已为 {score_collection_name} 创建分区 {partition_name}")
        except Exception as e:
            print(f"检查分区时出错: {str(e)}")
            score_collection.create_partition(partition_name)
            print(f"已为 {score_collection_name} 创建分区 {partition_name}")
        
        # 为向量字段创建索引
        index_params = {
            "metric_type": "L2",  # 使用欧氏距离
            "index_type": "IVF_FLAT",  # 使用IVF_FLAT索引类型
            "params": {"nlist": 100}  # 聚类中心数量
        }
        
        try:
            score_collection.create_index(field_name="score_vector", index_params=index_params)
            print(f"已为 {score_collection_name} 创建索引")
        except Exception as e:
            print(f"创建索引时出错: {str(e)}")
            # 索引可能已存在，继续执行
            
    except Exception as e:
        print(f"创建评分向量集合时出错: {str(e)}")
    
    # 创建增强特征向量集合
    try:
        # 5.2 增强特征向量集合字段定义
        enhanced_fields = [
            # ID字段
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            # 基础信息字段
            FieldSchema(name="university", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="university_en", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="country", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="country_en", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="continent", dtype=DataType.VARCHAR, max_length=50),
            # 排名字段
            FieldSchema(name="rank", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="rank_numeric", dtype=DataType.DOUBLE),
            # 向量字段 - 只保留一个向量字段
            FieldSchema(name="enhanced_vector", dtype=DataType.FLOAT_VECTOR, dim=10)
        ]
        
        # 创建集合模式
        enhanced_schema = CollectionSchema(fields=enhanced_fields, description="ARWU2024 University Rankings - Enhanced Vectors")
        
        # 创建集合
        enhanced_collection_name = "arwu_universities_2024_enhanced"
        try:
            # 如果集合已存在，获取它
            enhanced_collection = Collection(name=enhanced_collection_name)
            print(f"集合 {enhanced_collection_name} 已存在")
        except Exception:
            # 如果集合不存在，创建它
            enhanced_collection = Collection(name=enhanced_collection_name, schema=enhanced_schema)
            print(f"已创建新集合 {enhanced_collection_name}")
        
        collections["enhanced"] = enhanced_collection
        
        # 创建分区
        partition_name = "ARWU2024"
        try:
            partitions = enhanced_collection.partitions
            partition_names = [p.name for p in partitions]
            if partition_name not in partition_names:
                enhanced_collection.create_partition(partition_name)
                print(f"已为 {enhanced_collection_name} 创建分区 {partition_name}")
        except Exception as e:
            print(f"检查分区时出错: {str(e)}")
            enhanced_collection.create_partition(partition_name)
            print(f"已为 {enhanced_collection_name} 创建分区 {partition_name}")
        
        # 为向量字段创建索引
        index_params = {
            "metric_type": "L2",  # 使用欧氏距离
            "index_type": "IVF_FLAT",  # 使用IVF_FLAT索引类型
            "params": {"nlist": 100}  # 聚类中心数量
        }
        
        try:
            enhanced_collection.create_index(field_name="enhanced_vector", index_params=index_params)
            print(f"已为 {enhanced_collection_name} 创建索引")
        except Exception as e:
            print(f"创建索引时出错: {str(e)}")
            # 索引可能已存在，继续执行
            
    except Exception as e:
        print(f"创建增强特征向量集合时出错: {str(e)}")
    
    # 创建文本嵌入向量集合
    try:
        # 5.3 文本嵌入向量集合字段定义
        text_fields = [
            # ID字段
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            # 基础信息字段
            FieldSchema(name="university", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="university_en", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="country", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="country_en", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="continent", dtype=DataType.VARCHAR, max_length=50),
            # 排名字段
            FieldSchema(name="rank", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="rank_numeric", dtype=DataType.DOUBLE),
            # 向量字段 - 只保留一个向量字段
            FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=768)
        ]
        
        # 创建集合模式
        text_schema = CollectionSchema(fields=text_fields, description="ARWU2024 University Rankings - Text Vectors")
        
        # 创建集合
        text_collection_name = "arwu_universities_2024_text"
        try:
            # 如果集合已存在，获取它
            text_collection = Collection(name=text_collection_name)
            print(f"集合 {text_collection_name} 已存在")
        except Exception:
            # 如果集合不存在，创建它
            text_collection = Collection(name=text_collection_name, schema=text_schema)
            print(f"已创建新集合 {text_collection_name}")
        
        collections["text"] = text_collection
        
        # 创建分区
        partition_name = "ARWU2024"
        try:
            partitions = text_collection.partitions
            partition_names = [p.name for p in partitions]
            if partition_name not in partition_names:
                text_collection.create_partition(partition_name)
                print(f"已为 {text_collection_name} 创建分区 {partition_name}")
        except Exception as e:
            print(f"检查分区时出错: {str(e)}")
            text_collection.create_partition(partition_name)
            print(f"已为 {text_collection_name} 创建分区 {partition_name}")
        
        # 为文本向量创建HNSW索引(适合高维向量)
        text_index_params = {
            "metric_type": "IP",  # 内积距离(余弦相似度)
            "index_type": "HNSW",  # 使用HNSW索引
            "params": {
                "M": 16,  # HNSW图每个节点的最大连接数
                "efConstruction": 200  # 构建索引时的搜索宽度
            }
        }
        
        try:
            text_collection.create_index(field_name="text_vector", index_params=text_index_params)
            print(f"已为 {text_collection_name} 创建索引")
        except Exception as e:
            print(f"创建索引时出错: {str(e)}")
            # 索引可能已存在，继续执行
            
    except Exception as e:
        print(f"创建文本嵌入向量集合时出错: {str(e)}")
    
    return collections

def process_and_import_data(json_file_path, collections, partition_name="ARWU2024"):
    """处理并导入ARWU数据到Milvus集合的指定分区"""
    # 连接Milvus服务已在setup_milvus_collection中完成，此处不需要重复连接
    
    if not collections:
        print("没有可用的集合，无法导入数据")
        return None
        
    # 加载JSON数据
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 处理数据
    processed_data = []
    for idx, item in enumerate(data):
        # 处理基本信息
        processed_item = {
            "university": item["University"],
            "university_en": university_map.get(item["University"], ""),
            "country": item["Country"],
            "country_en": country_map.get(item["Country"], ""),
            "continent": continent_map.get(item["Country"], "Unknown"),
            "rank": item["Rank"],
            "region_rank": item["Region_Rank"]
        }
            
        # 处理排名
        rank_info = process_rank_range(item["Rank"])
        processed_item.update(rank_info)
            
        # 处理地区排名
        region_rank_info = process_rank_range(item["Region_Rank"])
        processed_item["region_rank_lower"] = region_rank_info["rank_lower"]
        processed_item["region_rank_upper"] = region_rank_info["rank_upper"]
        processed_item["region_rank_numeric"] = region_rank_info["rank_numeric"]
            
        # 处理评分指标
        for field in ["Total_Score", "Alumni_Award", "Prof_Award", "High_cited_Scientist", 
                     "NS_Paper", "Inter_Paper", "Avg_Prof_Performance"]:
            processed_item[field.lower()] = item.get(field, 0)
            
        processed_item = handle_missing_values(processed_item)
            
        # 生成向量
        processed_item["score_vector"] = create_basic_score_vector(item)
        processed_item["enhanced_vector"] = create_enhanced_vector(processed_item)
            
        try:
            # 创建文本描述
            text = f"{item['University']} {university_map.get(item['University'], '')} {item['Country']} {country_map.get(item['Country'], '')}"
            # 使用与向量化相同的BERT模型生成查询向量
            processed_item["text_embedding"] = create_text_embedding(text)
        except Exception as e:
            print(f"处理大学 {item['University']} 的文本向量时出错: {str(e)}")
            processed_item["text_embedding"] = [0] * 768  # BERT默认维度
            
        processed_data.append(processed_item)
        if (idx + 1) % 10 == 0:
            print(f"已处理 {idx + 1}/{len(data)} 条记录")
    
    # 向每个集合中导入数据
    success = True
    
    # 向评分向量集合导入数据
    if "score" in collections:
        score_collection = collections["score"]
        try:
            # 准备插入数据
            score_entities = {}
            for field in score_collection.schema.fields:
                if field.name != "id":  # 跳过ID字段，使用自动ID
                    if field.name == "score_vector":
                        # 处理向量字段
                        score_entities[field.name] = [item["score_vector"] for item in processed_data]
                    else:
                        # 获取字段名（转换为小写以匹配处理后的数据）
                        field_name_lower = field.name.lower()
                        
                        # 根据字段名获取对应的值
                        if field.name == "university":
                            score_entities[field.name] = [item["university"] for item in processed_data]
                        elif field.name == "university_en":
                            score_entities[field.name] = [item["university_en"] for item in processed_data]
                        elif field.name == "country":
                            score_entities[field.name] = [item["country"] for item in processed_data]
                        elif field.name == "country_en":
                            score_entities[field.name] = [item["country_en"] for item in processed_data]
                        elif field.name == "continent":
                            score_entities[field.name] = [item["continent"] for item in processed_data]
                        elif field.name == "rank":
                            score_entities[field.name] = [item["rank"] for item in processed_data]
                        elif field.name == "region_rank":
                            score_entities[field.name] = [item["region_rank"] for item in processed_data]
                        elif field.name == "rank_lower":
                            score_entities[field.name] = [item["rank_lower"] for item in processed_data]
                        elif field.name == "rank_upper":
                            score_entities[field.name] = [item["rank_upper"] for item in processed_data]
                        elif field.name == "rank_numeric":
                            score_entities[field.name] = [item["rank_numeric"] for item in processed_data]
                        elif field.name == "total_score":
                            score_entities[field.name] = [item["total_score"] for item in processed_data]
                        elif field.name == "alumni_award":
                            score_entities[field.name] = [item["alumni_award"] for item in processed_data]
                        elif field.name == "prof_award":
                            score_entities[field.name] = [item["prof_award"] for item in processed_data]
                        elif field.name == "high_cited_scientist":
                            score_entities[field.name] = [item["high_cited_scientist"] for item in processed_data]
                        elif field.name == "ns_paper":
                            score_entities[field.name] = [item["ns_paper"] for item in processed_data]
                        elif field.name == "inter_paper":
                            score_entities[field.name] = [item["inter_paper"] for item in processed_data]
                        elif field.name == "avg_prof_performance":
                            score_entities[field.name] = [item["avg_prof_performance"] for item in processed_data]
                        else:
                            # 对于其他字段，尝试使用小写名称
                            score_entities[field.name] = [item.get(field_name_lower, None) for item in processed_data]
            
            # 批量插入数据到指定分区
            insert_result = score_collection.insert(score_entities, partition_name=partition_name)
            print(f"成功向分区 {partition_name} 插入 {insert_result.insert_count} 条评分向量记录")
                
            # 刷新集合以确保数据可用
            score_collection.flush()
            score_collection.load()
        except Exception as e:
            print(f"导入数据到评分向量集合时出错: {str(e)}")
            success = False
    
    # 向增强特征向量集合导入数据
    if "enhanced" in collections:
        enhanced_collection = collections["enhanced"]
        try:
            # 准备插入数据
            enhanced_entities = {}
            for field in enhanced_collection.schema.fields:
                if field.name != "id":  # 跳过ID字段，使用自动ID
                    if field.name == "enhanced_vector":
                        # 处理向量字段
                        enhanced_entities[field.name] = [item["enhanced_vector"] for item in processed_data]
                    else:
                        # 获取字段名（转换为小写以匹配处理后的数据）
                        field_name_lower = field.name.lower()
                        
                        # 根据字段名获取对应的值
                        if field.name == "university":
                            enhanced_entities[field.name] = [item["university"] for item in processed_data]
                        elif field.name == "university_en":
                            enhanced_entities[field.name] = [item["university_en"] for item in processed_data]
                        elif field.name == "country":
                            enhanced_entities[field.name] = [item["country"] for item in processed_data]
                        elif field.name == "country_en":
                            enhanced_entities[field.name] = [item["country_en"] for item in processed_data]
                        elif field.name == "continent":
                            enhanced_entities[field.name] = [item["continent"] for item in processed_data]
                        elif field.name == "rank":
                            enhanced_entities[field.name] = [item["rank"] for item in processed_data]
                        elif field.name == "rank_numeric":
                            enhanced_entities[field.name] = [item["rank_numeric"] for item in processed_data]
                        else:
                            # 对于其他字段，尝试使用小写名称
                            enhanced_entities[field.name] = [item.get(field_name_lower, None) for item in processed_data]
            
            # 批量插入数据到指定分区
            insert_result = enhanced_collection.insert(enhanced_entities, partition_name=partition_name)
            print(f"成功向分区 {partition_name} 插入 {insert_result.insert_count} 条增强特征向量记录")
                
            # 刷新集合以确保数据可用
            enhanced_collection.flush()
            enhanced_collection.load()
        except Exception as e:
            print(f"导入数据到增强特征向量集合时出错: {str(e)}")
            success = False
    
    # 向文本嵌入向量集合导入数据
    if "text" in collections:
        text_collection = collections["text"]
        try:
            # 准备插入数据
            text_entities = {}
            for field in text_collection.schema.fields:
                if field.name != "id":  # 跳过ID字段，使用自动ID
                    if field.name == "text_vector":
                        # 处理向量字段
                        text_entities[field.name] = [item["text_embedding"] for item in processed_data]
                    else:
                        # 获取字段名（转换为小写以匹配处理后的数据）
                        field_name_lower = field.name.lower()
                        
                        # 根据字段名获取对应的值
                        if field.name == "university":
                            text_entities[field.name] = [item["university"] for item in processed_data]
                        elif field.name == "university_en":
                            text_entities[field.name] = [item["university_en"] for item in processed_data]
                        elif field.name == "country":
                            text_entities[field.name] = [item["country"] for item in processed_data]
                        elif field.name == "country_en":
                            text_entities[field.name] = [item["country_en"] for item in processed_data]
                        elif field.name == "continent":
                            text_entities[field.name] = [item["continent"] for item in processed_data]
                        elif field.name == "rank":
                            text_entities[field.name] = [item["rank"] for item in processed_data]
                        elif field.name == "rank_numeric":
                            text_entities[field.name] = [item["rank_numeric"] for item in processed_data]
                        else:
                            # 对于其他字段，尝试使用小写名称
                            text_entities[field.name] = [item.get(field_name_lower, None) for item in processed_data]
            
            # 批量插入数据到指定分区
            insert_result = text_collection.insert(text_entities, partition_name=partition_name)
            print(f"成功向分区 {partition_name} 插入 {insert_result.insert_count} 条文本嵌入向量记录")
                
            # 刷新集合以确保数据可用
            text_collection.flush()
            text_collection.load()
        except Exception as e:
            print(f"导入数据到文本嵌入向量集合时出错: {str(e)}")
            success = False
    
    if success:
        print("所有数据成功导入到Milvus")
        return collections
    else:
        print("部分数据导入失败，请检查错误信息")
        return collections

# 6. 实用查询示例
def search_similar_by_scores(collections, reference_university, top_k=5):
    """查找评分结构相似的大学"""
    if "score" not in collections:
        return f"评分向量集合不可用"
    
    score_collection = collections["score"]
    
    # 先获取参考大学的评分向量
    res = score_collection.query(
        expr=f'university == "{reference_university}"',
        output_fields=["score_vector"]
    )
    if not res:
        return f"未找到大学: {reference_university}"
        
    reference_vector = res[0]["score_vector"]
        
    # 使用向量搜索
    search_params = {"metric_type": "L2", "params": {"ef": 100}}
    results = score_collection.search(
        data=[reference_vector],
        anns_field="score_vector",
        param=search_params,
        limit=top_k+1,  # 多搜索一个，因为第一个可能是参考大学自己
        output_fields=["university", "university_en", "country", "rank"]
    )
        
    # 过滤掉参考大学自己
    filtered_results = [hit for hit in results[0] 
                        if hit.entity.get("university") != reference_university]
        
    return filtered_results[:top_k]

def search_by_description(collections, description, top_k=5):
    """根据文本描述搜索大学"""
    if "text" not in collections:
        return f"文本嵌入向量集合不可用"
    
    text_collection = collections["text"]
    
    # 使用与向量化相同的BERT模型生成查询向量
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = BertModel.from_pretrained("bert-base-multilingual-cased")
        
    inputs = tokenizer(description, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    query_vector = outputs.last_hidden_state[:, 0, :].numpy().flatten().tolist()
        
    # 执行向量搜索
    search_params = {"metric_type": "IP", "params": {"ef": 100}}
    results = text_collection.search(
        data=[query_vector],
        anns_field="text_vector",
        param=search_params,
        limit=top_k,
        output_fields=["university", "university_en", "country", "rank"]
    )
        
    return results[0]

def hybrid_search(collections, region="Asia", min_research_score=40.0, top_k=5):
    """结合属性过滤和向量相似度的混合查询"""
    if "score" not in collections:
        return f"评分向量集合不可用"
    
    score_collection = collections["score"]
    
    # 基于研究实力创建目标向量(重点关注研究相关指标)
    target_vector = [50.0, 20.0, 20.0, 40.0, 80.0, 80.0, 40.0]
        
    # 执行搜索
    search_params = {"metric_type": "L2", "params": {"ef": 100}}
    results = score_collection.search(
        data=[target_vector],
        anns_field="score_vector",
        param=search_params,
        expr=f'continent == "{region}"',
        limit=top_k,
        output_fields=["university", "university_en", "country", "rank", "ns_paper", "inter_paper"]
    )
        
    return results[0]

def milvus_pipeline(json_file_path=input_file):
    """完整的Milvus数据处理与导入流程"""
    print("开始Milvus数据处理与导入流程...")
        
    # 设置集合
    collections = setup_milvus_collection()
    if not collections:
        print("创建集合或连接到Milvus失败，终止流程")
        print("您可以继续使用数据处理功能，但向量数据库功能将不可用")
        print("要解决此问题，请确保:")
        print("1. Milvus服务器已运行并可访问")
        print("2. 正确配置了网络设置（特别是在Docker环境中）")
        print("3. 您可以尝试手动修改连接参数后重试")
        return None
        
    # 处理并导入数据
    collections = process_and_import_data(json_file_path, collections)
    if not collections:
        print("数据导入失败，终止流程")
        return None
        
    print("数据处理与导入完成，Milvus集合已准备就绪")
    return collections

# 主函数 - 仅数据处理
if __name__ == "__main__":
    # 处理基础数据
    processed_data = process_data()
    print("基础数据处理完成!")
    
    # # 尝试调用Milvus向量库功能（可选）
    # try:
    #     print("正在尝试连接Milvus向量数据库...")
    #     collections = milvus_pipeline()
    #     if collections:
    #         print("Milvus向量数据库功能已成功设置!")
    #         print("集合信息:")
    #         if "score" in collections:
    #             print(f"- 评分向量集合 (7维): {collections['score'].name}")
    #         if "enhanced" in collections:
    #             print(f"- 增强特征向量集合 (10维): {collections['enhanced'].name}")
    #         if "text" in collections:
    #             print(f"- 文本嵌入向量集合 (768维): {collections['text'].name}")
    #
    #         print("\n您现在可以使用以下功能:")
    #         print("1. 搜索评分相似大学: search_similar_by_scores(collections, '哈佛大学')")
    #         print("2. 文本描述搜索: search_by_description(collections, '亚洲顶尖研究型大学')")
    #         print("3. 混合搜索: hybrid_search(collections, region='Asia')")
    #     else:
    #         print("Milvus向量数据库功能不可用，仅完成了基础数据处理")
    # except Exception as e:
    #     print(f"Milvus向量数据库功能出错: {str(e)}")
    #     print("已完成基础数据处理，向量数据库功能不可用")