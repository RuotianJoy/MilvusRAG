import json
import re
import os
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import logging
from transformers import BertModel, BertTokenizer
import torch

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载环境变量
load_dotenv()

# 全局BERT模型和分词器
BERT_MODEL = None
BERT_TOKENIZER = None
VECTOR_DIM = 768  # BERT base模型的向量维度

def load_bert_model():
    """加载BERT模型和分词器"""
    global BERT_MODEL, BERT_TOKENIZER
    if BERT_MODEL is None or BERT_TOKENIZER is None:
        logging.info("加载BERT模型和分词器...")
        try:
            # 使用bert-base-uncased模型
            model_name = "bert-base-uncased"
            BERT_TOKENIZER = BertTokenizer.from_pretrained(model_name)
            BERT_MODEL = BertModel.from_pretrained(model_name)
            logging.info("BERT模型和分词器加载成功")
        except Exception as e:
            logging.error(f"加载BERT模型时出错: {e}")
            raise

def is_valid_record(record):
    """检查记录是否有效（至少需要有Name字段）"""
    return record and "Name" in record and record["Name"] and isinstance(record["Name"], str) and record["Name"].strip()

def clean_text(text):
    """清理文本，移除引用标记、HTML标签等"""
    if not text or not isinstance(text, str):
        return None
    
    # 移除引用标记 [1], [2] 等
    text = re.sub(r'\[\d+\]', '', text)
    
    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    
    # 清理多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text if text else None

def parse_location(location_str):
    """将位置字符串解析为城市、州和国家"""
    if not location_str:
        return {"city": None, "state": None, "country": None}
    
    parts = [p.strip() for p in location_str.split(',')]
    
    location = {
        "city": None,
        "state": None,
        "country": None
    }
    
    # 处理不同的位置格式
    if len(parts) >= 3:
        location["city"] = parts[0]
        location["state"] = parts[1]
        location["country"] = parts[2]
    elif len(parts) == 2:
        location["city"] = parts[0]
        location["state"] = parts[1]
        location["country"] = "United States"  # 默认美国
    elif len(parts) == 1:
        location["state"] = parts[0]
        location["country"] = "United States"  # 默认美国
    
    return location

def parse_enrollment(enrollment_str):
    """从入学人数字符串中提取纯数字"""
    if not enrollment_str:
        return None
    
    # 寻找所有数字
    numbers = re.findall(r'[\d,]+', enrollment_str)
    if not numbers:
        return None
    
    # 取第一个找到的数字
    num = numbers[0].replace(',', '')
    try:
        return int(num)
    except ValueError:
        return None

def parse_founded(founded_str):
    """将成立年份标准化为YYYY格式"""
    if not founded_str:
        return None
    
    # 寻找4位数年份
    year_match = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', founded_str)
    if year_match:
        return int(year_match.group(1))
    
    return None

def parse_campus_size(size_str):
    """统一转换校园大小为英亩"""
    if not size_str:
        return {"acres": None, "setting": None}
    
    result = {"acres": None, "setting": None}
    
    # 提取设置描述
    setting_patterns = ["urban", "suburban", "rural", "city", "town", "village"]
    for pattern in setting_patterns:
        if pattern in size_str.lower():
            result["setting"] = pattern
            break
    
    # 提取面积数值
    size_match = re.search(r'([\d,\.]+)\s*(acre|acres|hectare|hectares|km²|square\s*kilometers?|square\s*miles?)', size_str.lower())
    if not size_match:
        return result
    
    try:
        value = float(size_match.group(1).replace(',', ''))
        unit = size_match.group(2)
        
        # 转换为英亩
        if 'hectare' in unit:
            value = value * 2.47105  # 1公顷 = 2.47105英亩
        elif 'km²' in unit or 'square kilometer' in unit:
            value = value * 247.105  # 1平方公里 = 247.105英亩
        elif 'square mile' in unit:
            value = value * 640  # 1平方英里 = 640英亩
        
        result["acres"] = round(value, 2)
    except ValueError:
        pass
    
    return result

def extract_additional_fields(notes):
    """从Additional/Notes字段中提取关键子字段"""
    if not notes:
        return {}, {}
    
    # 定义要提取的关键字段及其正则表达式
    key_fields = {
        "religious_affiliation": [r'religious\s+affiliation[\s:]+([^,\n]+)', r'affiliated\s+with\s+([^,\n]+church|[^,\n]+religious)'],
        "academic_affiliations": [r'academic\s+affiliations?[\s:]+([^.\n]+)', r'member\s+of\s+([^.\n]+)'],
        "president": [r'president[\s:]+([^,\n]+)', r'chancellor[\s:]+([^,\n]+)'],
        "colors": [r'colors?[\s:]+([^.\n]+)'],
        "nickname": [r'nickname[\s:]+([^,\n]+)', r'known\s+as\s+([^,\n]+)'],
        "mascot": [r'mascot[\s:]+([^,\n]+)']
    }
    
    extracted = {}
    remaining = notes
    
    for field, patterns in key_fields.items():
        for pattern in patterns:
            match = re.search(pattern, notes, re.IGNORECASE)
            if match:
                value = clean_text(match.group(1))
                
                # 特殊处理列表类型字段
                if field == "academic_affiliations" or field == "colors":
                    items = [item.strip() for item in value.split(',') if item.strip()]
                    extracted[field] = items
                else:
                    extracted[field] = value
                
                # 从原文本中移除已提取的内容
                remaining = re.sub(pattern, '', remaining, flags=re.IGNORECASE)
                break
    
    # 处理剩余内容
    additional_info = {}
    for line in remaining.split('\n'):
        if ':' in line:
            parts = line.split(':', 1)
            key = clean_text(parts[0]).lower().replace(' ', '_')
            value = clean_text(parts[1])
            if key and value and key not in extracted:
                additional_info[key] = value
    
    return extracted, additional_info

def process_website(url):
    """处理网站URL，确保格式统一"""
    if not url:
        return None
    
    # 清理空格和引号
    url = url.strip().strip('"\'')
    
    # 添加https前缀
    if not (url.startswith('http://') or url.startswith('https://')):
        url = 'https://' + url
    
    return url

def get_embedding(text):
    """使用BERT获取文本嵌入向量"""
    global BERT_MODEL, BERT_TOKENIZER
    
    if not text or text.strip() == "":
        return np.zeros(VECTOR_DIM)  # 返回零向量
    
    # 确保模型已加载
    if BERT_MODEL is None or BERT_TOKENIZER is None:
        load_bert_model()
    
    try:
        # 预处理文本，截断过长的文本
        max_seq_length = 512  # BERT最大序列长度
        if len(text.split()) > max_seq_length * 0.8:  # 按词数粗略估计
            # 截断过长的文本
            words = text.split()
            text = " ".join(words[:int(max_seq_length * 0.8)])
            logging.warning(f"文本被截断，原长度：{len(words)} 词，截断后：{len(text.split())} 词")
        
        # 将文本转换为BERT输入格式
        inputs = BERT_TOKENIZER(text, return_tensors="pt", truncation=True, max_length=max_seq_length)
        
        # 使用模型生成嵌入
        with torch.no_grad():
            outputs = BERT_MODEL(**inputs)
        
        # 使用[CLS]标记的最后隐层状态作为文本表示
        embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
        
        return embedding
    
    except Exception as e:
        logging.error(f"获取向量嵌入时出错: {e}")
        return np.zeros(VECTOR_DIM)  # 出错时返回零向量

def create_text_for_vector(college, vector_type):
    """为不同类型的向量创建文本表示"""
    if vector_type == "basic":
        # 基本信息向量
        parts = [
            college.get("name", ""),
            college.get("type", ""),
            college.get("control", ""),
            college.get("location", {}).get("state", ""),
            college.get("region", "")
        ]
        return " ".join([str(part) for part in parts if part])
    
    elif vector_type == "detail":
        # 详细描述向量
        # 将所有文本字段组合成一段文本
        details = []
        # 添加基本字段
        details.append(f"Name: {college.get('name', '')}")
        
        loc = college.get('location', {})
        location_str = f"Location: {loc.get('city', '')} {loc.get('state', '')} {loc.get('country', '')}"
        details.append(location_str)
        
        details.append(f"Type: {college.get('type', '')}")
        details.append(f"Control: {college.get('control', '')}")
        details.append(f"Region: {college.get('region', '')}")
        details.append(f"Enrollment: {college.get('enrollment', '')}")
        details.append(f"Founded: {college.get('founded', '')}")
        
        campus = college.get('campus_size', {})
        campus_str = f"Campus: {campus.get('acres', '')} acres, {campus.get('setting', '')}"
        details.append(campus_str)
        
        # 添加其他重要字段
        if college.get('religious_affiliation'):
            details.append(f"Religious Affiliation: {college.get('religious_affiliation', '')}")
        
        if college.get('academic_affiliations'):
            affiliations = ", ".join(college.get('academic_affiliations', []))
            details.append(f"Academic Affiliations: {affiliations}")
        
        if college.get('president'):
            details.append(f"President: {college.get('president', '')}")
        
        # 添加额外信息
        additional = college.get('additional_info', {})
        for key, value in additional.items():
            details.append(f"{key.replace('_', ' ').title()}: {value}")
        
        return " ".join([str(detail) for detail in details if detail])
    
    elif vector_type == "feature":
        # 特色向量
        features = []
        if college.get('religious_affiliation'):
            features.append(f"Religious: {college.get('religious_affiliation', '')}")
        
        if college.get('mascot'):
            features.append(f"Mascot: {college.get('mascot', '')}")
        
        if college.get('nickname'):
            features.append(f"Nickname: {college.get('nickname', '')}")
        
        if college.get('colors'):
            colors = ", ".join(college.get('colors', []))
            features.append(f"Colors: {colors}")
        
        # 添加一些区分性特征
        features.append(f"Founded in {college.get('founded', '')}")
        
        campus = college.get('campus_size', {})
        if campus.get('setting'):
            features.append(f"Campus setting: {campus.get('setting', '')}")
        
        return " ".join([str(feature) for feature in features if feature])
    
    return ""

def process_data(input_file, output_file):
    """处理US高校维基百科数据"""
    logging.info(f"开始处理数据: {input_file}")
    
    try:
        # 加载BERT模型
        load_bert_model()
        
        # 读取原始数据
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 过滤无效记录（没有Name字段的记录）
        valid_data = [record for record in raw_data if is_valid_record(record)]
        invalid_count = len(raw_data) - len(valid_data)
        
        logging.info(f"成功读取 {len(raw_data)} 条记录，筛除 {invalid_count} 条无效记录，剩余 {len(valid_data)} 条有效记录进行处理")
        
        # 处理每一所学校的数据
        processed_data = []
        
        for idx, college in enumerate(tqdm(valid_data, desc="处理学校数据")):
            # 生成唯一ID
            college_id = f"us_college_{idx+1:05d}"
            
            # 清理和标准化数据
            name = clean_text(college.get("Name"))
            location = parse_location(clean_text(college.get("Location")))
            control = clean_text(college.get("Control"))
            college_type = clean_text(college.get("Type"))
            enrollment = parse_enrollment(clean_text(college.get("Enrollment")))
            state = clean_text(college.get("State"))
            region = clean_text(college.get("Region"))
            founded = parse_founded(clean_text(college.get("Founded")))
            campus_size = parse_campus_size(clean_text(college.get("Campus Size & Setting")))
            website = process_website(clean_text(college.get("Official Website")))
            
            # 处理Additional/Notes字段
            extracted_fields, additional_info = extract_additional_fields(clean_text(college.get("Additional/Notes")))
            
            # 创建标准化的学校数据结构
            processed_college = {
                "id": college_id,
                "name": name,
                "location": location,
                "control": control,
                "type": college_type,
                "enrollment": enrollment,
                "state": state,
                "region": region,
                "founded": founded,
                "campus_size": campus_size,
                "website": website,
                "religious_affiliation": extracted_fields.get("religious_affiliation"),
                "academic_affiliations": extracted_fields.get("academic_affiliations", []),
                "president": extracted_fields.get("president"),
                "colors": extracted_fields.get("colors", []),
                "nickname": extracted_fields.get("nickname"),
                "mascot": extracted_fields.get("mascot"),
                "additional_info": additional_info,
                "metadata": {
                    "page_title": clean_text(college.get("PageTitle")),
                    "meta_description": clean_text(college.get("MetaDescription")),
                    "wikipedia_url": clean_text(college.get("WikipediaURL"))
                }
            }
            
            # 生成向量嵌入
            basic_text = create_text_for_vector(processed_college, "basic")
            detail_text = create_text_for_vector(processed_college, "detail")
            feature_text = create_text_for_vector(processed_college, "feature")
            
            # 获取向量嵌入
            logging.info(f"生成第 {idx+1} 所学校的向量嵌入")
            basic_vector = get_embedding(basic_text)
            detail_vector = get_embedding(detail_text)
            feature_vector = get_embedding(feature_text)
            
            # 添加向量到学校数据
            processed_college["vectors"] = {
                "basic_vector": basic_vector.tolist(),  # 转换为列表以便JSON序列化
                "detail_vector": detail_vector.tolist(),
                "feature_vector": feature_vector.tolist()
            }
            
            processed_data.append(processed_college)
            
            # 每处理100条记录保存一次
            if (idx + 1) % 100 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                logging.info(f"已处理 {idx+1}/{len(valid_data)} 条记录，临时保存到 {output_file}")
        
        # 最终保存处理后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"数据处理完成，共处理 {len(processed_data)} 条记录，已保存到 {output_file}")
        return processed_data
        
    except Exception as e:
        logging.error(f"处理数据时出错: {e}")
        raise

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_file = os.path.join(project_root, 'DataOriginal\Data', 'US高校维基百科爬虫数据.json')
    output_file = os.path.join(project_root, 'DataProcessed', 'US高校维基百科数据_processed.json')
    
    process_data(input_file, output_file) 
