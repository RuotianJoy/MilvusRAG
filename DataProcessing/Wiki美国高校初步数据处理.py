import json
import re
import os
from tqdm import tqdm
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer

# 加载BERT模型
def load_bert_model():
    try:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        print("成功加载BERT模型")
        return model
    except Exception as e:
        print(f"加载BERT模型失败: {e}")
        return None

# 标准化网址
def normalize_url(url):
    if not url:
        return ""
    
    # 确保URL有协议前缀
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "http://" + url
    
    # 移除末尾的斜杠
    if url.endswith("/"):
        url = url[:-1]
    
    return url

# 处理学生数量为整数
def process_enrollment(enrollment):
    if not enrollment:
        return 0
    
    # 移除非数字字符
    digits_only = re.sub(r'[^\d]', '', enrollment)
    if digits_only:
        return int(digits_only)
    return 0

# 生成嵌入向量
def generate_embedding(text, model):
    try:
        if model:
            # 使用BERT模型生成嵌入
            embedding = model.encode(text)
            return embedding.tolist()
        else:
            # 返回一个全零向量作为替代
            return [0.0] * 384  # paraphrase-MiniLM-L6-v2模型的向量维度为384
    except Exception as e:
        print(f"生成嵌入向量失败: {e}")
        # 返回一个全零向量作为替代
        return [0.0] * 384

# 处理学校数据
def process_college_data(college, model=None):
    # 标准化字符串
    name = college.get("Name", "").strip()
    location = college.get("Location", "Unknown").strip().title() if college.get("Location") else "Unknown"
    control = college.get("Control", "").strip()
    college_type = college.get("Type", "").strip()
    enrollment = process_enrollment(college.get("Enrollment", "0"))
    state = college.get("State", "").strip().title() if college.get("State") else ""
    region = college.get("Region", "").strip()
    website = normalize_url(college.get("Website", ""))
    
    # 创建用于生成嵌入的文本
    embedding_text = f"{name} {college_type} {control} {state} {region}"
    
    # 生成嵌入向量
    if model:
        embedding = generate_embedding(embedding_text, model)
    else:
        # 使用随机向量代替BERT嵌入（用于测试）
        embedding = np.random.random(384).tolist()
    
    return {
        "name": name,
        "location": location,
        "control": control,
        "type": college_type,
        "enrollment": enrollment,
        "state": state,
        "region": region,
        "website": website,
        "text_vector": embedding
    }

# 检查记录是否为空数据
def is_valid_record(college):
    # 检查name字段是否为空
    name = college.get("Name", "").strip()
    return len(name) > 0

# 保存处理后的数据到JSON文件
def save_to_json(processed_data, output_file):
    # 将嵌入向量转换为列表，确保可以被JSON序列化
    serializable_data = []
    for item in processed_data:
        item_copy = item.copy()
        if isinstance(item_copy["text_vector"], np.ndarray):
            item_copy["text_vector"] = item_copy["text_vector"].tolist()
        serializable_data.append(item_copy)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        print(f"数据已成功保存到 {output_file}")
        return True
    except Exception as e:
        print(f"保存数据到JSON文件失败: {e}")
        return False

# 主处理函数
def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="处理Wiki美国高校数据并保存为JSON文件")
    parser.add_argument("--use-bert", action="store_true", help="是否使用BERT模型生成嵌入向量")
    parser.add_argument("--output", default="Wiki美国高校初步数据_processed.json", help="输出JSON文件路径")
    args = parser.parse_args()
    
    # 设置文件路径
    input_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "大学官方数据爬取", 
                            "直接从Wiki获取的US高校初步数据US高校初步数据_with_website.JSON")
    
    # 读取数据
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            colleges = json.load(f)
        total_count = len(colleges)
        print(f"成功加载数据，共有 {total_count} 所学校")
    except Exception as e:
        print(f"读取数据文件失败: {e}")
        return
    
    # 过滤空数据（name字段为空的记录）
    filtered_colleges = [college for college in colleges if is_valid_record(college)]
    filtered_count = len(filtered_colleges)
    removed_count = total_count - filtered_count
    print(f"过滤空数据完成：保留 {filtered_count} 条记录，移除 {removed_count} 条空记录")
    
    # 加载BERT模型（如果需要）
    model = None
    if args.use_bert:
        model = load_bert_model()
    
    # 处理数据
    processed_data = []
    
    print("开始处理学校数据...")
    for college in tqdm(filtered_colleges):
        processed_college = process_college_data(college, model)
        processed_data.append(processed_college)
    
    # 保存到JSON文件
    output_file = args.output
    save_to_json(processed_data, output_file)
    
    print("\n处理完成！数据已成功保存到JSON文件中。")
    print(f"处理后的数据包含 {len(processed_data)} 所学校的信息。")
    print(f"可以使用Wiki美国高校初步数据导入.py将数据导入到Milvus数据库中。")

if __name__ == "__main__":
    main()
