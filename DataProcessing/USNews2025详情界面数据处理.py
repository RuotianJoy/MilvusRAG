#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
import uuid
import os

def clean_rank_data(rank_str):
    """
    处理排名数据，将'#1 in Best Global Universities'转换为数字1，并检测是否为并列排名
    """
    if not rank_str or not isinstance(rank_str, str):
        return {"numeric_rank": None, "display_rank": rank_str, "is_tied": False}
    
    is_tied = "(tie)" in rank_str
    # 提取数字部分
    match = re.search(r'#(\d+)', rank_str)
    if match:
        numeric_rank = int(match.group(1))
        return {"numeric_rank": numeric_rank, "display_rank": rank_str, "is_tied": is_tied}
    return {"numeric_rank": None, "display_rank": rank_str, "is_tied": is_tied}

def clean_number(number_str):
    """
    将带逗号的数字字符串转换为整数，如"20,050" -> 20050
    """
    if not number_str or not isinstance(number_str, str):
        return number_str
    
    # 移除所有逗号
    cleaned = number_str.replace(",", "")
    
    # 尝试转换为整数
    try:
        return int(cleaned)
    except ValueError:
        # 如果是百分比形式，转换为小数
        if "%" in cleaned:
            try:
                return float(cleaned.replace("%", "")) / 100
            except ValueError:
                return number_str
        return number_str

def process_university_data(data_dict):
    """
    处理大学统计数据
    """
    if not data_dict:
        return {}
    
    # 映射字段名
    field_mapping = {
        "Total number of students": "total_students",
        "Number of international students": "international_students",
        "Total number of academic staff": "academic_staff",
        "Number of international staff": "international_staff",
        "Number of undergraduate degrees awarded": "undergraduate_degrees",
        "Number of master's degrees awarded": "master_degrees",
        "Number of doctoral degrees awarded": "doctoral_degrees",
        "Number of research only staff": "research_staff",
        "Number of new undergraduate students": "new_undergraduate_students",
        "Number of new master's students": "new_master_students",
        "Number of new doctoral students": "new_doctoral_students"
    }
    
    result = {}
    for original_key, clean_key in field_mapping.items():
        if original_key in data_dict:
            result[clean_key] = clean_number(data_dict[original_key])
    
    return result

def process_subject_rankings(rankings_list):
    """
    处理学科排名列表
    """
    if not rankings_list:
        return []
    
    result = []
    for subject_rank in rankings_list:
        if not isinstance(subject_rank, str):
            continue
            
        parts = subject_rank.split("in")
        if len(parts) == 2:
            rank_part = parts[0].strip()
            subject_part = parts[1].strip()
            is_tied = "(tie)" in subject_part
            subject_name = subject_part.replace("(tie)", "").strip()
            
            try:
                rank_value = int(rank_part.replace("#", ""))
                result.append({
                    "subject": subject_name,
                    "rank": rank_value,
                    "is_tied": is_tied
                })
            except ValueError:
                continue
    
    return result

def process_global_indicators(indicators_dict):
    """
    处理全球指标数据
    """
    if not indicators_dict:
        return {}
    
    # 映射字段名
    field_mapping = {
        "Global score": "global_score",
        "Global research reputation": "global_research_reputation_rank",
        "Regional research reputation": "regional_research_reputation_rank",
        "Publications": "publications_rank",
        "Books": "books_rank",
        "Conferences": "conferences_rank",
        "Normalized citation impact": "normalized_citation_impact_rank",
        "Total citations": "total_citations_rank",
        "Number of publications that are among the 10% most cited": "top_10_percent_cited_publications_count_rank",
        "Percentage of total publications that are among the 10% most cited": "top_10_percent_cited_publications_percentage_rank",
        "International collaboration - relative to country": "international_collaboration_relative_to_country_rank",
        "International collaboration": "international_collaboration_rank",
        "Number of highly cited papers that are among the top 1% most cited": "top_1_percent_cited_papers_count_rank",
        "Percentage of highly cited papers that are among the top 1% most cited": "top_1_percent_cited_papers_percentage_rank"
    }
    
    result = {}
    for original_key, clean_key in field_mapping.items():
        if original_key in indicators_dict:
            value = indicators_dict[original_key]
            if original_key == "Global score":
                try:
                    result[clean_key] = float(value)
                except (ValueError, TypeError):
                    result[clean_key] = value
            else:
                # 处理排名数据
                if isinstance(value, str) and value.startswith("#"):
                    try:
                        result[clean_key] = int(value.replace("#", ""))
                    except ValueError:
                        result[clean_key] = value
                else:
                    result[clean_key] = value
    
    return result

def process_university(university_data):
    """
    处理单个大学数据，转换为结构化格式
    """
    # 生成唯一ID，基于大学名称
    name = university_data.get("name", "")
    university_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, name.lower()))
    
    # 处理排名数据
    rank_info = clean_rank_data(university_data.get("rank", ""))
    
    # 处理大学统计数据
    university_stats = process_university_data(university_data.get("university_data", {}))
    
    # 处理学科排名
    subject_rankings = process_subject_rankings(university_data.get("subject_rankings", []))
    
    # 处理全球指标
    global_indicators = process_global_indicators(university_data.get("best_global_indicators", {}))
    
    # 构建最终结构
    processed_university = {
        "university_id": university_id,
        "name": name,
        "global_rank": rank_info,
        "summary": university_data.get("summary", ""),
        "university_data": university_stats,
        "subject_rankings": subject_rankings,
        "global_indicators": global_indicators
    }
    
    return processed_university

def process_usnews_data(input_file, output_file):
    """
    处理整个USNews数据文件
    """
    try:
        # 读取JSON数据
        with open(input_file, 'r', encoding='utf-8') as f:
            universities = json.load(f)
        
        # 处理每所大学数据
        processed_universities = []
        for university in universities:
            processed_university = process_university(university)
            processed_universities.append(processed_university)
        
        # 保存处理后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_universities, f, ensure_ascii=False, indent=2)
        
        print(f"数据处理完成，共处理 {len(processed_universities)} 所大学数据")
        return processed_universities
        
    except Exception as e:
        print(f"处理数据时发生错误: {str(e)}")
        return None

if __name__ == "__main__":
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    root_dir = os.path.dirname(current_dir)
    
    # 输入和输出文件路径
    input_file = os.path.join(root_dir, "第三方排名网站数据爬取", "USNews2025详情界面数据.json")
    output_file = os.path.join(current_dir, "USNews2025详情界面数据_processed.json")
    
    # 处理数据
    process_usnews_data(input_file, output_file) 