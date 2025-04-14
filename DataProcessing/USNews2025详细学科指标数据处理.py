#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
import uuid
import os
import numpy as np
from sentence_transformers import SentenceTransformer

class USNews2025SubjectDataProcessor:
    def __init__(self, input_file_path, output_file_path):
        """
        初始化处理器
        
        Args:
            input_file_path: 输入JSON文件路径
            output_file_path: 输出处理后的JSON文件路径
        """
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        
        # 初始化BERT模型用于嵌入
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 学校和学科ID映射
        self.school_ids = {}
        self.subject_ids = {}
        
    def load_data(self):
        """
        加载原始JSON数据
        
        Returns:
            加载的JSON数据
        """
        try:
            with open(self.input_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"成功加载原始数据，共 {len(data)} 所学校的学科指标数据")
            return data
        except Exception as e:
            print(f"加载数据文件失败: {str(e)}")
            return []
    
    def normalize_rank(self, rank_str, max_rank=1000):
        """
        归一化排名值
        
        Args:
            rank_str: 排名字符串，格式为 "#X"
            max_rank: 最大排名值，用于归一化
            
        Returns:
            归一化后的排名值 (0-1范围)
        """
        if not rank_str or not isinstance(rank_str, str):
            return -1  # 缺失值标记
            
        # 提取数字部分
        match = re.search(r'#(\d+)', rank_str)
        if match:
            rank = int(match.group(1))
            # 归一化: 1为最好，接近0为最差
            normalized = 1 - (rank - 1) / max_rank
            return max(0, normalized)  # 确保不小于0
        
        return -1  # 无法解析则返回-1
    
    def normalize_score(self, score_str):
        """
        归一化分数值
        
        Args:
            score_str: 分数字符串，如 "71.3"
            
        Returns:
            归一化后的分数 (0-1范围)
        """
        if not score_str or not isinstance(score_str, str):
            return -1  # 缺失值标记
            
        try:
            score = float(score_str)
            # 分数通常是0-100范围，归一化到0-1
            return score / 100.0
        except ValueError:
            return -1  # 无法解析则返回-1
    
    def get_text_embedding(self, text):
        """
        获取文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            文本的嵌入向量
        """
        if not text:
            # 如果文本为空，返回全零向量
            return np.zeros(384).tolist()  # 384是all-MiniLM-L6-v2模型的维度
        
        try:
            # 使用SentenceTransformer获取嵌入
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"获取嵌入向量失败: {str(e)}")
            # 返回全零向量作为备用
            return np.zeros(384).tolist()
    
    def get_or_create_school_id(self, school_name):
        """
        获取或创建学校ID
        
        Args:
            school_name: 学校名称
            
        Returns:
            学校ID
        """
        if school_name not in self.school_ids:
            self.school_ids[school_name] = str(uuid.uuid5(uuid.NAMESPACE_DNS, school_name.lower()))
        return self.school_ids[school_name]
    
    def get_or_create_subject_id(self, subject_name):
        """
        获取或创建学科ID
        
        Args:
            subject_name: 学科名称
            
        Returns:
            学科ID
        """
        if subject_name not in self.subject_ids:
            self.subject_ids[subject_name] = str(uuid.uuid5(uuid.NAMESPACE_DNS, subject_name.lower()))
        return self.subject_ids[subject_name]
    
    def extract_indicators_from_school_data(self, school_data):
        """
        从学校数据中提取所有学科指标
        
        Args:
            school_data: 单个学校的原始数据
            
        Returns:
            学科-指标关系列表
        """
        school_name = school_data.get("name", "")
        school_id = self.get_or_create_school_id(school_name)
        result = []
        
        # 遍历所有学科数据
        for subject_name, indicators in school_data.items():
            # 跳过学校名称字段
            if subject_name == "name":
                continue
                
            subject_id = self.get_or_create_subject_id(subject_name)
            
            # 提取overall_score
            overall_score = -1  # 默认为-1（缺失）
            overall_score_key = f"{subject_name} overall score"
            if overall_score_key in indicators:
                try:
                    overall_score = float(indicators[overall_score_key])
                except (ValueError, TypeError):
                    pass
            
            # 构建指标向量
            indicator_values = []
            indicator_names = []
            
            for indicator_name, indicator_value in indicators.items():
                # 处理indicator_name，移除学科名称前缀
                clean_indicator_name = indicator_name.replace(f"{subject_name} ", "")
                indicator_names.append(clean_indicator_name)
                
                # 处理不同类型的指标值
                if "rank" in clean_indicator_name.lower():
                    # 排名类指标，归一化
                    norm_value = self.normalize_rank(indicator_value)
                elif "score" in clean_indicator_name.lower():
                    # 分数类指标，归一化
                    norm_value = self.normalize_score(indicator_value)
                else:
                    # 其他类型，尝试转换为浮点数
                    try:
                        if isinstance(indicator_value, str) and indicator_value.startswith("#"):
                            norm_value = self.normalize_rank(indicator_value)
                        else:
                            value = float(indicator_value) if indicator_value else -1
                            norm_value = value / 100.0 if value > 0 else -1
                    except (ValueError, TypeError):
                        norm_value = -1
                
                indicator_values.append(norm_value)
            
            # 构建描述文本用于语义向量生成
            description = f"{school_name} {subject_name} with overall score {overall_score}"
            
            # 使用BERT生成语义向量
            text_embedding = self.get_text_embedding(description)
            
            # 构建最终的关系对象
            relation = {
                "relation_id": str(uuid.uuid4()),
                "school_id": school_id,
                "school_name": school_name,
                "subject_id": subject_id,
                "subject_name": subject_name,
                "overall_score": overall_score,
                "indicator_names": indicator_names,
                "indicator_vector": indicator_values,
                "text_embedding": text_embedding,
                "raw_data": indicators  # 保存原始JSON数据
            }
            
            result.append(relation)
        
        return result
    
    def extract_schools_and_subjects(self, data):
        """
        从处理后的关系数据中提取学校和学科表
        
        Args:
            data: 处理后的关系数据
            
        Returns:
            学校表和学科表
        """
        schools = {}
        subjects = {}
        
        for item in data:
            school_id = item["school_id"]
            school_name = item["school_name"]
            subject_id = item["subject_id"]
            subject_name = item["subject_name"]
            
            # 添加到学校表
            if school_id not in schools:
                schools[school_id] = {
                    "school_id": school_id,
                    "school_name": school_name
                }
            
            # 添加到学科表
            if subject_id not in subjects:
                subjects[subject_id] = {
                    "subject_id": subject_id,
                    "subject_name": subject_name
                }
        
        return list(schools.values()), list(subjects.values())
    
    def process_data(self):
        """
        处理数据的主函数
        """
        # 加载原始数据
        raw_data = self.load_data()
        if not raw_data:
            return False
        
        # 处理所有学校的数据
        all_relations = []
        for school_data in raw_data:
            relations = self.extract_indicators_from_school_data(school_data)
            all_relations.extend(relations)
        
        # 提取学校和学科表
        schools, subjects = self.extract_schools_and_subjects(all_relations)
        
        # 构建最终输出
        result = {
            "schools": schools,
            "subjects": subjects,
            "relations": all_relations
        }
        
        # 保存处理后的数据
        try:
            with open(self.output_file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"成功保存处理后的数据到 {self.output_file_path}")
            print(f"处理了 {len(schools)} 所学校，{len(subjects)} 个学科，共 {len(all_relations)} 条关系数据")
            return True
        except Exception as e:
            print(f"保存处理后数据失败: {str(e)}")
            return False

# 如果作为主程序运行
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_file = os.path.join(project_root, 'DataOriginal\Data', 'USNews2025详细学科指标数据.json')
    output_file = os.path.join(project_root, 'DataProcessed', 'USNews2025详细学科指标数据_processed.json')
    
    processor = USNews2025SubjectDataProcessor(input_file, output_file)
    processor.process_data() 
