#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer
import os
import configparser

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 配置文件路径
config_file = os.path.join(project_root, "Config", "Milvus.ini")

# 读取配置文件
def load_config():
    """读取配置文件"""
    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')
    return {
        'host': config.get('connection', 'host', fallback='localhost'),
        'port': config.get('connection', 'port', fallback='19530')
    }


class USNews2025SubjectDataTester:
    def __init__(self):
        """
        初始化测试器
        
        Args:
            host: Milvus服务器主机
            port: Milvus服务器端口
        """
        # 连接Milvus服务
        self.connect_to_milvus()
        
        # 初始化BERT模型用于查询嵌入
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 集合名称
        self.schools_collection_name = "usnews2025_schools"
        self.subjects_collection_name = "usnews2025_subjects"
        self.relations_collection_name = "usnews2025_school_subject_relations"
        self.relations_text_collection_name = "usnews2025_school_subject_relations_text"  # 新增的文本向量集合
    
    def connect_to_milvus(self):
        """
        连接到Milvus服务
        """
         # 加载配置
        milvus_config = load_config()
        host = milvus_config['host']
        port = milvus_config['port']
        
        try:
            connections.connect("default", host=self.host, port=self.port)
            print(f"已成功连接到Milvus服务器: {self.host}:{self.port}")
        except Exception as e:
            print(f"连接Milvus服务器失败: {str(e)}")
            raise
    
    def check_collections_exist(self):
        """
        检查集合是否存在
        
        Returns:
            是否所有集合都存在
        """
        collections = [
            self.schools_collection_name,
            self.subjects_collection_name,
            self.relations_collection_name,
            self.relations_text_collection_name  # 添加检查新集合
        ]
        
        for collection_name in collections:
            if not utility.has_collection(collection_name):
                print(f"集合 {collection_name} 不存在")
                return False
        
        print("所有集合均存在")
        return True
    
    def count_entities(self):
        """
        统计各集合中的实体数量
        """
        collections = [
            self.schools_collection_name,
            self.subjects_collection_name,
            self.relations_collection_name,
            self.relations_text_collection_name  # 添加新集合
        ]
        
        for collection_name in collections:
            if utility.has_collection(collection_name):
                collection = Collection(collection_name)
                collection.load()
                count = collection.num_entities
                print(f"集合 {collection_name} 中有 {count} 条数据")
                collection.release()
    
    def test_simple_query(self):
        """
        测试简单查询
        """
        if not utility.has_collection(self.schools_collection_name):
            print(f"集合 {self.schools_collection_name} 不存在，无法进行查询")
            return
        
        # 查询学校表中的前5个学校
        schools_collection = Collection(self.schools_collection_name)
        schools_collection.load()
        
        try:
            results = schools_collection.query(
                expr="",
                output_fields=["school_name"],
                limit=5
            )
            
            print("\n前5所学校:")
            for i, result in enumerate(results):
                print(f"{i+1}. {result['school_name']}")
        except Exception as e:
            print(f"查询学校失败: {str(e)}")
        finally:
            schools_collection.release()
        
        # 查询学科表中的前5个学科
        if not utility.has_collection(self.subjects_collection_name):
            print(f"集合 {self.subjects_collection_name} 不存在，无法进行查询")
            return
            
        subjects_collection = Collection(self.subjects_collection_name)
        subjects_collection.load()
        
        try:
            results = subjects_collection.query(
                expr="",
                output_fields=["subject_name"],
                limit=5
            )
            
            print("\n前5个学科:")
            for i, result in enumerate(results):
                print(f"{i+1}. {result['subject_name']}")
        except Exception as e:
            print(f"查询学科失败: {str(e)}")
        finally:
            subjects_collection.release()
    
    def test_vector_search(self, query_text="Harvard University Computer Science"):
        """
        测试向量搜索
        
        Args:
            query_text: 查询文本
        """
        # 使用关系文本集合进行向量搜索
        if not utility.has_collection(self.relations_text_collection_name):
            print(f"集合 {self.relations_text_collection_name} 不存在，无法进行向量搜索")
            return
        
        # 生成查询文本的向量表示
        query_vector = self.embedding_model.encode(query_text).tolist()
        
        # 加载关系文本集合
        relations_text_collection = Collection(self.relations_text_collection_name)
        relations_text_collection.load()
        
        try:
            # 基于文本向量进行搜索
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }
            
            text_results = relations_text_collection.search(
                data=[query_vector],
                anns_field="text_embedding",  # 在文本集合中搜索
                param=search_params,
                limit=5,
                output_fields=["relation_id", "school_id", "subject_id"]
            )
            
            print(f"\n使用查询文本 '{query_text}' 搜索最相似的5条记录:")
            
            if text_results and text_results[0]:
                # 获取关系ID和其他ID
                relation_ids = []
                school_ids = []
                subject_ids = []
                
                for hit in text_results[0]:
                    relation_ids.append(hit.entity.get("relation_id"))
                    school_ids.append(hit.entity.get("school_id"))
                    subject_ids.append(hit.entity.get("subject_id"))
                
                # 查询关系集合获取更多信息
                relations_collection = Collection(self.relations_collection_name)
                relations_collection.load()
                
                relation_results = relations_collection.query(
                    expr=f"id in {relation_ids}",
                    output_fields=["id", "school_id", "subject_id", "overall_score", "raw_data"]
                )
                
                # 创建关系映射，用relation_id作为键
                relations_dict = {r["id"]: r for r in relation_results}
                relations_collection.release()
                
                # 查询学校名称
                schools_collection = Collection(self.schools_collection_name)
                schools_collection.load()
                school_results = schools_collection.query(
                    expr=f"id in {school_ids}",
                    output_fields=["id", "school_name"]
                )
                schools_dict = {r["id"]: r["school_name"] for r in school_results}
                schools_collection.release()
                
                # 查询学科名称
                subjects_collection = Collection(self.subjects_collection_name)
                subjects_collection.load()
                subject_results = subjects_collection.query(
                    expr=f"id in {subject_ids}",
                    output_fields=["id", "subject_name"]
                )
                subjects_dict = {r["id"]: r["subject_name"] for r in subject_results}
                subjects_collection.release()
                
                # 显示结果
                for i, hit in enumerate(text_results[0]):
                    relation_id = hit.entity.get("relation_id")
                    school_id = hit.entity.get("school_id")
                    subject_id = hit.entity.get("subject_id")
                    
                    # 获取详细信息
                    relation_data = relations_dict.get(relation_id, {})
                    overall_score = relation_data.get("overall_score", "未知")
                    
                    school_name = schools_dict.get(school_id, "未知学校")
                    subject_name = subjects_dict.get(subject_id, "未知学科")
                    
                    print(f"{i+1}. 得分: {hit.score:.4f}")
                    print(f"   学校: {school_name}")
                    print(f"   学科: {subject_name}")
                    print(f"   总体评分: {overall_score}")
            else:
                print("没有找到结果")
                
        except Exception as e:
            print(f"向量搜索失败: {str(e)}")
        finally:
            relations_text_collection.release()
    
    def test_indicator_vector_search(self, target_indicator_vector=None):
        """
        测试指标向量搜索
        
        Args:
            target_indicator_vector: 目标指标向量，如果为None则使用一个已有向量
        """
        if not utility.has_collection(self.relations_collection_name):
            print(f"集合 {self.relations_collection_name} 不存在，无法进行指标向量搜索")
            return
        
        # 加载关系集合
        relations_collection = Collection(self.relations_collection_name)
        relations_collection.load()
        
        try:
            # 如果没有提供目标向量，首先获取一个现有的向量作为查询示例
            if target_indicator_vector is None:
                # 获取一个示例向量
                sample_results = relations_collection.query(
                    expr="",
                    output_fields=["id", "school_id", "subject_id", "indicator_vector"],
                    limit=1
                )
                
                if not sample_results:
                    print("无法获取示例向量，集合可能为空")
                    return
                
                target_indicator_vector = sample_results[0]["indicator_vector"]
                target_school_id = sample_results[0]["school_id"]
                target_subject_id = sample_results[0]["subject_id"]
                
                # 查询学校和学科名称
                schools_collection = Collection(self.schools_collection_name)
                schools_collection.load()
                school_result = schools_collection.query(
                    expr=f"id == '{target_school_id}'",
                    output_fields=["school_name"],
                    limit=1
                )
                target_school_name = school_result[0]["school_name"] if school_result else "未知学校"
                schools_collection.release()
                
                subjects_collection = Collection(self.subjects_collection_name)
                subjects_collection.load()
                subject_result = subjects_collection.query(
                    expr=f"id == '{target_subject_id}'",
                    output_fields=["subject_name"],
                    limit=1
                )
                target_subject_name = subject_result[0]["subject_name"] if subject_result else "未知学科"
                subjects_collection.release()
                
                print(f"\n使用 {target_school_name} 的 {target_subject_name} 指标向量作为查询目标")
            
            # 基于指标向量进行搜索
            search_params = {
                "metric_type": "L2",
                "params": {"ef": 64}
            }
            
            indicator_results = relations_collection.search(
                data=[target_indicator_vector],
                anns_field="indicator_vector",
                param=search_params,
                limit=5,
                output_fields=["id", "school_id", "subject_id", "overall_score"]
            )
            
            print("\n搜索最相似的5条指标向量记录:")
            
            if indicator_results and indicator_results[0]:
                # 获取学校和学科ID
                school_ids = []
                subject_ids = []
                
                for hit in indicator_results[0]:
                    school_ids.append(hit.entity.get("school_id"))
                    subject_ids.append(hit.entity.get("subject_id"))
                
                # 查询学校名称
                schools_collection = Collection(self.schools_collection_name)
                schools_collection.load()
                school_results = schools_collection.query(
                    expr=f"id in {school_ids}",
                    output_fields=["id", "school_name"]
                )
                schools_dict = {r["id"]: r["school_name"] for r in school_results}
                schools_collection.release()
                
                # 查询学科名称
                subjects_collection = Collection(self.subjects_collection_name)
                subjects_collection.load()
                subject_results = subjects_collection.query(
                    expr=f"id in {subject_ids}",
                    output_fields=["id", "subject_name"]
                )
                subjects_dict = {r["id"]: r["subject_name"] for r in subject_results}
                subjects_collection.release()
                
                # 显示结果
                for i, hit in enumerate(indicator_results[0]):
                    school_id = hit.entity.get("school_id")
                    subject_id = hit.entity.get("subject_id")
                    overall_score = hit.entity.get("overall_score")
                    
                    school_name = schools_dict.get(school_id, "未知学校")
                    subject_name = subjects_dict.get(subject_id, "未知学科")
                    
                    print(f"{i+1}. 距离: {hit.distance:.4f}")
                    print(f"   学校: {school_name}")
                    print(f"   学科: {subject_name}")
                    print(f"   总体评分: {overall_score}")
            else:
                print("没有找到结果")
                
        except Exception as e:
            print(f"指标向量搜索失败: {str(e)}")
        finally:
            relations_collection.release()
    
    def test_hybrid_search(self, subject_name="Computer Science", min_score=70.0):
        """
        测试混合查询
        
        Args:
            subject_name: 学科名称
            min_score: 最低总体评分
        """
        if not utility.has_collection(self.relations_collection_name):
            print(f"集合 {self.relations_collection_name} 不存在，无法进行混合查询")
            return
        
        # 首先查询学科ID
        if not utility.has_collection(self.subjects_collection_name):
            print(f"集合 {self.subjects_collection_name} 不存在，无法进行混合查询")
            return
            
        subjects_collection = Collection(self.subjects_collection_name)
        subjects_collection.load()
        
        try:
            subject_results = subjects_collection.query(
                expr=f'subject_name == "{subject_name}"',
                output_fields=["id", "subject_name"]
            )
            
            if not subject_results:
                print(f"未找到学科: {subject_name}")
                return
                
            subject_id = subject_results[0]["id"]
            
            # 使用标量过滤执行查询
            relations_collection = Collection(self.relations_collection_name)
            relations_collection.load()
            
            results = relations_collection.query(
                expr=f'subject_id == "{subject_id}" && overall_score >= {min_score}',
                output_fields=["school_id", "overall_score"],
                limit=10
            )
            
            # 查询学校名称
            if results:
                school_ids = [r["school_id"] for r in results]
                
                schools_collection = Collection(self.schools_collection_name)
                schools_collection.load()
                school_results = schools_collection.query(
                    expr=f"id in {school_ids}",
                    output_fields=["id", "school_name"]
                )
                schools_dict = {r["id"]: r["school_name"] for r in school_results}
                schools_collection.release()
                
                # 显示结果
                print(f"\n{subject_name} 学科总分 >= {min_score} 的学校:")
                
                # 按总体评分降序排列
                results.sort(key=lambda x: x["overall_score"], reverse=True)
                
                for i, result in enumerate(results):
                    school_id = result["school_id"]
                    overall_score = result["overall_score"]
                    school_name = schools_dict.get(school_id, "未知学校")
                    
                    print(f"{i+1}. {school_name} - 总体评分: {overall_score}")
            else:
                print(f"未找到符合条件的 {subject_name} 学科数据")
                
        except Exception as e:
            print(f"混合查询失败: {str(e)}")
        finally:
            if 'subjects_collection' in locals() and subjects_collection:
                subjects_collection.release()
            if 'relations_collection' in locals() and relations_collection:
                relations_collection.release()
    
    def run_all_tests(self):
        """
        运行所有测试
        """
        if not self.check_collections_exist():
            print("集合不存在，无法进行测试")
            return
        
        print("\n===== 测试1: 统计各集合中的实体数量 =====")
        self.count_entities()
        
        print("\n===== 测试2: 简单查询测试 =====")
        self.test_simple_query()
        
        print("\n===== 测试3: 文本向量搜索测试 =====")
        self.test_vector_search()
        
        print("\n===== 测试4: 指标向量搜索测试 =====")
        self.test_indicator_vector_search()
        
        print("\n===== 测试5: 混合查询测试 =====")
        self.test_hybrid_search()
        
        print("\n所有测试完成")


# 如果作为主程序运行
if __name__ == "__main__":
    tester = USNews2025SubjectDataTester(host="localhost", port="19530")
    tester.run_all_tests() 
