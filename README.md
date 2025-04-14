# MilvusRAG

## 项目介绍

MilvusRAG是一个基于Milvus向量数据库的检索增强生成系统，专门用于存储和检索大学排名数据。本项目利用向量搜索技术实现了高效的相似度匹配和语义搜索功能，可以用于教育领域的智能问答和数据分析。

## 主要功能

- 数据处理：处理ARWU（世界大学学术排名）和维基百科美国高校数据
- 向量生成：基于大学排名数据生成评分向量、增强向量和文本向量
- 向量存储：利用Milvus实现高效的向量存储和检索
- 相似度搜索：基于多种向量表示实现大学相似度比较
- 语义搜索：支持自然语言查询，找到语义相关的大学信息

## 数据源

- **ARWU排名数据**：世界大学学术排名(Academic Ranking of World Universities)数据
- **维基百科美国高校数据**：从维基百科采集的美国高校基本信息

## 技术栈

- **Python**：主要开发语言
- **Milvus**：向量数据库，用于存储和检索向量数据
- **PyMilvus**：Milvus的Python客户端
- **Transformers**：用于生成文本嵌入向量
- **BERT**：用于生成语义向量表示
- **NumPy**：用于数值计算和向量处理

## 项目结构

```
MilvusRAG/
├── DataOriginal/       # 原始数据目录
├── DataProcessed/      # 处理后的数据目录
├── DataProcessing/     # 数据处理脚本
│   ├── ARWU排名数据处理.py
│   └── Wiki美国高校初步数据处理.py
├── DataTesting/        # 测试数据目录
├── DataUploading/      # 数据上传脚本
│   ├── ARWU排名完整导入.py
│   └── Wiki美国高校初步数据导入.py
└── README.md           # 项目说明文档
```

## 安装与配置

### 前提条件

- Python 3.8+
- Milvus 2.0+

### 环境设置

1. 安装Milvus：

   按照[Milvus官方文档](https://milvus.io/docs/install_standalone-docker.md)安装Milvus服务器。

   ```bash
   # 使用Docker安装Milvus
   docker run -d --name milvus_standalone -p 19530:19530 -p 9091:9091 milvus/milvus:latest standalone
   ```

2. 安装Python依赖：

   ```bash
   pip install pymilvus numpy transformers torch tqdm
   ```

## 使用方法

### 数据处理

1. 处理ARWU排名数据：

   ```bash
   python DataProcessing/ARWU排名数据处理.py
   ```

2. 处理维基百科美国高校数据：

   ```bash
   python DataProcessing/Wiki美国高校初步数据处理.py
   ```

### 数据导入

1. 导入ARWU排名数据到Milvus：

   ```bash
   python DataUploading/ARWU排名完整导入.py
   ```

2. 导入维基百科美国高校数据到Milvus：

   ```bash
   python DataUploading/Wiki美国高校初步数据导入.py
   ```

## 检索示例

### 基于分数相似度搜索：

```python
# 查找与清华大学评分相似的大学
search_similar_by_scores(collections, "清华大学", top_k=5)
```

### 基于语义描述搜索：

```python
# 查找与描述匹配的大学
search_by_description(collections, "著名的亚洲研究型大学", top_k=5)
```

### 混合检索：

```python
# 在亚洲地区查找研究分数高于40的大学
hybrid_search(collections, region="Asia", min_research_score=40.0, top_k=5)
```

## 项目特点

1. **多维向量表示**：使用评分向量、增强向量和文本向量多角度表示大学特征
2. **高效检索**：基于Milvus实现高效的向量检索和相似度比较
3. **多语言支持**：支持中英文双语查询和检索
4. **地理位置过滤**：支持按地区和国家过滤搜索结果
5. **混合检索策略**：结合结构化查询和向量相似度搜索

## 拓展功能

- 增加更多数据源和排名系统
- 设计Web API接口提供检索服务
- 构建简单的前端界面进行交互式查询
- 整合到智能问答系统，实现教育领域的RAG应用