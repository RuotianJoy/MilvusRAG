# MilvusRAG

## 项目简介

MilvusRAG是一个基于Milvus向量数据库的检索增强生成(RAG)系统，专注于教育领域数据的智能检索与问答。项目集成了学术排名数据（ARWU）和美国高校信息，使用向量相似度搜索实现高效的数据检索和智能问答功能。本项目包含完整的数据处理、向量存储、检索和评估流程，并提供Web界面进行可视化交互。

## 核心特性

- **多维向量表示**：将大学信息转换为评分向量、增强向量和文本向量
- **语义搜索**：基于向量相似度的语义级搜索
- **多语言支持**：中英文双语数据处理和查询
- **自动评测**：基于ROUGE指标的RAG系统性能评估
- **精准检索**：结合关键词和向量搜索的混合检索策略
- **灵活配置**：通过配置文件管理系统参数
- **本地嵌入选项**：支持本地和API两种模式生成文本嵌入
- **可视化界面**：Web应用提供文件上传、测试执行、结果分析和数据浏览功能

## 项目架构

```
MilvusRAG/
├── Config/             # 配置文件目录
│   └── Milvus.ini      # Milvus连接配置
├── DataOriginal/       # 原始数据目录
├── DataProcessed/      # 处理后的数据目录
│   ├── ARWU2024_processed.json       # 处理后的ARWU排名数据
│   └── Wiki美国高校初步数据_processed.json  # 处理后的美国高校数据
├── DataProcessing/     # 数据处理脚本
│   ├── ARWU排名数据处理.py           # ARWU排名数据处理
│   └── Wiki美国高校初步数据处理.py    # 维基百科美国高校数据处理
├── DataUploading/      # 数据上传脚本
│   ├── ARWU排名完整导入.py           # ARWU排名数据导入Milvus
│   └── Wiki美国高校初步数据导入.py    # 美国高校数据导入Milvus
├── DataTesting/        # 数据测试目录
├── RAGTesting/         # RAG系统测试
│   ├── RAG_TestingAndEvaluation.py  # 更新版测试与评估脚本
│   ├── auto_test.py                # 自动化测试脚本
│   └── RAG测试问题库及答案.xlsx      # 测试问题和标准答案
├── WebSite/            # Web应用目录
│   ├── app.py                      # Flask应用主文件
│   ├── requirements.txt            # Web应用依赖包
│   ├── static/                     # 静态资源
│   ├── templates/                  # HTML模板
│   ├── uploads/                    # 上传文件和结果存储
│   └── data/                       # 数据和历史记录
├── TestMilvusConnect.py # Milvus连接测试工具
└── README.md            # 项目文档
```

## 技术栈

- **Python**：主要开发语言
- **Milvus**：向量数据库，用于高效存储和检索向量数据
- **PyMilvus**：Milvus的Python客户端
- **OpenAI API/Deepseek API**：用于生成文本嵌入和LLM推理
- **SentenceTransformers**：本地文本嵌入模型支持
- **ROUGE**：用于评估生成文本质量的指标
- **Pandas/NumPy**：用于数据处理和分析
- **Flask**：Web应用框架
- **Bootstrap**：前端UI框架
- **Chart.js**：数据可视化

## 数据流程

1. **数据处理**：原始数据 → 结构化数据 → 向量化表示
2. **数据导入**：向量化数据 → Milvus集合
3. **检索系统**：用户查询 → 向量化 → 相似度检索 → 相关上下文
4. **生成系统**：检索上下文 + 用户查询 → LLM → 智能回答
5. **效果评测**：使用ROUGE指标和关键词匹配评估系统性能
6. **结果展示**：Web界面呈现评测结果和可视化图表

## 快速开始

### 环境要求

- Python 3.10+
- Milvus 2.3.0+
- OpenAI API/Deepseek API访问权限

### 安装与配置

1. **安装Milvus**

   参考[Milvus官方文档](https://milvus.io/docs)安装Milvus服务

   ```bash
   # 使用Docker安装Milvus (推荐)
   docker run -d --name milvus_standalone -p 19530:19530 -p 9091:9091 milvus/milvus:latest standalone
   ```

2. **安装环境依赖包**

   ```bash
   pip install -r requirements.txt
   ```

3. **配置Milvus连接**

   修改`Config/Milvus.ini`配置文件中的连接参数

4. **测试Milvus连接**

   ```bash
   python TestMilvusConnect.py
   ```

### 数据处理与导入

1. **处理ARWU排名数据**

   ```bash
   python DataProcessing/ARWU排名数据处理.py
   ```

2. **将处理后的数据导入Milvus**

   ```bash
   python DataUploading/ARWU排名完整导入.py
   ```

3. **同样处理和导入维基百科美国高校数据**

   ```bash
   python DataProcessing/Wiki美国高校初步数据处理.py
   python DataUploading/Wiki美国高校初步数据导入.py
   ```

### 命令行测试与评估

使用评测脚本测试RAG系统性能：

```bash
python RAGTesting/RAG_TestingAndEvaluation.py
```

### 启动Web应用

1. **安装Web应用依赖**

   ```bash
   cd WebSite
   pip install -r requirements.txt
   ```

2. **运行Flask应用**

   ```bash
   python app.py
   ```

   应用将在 `http://localhost:5000` 上启动。

## Web应用功能

### 1. 测试文件上传

- 上传包含`Questions`和`Answers`列的CSV或Excel文件
- 文件必须至少包含这两列，每行代表一个测试问题和参考答案

### 2. 执行测试

- 上传后，点击"开始测试"按钮
- 系统将连接Milvus数据库，执行检索和生成
- 每个问题的测试结果会实时显示

### 3. 查看结果

- 测试完成后，可以查看详细的评估指标
- 支持下载CSV格式的完整结果
- 图表展示平均分数和精确率/召回率分布

### 4. 浏览Milvus数据

- 通过"Milvus数据"页面查看数据库内容
- 显示集合列表、字段结构和数据样本
- 支持查看索引信息和统计数据

## 评估指标

系统使用以下指标评估生成答案的质量：

- **ROUGE-1**：单词级别匹配度，衡量单个词语的召回率和精确率
- **ROUGE-2**：双词级别匹配度，考虑词序和上下文
- **ROUGE-L**：最长公共子序列匹配度，衡量句子结构相似性
- **关键词匹配率**：生成答案包含参考答案关键词的比例
- **精确率**：生成内容中正确信息的占比
- **召回率**：参考答案中被成功覆盖的信息占比

评估结果以CSV格式保存，包含每个问题的详细评分和总体平均分数。

## 注意事项

- Web应用默认只处理前10个测试问题，以避免长时间运行
- 确保Milvus服务正常运行且已导入数据
- 测试结果会保存在`WebSite/uploads`目录下的CSV文件中
- 在使用前请确保配置了正确的API密钥


