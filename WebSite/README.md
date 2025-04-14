# MilvusRAG Web应用

MilvusRAG Web应用是一个基于Flask的可视化界面，用于上传测试文件、执行测试并显示Milvus数据库内容。

## 功能特性

- 上传CSV或Excel测试文件
- 执行RAG测试评估
- 计算ROUGE指标和关键词匹配率
- 查看测试结果和图表
- 浏览Milvus数据库集合和数据

## 安装与配置

1. 安装依赖包：

```bash
pip install -r requirements.txt
```

2. 确保Milvus服务器正在运行，并根据需要修改`app.py`中的Milvus连接配置：

```python
# Milvus连接配置
MILVUS_HOST = '127.0.0.1'  # 默认本地主机
MILVUS_PORT = '19530'  # 默认端口
```

## 运行应用

```bash
python app.py
```

应用将在 `http://localhost:5000` 上启动。

## 使用说明

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

## 文件结构

```
WebSite/
├── app.py                # Flask应用主文件
├── requirements.txt      # 依赖包列表
├── static/               # 静态资源
│   └── css/              # 样式表
│       └── style.css     # 主样式表
├── templates/            # HTML模板
│   ├── index.html        # 首页模板
│   ├── process.html      # 处理页面模板
│   └── milvus.html       # Milvus数据页面模板
└── uploads/              # 上传文件和结果存储目录
```

## 注意事项

- 应用仅处理前10个测试问题，以避免长时间运行
- 确保Milvus服务正常运行且已导入数据
- 测试结果会保存在`uploads`目录下的CSV文件中 