# USNews2025数据整理规范指南

## 1. 数据概述

USNews2025初始数据包含全球大学排名信息，主要字段有：
- `name`: 大学名称
- `detail_url`: USNews网站上该大学的详情页URL
- `rank`: 在"Best Global Universities"中的排名

## 2. 数据标准化要求

### 2.1 基础数据清洗

- **大学名称标准化**
  - 移除名称中的多余空格
  - 确保名称大小写格式一致（保持原有大小写格式）
  - 解决名称中的特殊字符问题

- **排名数据处理**
  - 从排名字符串（如"#1 in Best Global Universities"）中提取数字部分
  - 对于并列排名（含"tie"标记），需记录原始排名值并标记为并列
  - 创建数值型排名字段，便于数据分析和搜索

- **URL标准化**
  - 确保所有URL格式一致
  - 验证URL有效性

### 2.2 数据扩展

从详情页获取的补充信息应包括但不限于：
- 学校所在国家/地区
- 学校类型（公立/私立）
- 学生人数
- 师生比例
- 国际学生比例
- 学校简介
- 特色学科/专业
- 录取率（如有）
- 学费信息（如有）

### 2.3 数据格式标准

最终处理后的数据应采用以下JSON格式：

```json
{
  "id": "唯一标识符",
  "name": "标准化后的大学名称",
  "original_name": "原始大学名称",
  "country": "国家/地区",
  "rank": {
    "numeric_rank": 1,  // 数值型排名
    "display_rank": "#1 in Best Global Universities",  // 显示用排名
    "is_tied": false  // 是否并列
  },
  "details": {
    "type": "公立/私立",
    "student_count": 30000,
    "faculty_count": 2000,
    "international_student_percentage": 20,
    "description": "学校简介文本",
    "notable_programs": ["专业1", "专业2", "专业3"],
    "admission_rate": 0.05,  // 如有
    "tuition": {  // 如有
      "domestic": 50000,
      "international": 60000,
      "currency": "USD"
    }
  },
  "urls": {
    "detail_url": "https://www.usnews.com/...",
    "official_website": "https://university.edu"  // 如有
  },
  "metadata": {
    "data_source": "USNews2025",
    "last_updated": "ISO格式日期时间",
    "version": "1.0"
  }
}
```

## 3. 向量化处理

### 3.1 文本向量化策略

为构建知识库，需要对以下字段进行向量化处理：

1. **基本信息向量**
   - 大学名称
   - 国家/地区
   - 学校类型

2. **详细描述向量**
   - 学校简介
   - 特色学科/专业描述

### 3.2 推荐的嵌入模型

- 对于英文内容：使用OpenAI的text-embedding-3-small或text-embedding-3-large模型
- 对于多语言内容：使用多语言嵌入模型如BERT-multilingual

### 3.3 向量维度建议

- 文本嵌入向量维度：768或1536（取决于所选模型）

## 4. Milvus数据库配置

### 4.1 集合设计

创建名为`usnews_universities_2025`的集合，包含以下字段：

- `id`: 主键，VARCHAR类型
- `name_vector`: 学校名称的向量表示，FLOAT_VECTOR类型, 维度与嵌入模型匹配
- `description_vector`: 学校描述的向量表示，FLOAT_VECTOR类型，维度与嵌入模型匹配
- `metadata`: 包含所有结构化数据的JSON字段，VARCHAR类型

### 4.2 索引配置

- 对向量字段使用IVF_FLAT或HNSW索引类型
- 建议参数：
  - IVF_FLAT: nlist=1024
  - HNSW: M=16, efConstruction=500

### 4.3 分区策略

按国家/地区对数据进行分区，例如：
- 北美高校
- 欧洲高校
- 亚洲高校
- 大洋洲高校
- 其他地区高校

## 5. 数据处理流程

1. **数据抓取**：从USNews网站获取初始排名数据
2. **数据扩充**：爬取各大学详情页获取补充信息
3. **数据清洗**：按照2.1节要求进行清洗
4. **数据标准化**：转换为2.3节定义的标准JSON格式
5. **向量化处理**：使用选定的嵌入模型生成向量
6. **数据导入**：将处理后的数据导入Milvus数据库

## 6. 质量控制

- 确保无重复记录
- 验证必填字段完整性
- 检查数值字段合理性（如排名应为正整数）
- 确保向量维度一致性
- 定期更新数据以保持信息时效性

## 7. 使用示例

```python
# 示例：如何向Milvus导入处理后的大学数据
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# 连接到Milvus服务器
connections.connect("default", host="localhost", port="19530")

# 定义集合字段
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
    FieldSchema(name="name_vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="description_vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
]

# 创建集合模式和集合
schema = CollectionSchema(fields, "USNews 2025大学排名数据集")
collection = Collection("usnews_universities_2025", schema)

# 创建向量索引
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 500}
}

collection.create_index("name_vector", index_params)
collection.create_index("description_vector", index_params)

# 加载集合
collection.load()

# 插入数据示例
collection.insert([
    ["harvard_1"],  # id
    [[0.1, 0.2, ...]],  # name_vector
    [[0.3, 0.4, ...]],  # description_vector
    [json.dumps(harvard_metadata)]  # metadata
])
``` 