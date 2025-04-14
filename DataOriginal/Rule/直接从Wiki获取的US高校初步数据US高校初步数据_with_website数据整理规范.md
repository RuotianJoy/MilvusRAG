# 美国高校数据整理规范

## 1. 数据概述

当前数据集包含从Wiki获取的美国高校基本信息，需要对这些数据进行标准化处理，以便于向量化并存储到Milvus向量数据库中。

## 2. 数据字段规范

### 2.1 必需字段

以下字段为必需字段，数据处理时必须确保其完整性：

| 字段名 | 数据类型 | 说明 | 示例 |
|--------|---------|------|------|
| Name | 字符串 | 学校全称 | "Albertus Magnus College" |
| Control | 字符串 | 管理类型 | "Public"或"Private" |
| Type | 字符串 | 学校类型 | "Masters university" |
| State | 字符串 | 所在州 | "Connecticut" |
| Region | 字符串 | 所在区域 | "Northeast" |

### 2.2 可选字段

以下字段为可选字段，如数据缺失需按照规则填充：

| 字段名 | 数据类型 | 说明 | 缺失处理 | 示例 |
|--------|---------|------|----------|------|
| Location | 字符串 | 所在城市 | 填充"Unknown" | "New Haven" |
| Enrollment | 字符串 | 学生数量 | 填充"0" | "1,275" |
| Website | 字符串 | 学校网站 | 填充空字符串"" | "http://albertus.edu" |

## 3. 数据清洗规则

### 3.1 字符串标准化
- 所有字符串字段去除前后空格
- 学校名称(Name)保持原有大小写
- 城市(Location)和州(State)首字母大写

### 3.2 数字处理
- Enrollment字段转换为整数，去除逗号和其他非数字字符
  - 例："1,275" → 1275

### 3.3 网址标准化
- 确保所有网址包含协议(http://或https://)
- 移除末尾的斜杠"/"

### 3.4 缺失值处理
- 缺失的Location字段填充为"Unknown"
- 缺失的Website字段填充为空字符串
- 缺失的Enrollment字段填充为"0"

## 4. 向量化方案

### 4.1 文本向量化
- 学校描述向量：将Name、Type、Control、State和Region合并成一段文本，使用文本嵌入模型(如OpenAI的text-embedding-ada-002)进行向量化
- 向量维度：根据所选模型确定(如OpenAI模型为1536维)

### 4.2 元数据存储
- 所有原始字段作为元数据存储，便于过滤和检索

## 5. Milvus数据库设计

### 5.1 集合(Collection)设计
```python
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

# 定义字段
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="control", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="enrollment", dtype=DataType.INT64),
    FieldSchema(name="state", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="website", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=1536)
]

# 定义集合模式
schema = CollectionSchema(fields, description="美国高校数据集")
collection = Collection("us_colleges", schema)
```

### 5.2 索引配置
```python
# 为向量字段创建索引
index_params = {
    "metric_type": "COSINE",  # 余弦相似度
    "index_type": "HNSW",     # HNSW索引
    "params": {"M": 8, "efConstruction": 200}
}
collection.create_index("text_vector", index_params)
```

## 6. 数据导入流程

1. 加载原始JSON数据
2. 按照上述规则清洗和标准化数据
3. 生成文本描述并向量化
4. 构建符合Milvus集合模式的数据
5. 批量导入Milvus集合

## 7. 查询示例

```python
# 按地区和学校类型查询
collection.query(
    expr="region == 'Northeast' and type == 'Masters university'",
    output_fields=["name", "location", "control", "enrollment", "website"]
)

# 向量相似度搜索（寻找相似学校）
collection.search(
    data=[target_vector],
    anns_field="text_vector",
    param={"metric_type": "COSINE", "params": {"ef": 10}},
    limit=5,
    expr="state == 'Connecticut'",  # 可选过滤条件
    output_fields=["name", "location", "type", "control", "enrollment"]
)
```

## 8. 维护计划

1. 定期（每学期或每年）更新学校数据
2. 检查并补充缺失的Website字段
3. 更新Enrollment数据以保持最新
4. 监控向量索引性能，根据需要重建索引 