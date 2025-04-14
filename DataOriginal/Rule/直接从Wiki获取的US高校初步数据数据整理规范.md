# US高校数据整理规范

## 1. 数据概述

本文档为美国高校数据的整理规范，用于指导将Wiki获取的US高校初步数据进行标准化处理，以便向量化并存储到Milvus向量数据库中。

## 2. 数据字段说明

当前数据包含以下字段：

| 字段名 | 字段类型 | 说明 | 示例 |
|-------|--------|------|------|
| Name | 文本 | 高校名称 | Albertus Magnus College |
| Location | 文本 | 高校所在城市 | New Haven |
| Control | 文本 | 管理类型（公立/私立） | Private |
| Type | 文本 | 高校类型 | Masters university |
| Enrollment | 文本 | 在校学生数量 | 1,275 |
| State | 文本 | 所在州 | Connecticut |
| Region | 文本 | 所在地区 | Northeast |

## 3. 数据标准化要求

### 3.1 数据格式统一

1. **高校名称(Name)**：保持原始格式，无需修改
2. **城市位置(Location)**：保持原始格式，无需修改
3. **管理类型(Control)**：统一为"Public"或"Private"两种值
4. **高校类型(Type)**：标准化为以下几种类型：
   - Associates college
   - Baccalaureate college
   - Baccalaureate/associate's college
   - Masters university
   - Doctoral university
   - Arts school
   - 其他类型需归类到以上几种之一
5. **在校学生数量(Enrollment)**：
   - 移除逗号，转换为整数类型
   - 例如："1,275" → 1275
6. **州(State)**：保持原始格式，确保统一州名全称
7. **地区(Region)**：标准化为以下几种类型：
   - Northeast
   - Midwest
   - South
   - West

### 3.2 数据清洗

1. **缺失值处理**：
   - Name字段为必填项，不允许为空
   - 其他字段若缺失，处理原则如下：
     - Location：标记为"Unknown"
     - Control：标记为"Unknown"
     - Type：标记为"Other"
     - Enrollment：标记为0
     - State：根据Location推断，若无法推断则标记为"Unknown"
     - Region：根据State推断，若无法推断则标记为"Unknown"

2. **数据格式一致性**：
   - 确保文本字段不包含首尾空格
   - 确保文本大小写格式一致（推荐采用原始Wiki格式）

3. **重复数据处理**：
   - 基于Name字段检测重复记录
   - 若存在重复，保留最完整的记录或最新的记录

## 4. 向量化处理

### 4.1 文本字段向量化

1. **向量化模型选择**：
   - 推荐使用通用文本嵌入模型如BERT或Sentence-BERT
   - 对于专业教育领域，可考虑使用领域适应的模型

2. **向量化策略**：
   - **基础信息向量**：将Name、Type、Control合并为一个文本进行向量化
   - **地理位置向量**：将Location、State、Region合并为一个文本进行向量化
   - 根据需要可以生成多种向量表示同一学校的不同方面

3. **向量维度**：
   - 建议使用768维或1536维向量（取决于所选模型）

### 4.2 数值字段处理

1. **Enrollment字段**：
   - 归一化处理：将数值映射到[0,1]区间
   - 分箱处理：根据学生数量分为小型(≤2,000)、中型(2,001-10,000)、大型(>10,000)

## 5. Milvus数据库配置

### 5.1 集合设计

创建至少一个集合用于存储学校数据：

```python
# 示例代码
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
    FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="location_vector", dtype=DataType.FLOAT_VECTOR, dim=768)
]

# 创建集合
schema = CollectionSchema(fields)
collection = Collection(name="us_colleges", schema=schema)
```

### 5.2 索引配置

为向量字段创建索引以加速搜索：

```python
# 示例索引参数
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}

# 创建索引
collection.create_index(field_name="text_vector", index_params=index_params)
collection.create_index(field_name="location_vector", index_params=index_params)
```

## 6. 数据导入流程

1. **数据预处理**：
   - 读取原始JSON数据
   - 按照3.1和3.2章节要求进行数据清洗和标准化
   - 生成向量表示

2. **数据导入**：
   - 批量导入数据到Milvus
   - 建议批次大小：1000条记录

3. **数据验证**：
   - 导入后验证记录总数
   - 进行简单搜索测试，验证向量搜索功能

## 7. 数据使用建议

1. **搜索参数建议**：
   - 相似度搜索推荐使用余弦相似度(COSINE)
   - topK参数建议设置为5-20之间，根据实际应用场景调整

2. **混合检索**：
   - 结合向量检索和属性过滤（如按State、Control等过滤）
   - 示例：查找与"liberal arts college"语义相似且位于"Northeast"地区的学校

## 8. 后续数据扩展

未来可考虑扩展以下字段，丰富数据内容：

1. **建校时间**：学校成立年份
2. **学校网址**：官方网站URL
3. **特色专业**：学校的优势或特色学科
4. **学费信息**：年度学费统计
5. **师生比例**：教师与学生的比例数据
6. **排名信息**：不同排名体系下的学校排名

## 9. 更新维护计划

1. **定期更新**：建议每学年更新一次数据
2. **增量更新策略**：只更新有变化的记录，保留历史数据
3. **版本控制**：对每次更新的数据集进行版本标记 