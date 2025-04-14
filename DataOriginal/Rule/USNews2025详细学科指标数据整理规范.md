# USNews2025学科指标数据整理规范

本文档定义了将USNews2025详细学科指标数据向量化并存储到Milvus向量数据库的数据整理规范。

## 1. 数据结构

### 1.1 原始数据结构
原始数据为JSON格式，结构如下：
```json
[
  {
    "name": "学校名称",
    "学科名称1": {
      "学科名称1 overall score": "分数值",
      "指标1": "排名或数值",
      "指标2": "排名或数值",
      ...
    },
    "学科名称2": {
      ...
    }
  },
  {
    "name": "另一所学校名称",
    ...
  }
]
```

### 1.2 向量化后的数据结构

将原始数据转换为以下结构：

#### 1.2.1 学校基本信息表
```
school_id: 唯一标识符 (UUID或自增ID)
school_name: 学校名称 (如"Harvard University")
```

#### 1.2.2 学科信息表
```
subject_id: 唯一标识符
subject_name: 学科名称 (如"Artificial Intelligence")
```

#### 1.2.3 学校-学科关系表
```
relation_id: 唯一标识符
school_id: 学校ID
subject_id: 学科ID
overall_score: 总体评分
indicator_vector: 指标的向量表示
raw_data: 原始JSON数据(可选，用于数据恢复)
```

## 2. 数据向量化方法

### 2.1 指标数值处理

1. **排名数据处理**：
   - 将"#X"格式的排名转换为数值，如"#1" → 1
   - 为保持一致性，可将排名归一化到[0,1]区间：normalized_rank = 1 - (rank-1)/max_rank
   - 缺失值处理：可使用平均值、中位数或-1标记

2. **分数数据处理**：
   - 保持原始分数值，如"71.3" → 71.3
   - 可选择性地将分数标准化到[0,1]区间：normalized_score = score/100

### 2.2 向量构建方法

每个学校-学科组合构建以下向量：

1. **完整指标向量**：
   - 将所有指标值按固定顺序拼接成一个向量
   - 向量维度：指标数量（约10-15个指标）
   - 示例：[overall_score, global_rank, regional_rank, publications_rank, ...]

2. **语义描述向量**（使用预训练模型）：
   - 使用BERT/OpenAI Embedding等预训练模型，将学校+学科的文本描述转换为向量
   - 维度：取决于所选模型（如OpenAI的ada-002模型为1536维）
   - 输入文本示例："Harvard University Artificial Intelligence with overall score 63.3"

## 3. Milvus数据库设计

### 3.1 集合(Collection)设计

#### 3.1.1 schools_subjects集合
```
名称: schools_subjects
主键: id (自动生成)
字段:
- school_id: 学校ID (索引)
- school_name: 学校名称
- subject_id: 学科ID (索引)
- subject_name: 学科名称
- overall_score: 总体评分 (Float)
- indicator_vector: 指标向量 (Float向量，维度=指标数量)
- text_embedding: 文本语义向量 (Float向量，维度取决于预训练模型)
- raw_data: 原始JSON数据 (JSON格式，可选)
```

### 3.2 索引策略

1. **向量索引**：
   - indicator_vector: 使用L2距离或内积距离的HNSW/IVF_FLAT索引
   - text_embedding: 使用余弦相似度的HNSW索引

2. **标量索引**：
   - school_id, subject_id: 使用等值索引
   - overall_score: 使用范围索引

## 4. 数据处理流程

1. **数据提取**：
   - 解析JSON文件，提取学校和学科信息
   - 生成唯一ID

2. **数据转换**：
   - 将排名和分数转换为数值
   - 构建指标向量
   - 生成文本描述并转换为语义向量

3. **数据加载**：
   - 批量插入Milvus集合
   - 建立索引

4. **数据验证**：
   - 抽样验证向量是否正确表示原始数据
   - 测试简单查询确保数据可检索

## 5. 查询示例

### 5.1 基于相似度的查询

查询与指定学校-学科最相似的其他学校-学科组合：

```python
# 假设我们有Harvard University的Computer Science指标向量
search_params = {
  "metric_type": "L2",
  "params": {"nprobe": 10}
}
results = collection.search(
  data=[target_vector],
  anns_field="indicator_vector",
  param=search_params,
  limit=5,
  expr="school_id != 'harvard_id'"
)
```

### 5.2 混合查询

结合向量相似性和标量过滤的查询：

```python
# 查找overall_score>80且与目标向量相似的计算机科学专业
results = collection.search(
  data=[target_vector],
  anns_field="indicator_vector",
  param=search_params,
  limit=10,
  expr="subject_name == 'Computer Science' and overall_score > 80"
)
```

## 6. 注意事项

1. **数据一致性**：确保所有学校的指标向量维度一致，缺失值需要适当填充
2. **向量维度**：根据实际指标数量和预训练模型选择合适的向量维度
3. **性能优化**：大量数据时考虑分片和负载均衡
4. **更新策略**：定义数据更新周期和增量更新方案 