# USNews2025大学数据向量化规范

## 1. 数据概述

USNews2025详情界面数据包含了全球大学的综合排名信息，每所大学包含以下主要数据模块：

- 基本信息：大学名称、全球排名、详情URL
- 学校概述：学校历史、特点、规模等文字描述
- 大学统计数据：学生数量、国际学生比例、教职工数量等
- 学科排名：各个学科的具体排名
- 全球指标：研究声誉、出版物、引用等评分指标

## 2. 数据清洗与预处理

### 2.1 基础数据清洗

1. **排名数据处理**
   - 将形如"#1 in Best Global Universities"的排名提取为数字1
   - 识别并标记带"tie"的并列排名

2. **数值型数据转换**
   - 将"20,050"形式的数字转换为整数20050
   - 将百分比数据转换为小数形式

3. **文本标准化**
   - 统一学校名称的表示方式
   - 处理文本中的特殊字符
   - 为缺失字段填充默认值

### 2.2 数据结构化转换

将原始JSON数据转换为以下结构：

```json
{
  "university_id": "唯一ID",
  "name": "大学名称",
  "global_rank": {
    "numeric_rank": 1,
    "display_rank": "#1 in Best Global Universities",
    "is_tied": false
  },
  "summary": "学校概述文本",
  "university_data": {
    "total_students": 20050,
    "international_students": 4924,
    "academic_staff": 2235,
    "international_staff": 516,
    "undergraduate_degrees": 1476,
    "master_degrees": 4626,
    "doctoral_degrees": 1444,
    "research_staff": 1883,
    "new_undergraduate_students": 1401,
    "new_master_students": 3574,
    "new_doctoral_students": 1366
  },
  "subject_rankings": [
    {
      "subject": "Biology and Biochemistry",
      "rank": 1,
      "is_tied": false
    },
    {
      "subject": "Computer Science",
      "rank": 23,
      "is_tied": true
    }
    // 更多学科...
  ],
  "global_indicators": {
    "global_score": 100,
    "global_research_reputation_rank": 1,
    "regional_research_reputation_rank": 1,
    "publications_rank": 1,
    "books_rank": 3,
    "conferences_rank": 68,
    "normalized_citation_impact_rank": 29,
    "total_citations_rank": 1,
    "top_10_percent_cited_publications_count_rank": 1,
    "top_10_percent_cited_publications_percentage_rank": 17,
    "international_collaboration_relative_to_country_rank": 161,
    "international_collaboration_rank": 788,
    "top_1_percent_cited_papers_count_rank": 1,
    "top_1_percent_cited_papers_percentage_rank": 22
  }
}
```

## 3. 向量化策略

### 3.1 向量化分组

将大学数据分为5个模块进行独立向量化：

1. **基础信息向量**：包含大学名称、排名等基本信息
2. **概述向量**：基于学校概述文本
3. **统计数据向量**：基于大学统计数字指标
4. **学科排名向量**：基于所有学科排名情况
5. **全球指标向量**：基于全球评估指标数据

### 3.2 向量化方法

1. **文本数据向量化**
   - 推荐模型：OpenAI text-embedding-3-large (1536维)或text-embedding-3-small (1536维)
   - 备选模型：Sentence-BERT模型(768维)

2. **数值数据向量化**
   - 对所有数值指标进行归一化处理(Min-Max或Z-score)
   - 对排名数据应用非线性变换(如对数变换)增加区分度

3. **混合数据向量化**
   - 对于包含文本和数值的数据，先分别向量化再拼接
   - 必要时通过PCA等降维方法调整维度

### 3.3 向量维度规范

- 文本向量：使用1536维(OpenAI模型)或768维(BERT类模型)
- 数值向量：根据特征数量决定，建议不超过100维
- 混合向量：根据实际需求控制在2048维以内

## 4. Milvus数据库设计

### 4.1 集合设计

设计5个独立集合存储不同类型的向量：

1. `university_base`：存储基础信息及其向量
2. `university_summary`：存储概述信息及其向量
3. `university_statistics`：存储统计数据及其向量
4. `university_subjects`：存储学科排名及其向量
5. `university_indicators`：存储全球指标及其向量

### 4.2 字段设计示例

以`university_base`集合为例：

```python
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="rank", dtype=DataType.INT64),
    FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
]

schema = CollectionSchema(fields, "大学基础信息及向量")
collection = Collection("university_base", schema)
```

### 4.3 索引配置

```python
# 向量索引参数
index_params = {
    "metric_type": "COSINE",  # 或L2
    "index_type": "HNSW",     # 高效的图索引
    "params": {
        "M": 16,              # 每个节点的最大边数
        "efConstruction": 500 # 构建索引时的候选集大小
    }
}

# 创建索引
collection.create_index("embedding", index_params)
```

## 5. 数据查询模式

### 5.1 基本查询方式

1. **相似大学查询**
```python
# 查找与哈佛大学相似的大学
query_embedding = get_university_embedding("Harvard University")
search_params = {"metric_type": "COSINE", "params": {"ef": 100}}
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param=search_params,
    limit=10,
    expr=None
)
```

2. **混合条件查询**
```python
# 查找与哈佛大学相似且位于欧洲的大学
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param=search_params,
    limit=10,
    expr="region == 'Europe'"
)
```

### 5.2 跨集合查询策略

通过共享ID在多个集合间进行关联查询：

1. 在`university_base`集合中检索相似大学获取ID列表
2. 使用这些ID在其他集合中查询详细信息
3. 合并结果返回完整大学信息

## 6. 实用代码示例

### 6.1 数据预处理与向量化

```python
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 处理单所大学数据
def process_university(university_data):
    # 基本信息提取
    name = university_data.get("name", "")
    rank_str = university_data.get("rank", "")
    rank_num = int(rank_str.replace("#", "").split(" ")[0]) if rank_str else 0
    
    # 处理概述
    summary = university_data.get("summary", "")
    
    # 处理大学统计数据
    university_stats = university_data.get("university_data", {})
    processed_stats = {}
    for key, value in university_stats.items():
        if isinstance(value, str) and "," in value:
            processed_stats[key] = int(value.replace(",", ""))
        else:
            processed_stats[key] = value
    
    # 处理学科排名
    subject_rankings = []
    for subject_rank in university_data.get("subject_rankings", []):
        parts = subject_rank.split("in")
        if len(parts) == 2:
            rank_part = parts[0].strip()
            subject_part = parts[1].strip()
            is_tied = "(tie)" in subject_part
            subject_name = subject_part.replace("(tie)", "").strip()
            rank_value = int(rank_part.replace("#", ""))
            subject_rankings.append({
                "subject": subject_name,
                "rank": rank_value,
                "is_tied": is_tied
            })
    
    # 生成向量
    base_text = f"{name} {rank_str}"
    base_embedding = model.encode(base_text)
    summary_embedding = model.encode(summary)
    
    return {
        "id": name.lower().replace(" ", "_"),
        "base": {
            "name": name,
            "rank": rank_num,
            "embedding": base_embedding.tolist()
        },
        "summary": {
            "text": summary,
            "embedding": summary_embedding.tolist()
        },
        "stats": processed_stats,
        "subjects": subject_rankings,
        "indicators": university_data.get("best_global_indicators", {})
    }
```

### 6.2 Milvus数据导入

```python
from pymilvus import connections, Collection

# 连接Milvus
connections.connect("default", host="localhost", port="19530")

# 加载集合
base_collection = Collection("university_base")
base_collection.load()

# 准备插入数据
ids = []
names = []
ranks = []
embeddings = []

for university in processed_universities:
    ids.append(university["id"])
    names.append(university["base"]["name"])
    ranks.append(university["base"]["rank"])
    embeddings.append(university["base"]["embedding"])

# 执行插入
base_collection.insert([
    ids,
    names,
    ranks,
    embeddings
])
```

## 7. 更新与维护策略

1. **数据更新频率**
   - 每年随USNews排名更新进行一次全量更新
   - 对于重要数据变更可进行增量更新

2. **版本控制**
   - 为每次更新的数据标记版本号（如"2025.1"、"2025.2"）
   - 在元数据中记录更新时间和更新内容

3. **数据备份**
   - 定期备份原始数据和向量数据
   - 对于重要版本保留快照

## 8. 注意事项

1. 向量化前确保数据完整性和一致性
2. 对于不同语言的数据，选择适当的多语言向量模型
3. 根据实际查询需求调整向量索引参数
4. 大规模数据导入时注意分批处理，避免内存溢出
5. 定期检查向量质量，必要时更新向量化模型 