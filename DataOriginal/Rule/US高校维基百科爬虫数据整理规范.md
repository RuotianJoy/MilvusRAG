# US高校维基百科数据整理规范

## 1. 数据概述

当前数据集包含从维基百科爬取的美国高校信息，以JSON格式存储。每所高校包含以下主要字段：
- 基本信息：Name, Location, Control, Type, Enrollment, State, Region, Founded, Campus Size & Setting, Official Website
- 附加信息：Additional/Notes（包含多个子字段）
- 元数据：PageTitle, MetaDescription, WikipediaURL

## 2. 数据清洗要求

### 2.1 字段标准化
- 所有字段名保持一致性，使用驼峰命名法（如`campusSize`）或下划线命名法（如`campus_size`）
- 空字段处理：空值统一用`null`表示，避免使用空字符串、"N/A"或其他非标准表示
- 删除字段值中的引用标记（如`[1]`, `[2]`等）
- 移除字段值中的HTML标签

### 2.2 特定字段处理
- **Name**: 保留学校官方全名，移除引号或其他修饰符
- **Location**: 拆分为city, state, country三个子字段
- **Enrollment**: 提取纯数字，移除文本描述和引用标记
- **Founded**: 标准化为年份（YYYY格式）
- **Campus Size**: 统一转换为英亩(acres)，并提取纯数字
- **Official Website**: 确保格式统一，添加https://前缀

### 2.3 Additional/Notes字段处理
- 将Additional/Notes内的关键子字段提升为顶级字段
- 推荐提升的字段：Religious affiliation, Academic affiliations, President, Colors, Nickname, Mascot
- 其他次要字段可保留在Additional/Notes中

## 3. 数据结构化

### 3.1 标准化JSON格式
```json
{
  "id": "唯一标识符",
  "name": "学校名称",
  "location": {
    "city": "城市",
    "state": "州",
    "country": "国家"
  },
  "control": "公立/私立",
  "type": "学校类型",
  "enrollment": 数字,
  "region": "地区",
  "founded": 年份,
  "campus_size": {
    "acres": 数字,
    "setting": "环境描述"
  },
  "website": "官方网站URL",
  "religious_affiliation": "宗教关联",
  "academic_affiliations": ["学术联盟1", "学术联盟2"],
  "president": "校长姓名",
  "colors": ["颜色1", "颜色2"],
  "nickname": "昵称",
  "mascot": "吉祥物",
  "additional_info": {
    "其他字段1": "值1",
    "其他字段2": "值2"
  },
  "metadata": {
    "page_title": "维基页面标题",
    "meta_description": "元描述",
    "wikipedia_url": "维基百科URL"
  }
}
```

## 4. 向量化处理

### 4.1 向量嵌入方法
- 为每所学校生成以下向量表示：
  - **基本信息向量**: 基于学校名称、类型、控制类型、州和地区生成
  - **详细描述向量**: 基于学校的全部文本信息生成
  - **特色向量**: 基于学校的独特特征（如宗教关联、吉祥物、昵称等）生成

### 4.2 向量化文本预处理
- 移除停用词
- 执行词干提取或词形还原
- 标准化大小写
- 移除特殊字符和数字（除非有特定含义）

### 4.3 推荐的嵌入模型
- 对于英文文本：使用OpenAI text-embedding-ada-002或text-embedding-3-small模型
- 对于混合文本：使用多语言模型如LaBSE或M-BERT

## 5. Milvus数据库设计

### 5.1 集合设计
```
collection_name: us_colleges
```

### 5.2 字段设计
```
- id: VARCHAR, 主键
- name: VARCHAR, 索引
- state: VARCHAR, 索引
- control: VARCHAR, 索引
- type: VARCHAR, 索引
- enrollment: INT64
- founded: INT64
- basic_vector: FLOAT_VECTOR(1536), 基本信息向量
- detail_vector: FLOAT_VECTOR(1536), 详细描述向量
- feature_vector: FLOAT_VECTOR(1536), 特色信息向量
- json_data: VARCHAR, 存储完整JSON数据
```

### 5.3 索引配置
```
向量索引类型: HNSW
向量索引参数: 
  - M: 16
  - efConstruction: 200
标量索引: 
  - name, state, control, type 字段建立标量索引
```

## 6. 数据处理流程

1. **原始数据提取**：从JSON文件读取原始数据
2. **数据清洗**：按照2.1-2.3章节执行数据清洗
3. **数据结构化**：按照3.1章节格式重组数据结构
4. **向量生成**：
   - 构建文本表示
   - 使用嵌入模型生成向量
5. **数据导入**：
   - 创建Milvus集合
   - 批量导入数据
6. **验证与测试**：
   - 进行简单查询测试
   - 验证向量相似度搜索效果

## 7. 质量控制

- 定期检查数据完整性
- 抽样验证向量质量
- 监控数据更新与同步
- 定期更新向量表示以反映最新的数据变化

## 8. 注意事项

- 保持原始数据的备份
- 记录所有数据转换和清洗的步骤
- 文档化向量模型的选择和参数
- 确保遵循数据隐私和使用规范 