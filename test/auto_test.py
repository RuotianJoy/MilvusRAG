# 一个能够从文件中读取问题和答案的agent，具有知识库功能，可选择记忆（默认关闭），在回答完毕后将计算测评指标并将计算结果存储到CSV文件中
import os
import pandas as pd
import openpyxl
from llama_index.llms.openai import OpenAI
from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema
import openai
from openai import OpenAI as OPENAI
from pymilvus import utility
from llama_index.core.agent import ReActAgent
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_precision, context_recall
)
# Faithfulness（忠实度）:答案是否基于检索到的上下文,避模型生成与检索信息不符的幻觉
# Answer Relevancy（答案相关性）:衡量答案是否与问题相关，确保模型没有答非所问
# Context Precision（上下文精确度）：衡量检索到的上下文是否包含了正确的信息，避免噪音
# Context Recall（上下文召回率）：衡量检索到的上下文是否足够全面，避免信息缺失

from ragas import evaluate
from ragas import EvaluationDataset, SingleTurnSample

from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import time

os.environ["OPENAI_API_KEY"] = "YOUR_KEY"

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

client = OPENAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="Your_key",
    base_url="base_url"
)

# 这里的USER_id和Conversation_id都是固定值，但是应用在现实中，应该是由前端发回的值用于存储与判断
USER_id = 123456789
Conversation_id = 1234

print(f"user_id == {USER_id} && conversation_id == {Conversation_id}")
# 连接到 Milvus


def connect_to_milvus():
    connections.connect("default", host="localhost", port="19530")


# 获取文本的嵌入表示
def get_embedding(text):
    model = "text-embedding-ada-002"
    vector = client.embeddings.create(input=[text], model=model).data[0].embedding
    # 返回文本的向量
    return vector


# 将对话文本和对应的嵌入存入 Milvus 数据库
def insert_conversation(collection, text):
    embedding = get_embedding(text)
    collection.insert([{"embedding": embedding, "text": text, "user_id": USER_id, "conversation_id": Conversation_id}])


# 使用用户输入（查询）生成的嵌入，检索最相关的对话历史（通过 Milvus 向量搜索）
def search_relevant_memory(collection, query, top_k=5):

    query_embedding = get_embedding(query)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        output_fields=["text"],
        param=search_params,
        limit=top_k,
        expr=f"user_id == {USER_id} && conversation_id == {Conversation_id}"
    )
    retrieved_texts = []
    for hits in results[0]:
        retrieved_texts.append(hits.entity.get("text"))
    return retrieved_texts


def search_relevant_knowledge(collection, query, top_k=5):
    # 初始化 filter_sentence 为 None
    filter_sentence = None

    # 判断 query 中包含哪个公司，并构造相应的过滤条件
    # 这里的判断后期需要根据数据进行改动

    if "电商A" in query and "电商B" in query:
        filter_sentence = "company_id == '电商A' || company_id == '电商B'"
    elif "电商A" in query:
        filter_sentence = 'company_id == "电商A"'
    elif "电商B" in query:
        filter_sentence = "company_id == '电商B'"

    print(f"filter:{filter_sentence}")

    query_embedding = get_embedding(query)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        output_fields=["text", "company_id"],
        param=search_params,
        limit=top_k,
        expr=filter_sentence
    )

    # print(results)
    retrieved_texts = []

    for hits in results[0]:
        # 获取 text 和 company_id 字段，并将它们一起存储
        text = hits.entity.get("text")
        company_id = hits.entity.get("company_id")

        # 将文本和 company_id 一起存入 retrieved_texts
        retrieved_texts.append(text + "消息来源于：" + company_id + "\n")

    return retrieved_texts


# 用于从 Milvus 中检索相关对话，并将其作为上下文提供给对话处理
def load_memory_variables(collection, inputs, top_k=5):
    collection.load()
    user_input = inputs.get("input", "")
    relevant_texts = search_relevant_memory(collection, user_input, top_k)
    context = "\n".join(relevant_texts)
    return {"history": context}


# 用于从 Milvus 中检索相关知识，并将其作为上下文提供给对话处理
def load_knowledge_variables(collection, inputs, top_k=2):
    collection.load()
    user_input = inputs.get("input", "")
    relevant_texts = search_relevant_knowledge(collection, user_input, top_k)
    context = "\n".join(relevant_texts)
    return {"knowledge": context}


# 保存对话上下文（用户输入和机器人输出）
def save_context(collection, inputs, outputs):
    user_input = inputs.get("input", "")
    bot_response = outputs.get("output", "")
    insert_conversation(collection, f"用户: {user_input}")
    insert_conversation(collection, f"机器人: {bot_response}")


def main():
    # 1. 连接到 Milvus Lite
    connect_to_milvus()

    # 2. 定义并创建集合
    collection_name = "user_conversation_memory"
    if collection_name not in utility.list_collections():
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_id", dtype=DataType.INT64),  # 用户的ID
            FieldSchema(name="conversation_id", dtype=DataType.INT64),  # 对话的ID
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),  # OpenAI 生成的嵌入维度
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
        ]
        schema = CollectionSchema(fields, description="Conversation Memory Collection")
        collection = Collection(name=collection_name, schema=schema)
        # 为 "embedding" 字段创建索引
        index_params = {
            "index_type": "IVF_FLAT",  # 使用 IVF_FLAT 索引类型，可以根据需要选择其他类型
            "metric_type": "L2",  # 使用 L2 距离度量，其他类型可选
            "params": {"nlist": 100}  # 索引的参数
        }

        # 创建索引
        collection.create_index(field_name="embedding", index_params=index_params)

    else:
        collection = Collection(name=collection_name)

    know_collection = Collection(name="pdf_know_collection")

    # 3. 设置 OpenAI API 密钥（如果使用 OpenAI 的远程模型）
    openai.api_key = "Your_KEY"

    # 4. 创建 OpenAI 的 LLM 接口
    llm = OpenAI(model="gpt-4o-mini")

    agent = ReActAgent.from_tools(llm=llm, verbose=True)    # 可以看到思考过程
    # agent = ReActAgent.from_tools(llm=llm, verbose=False)     # 只有输入输出

    prompt = "注意：输入是人类语言，无需使用工具。"

    # Excel 文件路径
    xlsx_path = r"RAG测试问题库及答案.xlsx"  # 请替换成实际路径
    df = pd.read_excel(xlsx_path)

    # 评测数据存储
    samples = []
    spend_time = []
    for idx, row in df.iterrows():
        user_input = str(row["Questions"])
        reference_answer = str(row["Answers"])

        start_time = time.time()  # 记录开始时间
        # 通过 OpenAI LLM 处理对话
        relevant_history = load_memory_variables(collection, {"input": user_input}, top_k=1)
        relevant_knowledge = load_knowledge_variables(know_collection, {"input": user_input}, top_k=1)
        bot_response = agent.chat("\n过往聊天记录：\n" + relevant_history.get("history", "") + "\n相关知识背景：\n" + relevant_knowledge.get("knowledge", "") + prompt + "\n用户的输入为：" + user_input)
        # 这里的最后一句话要用prompt代替

        # 如果想要将此次对话记录进记忆，可以将下面这行代码解除注释，此处问题上下文并无多少关联，因此未进行记忆
        # save_context(collection, {"input": user_input}, {"output": bot_response})

        end_time = time.time()  # 记录结束时间
        retrieval_time = end_time - start_time  # 计算检索时间
        spend_time.append(retrieval_time)

        sample = SingleTurnSample(
            user_input=user_input,
            # user_input（用户输入）
            reference=reference_answer,
            # reference（参考答案），可以为空，但为空时 有的指标会异常，比如'context_recall'会显示nan
            response=str(bot_response),
            # response（模型生成的答案）
            retrieved_contexts=["\n过往聊天记录：\n" + relevant_history.get("history", "") + "\n相关知识背景：\n" + relevant_knowledge.get("knowledge", "")],
            # retrieved_contexts（检索到的上下文）
        )

        samples.append(sample)
        # 所以指标都是针对单轮RAG的
        print(f"用户：{user_input}")
        print(f"机器人: {bot_response}")

    # 执行 Ragas 评测
    # 创建 Ragas Dataset
    eval_dataset = EvaluationDataset(samples=samples)

    # 选择评估指标
    metrics = [context_precision, faithfulness, answer_relevancy, context_recall]
    results = evaluate(dataset=eval_dataset, metrics=metrics, llm=evaluator_llm,)

    # 输出评测结果
    print("评估结果：")
    print(results)

    # 将评估结果转换为 DataFrame
    df = results.to_pandas()

    # 添加检索时间到 DataFrame
    df['retrieval_time'] = spend_time  # 从每个 sample 中提取时间并添加到 DataFrame

    # ==== 清洗函数（防止格式错乱）====
    def clean_cell(value):
        if isinstance(value, list):
            return "\n".join(str(v).strip().replace("\n", " ") for v in value)
        elif isinstance(value, str):
            return value.replace("\n", " ").replace("\r", " ").strip()
        else:
            return value

    df = df.applymap(clean_cell)

    # ==== 写入 CSV，防止乱码 & 重复列名 ====
    csv_file = "evaluation_results.csv"
    file_exists = os.path.exists(csv_file)

    df.to_csv(
        csv_file,
        mode="a",
        header=not file_exists,  # 只在第一次写入时保留列名
        index=False,
        encoding="utf-8-sig"  # 保证 Excel 识别正常
    )

    print(f"评估结果已追加到 {csv_file}")


if __name__ == "__main__":
    main()

