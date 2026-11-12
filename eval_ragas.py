"""
    使用Ragas对当前RAG应用进行评估
"""
import os
from logging import raiseExceptions

from datasets import Dataset
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy

from mysql_db import mysql
from rag_chain import rag_chain

# --------- 配置 ------------
KB_NAME = "财务年报1"  # 与MySQL中的名称一致
SESSION_ID = "ragas_eval_session"

# ---------- 真实问题列表 ----------
questions = [
    "公司2019年归属于上市公司股东的净利润是多少？相比2018年增长了多少？",
    "公司2019年营业收入是多少？主要增长原因是什么？",
    "公司2019年的基本每股收益和加权平均净资产收益率分别是多少？",
    "公司2019年度利润分配预案是什么？每10股派发现金红利多少？",
    "公司在2019年年度报告中披露了哪些主要风险因素？（至少列出三项）"
]

# ---------- 参考答案（从年报中提取） --------------
ground_truth = [
    "2019年归属于上市公司股东的净利润为105,233,212.02元（约1.05亿元），较2018年增长789.42%。",
    "2019年营业收入为2,001,939,841.94元（约20.02亿元），同比增长11.84%。主要原因是客户订单增加及人民币汇率影响。",
    "2019年基本每股收益为0.57元/股，加权平均净资产收益率为15.07%。",
    "以总股本185,391,680股为基数，向全体股东每10股派发现金红利1元（含税），不送红股，不以公积金转增股本。",
    "主要风险包括：国际方面：中美贸易战影响以及部分国家采取贸易保护主义措施，如提高进口关税等手段打压中国家电产品；全球金融危机导致国际贸易保护抬头，增加了关税与非关税壁垒，对家电出口构成威胁。国内方面：随着国内生产发展和生活水平提升，市场竞争加剧。如果国内市场开拓不力，公司将面临销售收入下降的风险。汇率波动：人民币被纳入SDR后跨境资本流动性增加，加剧了人民币汇率波动性。由于公司产品主要面向出口市场，汇率波动对公司经营活动有较大影响。"
]


# -------------- 生成answers 和 contexts（同步）
def generate_answers_and_contexts(kb_id, questions):
    answers = []
    contexts = []
    for q in questions:
        # 获取检索到的上下文
        context = rag_chain.retrieve(kb_id, q)
        contexts.append([context])

        # 同步生成答案（rag_chain.chat 是生成器）
        full_answer = ""
        for chunk in rag_chain.chat(kb_id, SESSION_ID, q):
            full_answer += chunk
        answers.append(full_answer)
    return answers, contexts


# ------------- 辅助函数：获取知识库ID
def get_kb_id_by_name(kb_name):
    kbs = mysql.get_all_kb()
    for kid, name in kbs:
        if name == kb_name:
            return kid
    raise ValueError(f"知识库 {kb_name} 不存在，请现在前端创建并上传年报PDF")


# -------- 主流程评估（同步）----------
def main():
    # 1. 获取知识库ID
    kb_id = get_kb_id_by_name(KB_NAME)
    print(f"使用知识库：{KB_NAME}（ID：{kb_id}）")

    # 2. 生成真实的 answers 和 contexts
    print("正在生成答案和检索上下文...")
    answers, contexts = generate_answers_and_contexts(kb_id, questions)

    # 3. 构建DataSet
    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truth
    }
    dataset = Dataset.from_dict(data)

    # 4. 初始化评估用的 LLM 和 Embeddings（使用阿里云 DashScope）
    llm = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-max",
        temperature=0.1
    )
    embeddings = DashScopeEmbeddings(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="text-embedding-v3"
    )
    llm_wrapper = LangchainLLMWrapper(llm)
    embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)

    # 5. 执行评估
    print("正在评估...")
    result = evaluate(
        llm=llm_wrapper,
        embeddings=embeddings_wrapper,
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ],
        raise_exceptions=False
    )

    # 6. 保存结果
    df = result.to_pandas()
    print(df)
    df.to_csv("ragas_eval_real_2019.csv", index=True)
    print("评估完成，结果已保存至 ragas_eval_real_2019.csv")


if __name__ == '__main__':
    main()
