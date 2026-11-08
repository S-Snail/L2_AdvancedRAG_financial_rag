"""
1. 上传文档分块 --- 完成了
2. 存储 --- 完成了
3. 检索 --- 提示词，检索方法（混合检索，重排序，上下文压缩过滤），llm
"""
import hashlib
import os

import redis
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from chat_manager import chat_manager
from config import REDIS_URL, CACHE_TTL
from models import get_ali_rerank
from vector_store import vector_store

redis_client = redis.from_url(REDIS_URL)


class RAGChain:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-max",
            temperature=0.1
        )
        self.prompt = self._get_finance_prompt()
        self.base_chain = self.prompt | self.llm | StrOutputParser()
        self.rag_chain = chat_manager.wrap_chain(self.base_chain)

        # 初始化阿里重排序模型，作为上下文压缩器（保留前5个最相关文档）
        self.reranker = get_ali_rerank(top_n=5)

    def _get_finance_prompt(self):
        return ChatPromptTemplate([
            ("system", """你是专业财务分析师，仅依据提供的文档回答问题。
            文档片段：{context}
            要求：准确、严谨、简洁，无信息时明确说明
            """),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")
        ])

    def _get_hybrid_retriever(self, kb_id):
        """原来的混合检索器"""
        db = vector_store.get_db(kb_id)
        docs = db.get()["documents"]
        if not docs:
            return db.as_retriever(docs)
        chroma_ret = db.as_retriever(search_kwargs={"k": 8})  # 多召回一些，供重排序选择
        bm25_ret = BM25Retriever.from_texts(docs)
        bm25_ret.k = 8
        return EnsembleRetriever(retrievers=[chroma_ret, bm25_ret], weights=[0.7, 0.3])

    def _get_compressed_retriever(self, kb_id):
        """包装后的压缩检索器（自动重排序）"""
        base_retriever = self._get_hybrid_retriever(kb_id)
        compressed_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker,  # 阿里重排序模型作为压缩器
            base_retriever=base_retriever
        )
        return compressed_retriever

    def retrieve(self, kb_id, query):
        """返回检索并重排序后的上下文字符串，带缓存"""
        cache_key = f"rag:{kb_id}:{hashlib.md5(query.encode()).hexdigest()}"
        if redis_client.exists(cache_key):
            return redis_client.get(cache_key)

        # 使用压缩检索器（自动重排序）
        compressed_retriever = self._get_compressed_retriever(kb_id)
        docs = compressed_retriever.invoke(query)  # 返回重排序后的文档列表
        context = "\n".join([d.page_content for d in docs])
        # 保存到缓存
        redis_client.setex(cache_key, CACHE_TTL, context)
        return context

    def chat(self, kb_id, session_id, question):
        """"""
        context = self.retrieve(kb_id, question)
        response = self.rag_chain.stream(
            {"input": question, "context": context, "chat_history": []},
            config={"configurable": {"session_id": session_id}}
        )
        for token in response:
            yield str(token)

    def chat_with_citations(self, kb_id, session_id, question):
        """
        返回（answer_generator,citations_list）
        - answer_generator：生成答案 token 的生成器
        - citation_list：重排序后保留的文档片段列表，每个元素为dict {content,source,page}
        """
        # 1.使用压缩检索器获取重排序后的文档
        compressed_retriever = self._get_compressed_retriever(kb_id)
        docs = compressed_retriever.invoke(question)

        # 构建上下文和引用
        context = "\n".join([d.page_content for d in docs])
        citations = []
        for doc in docs:
            citations.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", "未知文件"),
                "page": doc.metadata.get("page", "")
            })

        # 生成答案流
        response = self.rag_chain.stream(
            {"input": question,
             "context": context,
             "chat_history": []},
            config={"configurable": {"session_id": session_id}}
        )
        return response, citations


rag_chain = RAGChain()
