import chromadb
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from loguru import logger

from config import CHROMA_PERSIST_DIR, DASHSCOPE_API_KEY, EMBEDDING_MODEL


class OptimizedChroma:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            cls._instance.embedding = DashScopeEmbeddings(
                dashscope_api_key=DASHSCOPE_API_KEY,
                model=EMBEDDING_MODEL
            )
        return cls._instance

    def get_db(self, kb_id):
        db = Chroma(
            client=self.client,
            collection_name=kb_id,
            embedding_function=self.embedding,
            persist_directory=f"{CHROMA_PERSIST_DIR}/{kb_id}"
        )
        # 优化索引
        try:
            coll = self.client.get_collection(kb_id)
            coll.modify(metadata={
                "hnsw:space": "cosine",
                "hnsw:m": 8,
                "hnsw:ef_construction": 64,
                "hnsw:ef": 32
            })
        except:
            pass
        return db

    def add_document(self, kb_id, docs):
        db = self.get_db(kb_id)
        db.add_documents(docs)
        db.persist()
        logger.info(f"向量化完成，共{len(docs)}条片段")

    def delete_kb(self, kb_id):
        try:
            self.client.delete_collection(kb_id)
            logger.info(f"删除 Chroma 集合成功：{kb_id}")
            return True
        except Exception as e:
            logger.error(f"删除 Chroma 集合失败：{kb_id}，错误：{e}")
            return False


vector_store = OptimizedChroma()
