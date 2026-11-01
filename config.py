import os

from dotenv import load_dotenv

load_dotenv()

# 大模型配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
LLM_MODEL = "qwen-max"
EMBEDDING_MODEL = "text_embedding_v3"

# 存储配置
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "qwe123"
MYSQL_DB = "financial_rag"

# 系统配置
MEMORY_WINDOW = 6  # 保留6条对话历史
CACHE_TTL = 3600  # 缓存1小时
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
