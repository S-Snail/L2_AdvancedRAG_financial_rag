import os

from langchain_community.document_compressors import DashScopeRerank

from config import RERANK_MODEL

ALI_TONGYI_EMBEDDING = "text-embedding-v3"
ALI_TONGYI_API_KEY_SYSVAR_NAME = "DASHSCOPE_API_KEY"


def get_ali_rerank(top_n=5):
    '''
    通过LangChain获得一个阿里重排序模型的实例
    基于阿里云的DasgScope大模型实现，通过语义理解能力对文档与查询的匹配度进行二次评估。
    相比传统检索，语义重排序会增加一定延迟，适合对精度要求高的场景。
    :param top_n:
    :return: 阿里通义千问嵌入模型的实例
    '''
    rerank = DashScopeRerank(
        dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_SYSVAR_NAME),
        top_n=top_n
    )
    rerank.model = RERANK_MODEL
    return rerank
