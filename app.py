# ---------- 知识库操作 --------
# from mysql_db import mysql
import uuid

from loguru import logger

from mysql_db import mysql
from document_loader import processor
from vector_store import vector_store


def upload_file(kb_name, file):
    """上传文档到选中的知识库（如果知识库不存在则自动创建）
        Args:
            kb_name: 知识库名称
            file:上传的文件
    """
    if not kb_name:
        return "请先选择或输入知识库名称"
    if file is None:
        return "请选择文件"

    # 获取原始文件名（Gradio 上传组件提供的原始文件名）
    original_filename = file.orig_name if hasattr(file, 'orig_name') else file.name

    kbs = mysql.get_all_kb()
    kb_id = None
    for kid, name in kbs:
        if name == kb_name:
            kb_id = kid
            continue

    if not kb_id:
        kb_id = str(uuid.uuid4())
        mysql.insert_kb(kb_id, kb_name)
        logger.info(f"自动创建知识库：{kb_name} (id = {kb_id})")

    docs = processor.load_file(file.name, original_filename)  # 传递原始文件名
    if not docs:
        return "文件解析失败或为空"
    vector_store.add_document(kb_id, docs)
    mysql.insert_doc(str(uuid.uuid4()), kb_id, original_filename)  # 存储原始文件名
    return f"上传成功，共 {len(docs)} 个片段 （知识库：{kb_name}）"


def delete_kb(kb_name):
    """删除选中的知识库（数据库 + 向量库）"""
    if not kb_name:
        return "请选选择要删除的数据库"
    kbs = mysql.get_all_kb()
    kb_id = None
    for kid, name in kbs:
        if name == kb_name:
            kb_id = kid
            break
    if not kb_id:
        return "知识库不存在，请刷新后重试"

    vector_store.delete_kb(kb_id)
    mysql.delete_kb(kb_id)
    logger.info(f"已删除知识库：{kb_name} (id = {kb_id})")
    return f"已删除知识库：{kb_name}"


def get_kb_list():
    """获取所有知识库名称列表"""
    kbs = mysql.get_all_kb()
    logger.info(f"当前知识库列表：{kbs}")
    return [name for _, name in kbs]

# ---- 对话逻辑---------
# TODO 未完，待续。。。