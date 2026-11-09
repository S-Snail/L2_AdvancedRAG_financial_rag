# ---------- 知识库操作 --------
# from mysql_db import mysql
import uuid
from random import choices

import gradio as gr
from loguru import logger
from openpyxl.styles.builtins import output

from mysql_db import mysql
from document_loader import processor
from rag_chain import rag_chain
from vector_store import vector_store

SESSION_ID = str(uuid.uuid4())


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
def chat_response(kb_name, message, history):
    """
    history: list of dicts with keys 'role' and 'content'
    :param kb_name:
    :param message:
    :param history:
    :return:
    """
    if not message:
        return history

    if history is None:
        history = []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    yield history

    # 获取知识库 ID
    kb_id = None
    for k_id, name in mysql.get_all_kb():
        if name == kb_name:
            kb_id = k_id
            break
    if not k_id:
        history[-1]["content"] = "请先选择有效的知识库"
        yield history
        return

    full_answer = ""
    try:
        # 调用带引用的方法
        resp_gen, citations = rag_chain.chat_with_citations(kb_id, SESSION_ID, message)
        for chunk in resp_gen:
            full_answer += str(chunk)
            history[-1]["content"] = full_answer
            yield history

        # 生成引用文本
        if citations:
            ref_text = "\n\n---\n**参考来源：**\n"
            for i, cit in enumerate(citations, 1):
                page_info = f" 第{cit['page']}页" if cit['page'] else ""
                ref_text += f"\n{i}. **{cit['source']}**{page_info}\n > {cit['content']}\n"
            # 将引用增加到答案末尾
            history[-1]["content"] = full_answer + ref_text
            yield history
        else:
            # 没有引用时，确保已输出完整答案
            yield history
    except Exception as e:
        logger.error(f"对话出错：{e}")
        history[-1]["content"] = f"出错：{str(e)}"
        yield history


# ------------Gradio 界面 ----------------
with gr.Blocks(title="金融文档智能问答助手") as demo:
    gr.Markdown("# 金融文档智能问答助手")
    with gr.Row():
        with gr.Column(scale=1):
            kb_select = gr.Dropdown(
                label="选择知识库",
                choices=get_kb_list(),
                interactive=True,
                allow_custom_value=True  # 允许手动输入
            )
            file_upload = gr.File(label="上传文档", file_types=[".pdf", ".txt", ".docx"])
            upload_btn = gr.Button("上传文档")
            delete_btn = gr.Button("删除知识库", variant="stop")
            refresh_btn = gr.Button("刷新知识库列表")
            result_msg = gr.Textbox(label="操作结果", interactive=False, lines=3)

        with gr.Column(3):
            chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox(label="提问", placeholder="请输入您的问题...")
            clear_btn = gr.Button("清空对话")

    # 事件绑定
    upload_btn.click(
        upload_file,
        inputs=[kb_select, file_upload],
        outputs=result_msg
    ).then(
        lambda: gr.update(choices=get_kb_list()),
        outputs=kb_select
    )

    delete_btn.click(
        delete_kb,
        inputs=kb_select,
        outputs=result_msg
    ).then(
        lambda: gr.update(choices=get_kb_list()),
        outputs=kb_select
    )

    msg.submit(
        chat_response,
        inputs=[kb_select, msg, chatbot],
        outputs=chatbot
    ).then(
        lambda: "", None, msg
    )

    clear_btn.click(
        lambda: gr.update(choices=get_kb_list()),
        outputs=kb_select
    )

if __name__ == '__main__':
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860)
