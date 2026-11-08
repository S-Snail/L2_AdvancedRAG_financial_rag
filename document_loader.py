import fitz
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    def __init__(self):
        self.text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_file(self, file_path, original_filename=None):
        """
        加载并切分文档，original_filename用于元数据中的source
        :param file_path:
        :param original_filename:
        :return:
        """
        try:
            # PDF财务文档专用解析器
            if file_path.endswith(".pdf"):
                return self._load_pdf(file_path, original_filename)
            # 通用加载文档
            loader = UnstructuredFileLoader(file_path)
            docs = loader.load()
            # 更新元数据中的source为原始文件名
            for doc in docs:
                doc.metadata["source"] = original_filename if original_filename else file_path
            return self.text_spliter.split_documents(docs)
        except Exception as e:
            logger.error(f"文件加载失败：{e}")
            return []

    def _load_pdf(self, file_path, original_filename):
        """
        财务PDF专用解析（表格+文本），使用原始文件名作为source
        :param file_path:
        :param original_filename:
        :return:
        """
        docs = []
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            text = page.get_text()
            # 提取表格
            try:
                tables = page.find_tables()
                for table in tables:
                    df = table.to_pandas()
                    text += "\n表格数据：\n" + df.to_string()
            except:
                pass
            # 添加元数据
            from langchain_core.documents import Document
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": original_filename if original_filename else file_path,
                    "page": page_num + 1,
                    "type": "financial_pdf"
                }
            ))
        return self.text_spliter.split_documents(docs)


processor = DocumentProcessor()
