import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownTextSplitter
from rag_qa.text_splitters import ChineseRecursiveTextSplitter, AliTextSplitter
from rag_qa.document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader
from datetime import datetime
from base.config import Config
from base.logger import logger

config = Config()
document_loaders = {
    '.txt': TextLoader,
    '.pdf': OCRPDFLoader,
    '.docx': OCRDOCLoader,
    '.ppt': OCRPPTLoader,
    '.pptx': OCRPPTLoader,
    '.jpg': OCRIMGLoader,
    '.png': OCRIMGLoader,
    '.md': UnstructuredMarkdownLoader,
}

def load_document_from_directory(path:str):
    '''
    从指定目录加载所有支持类型的文件，把每个文档添加到数据库
    :param path: 目标绝对路径
    :return: 加载完成的文档列表 为每个元素添加 Langchain Document对象含 page_content 和 meta_data
    '''

    documents = []

    supported_extension = document_loaders.keys()

    source = Path(path).name.replace('_data', '')

    for root, _, files in os.walk(path):
        for file in files:
            file_path = Path(root) / file
            # print(file_path)

            file_extension = file_path.suffix.lower()

            if file_extension in supported_extension:
                try:
                    loader_class = document_loaders[file_extension]
                    if file_extension == '.txt':
                        loader = loader_class(file_path, encoding='utf-8')
                    else:
                        loader = loader_class(file_path)
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        doc.metadata['source'] = source
                        doc.metadata['file_path'] = file_path
                        doc.metadata['timestamp'] = datetime.now().isoformat()
                    documents.extend(loaded_docs)
                    logger.info(f'加载文件成功:{file_path}')
                except Exception as e:
                    logger.error(f'加载文件异常: {file_path} 异常信息 {e}')
            else:
                logger.warning(f'不支持文件类型 {e}')

    return documents

def process_document(path, parent_chunk_size=config.RETRIEVAL_PARENT_CHUNK_SIZE, child_chunk_size=config.RETRIEVAL_CHILD_CHUNK_SIZE, chunk_overlap=config.RETRIEVAL_CHUNK_OVERLAP):
    '''

    :param path: 文档目录路径
    :param parent_chunk_size: 父块切分长度
    :param child_chunk_size:  子块切分长度
    :param chunk_overlap:  子块重叠长度
    :return:
            child_chunk 每个子块包含父块的信息
    '''

    documents = load_document_from_directory(path)

    logger.info(f'加载文档的数量: {len(documents)}')

    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    child_chunks = []

    for i, doc in enumerate(documents):
        file_extension = doc.metadata['file_path'].suffix.lower()
        is_markdown = (file_extension == '.md')
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter

        logger.info(f"处理文档: {doc.metadata['file_path']}, 使用切分器:{'Markdown' if is_markdown else 'ChineseRecursive'}")

        parent_docs = parent_splitter_to_use.split_documents([doc])

        for j, parent_doc in enumerate(parent_docs):
            parent_id = f'doc_{i}_parent_{j}'
            parent_doc.metadata['parent_id'] = parent_id
            parent_doc.metadata['content'] = parent_doc.page_content #存储父块完整信息

            sub_chunks = child_splitter_to_use.split_documents([parent_doc])

            for k, sub_chunk in enumerate(sub_chunks):
                sub_chunk.metadata['parent_id'] = parent_id    #关联父块ID
                sub_chunk.metadata['parent_content'] = parent_doc.page_content # 存储父块内容
                sub_chunk.metadata['id'] = f'{parent_id}_child_{k}'

                child_chunks.append(sub_chunk)

    logger.info(f'切分完成的子块数量: {len(child_chunks)}')

    return child_chunks










if __name__ == '__main__':
    # load_document_from_directory(str(Config().ROOT_DIR / 'rag_qa' / 'data' / 'ai_data'))
    chunk = process_document(str(Config().ROOT_DIR / 'rag_qa' / 'data' / 'ai_data'))
    print(len(chunk))
    print(chunk[0])
