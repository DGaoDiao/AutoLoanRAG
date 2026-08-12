#向量存储

from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import hashlib
from base.config import Config
from base.logger import logger
import sys, pathlib
import torch.cuda
from rag_qa.core.document_processor import *
import numpy as np

config = Config()


class VectorStore:
    def __init__(self,
                 collection_name = config.MILVUS_COLLECTION_NAME,
                 host = config.MILVUS_HOST,
                 port = config.MILVUS_PORT,
                 database = config.MILVUS_DATABASE,
                 ):
        """??????
        
        params:
            collection_name: ?????
            host: ?????
            port: ?????
            database: ?????
        
        return:
            ??"""
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'设用设备{self.device}')
        #加载模型 指定运行设备 模型用于计算 ’查询-文档‘的相关性得分
        self.reranker = CrossEncoder(f'{Config().ROOT_DIR}/model/bge-reranker-v2-m3',device=self.device)
        self.embedding_function = BGEM3EmbeddingFunction(f'{Config().ROOT_DIR}/model/bge-m3',
                                                        use_fp16 = (self.device == 'cuda'), # GPU时启用半精度计算（减少内存占用，提升速度），CPU时禁用
                                                        device=self.device)

        self.dense_dim = self.embedding_function.dim['dense']


    def __enter__(self):
        """?? __enter__ ???
        
        params:
            ??
        
        return:
            ??????"""
        uri = f"http://{self.host}:{self.port}"
        bootstrap_client = MilvusClient(uri=uri)
        try:
            if self.database not in bootstrap_client.list_databases():
                bootstrap_client.create_database(self.database)
                self.logger.info(f'已创建 Milvus 数据库: {self.database}')
        finally:
            bootstrap_client.close()
        self.client = MilvusClient(uri=uri, db_name=self.database)
        self._create_or_load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """?? __exit__ ???
        
        params:
            exc_type: ?????
            exc_val: ?????
            exc_tb: ?????
        
        return:
            ??????"""
        self.client.close()
        self.logger.info(f'已关闭客户端，释放资源')

    def _create_or_load(self):
        # 检查指定集合是否存在
        """?? _create_or_load ???
        
        params:
            ??
        
        return:
            ??????"""
        if not self.client.has_collection(self.collection_name):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field(field_name='id', datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            schema.add_field(field_name='sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name='parent_id', datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name='parent_content', datatype=DataType.VARCHAR,  max_length=65535)
            schema.add_field(field_name='source', datatype=DataType.VARCHAR,  max_length=100)
            schema.add_field(field_name='timestamp', datatype=DataType.VARCHAR, max_length=100)
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name='dense_vector',
                index_name = 'dense_index',
                index_type = 'IVF_FLAT',
                metric_type = 'COSINE',
                params={'nlist': 128}
            )
            index_params.add_index(
                field_name = 'sparse_vector',
                index_name = 'sparse_index',
                index_type = 'SPARSE_INVERTED_INDEX',
                metric_type = 'IP',
                params={'drop_ratio_build': 0.2}
            )
            self.client.create_collection(self.collection_name, schema=schema, index_params=index_params)
            logger.info(f'已创建集合 {self.collection_name}')
        else:
            logger.info(f'已加载集合 {self.collection_name}')
        self.client.load_collection(self.collection_name)


    # 将子块转成向量加入数据库
    def add_document(self, documents):
        # 提取所有文档内容 用bge-m3词嵌入
        """?? add_document ???
        
        params:
            documents: ?????
        
        return:
            ??????"""
        texts = [doc.page_content for doc in documents]

        embeddings = self.embedding_function(texts)

        data = []

        for i, doc in enumerate(documents):
            text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            # BGM-M3 返回的稀疏向量是矩阵，需要转换为Milvus支持的字典格式
            sparse_vector = {}
            row = embeddings['sparse'][[i]]

            indices = row.indices
            values = row.data

            # 组装稀疏向量字典， 将索引和权重配对
            for idx, value in zip(indices, values):
                sparse_vector[idx] = value

            data.append({
                'id': text_hash,                                                               # 唯一ID
                'text': doc.page_content,                                                      # 文档内容
                'dense_vector': np.array(embeddings['dense'][i], dtype=np.float32),            # 稠密向量 模型生成
                'sparse_vector': sparse_vector,                                                # 稀疏向量 组装后的字典
                'parent_id': doc.metadata['parent_id'],                                        # 父文档的ID从原数据获取
                'parent_content': doc.metadata['parent_content'],                              # 父文档内容  从原数据获取 用于上下文补充
                'source': doc.metadata.get('source', 'unknown'),                               # 学科类别
                'timestamp': doc.metadata.get('timestamp', 'unknown'),                         # 时间戳
            })

        # 插入数据到milvus 仅当data非空时执行
        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)
            logger.info(f'已插入/更新 {len(data)}条数据到集合 {self.collection_name}')






    # 混合检索
    def hybrid_search(self, query, k=config.RETRIEVAL_RETRIEVAL_K, m=config.RETRIEVAL_CANDIDATE_M, source_filter = None):
        """该函数用于执行混合检索+结果重排序,返回精准父文档
                        :param query: 用户查询文本
                        :param k: 混合解锁返回TopK子块数量
                        :param m:
                        :param source_filter: 学科过滤条件
                        :return: 返回重排序后的top-m个父文档
                
                params:
                    query: ?????
                    k: ?????
                    m: ?????
                    source_filter: ?????
        
        return:
            ??????"""
        query_embeddings = self.embedding_function([query])
        dense_query_vector = query_embeddings['dense'][0]
        dense_query_vector = dense_query_vector.astype(np.float32)
        sparse_query_vector = {}
        row = query_embeddings['sparse'][[0]]
        indices = row.indices
        values = row.data

        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value

        filter_expr = f"source == '{source_filter}'" if source_filter else ""

        dense_request = AnnSearchRequest(
            data = [dense_query_vector],      #查询向量
            anns_field='dense_vector',
            param={'metric_type': 'COSINE', 'params': {'nprobe': 10}},
            limit = k,
            expr=filter_expr
        )

        sparse_request = AnnSearchRequest(
            data = [sparse_query_vector],
            anns_field='sparse_vector',
            param={'metric_type': 'IP'},
            limit = k,
            expr=filter_expr
        )

        # 权重逻辑 稠密向量侧重语义  稀疏向量侧重关键词
        ranker = WeightedRanker(1.0, 0.7)

        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs = [dense_request, sparse_request],
            ranker = ranker,
            limit=k,
            output_fields=['text', 'parent_id', 'parent_content', 'source', 'timestamp']
        )[0]

        sub_chunks = [self._doc_from_hit(hit['entity']) for hit in results]


        #去重父块
        parent_docs = self._get_unique_parent_docs(sub_chunks)

        # 重排序逻辑： 父文档数量<m时无需排序直接返回:
        if len(parent_docs) < m:
            return parent_docs[:m]

        if parent_docs:
            # 构筑重排序逻辑 构建'查询-文档'配对列表,重排序模型需要改格式输入, 每个配对为 [query, doc_content]
            pairs = [[query, doc.page_content] for doc in parent_docs]
            scores = self.reranker.predict(pairs)

            ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        else:
            # 返回空列表
            ranked_parent_docs = []

        return ranked_parent_docs[:m]

    # 拿到父块
    def _get_unique_parent_docs(self, sub_chunks):
        """?? _get_unique_parent_docs ???
        
        params:
            sub_chunks: ?????
        
        return:
            ??????"""
        parent_contents = set()
        unique_docs = []
        for chunk in sub_chunks:
            parent_content = chunk.metadata.get('parent_content', chunk.page_content)
            if parent_content and parent_content not in parent_contents:
                unique_docs.append(Document(page_content=chunk.page_content, metadata=chunk.metadata))
                parent_contents.add(parent_content)
        return unique_docs

    # 把父块转成 langchain document对象
    def _doc_from_hit(self, hit):
        """?? _doc_from_hit ???
        
        params:
            hit: ?????
        
        return:
            ??????"""
        return Document(
            page_content=hit.get('text'),
            metadata={
                'parent_id': hit.get('parent_id'),
                'parent_content': hit.get('parent_content'),
                'source': hit.get('source'),
                'timestamp': hit.get('timestamp'),
            }
        )


if __name__ == '__main__':
    with VectorStore() as vector_store:
        res = vector_store.hybrid_search('rag是什么')
        print(res, len(res))
