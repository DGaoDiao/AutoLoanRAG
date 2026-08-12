from rank_bm25 import BM25Okapi
import numpy as np
from mysql_qa.utils.preprocess import preprocess_text
from base.logger import logger
from mysql_qa.db.mysql_client import MySqlClient
from mysql_qa.cache.redis_client import RedisClient


class BM25Search(object):
    def __init__(self, redis_client, mysql_client):
        """??????
        
        params:
            redis_client: ?????
            mysql_client: ?????
        
        return:
            ??"""
        self.logger = logger
        self.redis_client = redis_client
        self.mysql_client = mysql_client
        self.bm25 = None
        self.question = None
        self.origianl_question = None
        self._load_data()

    def _load_data(self):
        """?? _load_data ???
        
        params:
            ??
        
        return:
            ??????"""
        original_key = 'auto_loan:qa_original_questions:v1'
        tokenized_key = 'auto_loan:qa_tokenized_questions:v1'
        self.origianl_question = self.redis_client.get_data(original_key)
        tokenized_question = self.redis_client.get_data(tokenized_key)

        if not self.origianl_question or not tokenized_question:
            rows = self.mysql_client.fetch_question()
            self.origianl_question = [row[0] for row in rows]
            if not self.origianl_question:
                self.logger.warning('未从mysql中加载到任何数据')
                return

            tokenized_question = [preprocess_text(q) for q in self.origianl_question]
            self.redis_client.set_data(original_key, self.origianl_question)
            self.redis_client.set_data(tokenized_key, tokenized_question)

        self.question = tokenized_question
        self.bm25 = BM25Okapi(self.question)
        self.logger.info('BM25模型初始化成功')

    def _softmax(self, scores):
        """?? _softmax ???
        
        params:
            scores: ?????
        
        return:
            ??????"""
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / np.sum(exp_scores)

    def search(self, query, threshold=0.60):
        """根据输入查询最相似问题，并返回对应答案
                        :param query: 查询文本
                        :param threshold: 相似度阈值
                        :return: 匹配成功(答案, False), 未匹配(None, True) True代表新查询, False代表旧查询
                
                params:
                    query: ?????
                    threshold: ?????
        
        return:
            ??????"""

        if not query or not isinstance(query, str):
            self.logger.error('无效查询: 查询为空或者为非字符串')
            return None, True

        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            #缓存命中
            return cached_answer, False

        try:
            if self.bm25 is None or not self.origianl_question:
                return None, True
            query_tokens = preprocess_text(query)
            scores = self.bm25.get_scores(query_tokens)
            # 对分数归一化处理
            softmax_scores = self._softmax(scores)
            best_idx = softmax_scores.argmax()
            best_score = softmax_scores[best_idx]
            if best_score > threshold:
                original_question = self.origianl_question[best_idx]
                answer = self.mysql_client.fetch_answer(original_question)
                if answer:
                    self.redis_client.set_data(f'auto_loan:answer:{query}', answer)
                    self.logger.info(f'搜索成功, 相似度:{best_score:.3f}')
                    return answer, False

            self.logger.info(f'未找到可靠答案, 最高匹配度{best_score:.3f}')
            return None, True

        except Exception as e:
            self.logger.error(f'数据库查询异常 {e}')
            return None, True

if __name__ == '__main__':
    pass
