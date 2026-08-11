# 该脚本用于实现 RAG的核心逻辑
from rag_qa.core.prompts import RAGPrompts                    # 导入提示词模板
import time                                                   # 计算时间
import sys, os                                                #导入路径处理和系统配置
from base.config import Config
from base.logger import logger
from rag_qa.core.query_classifer import QueryClassifier       #查询分类器
from rag_qa.core.stragety_selector import StrategySelector    #导入策略选择器
from rag_qa.core.vector_store import VectorStore


class RAGSystem:
    def __init__(self, vector_sotre, llm):
        '''
        初始化RAG系统
        :param vector_sotre: 向量数据库对象->用于存储和检索文档
        :param llm: 大语言模型调用函数
        '''
        self.vector_store = vector_sotre
        self.llm = llm
        self.rag_prompt = RAGPrompts.rag_prompt()
        self.query_classifier = QueryClassifier()
        self.strategy_selector = StrategySelector()

    def _llm_text(self, prompt):
        """Collect a streaming or non-streaming model response for retrieval steps."""
        response = self.llm(prompt)
        if isinstance(response, str):
            return response
        return ''.join(part for part in response if part)


    def _retrieve_with_hyde(self,query, source_filter = None):
        logger.info(f"使用HyDE策略进行检索，查询'{query}'")
        hyde_prompt_template = RAGPrompts.hyde_prompt()
        try:
            hyde_answer = self._llm_text(hyde_prompt_template.format(query=query)).strip()
            logger.info(f'HyDE 生成的假设答案:{hyde_answer}')
            # 使用假设答案进行检索，并返回检索结果
            return self.vector_store.hybrid_search(hyde_answer, source_filter=source_filter)
        except Exception as e:
            logger.error(f'HyDE 策略执行失败: {e}')
            return []

    def _retrieve_with_subqueries(self, query, source_filter = None):
        '''

        :param query:
        :return:
        '''
        logger.info(f'使用子查询策略进行检索 查询: {query}')
        subquery_prompt_template = RAGPrompts.subquery_prompt()
        try:
            subquery_text = self._llm_text(subquery_prompt_template.format(query=query)).strip()
            subqueries = [q.strip() for q in subquery_text.split('\n') if q.strip()]
            logger.info(f'生成的子查询: {subqueries}')
            if not subqueries:
                logger.warning(f'未能生成有效的子查询')
                return []
            all_docs = []
            for sub_q in subqueries:
                docs = self._retrieve_with_hyde(sub_q, source_filter=source_filter)
                all_docs.extend(docs)
                logger.info(f'子查询: {sub_q} 检索到 {len(docs)}个文档')

            unique_docs_dict = {doc.page_content: doc for doc in all_docs} #生成集合去重
            unique_docs = list(unique_docs_dict.values())
            return unique_docs
        except Exception as e:
            logger.error(f'子查询策略执行失败: {e}')
            return []

    def _retrieve_with_backtracking(self, query, source_filter = None):
        logger.info(f'使用回溯策略进行检索 查询: {query}')
        backtracking_prompt = RAGPrompts.backtracking_prompt()
        try:
            backtrack_answer = self._llm_text(backtracking_prompt.format(query=query)).strip()
            logger.info(f'生成回溯查询的: {backtrack_answer}')
            if not backtrack_answer:
                logger.warning(f'未能生成有效的回溯查询')
                return []
            return self.vector_store.hybrid_search(backtrack_answer, source_filter=source_filter)
        except Exception as e:
            logger.error(f'回溯查询策略执行失败: {e}')
            return []

    def retrieve(self, query, source_filter = None, strategy=None):
        '''

        :param query:
        :param source_filter:
        :param strategy:
        :return:
        '''
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)
        ranked_chunks = []
        if strategy == '假设问题检索':
            ranked_chunks = self._retrieve_with_hyde(query, source_filter=source_filter)
        elif strategy == '子查询检索':
            ranked_chunks = self._retrieve_with_subqueries(query, source_filter=source_filter)
        elif strategy == '回溯问题检索':
            ranked_chunks = self._retrieve_with_backtracking(query, source_filter=source_filter)
        else:
            logger.info(f'使用直接检索策略 检索: {query}')
            ranked_chunks = self.vector_store.hybrid_search(query, source_filter=source_filter)
        return ranked_chunks

    # 优化
    def generata_answer(self, query, source_filter = None, history=None):
        '''
        rag系统对外的核心接口 接受用户查询， 自动完成查询分类，策略选择，文档检索，答案生成 全流程
        :param query:
        :param source_filter:
        :return: 生成答案的最终文本
        '''
        start_time = time.time()
        logger.info(f'开始处理查询: {query}, 学科过滤:{source_filter}')

        if history is not None and not isinstance(history, list):
            logger.warning(f'无效的历史格式: {type(history)}, 忽略历史')
            history = []
        elif history:
            for h in history:
                if not (isinstance(h, dict) and 'question' in h and 'answer' in h):
                    logger.warning(f'无效的历史条目: {h}, 忽略历史')
                    history = []
                    break

        history_context = ''
        if history:
            history_context = '\n'.join(
                [f"Q: {h['question']} \nA: {h['answer']}" for h in history]
            )
            logger.info(f'使用对话历史 {history_context[:100]}...')


        # 调用查询分类器 做意图识别
        query_category = self.query_classifier.predict_category(query)
        # 若为通用知识
        if query_category == '通用知识':
            logger.info(f'查询为通用知识, 直接调用LLM生成答案')

            prompt_input = self.rag_prompt

            try:
                answer = self.llm(prompt_input.format(history = history_context, context='', question = query, phone = Config().APP_CUSTOMER_SERVICE_PHONE))
            except Exception as e:
                logger.error(f'直接调用LLM失败: {e}')
                answer = f'抱歉处理您的通用知识失败，请联系人工客服: {Config().APP_CUSTOMER_SERVICE_PHONE}'
            processing_time = time.time() - start_time
            logger.info(f'查询完成，耗时: {processing_time:.4f}s')
            return answer
        # 若为专业咨询 执行完整的RAG流程 策略选择 文档检索 结合上下文生成
        strategy = self.strategy_selector.select_strategy(query)

        chunks = self.retrieve(query, source_filter=source_filter, strategy=strategy)

        #检索到文档 进行拼接
        if chunks:
            context = '\n\n'.join([doc.page_content for doc in chunks])
            logger.info(f'构建上下文内容完成， 包含 {len(chunks)}块文档')
        else:
            context = ''
            logger.info(f'未检索到相关文档, 上下文为空')

        prompt_input = self.rag_prompt
        try:
            answer = self.llm(prompt_input.format(history = history_context, context=context, question=query, phone=Config().APP_CUSTOMER_SERVICE_PHONE))
        except Exception as e:
            logger.error(f'直接调用LLM失败: {e}')
            answer = f'抱歉处理您的专业知识失败，请联系人工客服: {Config().APP_CUSTOMER_SERVICE_PHONE}'

        processing_time = time.time() - start_time
        logger.info(f'查询完成，耗时: {processing_time:.4f}s')
        return answer




if __name__ == '__main__':
    with VectorStore() as vector_store:
        llm = StrategySelector().call_dashscope
        rag_system = RAGSystem(vector_store, llm=llm)
        res = rag_system.generata_answer('介绍下战锤这个IP')
        print(res)
