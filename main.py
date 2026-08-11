from idlelib import history

from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MySqlClient
from mysql_qa.retrieval.bm25_search import BM25Search
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_system import RAGSystem
from base.config import Config
from base.logger import logger
from openai import OpenAI
import time
import pymysql
import uuid


class IntegratedQASystem:
    def __init__(self, mysql_client, redis_client, vector_store):
        self.logger = logger
        self.config = Config()
        self.mysql_client = mysql_client
        self.redis_client = redis_client
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)
        try:
            self.client = OpenAI(
                base_url=self.config.LLM_DASHSCOPE_BASE_URL,
                api_key=self.config.LLM_DASHSCOPE_API_KEY,
            )
        except Exception as e:
            logger.error(f'OpenAI客户端初始化失败: {e}')
            raise
        self.vector_store = vector_store
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)

        # 优化 初始化历史对话表: 在MySql中创建历史对话表
        self.init_conversation_table()
        self.limit = self.config.MYSQL_LIMIT

        # 优化

    def init_conversation_table(self):
        '''
        初始化MySQL中的Conversation表, 用于存储历史对话
        :return:
        '''
        try:
            self.mysql_client.cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations(
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
                    created_at DATETIME NOT NULL,
                    INDEX idx_session_id (session_id))
                ''')
            self.mysql_client.connection.commit()
            self.logger.info('对话历史初始表创建成功')
        except Exception as e:
            self.logger.error(f'创建对话表失败 {e}')

    def call_dashscope(self, query):
        try:
            completion = self.client.chat.completions.create(
                model=Config().LLM_MODEL,
                messages = [
                    {'role': 'system', 'content': '你是一个靠谱的助手，能够根据用户输入的内容严格执行并返回可靠的结果'},
                    {'role': 'user', 'content': query}
                ],
                temperature = 0.1,
                timeout = 30,
                stream = True
            )
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield content
        except Exception as e:
            logger.error(f'Dashscope API 调用失败: {e}')


    # 优化 获取最近聊天记录的内部方法
    def _fetch_recent_history(self, session_id, limit = Config().MYSQL_LIMIT):
        try:
            self.mysql_client.cursor.execute('''
                SELECT question, answer
                FROM conversations
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            ''', (session_id, limit,)
            )
            history = [{'question': i[0], 'answer': i[1]} for i in self.mysql_client.cursor.fetchall()]
            return history[::-1]
        except Exception as e:
            self.logger.error(f'获取历史对话失败: {e}')
            return []

    #优化 对外拿对话历史
    def get_session_history(self, session_id):
        return self._fetch_recent_history(session_id)

    # 优化 更新会话历史
    def update_session_history(self, session_id, question, answer) -> list:
        '''
        更新会话历史 只保留最近的limit轮
        :param session_id: 会话的唯一标识
        :param question:  用户的问题
        :param answer:    问题的答案
        :return:
        '''
        try:
            self.mysql_client.cursor.execute('''
                 INSERT INTO conversations(session_id, question, answer,created_at)
                 VALUES(%s, %s, %s, NOW())''', (session_id, question, answer))
            #删除超过 限制的对话历史
            self.mysql_client.cursor.execute('''
                 DELETE FROM conversations
                 WHERE session_id = %s
                     AND id NOT IN (
                                    SELECT id 
                                    FROM (
                                         SELECT id
                                         FROM conversations
                                         WHERE session_id = %s
                                         ORDER BY created_at DESC        
                                         LIMIT %s
                                         ) AS tmp
                                    )
                    ''', (session_id, session_id, Config().MYSQL_LIMIT,)
                 )
            self.mysql_client.connection.commit()
            self.logger.info(f'会话历史更新成功 {session_id}')
            return self._fetch_recent_history(session_id)
        except Exception as e:
            self.logger.error(f'更新会话历史失败 {e}')
            self.mysql_client.connection.rollback()
            raise

    #清除指定id的素有会话历史
    def clear_session_history(self, session_id):
        try:
            self.mysql_client.cursor.execute('''
            DELETE FROM conversations WHERE session_id = %s
            ''', (session_id,))
            self.mysql_client.connection.commit()
            self.logger.info(f'会话{session_id}历史已被清除')
            return True
        except Exception as e:
            self.logger.error(f'清除会话历史失败{e}')
            self.mysql_client.connection.rollback()
            return False


    def query(self, query, source_filter = None, session_id = None):
        '''
        先查mysql和bm25 如果没有就走 RAG
        :param query:用户的问题
        :param source_filter:来源过滤条件
        :return:
        '''
        start_time = time.time()
        if source_filter not in (None, *self.config.APP_VALID_SOURCES):
            self.logger.warning(f'忽略无效的汽车金融知识类别: {source_filter}')
            source_filter = None
        self.logger.info(f'开始处理问题: {query}, 来源过滤条件: {source_filter or "不限"}')

        history = self.get_session_history(session_id) if session_id else None
        answer, need_rag = self.bm25_search.search(query)

        if answer:
            # 拿到了答案 直接返回
            self.logger.info(f'找到答案 {answer:50}')
            if session_id:
                self.update_session_history(session_id, query, answer)
            processing_time = time.time() - start_time
            self.logger.info('处理完成, 用时: {processing_time:.3f}s')
            yield answer, True
        elif need_rag:
            #  调用RAG生成
            self.logger.info('BM25答案不靠谱需要RAG系统来处理')
            collected_answer = ''
            for token in self.rag_system.generata_answer(query, source_filter, history=history):
                collected_answer += token
                yield token, False
            if session_id:
                self.update_session_history(session_id, query, collected_answer)
            self.logger.info(f'RAG系统生成了答案: {collected_answer:50}')
            processing_time = time.time() - start_time
            self.logger.info(f'处理完成, 用时: {processing_time:.3f}s')
            yield '', True
        else:
            # 啥都没有，告诉用户没答案
            self.logger.info(f'BM25没找到任何答案')
            processing_time = time.time() - start_time
            self.logger.info(f'处理完成, 用时: {processing_time:.3f}s')
            yield '没有找到和你问题相关的答案', True


def main():
    '''
    系统入口 用户可以输入问题和来源
    :return: 空
    '''
    session_id = str(uuid.uuid4())
    with MySqlClient() as mysql_client:
        with RedisClient() as redis_client:
            with VectorStore() as vector_store:
                qa_system = IntegratedQASystem(mysql_client, redis_client, vector_store)
                qa_system.logger.info('系统初始化完成')

                try:
                    print('\n集成问答系统')
                    print(f'支持的来源: {qa_system.config.APP_VAILD_SOURCES}')
                    print('输入问题查看答案，或者按exit退出系统')
                    while True:
                        query = input('请录入您的问题: ').strip()

                        if query.lower() == 'exit':
                            qa_system.logger.info('用户输入了exit, 准备退出系统')
                            print('再见, 感谢您的使用')
                            break
                        source_filter = input(f'请输入来源过滤(可选，支持:{"./".join(qa_system.config.APP_VAILD_SOURCES)}, 直接回车表示不限制): )').strip()
                        if source_filter:
                            if source_filter not in qa_system.config.APP_VAILD_SOURCES:
                                # 说明用户输入过滤原不在列表中 我们不需要过滤
                                qa_system.logger.warning(f'用户输入了无效的来源 {source_filter}')
                                source_filter = None
                            else:
                                qa_system.logger.info('用户选择了来源过滤: {source_filter}')

                        answer = ''
                        for token, is_complete in qa_system.query(query, source_filter, session_id):
                            if token:
                                answer += token
                            if is_complete:
                                print()
                                break

                except Exception as e:
                    qa_system.logger.error(f'系统处理异常: {e} 可能需要重启系统')
                    print(f'处理问题时出错: {e}')

if __name__ == '__main__':
    main()
