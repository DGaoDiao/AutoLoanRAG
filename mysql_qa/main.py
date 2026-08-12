from mysql_qa.db.mysql_client import MySqlClient
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.retrieval.bm25_search import BM25Search
from base.logger import logger
import time

class MySqlQASystem:
    def __init__(self,mysql_client, redis_client):
        """??????
        
        params:
            mysql_client: ?????
            redis_client: ?????
        
        return:
            ??"""
        self.logger = logger
        self.mysql_client = mysql_client
        self.redis_client = redis_client
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

    def query(self, query):
        """处理用户查询, 通过BM25查询搜索mysql获取答案, 返回结果并记录
                        :param query:
                        :return:
                
                params:
                    query: ?????
        
        return:
            ??????"""
        start_time = time.time()
        self.logger.info(f'处理查询:{query}')
        answer, _ = self.bm25_search.search(query)
        if answer:
            self.logger.info(f'MySQL答案:{answer}')
        else:
            self.logger.info('SQL中未找到答案, 需要调用RAG系统')
            answer = 'SQL中未找到答案'
        process_time = time.time() - start_time
        self.logger.info(f'查询耗时{process_time:.3f}秒')
        return answer

def main():
    """?? main ???
    
    params:
        ??
    
    return:
        ??????"""
    with MySqlClient() as mysql_client:
        with RedisClient() as redis_client:
            mysql_qa = MySqlQASystem(mysql_client, redis_client)

            try:
                time.sleep(1)
                print('\n欢迎使用伺服头骨')
                time.sleep(1)
                print('输入查询进行回答，输入exit退出')
                while True:
                    time.sleep(1)
                    query = input('\n请输入查询: ')
                    if query.lower() == 'exit':
                        logger.info('退出MySql系统')
                        print('再见, 为了帝皇')
                        break
                    answer = mysql_qa.query(query)
            except Exception as e:
                logger.error(f'系统错误{e}')
if __name__ == '__main__':
    main()

