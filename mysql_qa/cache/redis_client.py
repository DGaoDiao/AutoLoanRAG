import redis
import json
import sys
from pathlib import Path
from base.logger import logger
from base.config import Config

class RedisClient():
    def __init__(self):
        """初始化对象。
                
                params:
                    无。
                
                return:
                    无。"""
        self.logger = logger

    def __enter__(self):
        """执行 __enter__ 函数。
                
                params:
                    无。
                
                return:
                    函数返回值。"""
        try:
            self.client = redis.StrictRedis(
                host=Config().REDIS_HOST,
                port=Config().REDIS_PORT,
                db=Config().REDIS_DB,
                password=Config().REDIS_PASSWORD,
                decode_responses=True)
            return self
        except redis.RedisError as e:
            self.logger.error(f'Redis连接错误 {e}')
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """执行 __exit__ 函数。
                
                params:
                    exc_type: 参数说明。
                    exc_val: 参数说明。
                    exc_tb: 参数说明。
                
                return:
                    函数返回值。"""
        self.client.close()

    def set_data(self, key, value):
        """执行 set_data 函数。
                
                params:
                    key: 参数说明。
                    value: 参数说明。
                
                return:
                    函数返回值。"""
        try:
            self.client.set(key, json.dumps(value, ensure_ascii=False))
            self.logger.info(f'Redis存储成功 {key}')
        except redis.RedisError as e:
            self.logger.error(f'Redis存储失败 {e}')

    def get_data(self, key):
        """执行 get_data 函数。
                
                params:
                    key: 参数说明。
                
                return:
                    函数返回值。"""
        try:
            data = self.client.get(key)
            self.logger.info(f'成功获取数据')
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(f'获取数据失败 {e}')
            return None

    def get_answer(self, query):
        """执行 get_answer 函数。
                
                params:
                    query: 参数说明。
                
                return:
                    函数返回值。"""
        try:
            answer = self.client.get(f'auto_loan:answer:{query}')
            if answer:
                self.logger.info(f'从Redis中获取答案: {query}')
                return json.loads(answer)
            return None
        except redis.RedisError as e:
            self.logger.error(f'答案获取失败 {e}')
            return None

if __name__ == '__main__':
    with RedisClient() as client:
        print(client.logger)
