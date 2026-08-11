import configparser #解析INI格式的文件
import os           #导入路径操作模块
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
CONFIG_PATH = ROOT_DIR / "config.ini"

class Config:
    def __init__(self, config_file=str(CONFIG_PATH)):
        # 创建配置文件解析器
        self.config = configparser.ConfigParser()
        # 读取配置文件
        self.config.read(config_file, encoding='utf-8')
        # 解析并存储参数
        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='123456')
        self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='qa')
        self.MYSQL_LIMIT = int(self.config.get('mysql', 'limit', fallback=5))

        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')
        self.REDIS_PORT = self.config.get('redis', 'port', fallback=6379)
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='1234')
        self.REDIS_DB = self.config.get('redis', 'db', fallback='0')

        self.MILVUS_HOST = self.config.get('milvus', 'host', fallback='localhost')
        self.MILVUS_PORT = self.config.get('milvus', 'port', fallback=19530)
        self.MILVUS_DATABASE = self.config.get('milvus', 'database_name', fallback='AutoLoanRAG')
        self.MILVUS_COLLECTION_NAME = self.config.get('milvus', 'collection_name', fallback='auto_loan_rag')

        self.LLM_MODEL = self.config.get('llm', 'model', fallback='deepseek-r1:8b')
        self.LLM_DASHSCOPE_API_KEY = self.config.get('llm', 'dashscope_api_key', fallback='ollama')
        self.LLM_DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url', fallback='http://localhost:11434/v1')

        self.RETRIEVAL_PARENT_CHUNK_SIZE = int(self.config.get('retrieval', 'parent_chunk_size', fallback=1200))
        self.RETRIEVAL_CHILD_CHUNK_SIZE = int(self.config.get('retrieval', 'child_chunk_size', fallback=300))
        self.RETRIEVAL_CHUNK_OVERLAP = int(self.config.get('retrieval', 'chunk_overlap', fallback=50))
        self.RETRIEVAL_RETRIEVAL_K = int(self.config.get('retrieval', 'retrieval_k', fallback=5))
        self.RETRIEVAL_CANDIDATE_M = int(self.config.get('retrieval', 'candidate_m', fallback=2))

        sources = self.config.get(
            'app',
            'valid_sources',
            fallback='policy,product,application,contract,repayment,risk,vehicle',
        )
        self.APP_VALID_SOURCES = [item.strip() for item in sources.split(',') if item.strip()]
        # Keep the old misspelled name for compatibility with existing callers.
        self.APP_VAILD_SOURCES = self.APP_VALID_SOURCES
        self.APP_CUSTOMER_SERVICE_PHONE = self.config.get(
            'app', 'customer_service_phone', fallback='400-000-0000'
        )


        self.ROOT_DIR = ROOT_DIR



        self.LOG_DIR = ROOT_DIR / 'log'



if __name__ == "__main__":
    config = Config(CONFIG_PATH)
    print(config.REDIS_PORT)
    print(config.ROOT_DIR)
    print(config.LOG_DIR)
