import pymysql
import pandas as pd
from pathlib import Path
from base.config import Config
from base.logger import logger

CURRENT_DIR = Path(__file__).parent.absolute()

DATA_DIR = CURRENT_DIR.parent / "data"
DATA_PATH = DATA_DIR / 'JP学科知识问答.csv'
AUTO_LOAN_DATA_PATH = DATA_DIR / 'auto_loan_qa.csv'
QA_TABLE = 'auto_loan_qa'

class MySqlClient():
    def __init__(self):
        self.logger = logger

    def __enter__(self):
        try:
            #建立mysql连接
            self.connection = pymysql.connect(
                host = Config().MYSQL_HOST,
                user = Config().MYSQL_USER,
                password = Config().MYSQL_PASSWORD,
                database = Config().MYSQL_DATABASE,
            )
            self.cursor = self.connection.cursor()
            self.create_table()
            self.seed_auto_loan_data()
            self.logger.info('mysql连接成功')
            return self
        except pymysql.MySQLError as e:
            self.logger.error(f'MySQL数据库异常:{e}')
            raise e  #让调用方感知调用失败

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cursor.close()
        self.connection.close()

    def __existing_detection(self):
        if not self.connection:
            self.logger.error('数据库未连接')
            return False
        return True


    def create_table(self):
        if not self.__existing_detection():
            return
        query = """
        CREATE TABLE IF NOT EXISTS auto_loan_qa (
            id INT AUTO_INCREMENT PRIMARY KEY, 
            subject_name VARCHAR(100),
            question VARCHAR(1000),
            answer TEXT,
            UNIQUE KEY uq_auto_loan_question (question(255)))
        """
        try:
            self.cursor.execute(query)
            self.connection.commit()
            self.logger.info('创建表成功')
        except pymysql.MySQLError as e:
            self.logger.error(f'创建表失败 {e}')
            raise e

    def insert_data(self, data_path):
        if not self.__existing_detection():
            return
        try:
            data = pd.read_csv(data_path)
            for _, row in data.iterrows():
                insert_query = f"""
                INSERT INTO auto_loan_qa (subject_name, question, answer)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    subject_name = VALUES(subject_name), answer = VALUES(answer)
                """
                self.cursor.execute(insert_query, (row['subject_name'], row['question'], row['answer']))
            self.connection.commit()
            self.logger.info('数据插入成功')
        except Exception as e:
            self.logger.error(f'数据插入失败 {e}')
            self.connection.rollback()
            raise

    def seed_auto_loan_data(self):
        """Idempotently load the bundled automotive-loan FAQ dataset."""
        if AUTO_LOAN_DATA_PATH.exists():
            self.insert_data(AUTO_LOAN_DATA_PATH)

    def fetch_question(self):
        if not self.__existing_detection():
            return
        try:
            self.cursor.execute('SELECT question FROM auto_loan_qa ORDER BY id')
            results = self.cursor.fetchall()
            self.logger.info('数据查询成功')
            return results
        except pymysql.MySQLError as e:
            logger.error(f'数据查询失败 {e}')
            return []

    def fetch_answer(self, question):
        if not self.__existing_detection():
            return
        try:
            self.cursor.execute('SELECT answer FROM auto_loan_qa WHERE question = %s', (question,))
            results = self.cursor.fetchone()
            return results[0] if results else None

        except pymysql.MySQLError as e:
            self.logger.error(f'答案获取失败 {e}')
            return None



if __name__ == '__main__':
    with MySqlClient() as mysql_client:
        mysql_client.create_table()
        print(mysql_client.logger)


