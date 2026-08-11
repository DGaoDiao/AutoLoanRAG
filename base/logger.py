import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve()
from base.config import *

LOG_DIR = Config().LOG_DIR
LOG_DIR.mkdir(exist_ok=True)

def setup_logger(name: str = 'AutoLoanRAG', path=LOG_DIR):
    '''
    创建并返回一个日志记录器，支持同时输出到文件和控制台
    :param name:
    :param path:保存路径
    :return: 日志记录器
    '''
    # 初始化日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 初始化终端处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    #初始化文件处理器
    file_handler = logging.FileHandler(str(path/'app.log'), encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    #设置格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    #加入格式
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)


    #绑定格式
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()

