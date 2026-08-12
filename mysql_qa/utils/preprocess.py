import jieba
from base.logger import logger


def preprocess_text(text):
    """执行 preprocess_text 函数。
        
        params:
            text: 参数说明。
        
        return:
            函数返回值。"""
    logger.info(f'开始预处理文本')
    try:
        return jieba.lcut(text.lower())
    except AttributeError as e:
        logger.error(f'文本预处理失败 {e}')
        return None
