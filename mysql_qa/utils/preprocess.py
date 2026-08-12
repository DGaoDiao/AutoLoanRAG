import jieba
from base.logger import logger


def preprocess_text(text):
    """?? preprocess_text ???
    
    params:
        text: ?????
    
    return:
        ??????"""
    logger.info(f'开始预处理文本')
    try:
        return jieba.lcut(text.lower())
    except AttributeError as e:
        logger.error(f'文本预处理失败 {e}')
        return None