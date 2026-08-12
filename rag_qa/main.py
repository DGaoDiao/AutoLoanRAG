# 该脚本是RAG的主入口
from base.config import Config
from base.logger import logger
from rag_qa.core.document_processor import process_document
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_system import RAGSystem
from openai import OpenAI
from pymilvus.exceptions import MilvusException
from pathlib import Path

config = Config()

def main(query_mode = True, directory_path = Config().ROOT_DIR/'rag_qa'/'data'):
    """函数功能: 系统主函数，控制两种运行模式
                数据处理模式: query_mode = False 解析指定目录的文档，分割未文档块并存入向量数据库
                交互式查询模式: query_mode = True 接受用户问题，调用RAG系统生成答案并展示
            :param query_mode:
            :param directory_path: 需要处理的文档所在路径
            :return:
        
        params:
            query_mode: ?????
            directory_path: ?????
    
    return:
        ??????"""

    try:
        client = OpenAI(
            api_key = Config().LLM_DASHSCOPE_API_KEY,
            base_url = Config().LLM_DASHSCOPE_BASE_URL
        )
    except Exception as e:
        logger.error(f'初始化 OpenAI客户端失败 {e}')

        if query_mode:
            print('错误: 无法初始化语言模型客户端, 无法进入查询模式')
            return

        client = None

    #封装大模型调用逻辑
    def call_dashscope(prompt):
        """?? call_dashscope ???
        
        params:
            prompt: ?????
        
        return:
            ??????"""
        if not client:
            logger.error('LLM客户端未初始化, 无法调用 call_dashscope')
            return f'错误: LLM客户端不可用'
        try:
            completion = client.chat.completions.create(
                model = Config().LLM_MODEL,
                messages = [
                    {'role': 'system', 'content': '你是一个有用的助手'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content
            else:
                logger.error('LLM API 调用返回无效响应或空消息')
                return '错误: LLM返回无响应'
        except Exception as e:
            logger.error(f'LLM API (call_dashscope) 调用失败): {e}')
            return f'错误: 调用LLM失败 - {e}'


    try:
        with VectorStore(
                collection_name=Config().MILVUS_COLLECTION_NAME,
                host = Config().MILVUS_HOST,
                port = Config().MILVUS_PORT,
                database = Config().MILVUS_DATABASE
            ) as vector_store:
            #数据处理模式
            if not query_mode:
                logger.info('进入数据处理模式')
                total_chunk_added=  0
                for source_dir in Config().APP_VAILD_SOURCES:
                    dir_path = Path(directory_path/ source_dir)
                    if dir_path.exists():
                        logger.info(f'开始处理目录: {dir_path}')
                        try:
                            chunks = process_document(dir_path)
                            if chunks:
                                vector_store.add_document(chunks)
                                total_chunk_added += len(chunks)
                                logger.info(f'成功处理目录: {dir_path} 添加了{len(chunks)}个文档块')
                            else:
                                logger.info(f'目录: {dir_path} 为空或者没有有效文档')
                        except Exception as e:
                            logger.error(f'处理目录 {dir_path}时出错: {e}')

                    else:
                        logger.warning(f'目录{dir_path}不存在 已经跳过')
                logger.info('数据处理完毕，一共添加了{total_chunk_added}个文档块')
            else:
                #交互模式
                if not client:
                    print('错误: 查询模式需要语言模型客户端，但初始化失败。')
                    return
                logger.info('进入交互模式查询')
                try:
                    rag_system = RAGSystem(vector_store, call_dashscope)
                except Exception as e:
                    logger.error(f'初始化 RAGSystem 失败:{e}')
                    print('错误：无法初始化RAG系统 无法进入查询模式')
                    return
                valid_source = Config().APP_VAILD_SOURCES
                print('\n欢迎使用汽车贷款智能顾问')
                print(f'支持的知识类别: {valid_source}')
                print("输入您的问题，或输入'exit'退出")

                while True:
                    query = input('请输入您的问题: ')
                    if query.lower() == 'exit':
                        logger.info('用户退出查询模式')
                        print('再见')
                        break

                    source_filter_input = input(f"请输入知识类别 ({'/'.join(valid_source)}) (直接回车默认不过滤)：").strip()
                    source_filter = None  # 默认不过滤
                    if source_filter_input:
                        if source_filter_input in valid_source:
                            source_filter = source_filter_input
                            logger.info(f'用户选择了学科过滤: {source_filter}')
                        else:
                            logger.warning(f"无效的学科选择 '{source_filter}'")
                            print(f"提示：输入的学科 '{source_filter_input}' 无效，将不过滤。")

                    try:
                        print("正在生成答案，请稍候...")
                        answer = rag_system.generata_answer(query, source_filter=source_filter)
                        print("-" * 30)
                        print(f"问题: {query}")
                        print(f"回答: {answer}")
                        print("-" * 30)
                    except Exception as e:
                        logger.error(f"处理查询 '{query}' 时失败: {str(e)}")
                        print(f"抱歉，处理您的问题时遇到了错误，请稍后重试或联系管理员。\n")

    except MilvusException as e:
        logger.error(f'向量数据库连接失败: {e}')
        return

if __name__ == '__main__':

    import argparse
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Auto Loan RAG System Main Entry Point')
    parser.add_argument('--data-processing', action='store_true', help='Run in data processing mode instead of query mode')
    parser.add_argument('--data_dir', type=str, default=Config().ROOT_DIR/'rag_qa'/'data', help='Path to the data directory')
    args = parser.parse_args()
    main(query_mode=(not args.data_processing), directory_path=args.data_dir)


