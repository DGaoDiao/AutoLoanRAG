from langchain_core.prompts import PromptTemplate

from base.config import Config
from base.logger import logger

from openai import OpenAI


class StrategySelector:
    def __init__(self):
        """初始化对象。
                
                params:
                    无。
                
                return:
                    无。"""
        self.client = OpenAI(
            api_key=Config().LLM_DASHSCOPE_API_KEY,
            base_url=Config().LLM_DASHSCOPE_BASE_URL,
        )
        self.strategy_prompt_template = self._get_strategy_prompt()

    def call_dashscope(self, prompt):
        """执行 call_dashscope 函数。
                
                params:
                    prompt: 参数说明。
                
                return:
                    函数返回值。"""
        try:
            completion = self.client.chat.completions.create(
                model=Config().LLM_MODEL,
                messages = [
                    {'role': 'system', 'content': '你是一个有用的助手，能够根据用户输入的Prompt严格执行并返回可靠的结果'},
                    {'role': 'user', 'content': prompt}
                ],
                temperature = 0.1
            )
            return completion.choices[0].message.content if completion.choices else '直接检索'
        except Exception as e:
            logger.error(f'Dashscope API 调用失败: {e}')

            return '直接检索'

    def _get_strategy_prompt(self):
        """执行 _get_strategy_prompt 函数。
                
                params:
                    无。
                
                return:
                    函数返回值。"""
        return PromptTemplate(
            template='''
            你是一个专业的问题分析师，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                    * 示例：
                        * 查询：AI 学科学费是多少？
                        * 策略：直接检索
                    * 查询：JAVA的课程大纲是什么？
                        * 策略：直接检索
            2.  **假设问题检索（HyDE）：**
                * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                * 适用场景：适用于查询较为抽象，问题较为开放，直接检索效果不佳的问题，例如：
                    * 示例：
                        * 查询：人工智能在教育领域的应用有哪些？
                        * 策略：假设问题检索
            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                    * 示例：
                        * 查询：比较 Milvus 和 Zilliz Cloud 的优缺点。
                        * 策略：子查询检索
            4.  **回溯问题检索：**
                * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                * 适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题，例如：
                    * 示例：
                        * 查询：我有一个包含 100 亿条记录的数据集，想把它存储到 Milvus 中进行查询。可以吗？
                        * 策略：回溯问题检索

            你可以先考虑问题是否足够简单，然后再考虑问题是否复杂，如果是一个复杂问题，考虑下是问题有深度还是有广度，然后再考虑问题是不是一个开放性的问题
            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
            ''',
            input_variables=['query']
        )

    # 选择检索策略 -> 选择检索策略的核心方法 -> 整合模板和大模型调用, 返回最终策略.
    def select_strategy(self, query):
        """函数作用: 根据用户查询, 选择最合适的检索增强策略.
                                :param query: 用户输入的查询文本(字符串)
                                :return: 字符串 -> 选中的检索策略名称 -> 例如: 直接检索, 子查询检索...
                        
                        params:
                            query: 参数说明。
                
                return:
                    函数返回值。"""
        # 1. 格式化提示模板: 将用户查询填充到提示模板的query为止, 生成发给大模型的完整提示, 调用大模型获取策略.
        strategy = self.call_dashscope(self.strategy_prompt_template.format(query=query)).strip()
        # 2. 记录日志.
        logger.info(f"为查询 '{query}' 选择的检索策略：{strategy}")
        # 3. 返回选中的策略.
        return strategy

if __name__ == '__main__':
    strategy_selector = StrategySelector()
    res = strategy_selector.select_strategy('如何进行时间管理')
