import textwrap
import time
from typing import List

# 兼容当前 LangChain 版本：老式 Agent 接口在 langchain_classic 中
from langchain_classic.agents import (
    Tool,               # 可用工具
    AgentExecutor,      # Agent 执行器
    create_react_agent,  # 快速创建 ReAct Agent 的工厂函数
)
# PromptTemplate: 管理 LLM 的提示词
from langchain_core.prompts import PromptTemplate
# 通义千问模型
from langchain_community.llms import Tongyi  # 导入通义千问 Tongyi 模型
# LLM 抽象基类在 langchain_core.language_models 中
from langchain_core.language_models import BaseLanguageModel as BaseLLM


# 定义了 LLM 的 Prompt Template（用于公司信息问答）
CONTEXT_QA_TMPL = """
根据以下提供的信息，回答用户的问题
信息：{context}

问题：{query}
"""
CONTEXT_QA_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template=CONTEXT_QA_TMPL,
)


# 输出结果显示：每行最多 60 字符，每个字符停留 0.1 秒（动态显示效果）
def output_response(response: str) -> None:
    """以打字机效果打印模型回答。"""
    if not response:
        return
    # 每行最多 60 个字符
    for line in textwrap.wrap(response, width=60):
        for word in line.split():
            for char in word:
                print(char, end="", flush=True)
                time.sleep(0.1)
            print(" ", end="", flush=True)  # 单词之间加空格
        print()  # 换行
    # 一次问答结束的分隔线
    print("----------------------------------------------------------------")


# 模拟公司产品和公司介绍的数据源
class TeslaDataSource:
    """封装与特斯拉相关的“自有知识库”与工具函数。"""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    # 工具 1：产品描述
    def find_product_description(self, product_name: str) -> str:
        """根据产品名称返回产品描述（简单字典模拟）。"""
        product_info = {
            "Model 3": "具有简洁、动感的外观设计，流线型车身和现代化前脸。定价23.19-33.19万",
            "Model Y": "在外观上与Model 3相似，但采用了更高的车身和更大的后备箱空间。定价26.39-36.39万",
            "Model X": "拥有独特的翅子门设计和更加大胆的外观风格。定价89.89-105.89万",
        }
        # 基于产品名称 => 产品描述（增加一定的容错能力，避免因为多余字符或大小写导致匹配失败）
        name = product_name.strip()
        # 1）先尝试精确匹配
        if name in product_info:
            return product_info[name]
        # 2）再做一次不区分大小写和包含关系的模糊匹配
        lower_name = name.lower()
        for key, value in product_info.items():
            if key.lower() in lower_name or lower_name in key.lower():
                return value
        return "没有找到这个产品"

    # 工具 2：公司介绍
    def find_company_info(self, query: str) -> str:
        """模拟公司介绍文档数据库，让 LLM 根据信息回答问题。"""
        context = """
        特斯拉最知名的产品是电动汽车，其中包括Model S、Model 3、Model X和Model Y等多款车型。
        特斯拉以其技术创新、高性能和领先的自动驾驶技术而闻名。公司不断推动自动驾驶技术的研发，
        并在车辆中引入了各种驾驶辅助功能，如自动紧急制动、自适应巡航控制和车道保持辅助等。
        """
        # prompt 模板 = 上下文 context + 用户的 query
        prompt = CONTEXT_QA_PROMPT.format(query=query, context=context)
        # 使用 LLM 进行推理（Tongyi 基于 invoke 调用）
        return self.llm.invoke(prompt)


# ReAct Agent 的中文提示词模板
AGENT_TMPL = """你是一个善于使用工具来回答问题的智能助手。下面是你可以使用的工具：

{tools}

当你需要回答用户问题时，请严格按照以下用 --- 括起来的格式思考和输出：

---
Question: 我需要回答的问题
Thought: 针对上述问题，我需要做些什么（可以是中文思考）
Action: 从 "{tool_names}" 中选择一个最合适的工具名
Action Input: 调用该工具所需要的输入（通常是用户的原始问题或其中的关键字段）
Observation: 该工具返回的结果
...（以上 Thought / Action / Action Input / Observation 的组合可以重复多次，用于多步推理）
Thought: 我现在知道最终答案
Final Answer: 用自然语言给出用户问题的最终答案（只写答案本身，不要再解释思考过程）
---

现在开始回答，记得在给出最终答案前，需要按照指定格式进行逐步推理。

Question: {input}
{agent_scratchpad}
"""


if __name__ == "__main__":
    # 定义 LLM（通义千问）
    llm = Tongyi(model_name="qwen-turbo", dashscope_api_key="YOUR_DASHSCOPE_API_KEY")

    # 自有数据源
    tesla_data_source = TeslaDataSource(llm)

    # 定义工具列表
    tools: List[Tool] = [
        Tool(
            name="查询产品名称",
            func=tesla_data_source.find_product_description,
            description="当用户询问具体产品（如 Model 3、Model Y、Model X）的描述、价格、特点等信息时使用此工具。输入的是产品名称，例如：Model 3。",
        ),
        Tool(
            name="公司相关信息",
            func=tesla_data_source.find_company_info,
            description="当用户询问关于特斯拉公司本身的问题时使用此工具，例如：公司介绍、公司历史、公司技术、公司业务、公司特点、特斯拉怎么样、特斯拉公司如何等。输入的是用户的完整问题。",
        ),
    ]

    # 将上面的字符串模板封装为 PromptTemplate，供 ReAct Agent 使用
    react_prompt = PromptTemplate.from_template(AGENT_TMPL)

    # 使用 LangChain 提供的 ReAct 工厂方法创建 Agent（内部会自动处理中间步骤与工具调用）
    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

    # 创建 Agent 执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,          # 关闭中间推理日志，只保留最终答案
        max_iterations=2,       # 最多允许 2 轮推理（一次调用工具，一次给最终答案）
        return_intermediate_steps=True,  # 返回中间步骤，便于在没有 Final Answer 时兜底使用 Observation
        # 当解析 LLM 输出失败时，不直接抛错，而是把错误信息作为 Observation 反馈给 LLM，提示它按 ReAct 格式重写
        handle_parsing_errors=(
            "输出格式不符合要求。请严格按照给定的 ReAct 模板输出："
            "必须包含 Question、Thought、Action、Action Input、Observation 和 Final Answer 字段，"
            "且每一轮思考都要按照模板中的格式书写。"
        ),
    )

    # 交互式主循环：可以一直提问，直到 Ctrl+C
    while True:
        try:
            user_input = input("请输入您的问题：")
            if not user_input.strip():
                continue

            # ReAct Agent 的推荐用法：使用 invoke，并传入 {"input": ...}
            result = agent_executor.invoke({"input": user_input})
            # 优先使用 Agent 的输出；如果因为迭代上限导致输出为提示语，
            # 或者根本没有 Final Answer（只是在不断调用工具），
            # 则退化为使用「最后一次 Observation」作为回答，避免多次重复推理的输出
            answer = (result.get("output") or "").strip()
            if (
                ("Agent stopped due to iteration limit" in answer or not answer)
                and "intermediate_steps" in result
                and result["intermediate_steps"]
            ):
                # 取最后一次工具调用的 Observation 作为最终回答
                _last_action, last_observation = result["intermediate_steps"][-1]
                answer = str(last_observation)
            output_response(answer)
        except KeyboardInterrupt:
            break


