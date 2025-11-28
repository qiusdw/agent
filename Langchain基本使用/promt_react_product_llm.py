import re
from typing import List, Union
# Python内置模块，用于格式化和包装文本
import textwrap
import time

# 兼容当前 LangChain 版本：老式 Agent 接口在 langchain_classic 中
from langchain_classic.agents import (
    Tool,               # 可用工具
    AgentExecutor,      # Agent执行
    LLMSingleActionAgent,  # 定义Agent
    AgentOutputParser,  # 输出结果解析
)
# Prompt 相关类在 langchain_core.prompts 中
from langchain_core.prompts import StringPromptTemplate, PromptTemplate
# LLMChain 在 langchain_classic.chains.llm 中
from langchain_classic.chains.llm import LLMChain
# Agent执行、Agent结束类型在 langchain_core.agents 中
from langchain_core.agents import AgentAction, AgentFinish
# 通义千问模型
from langchain_community.llms import Tongyi  # 导入通义千问Tongyi模型
# LLM 抽象基类在 langchain_core.language_models 中
from langchain_core.language_models import BaseLanguageModel as BaseLLM

# 定义了LLM的Prompt Template
CONTEXT_QA_TMPL = """
根据以下提供的信息，回答用户的问题
信息：{context}

问题：{query}
"""
CONTEXT_QA_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template=CONTEXT_QA_TMPL,
)

# 输出结果显示，每行最多60字符，每个字符显示停留0.1秒（动态显示效果）
def output_response(response: str) -> None:
    if not response:
        exit(0)
    # 每行最多60个字符
    for line in textwrap.wrap(response, width=60):
        for word in line.split():
            for char in word:
                print(char, end="", flush=True)
                time.sleep(0.1)  # Add a delay of 0.1 seconds between each character
            print(" ", end="", flush=True)  # Add a space between each word
        print()  # Move to the next line after each line is printed
    # 遇到这里，这个问题的回答就结束了
    print("----------------------------------------------------------------")

# 模拟公司产品和公司介绍的数据源
class TeslaDataSource:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    # 工具1：产品描述
    def find_product_description(self, product_name: str) -> str:
        """模拟公司产品的数据库"""
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

    # 工具2：公司介绍
    def find_company_info(self, query: str) -> str:
        """模拟公司介绍文档数据库，让llm根据信息回答问题"""
        context = """
        特斯拉最知名的产品是电动汽车，其中包括Model S、Model 3、Model X和Model Y等多款车型。
        特斯拉以其技术创新、高性能和领先的自动驾驶技术而闻名。公司不断推动自动驾驶技术的研发，并在车辆中引入了各种驾驶辅助功能，如自动紧急制动、自适应巡航控制和车道保持辅助等。
        """
        # prompt模板 = 上下文context + 用户的query
        prompt = CONTEXT_QA_PROMPT.format(query=query, context=context)
        # 使用LLM进行推理（新版 Tongyi 需要使用 invoke 调用）
        return self.llm.invoke(prompt)


AGENT_TMPL = """按照给定的格式回答以下问题。你可以使用下面这些工具：

{tools}

回答时需要遵循以下用---括起来的格式：

---
Question: 我需要回答的问题
Thought: 回答这个上述我需要做些什么
Action: "{tool_names}" 中的一个工具名
Action Input: 选择这个工具所需要的输入
Observation: 选择这个工具返回的结果
...（这个 思考/行动/行动输入/观察 可以重复N次）
Thought: 我现在知道最终答案
Final Answer: 原始输入问题的最终答案
---

现在开始回答，记得在给出最终答案前，需要按照指定格式进行一步一步的推理。

Question: {input}
{agent_scratchpad}
"""


class CustomPromptTemplate(StringPromptTemplate):
    template: str  # 标准模板
    tools: List[Tool]  # 可使用工具集合

    def format(self, **kwargs) -> str:
        """
        按照定义的 template，将需要的值都填写进去。
        Returns:
            str: 填充好后的 template。
        """
        # 取出中间步骤并进行执行
        intermediate_steps = kwargs.pop("intermediate_steps")  
        print('intermediate_steps=', intermediate_steps)
        print('='*30)
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # 记录下当前想法 => 赋值给agent_scratchpad
        kwargs["agent_scratchpad"] = thoughts  
        # 枚举所有可使用的工具名+工具描述
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )  
        # 枚举所有的工具名称
        kwargs["tool_names"] = ", ".join(
            [tool.name for tool in self.tools]
        )  
        cur_prompt = self.template.format(**kwargs)
        #print(cur_prompt)
        return cur_prompt

"""
    对Agent返回结果进行解析，有两种可能：
    1）还在思考中 AgentAction
    2）找到了答案 AgentFinal
"""
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """
        解析 llm 的输出，根据输出文本找到需要执行的决策。
        Args:
            llm_output (str): _description_
        Raises:
            ValueError: _description_
        Returns:
            Union[AgentAction, AgentFinish]: _description_
        """
        # 如果句子中包含 Final Answer 则代表已经完成
        if "Final Answer:" in llm_output:
            # 直接返回完整的 LLM 输出，这样最终打印结果就会按我们在模板中规定的格式展示
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )

        # 需要进行 AgentAction
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"  # 解析 action_input 和 action
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # 有些模型会在 Action Input 后面直接接上换行和 Observation 说明，这里做一次裁剪，保证传给工具的是真正的输入
        if "\nObservation" in action_input:
            action_input = action_input.split("\nObservation", 1)[0]
        # Agent执行
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )

if __name__ == "__main__":
    # 从环境变量获取通义千问 API 密钥，避免在代码中暴露真实密钥
    import os
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    # 定义LLM
    llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=api_key)  # 使用通义千问qwen-turbo模型
    # 自有数据
    tesla_data_source = TeslaDataSource(llm)
    # 定义的Tools
    tools = [
        Tool(
            name="查询产品名称",
            func=tesla_data_source.find_product_description,
            description="当用户询问具体产品（如Model 3、Model Y、Model X）的描述、价格、特点等信息时使用此工具。输入的是产品名称，例如：Model 3",
        ),
        Tool(
            name="公司相关信息",
            func=tesla_data_source.find_company_info,
            description="当用户询问关于特斯拉公司本身的问题时使用此工具，例如：公司介绍、公司历史、公司技术、公司业务、公司特点、特斯拉怎么样、特斯拉公司如何等。输入的是用户的完整问题。",
        ),
    ]
    
    # 用户定义的模板
    agent_prompt = CustomPromptTemplate(
        template=AGENT_TMPL,
        tools=tools,
        input_variables=["input", "intermediate_steps"],
    )
    # Agent返回结果解析
    output_parser = CustomOutputParser()
    # 最常用的Chain, 由LLM + PromptTemplate组成
    llm_chain = LLMChain(llm=llm, prompt=agent_prompt)
    # 定义的工具名称
    tool_names = [tool.name for tool in tools]
    # 定义Agent = llm_chain + output_parser + tools_names
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )
    # 定义Agent执行器 = Agent + Tools
    # verbose=False：不打印中间推理过程日志
    # max_iterations=2：最多推理两轮（一次调用工具，一次给出最终答案）
    # return_intermediate_steps=True：同时返回中间步骤，便于在没有 Final Answer 时兜底使用最后一次 Observation
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=2,
        return_intermediate_steps=True,
    )

    # 主过程：可以一直提问下去，直到Ctrl+C
    while True:
        try:
            user_input = input("请输入您的问题：")
            # 使用 invoke（新推荐方式）
            result = agent_executor.invoke({"input": user_input})
            # 优先使用 Agent 的 output；如果因为迭代上限导致输出为提示语，
            # 或者根本没有 Final Answer（只是在不断调用工具），
            # 则手动构造符合模板格式的完整输出
            response = (result.get("output") or "").strip()
            if (
                ("Agent stopped due to iteration limit" in response or not response)
                and "intermediate_steps" in result
                and result["intermediate_steps"]
            ):
                # 从最后一次工具调用中提取完整格式，并手动补充 Final Answer
                last_action, last_observation = result["intermediate_steps"][-1]
                # 从 last_action.log 中提取已有的格式（Question / Thought / Action / Action Input）
                action_log = last_action.log.strip()
                # 确保 action_log 以 --- 开头，如果没有则添加
                if not action_log.startswith("---"):
                    action_log = "---\n" + action_log
                # 手动构造完整的模板格式输出，按照图片中的样式：Observation -> Thought: 我现在知道最终答案 -> Final Answer（结尾没有---）
                response = f"""{action_log}
Observation: {last_observation}
Thought: 我现在知道最终答案
Final Answer: {last_observation}"""
            output_response(response)
        except KeyboardInterrupt:
            break