#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from langchain_community.llms import Tongyi  # 导入通义千问Tongyi模型
import dashscope


class SimpleConversationChain:
    """简易对话链，模拟 ConversationChain 的基本行为。

    使用一个简单的对话历史，将历史轮次和本次输入拼成一个长 Prompt 交给 LLM。
    """

    def __init__(self, llm, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
        # 存储对话历史，结构为 [{"human": str, "ai": str}, ...]
        self.history = []

    def predict(self, input: str) -> str:
        """生成对当前输入的回复，并自动维护对话历史。"""
        # 构造对话上下文
        history_text = ""
        if self.history:
            lines = []
            for turn in self.history:
                lines.append(f"Human: {turn['human']}")
                lines.append(f"AI: {turn['ai']}")
            history_text = "\n".join(lines) + "\n"

        prompt = history_text + f"Human: {input}\nAI:"

        if self.verbose:
            print("=== 当前对话上下文 ===")
            print(history_text)
            print("=== 本次输入 ===")
            print(input)
            print("====================")

        # 调用底层 LLM
        response = self.llm.invoke(prompt)
        response_text = str(response)

        # 追加到历史
        self.history.append({"human": input, "ai": response_text})
        return response_text

# 从环境变量获取 dashscope 的 API Key
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

# 加载 Tongyi 模型
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=api_key)  # 使用通义千问qwen-turbo模型

# 使用自定义的带 memory 的对话链（替代旧版的 ConversationChain）
conversation = SimpleConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="你好，我是张三!")
print(output)


# In[2]:


output = conversation.predict(input="你记得我叫什么名字吗？")
print(output)

