from langchain_classic.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Tongyi
from langchain_classic.memory import ConversationBufferMemory
from typing import Dict, Any
import json
import os
import dashscope

# 从环境变量获取 dashscope 的 API Key
api_key = os.environ.get("DASHSCOPE_API_KEY")
dashscope.api_key = api_key


# ===================== 基础工具定义 =====================

class SymptomCollectTool:
    """症状归纳工具：从用户的自然语言描述中，抽取关键信息并结构化"""

    def __init__(self):
        self.name = "症状归纳"
        self.description = "根据用户对网络故障的自然语言描述，抽取关键信息并输出结构化的症状摘要"

    def run(self, description: str) -> str:
        """将自然语言描述转为结构化症状信息（JSON 字符串）

        参数:
            description: 用户描述的网络故障情况
        返回:
            JSON 字符串，包含主要症状字段
        """
        # 这里不调用外部模型，只做简单示例归纳
        result: Dict[str, Any] = {
            "raw_description": description,
            "possible_scope": [],
            "keywords": [],
        }

        text = description
        keywords = []
        if "打不开" in text or "无法访问" in text or "超时" in text:
            keywords.append("连接失败")
        if "慢" in text or "很卡" in text or "延迟" in text:
            keywords.append("延迟高")
        if "偶尔" in text or "有时候" in text or "间歇" in text:
            keywords.append("间歇性故障")
        if "WiFi" in text or "无线" in text:
            result["possible_scope"].append("接入层/WiFi")
        if "内网" in text:
            result["possible_scope"].append("局域网/内网")
        if "外网" in text or "互联网" in text:
            result["possible_scope"].append("出口/运营商")

        result["keywords"] = list(set(keywords))
        return json.dumps(result, ensure_ascii=False, indent=2)


class ConnectivityCheckTool:
    """连通性检查工具：根据症状信息，模拟 ping/端口连通性分析"""

    def __init__(self):
        self.name = "连通性检查"
        self.description = "根据症状摘要（JSON）和目标地址，给出连通性检查结论（是否可达、是否丢包严重等）"

    def run(self, symptom_summary_json: str, target: str) -> str:
        """模拟连通性检查

        参数:
            symptom_summary_json: 上一步症状归纳工具的 JSON 字符串输出
            target: 需要检查的目标地址（域名或 IP）
        返回:
            文本形式的连通性分析结论
        """
        try:
            summary = json.loads(symptom_summary_json)
        except Exception:
            summary = {"raw_description": symptom_summary_json}

        desc = summary.get("raw_description", "")
        analysis = [f"目标地址: {target}"]

        # 简单规则模拟
        if any(k in desc for k in ["完全", "一直", "总是", "始终"]) and any(
            k in desc for k in ["打不开", "无法访问", "超时"]
        ):
            analysis.append("模拟结论: 目标基本不可达，疑似网络中断或 DNS/路由异常。")
        elif "有时候" in desc or "偶尔" in desc:
            analysis.append("模拟结论: 连通性间歇异常，可能存在丢包或链路抖动。")
        elif "慢" in desc or "很卡" in desc or "延迟" in desc:
            analysis.append("模拟结论: 连通性基本可达，但存在较高时延或带宽瓶颈。")
        else:
            analysis.append("模拟结论: 暂无法仅根据描述判定连通性，需要进一步排查。")

        # 帮助后续工具的“结构化提示”
        analysis.append("建议下一步: 根据连通性情况做路由/链路层面排查。")
        return "\n".join(analysis)


class PathAnalysisTool:
    """路径分析工具：根据连通性结论，推测问题大致位于哪一跳/哪一层"""

    def __init__(self):
        self.name = "路径分析"
        self.description = "根据连通性检查结果，推测故障可能在本地网络、网关、运营商还是远端服务侧"

    def run(self, connectivity_result: str) -> str:
        """路径/范围分析

        参数:
            connectivity_result: 连通性检查工具的输出文本
        返回:
            文本形式的路径/范围分析结果
        """
        text = connectivity_result
        result_lines = ["路径/范围分析结果:"]

        if "基本不可达" in text or "网络中断" in text:
            result_lines.append("- 故障范围: 可能在本地网关、出口路由器或运营商侧链路。")
        elif "间歇异常" in text or "丢包" in text or "抖动" in text:
            result_lines.append("- 故障范围: 可能是无线信号不稳、交换机负载高或运营商链路抖动。")
        elif "较高时延" in text or "带宽瓶颈" in text:
            result_lines.append("- 故障范围: 更可能是带宽不足或远端服务负载高。")
        else:
            result_lines.append("- 故障范围: 信息不足，建议补充更多日志与现象。")

        result_lines.append("建议下一步: 结合日志/设备告警做根因分析。")
        return "\n".join(result_lines)


class LogAnalysisTool:
    """日志分析工具：对路由器/交换机/应用日志进行模式识别"""

    def __init__(self):
        self.name = "日志分析"
        self.description = "分析用户提供的网络设备或应用日志，提取关键错误信息和时间点"

    def run(self, log_text: str) -> str:
        """简单的关键字日志分析

        参数:
            log_text: 日志文本
        返回:
            提炼后的日志关键信息
        """
        if not log_text.strip():
            return "未提供任何日志内容。"

        lines = log_text.splitlines()
        important = []
        keywords = ["timeout", "超时", "reset", "丢包", "down", "unreachable", "错误"]

        for line in lines:
            if any(k.lower() in line.lower() for k in keywords):
                important.append(line)

        if not important:
            return "在提供的日志中未发现典型错误关键字，请人工进一步确认。"

        result = ["日志关键信息摘要:"]
        result.extend(f"- {l}" for l in important[:30])
        if len(important) > 30:
            result.append(f"... 还有 {len(important) - 30} 条类似记录未全部列出。")
        return "\n".join(result)


class RootCauseInferenceTool:
    """根因推断工具：综合症状、连通性、路径和日志分析结果，给出根因猜测"""

    def __init__(self):
        self.name = "根因推断"
        self.description = "输入前面多个工具的分析结果文本，输出可能的根因和置信度（示例规则推断）"

    def run(
        self,
        symptom_summary: str,
        connectivity_result: str,
        path_analysis: str,
        log_summary: str,
    ) -> str:
        """示例性的规则根因推断（在真实场景可用 LLM 进一步增强）

        参数:
            symptom_summary: 症状归纳工具输出
            connectivity_result: 连通性检查工具输出
            path_analysis: 路径分析工具输出
            log_summary: 日志分析工具输出
        返回:
            文本形式的根因推断结果
        """
        text_all = "\n".join([symptom_summary, connectivity_result, path_analysis, log_summary])
        reasons = []

        if "WiFi" in symptom_summary or "无线" in symptom_summary:
            if "间歇" in symptom_summary or "抖动" in connectivity_result:
                reasons.append("无线信号质量差或干扰严重，导致间歇性丢包。")

        if "出口" in path_analysis or "运营商" in path_analysis:
            if "timeout" in log_summary.lower() or "超时" in log_summary:
                reasons.append("运营商链路质量问题或出口带宽不足，导致访问外网超时。")

        if "带宽瓶颈" in connectivity_result or "带宽不足" in path_analysis:
            reasons.append("带宽资源不足或高峰期拥塞，导致整体网络变慢。")

        if not reasons:
            reasons.append("根据当前信息暂无法给出高置信度根因，建议继续补充抓包/更多日志。")

        result = ["综合根因推断结果:"]
        for i, r in enumerate(reasons, start=1):
            result.append(f"{i}. {r}")
        return "\n".join(result)


class SolutionSuggestionTool:
    """解决方案推荐工具：针对推断出的根因给出排查步骤与缓解措施"""

    def __init__(self):
        self.name = "解决方案推荐"
        self.description = "根据根因推断结果文本，给出分步骤的排查建议和临时缓解方案"

    def run(self, root_cause_text: str) -> str:
        """根据根因给出解决方案建议

        参数:
            root_cause_text: 根因推断工具输出
        返回:
            文本形式的解决方案建议
        """
        suggestions = ["针对以上可能根因的建议操作步骤："]

        if "无线信号质量差" in root_cause_text:
            suggestions.append(
                "- 检查无线 AP 部署位置，尽量减少墙体遮挡，优化信道、降低干扰；必要时增加 AP 数量。"
            )
            suggestions.append("- 在故障终端附近使用测速工具，对比有线与无线网络性能。")

        if "出口带宽不足" in root_cause_text or "运营商链路质量问题" in root_cause_text:
            suggestions.append("- 在出口处抓取流量，确认是否存在大量占用带宽的业务或异常流量。")
            suggestions.append("- 与运营商联系，核实时延/丢包情况，并考虑升级带宽或更换线路。")

        if "带宽资源不足" in root_cause_text or "高峰期拥塞" in root_cause_text:
            suggestions.append("- 评估当前带宽利用率，必要时启用 QoS 对关键业务做优先级保障。")
            suggestions.append("- 尝试错峰访问或限制大流量下载/视频等非关键业务。")

        if len(suggestions) == 1:
            suggestions.append("- 建议进一步补充抓包数据（如 tcpdump）、路由/交换日志，以便更精确定位。")

        return "\n".join(suggestions)


# ===================== Agent 构建 =====================


def create_network_diagnosis_agent():
    """创建网络故障诊断 Agent（ReAct 风格，可根据需要调用/串联多个工具）"""
    # 实例化工具
    symptom_tool = SymptomCollectTool()
    connectivity_tool = ConnectivityCheckTool()
    path_tool = PathAnalysisTool()
    log_tool = LogAnalysisTool()
    root_cause_tool = RootCauseInferenceTool()
    solution_tool = SolutionSuggestionTool()

    tools = [
        Tool(
            name=symptom_tool.name,
            func=symptom_tool.run,
            description=(
                "将用户对网络故障的自然语言描述转为结构化的症状摘要（JSON 字符串），"
                "通常作为后续连通性检查等工具的输入。"
            ),
        ),
        Tool(
            name=connectivity_tool.name,
            func=connectivity_tool.run,
            description=(
                "根据【症状归纳】输出的 JSON 和目标地址，模拟给出连通性检查结论。"
                "第一个参数应传入症状摘要 JSON 字符串，第二个参数是目标地址。"
            ),
        ),
        Tool(
            name=path_tool.name,
            func=path_tool.run,
            description=(
                "输入为【连通性检查】的结果文本，输出可能的故障范围与路径分析。"
            ),
        ),
        Tool(
            name=log_tool.name,
            func=log_tool.run,
            description="输入网络设备/系统/应用日志文本，输出提炼后的关键错误信息。",
        ),
        Tool(
            name=root_cause_tool.name,
            func=root_cause_tool.run,
            description=(
                "综合多个步骤的结果推断根因。需要按顺序传入 4 个文本参数："
                "1) 症状摘要(JSON 或文本)，2) 连通性检查结果，"
                "3) 路径分析结果，4) 日志关键信息摘要。"
            ),
        ),
        Tool(
            name=solution_tool.name,
            func=solution_tool.run,
            description="输入【根因推断】结果文本，输出分步骤的排查和缓解建议。",
        ),
    ]

    # 初始化 LLM
    llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=api_key)

    # 提示模板：强调工具串联关系与使用场景
    prompt = PromptTemplate.from_template(
        """你是一个专业的网络故障诊断工程师，可以使用以下工具来协助排查问题：
{tools}
可用工具名称: {tool_names}

工具之间可以串联使用，例如：
- 先用“症状归纳”提炼用户描述，再把它作为输入，调用“连通性检查”；
- 再把“连通性检查”的输出交给“路径分析”；
- 结合用户提供的日志，用“日志分析”提取关键信息；
- 最后用“根因推断”与“解决方案推荐”给出完整诊断结论。

在排错时请遵循以下原则：
1. 先弄清楚故障现象（优先考虑使用“症状归纳”）。
2. 如果与访问慢/打不开有关，考虑使用“连通性检查”和“路径分析”。
3. 如果用户提供了日志，再使用“日志分析”。
4. 在已经有足够信息时，再调用“根因推断”和“解决方案推荐”。
5. 工具可以被多次调用，你可以根据前一次的观察结果调整后续步骤。

使用以下 ReAct 格式与我交互：
问题: 你需要解决的网络故障问题
思考: 你当前的分析思路
行动: 要使用的工具名称，必须是 [{tool_names}] 中的一个
行动输入: 传给工具的输入
观察: 工具返回的结果
... (思考 / 行动 / 观察 可以重复多次)
思考: 我已经有了完整的诊断结论和建议
回答: 给出对网络故障的分析结论和具体处置建议

现在开始排查！
问题: {input}
思考: {agent_scratchpad}"""
    )

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=ConversationBufferMemory(memory_key="chat_history"),
        verbose=True,
        handle_parsing_errors=False,
    )

    return agent_executor


def diagnose_network_issue(problem_description: str, logs: str = "") -> str:
    """对网络故障进行一站式诊断

    参数:
        problem_description: 用户对故障现象的自然语言描述
        logs: 可选的日志文本（如路由器日志、系统日志、应用日志等）
    返回:
        Agent 的最终回答
    """
    try:
        agent = create_network_diagnosis_agent()

        # 为了让 Agent 知道是否有日志，把日志一起放进输入中
        if logs:
            user_input = (
                f"故障描述：{problem_description}\n\n"
                f"以下是相关日志（可按需使用“日志分析”工具）：\n{logs}"
            )
        else:
            user_input = f"故障描述：{problem_description}\n（当前未提供日志，如需要可以提醒我补充。）"

        response = agent.invoke({"input": user_input})
        return response.get("output", str(response))
    except Exception as e:
        return f"诊断过程中发生错误: {str(e)}"


if __name__ == "__main__":
    # 示例：简单的网络故障描述
    demo_problem = (
        "办公室的 WiFi 最近经常出现访问外网很慢，有时候网页直接打不开，"
        "视频会议也经常卡顿，重启路由器后会好一会儿，但过一段时间又变差。"
    )
    print("示例故障描述:\n", demo_problem)
    print("\n诊断结果:\n")
    print(diagnose_network_issue(demo_problem))


