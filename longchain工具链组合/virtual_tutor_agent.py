from langchain_classic.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Tongyi
from langchain_classic.memory import ConversationBufferMemory
from typing import Dict, Any, List
import os
import json
import dashscope

# 从环境变量获取 dashscope 的 API Key
api_key = os.environ.get("DASHSCOPE_API_KEY")
dashscope.api_key = api_key


# ===================== 自定义工具定义 =====================

class LearningProfileTool:
    """学习画像构建工具：根据学生自述，整理当前水平、目标与偏好"""

    def __init__(self):
        self.name = "学习画像分析"
        self.description = "根据学生的自我介绍与学习目标，提取当前水平、目标、薄弱点和偏好（输出为JSON字符串，用于后续工具输入）"

    def run(self, description: str) -> str:
        """根据学生描述构建学习画像（简化规则示例）

        参数:
            description: 学生的自我介绍、学习背景与目标
        返回:
            JSON 字符串形式的学习画像
        """
        profile: Dict[str, Any] = {
            "raw_description": description,
            "level": "未知",
            "goals": [],
            "weak_points": [],
            "preferences": [],
        }

        text = description

        # 简单规则抽取（水平特征）
        if any(k in text for k in ["入门", "零基础", "小白"]):
            profile["level"] = "入门/零基础"
        elif any(k in text for k in ["基础还可以", "有一定基础", "学过一段时间"]):
            profile["level"] = "初级/有基础"
        elif any(k in text for k in ["想进阶", "想提高", "中级"]):
            profile["level"] = "中级"
        elif "备考" in text or "考试" in text:
            profile["level"] = "应试阶段（水平待估计）"

        # 目标
        if "考试" in text or "备考" in text or "通过" in text:
            profile["goals"].append("通过相关考试")
        if "面试" in text or "求职" in text or "跳槽" in text:
            profile["goals"].append("提升求职/面试能力")
        if "兴趣" in text or "爱好" in text:
            profile["goals"].append("兴趣驱动学习")

        # 偏好
        if "视频" in text:
            profile["preferences"].append("偏好视频学习")
        if "刷题" in text:
            profile["preferences"].append("偏好刷题练习")
        if "系统" in text or "体系" in text:
            profile["preferences"].append("偏好系统化梳理")

        # 薄弱点（简单关键词）
        if "基础不好" in text or "不扎实" in text:
            profile["weak_points"].append("基础知识不扎实")
        if "记不住" in text or "容易忘" in text:
            profile["weak_points"].append("记忆与复习规划不足")
        if "做题" in text or "不会做题" in text:
            profile["weak_points"].append("解题能力薄弱")

        return json.dumps(profile, ensure_ascii=False, indent=2)


class KnowledgeAssessmentTool:
    """知识诊断工具：根据学生的回答或自述，推测当前掌握情况"""

    def __init__(self):
        self.name = "知识诊断"
        self.description = "根据学习画像（JSON）和学生近期表现描述，给出当前知识掌握情况的诊断总结"

    def run(self, profile_json: str, performance: str) -> str:
        """结合画像和表现描述，给出诊断结论

        参数:
            profile_json: 学习画像工具的输出 JSON 字符串
            performance: 学生最近的作业/测验/学习表现描述
        返回:
            文本形式的诊断总结
        """
        try:
            profile = json.loads(profile_json)
        except Exception:
            profile = {"raw_description": profile_json}

        lines: List[str] = ["知识掌握诊断结果:"]
        level = profile.get("level", "未知")
        weak = profile.get("weak_points", [])

        lines.append(f"- 画像判断水平: {level}")
        if weak:
            lines.append(f"- 明显薄弱点: {', '.join(weak)}")

        # 根据 performance 做一些简单推断
        if any(k in performance for k in ["错很多", "错得很多", "大部分不会"]):
            lines.append("- 近期练习情况: 错题较多，总体掌握度偏低。")
        elif any(k in performance for k in ["偶尔出错", "有些题不会"]):
            lines.append("- 近期练习情况: 整体还可以，但关键知识点存在漏洞。")
        elif any(k in performance for k in ["基本都会", "掌握还可以"]):
            lines.append("- 近期练习情况: 基础掌握尚可，可以向进阶内容过渡。")
        else:
            lines.append("- 近期练习情况: 描述信息有限，建议通过小测进一步量化。")

        lines.append("建议下一步：针对薄弱点设计针对性练习与复习计划。")
        return "\n".join(lines)


class StudyPlanTool:
    """学习计划规划工具：根据画像与诊断，制定阶段性学习计划"""

    def __init__(self):
        self.name = "学习计划制定"
        self.description = "综合学习画像与知识诊断结果，生成分阶段的学习规划（可按天/周拆分）"

    def run(self, profile_json: str, diagnosis_text: str, duration_weeks: int = 4) -> str:
        """生成一个简化版学习计划

        参数:
            profile_json: 学习画像 JSON
            diagnosis_text: 知识诊断文本
            duration_weeks: 规划时长（周）
        返回:
            文本形式的阶段性学习计划
        """
        try:
            profile = json.loads(profile_json)
        except Exception:
            profile = {"level": "未知", "goals": [], "weak_points": []}

        level = profile.get("level", "未知")
        goals = profile.get("goals", [])
        weak = profile.get("weak_points", [])

        lines: List[str] = [f"{duration_weeks} 周学习计划（概要）:"]
        lines.append(f"- 当前水平: {level}")
        if goals:
            lines.append(f"- 学习目标: {', '.join(goals)}")
        if weak:
            lines.append(f"- 重点攻克薄弱点: {', '.join(weak)}")

        # 简单按周拆分
        for week in range(1, duration_weeks + 1):
            if week == 1:
                lines.append(
                    f"\n第 {week} 周：夯实基础\n"
                    f"- 梳理基础概念与公式，整理错题本。\n"
                    f"- 每天安排 30~60 分钟复习 + 20~30 分钟练习题。"
                )
            elif week == 2:
                lines.append(
                    f"\n第 {week} 周：针对薄弱点专项突破\n"
                    f"- 根据诊断结果与错题本，挑选 1~2 个薄弱模块集中练习。\n"
                    f"- 每天安排 40 分钟专题练习 + 20 分钟总结。"
                )
            elif week == 3:
                lines.append(
                    f"\n第 {week} 周：综合训练与模拟\n"
                    f"- 进行小型模拟测试，按考试或实战场景出题。\n"
                    f"- 复盘模拟结果，继续补齐知识漏洞。"
                )
            else:
                lines.append(
                    f"\n第 {week} 周：查漏补缺与巩固提升\n"
                    f"- 回顾前几周的学习内容，重新做典型错题。\n"
                    f"- 适当增加难度，尝试进阶/拓展题目。"
                )

        lines.append("\n建议结合个人时间安排做细化（按每天/具体时间段分配）。")
        return "\n".join(lines)


class ResourceRecommendationTool:
    """学习资源推荐工具：根据计划与偏好推荐资源"""

    def __init__(self):
        self.name = "学习资源推荐"
        self.description = "根据学习计划与学生偏好，推荐合适的学习资源类型与使用策略"

    def run(self, profile_json: str, plan_text: str, subject: str) -> str:
        """给出资源推荐（不依赖真实外部资源，仅做类型建议）

        参数:
            profile_json: 学习画像 JSON
            plan_text: 学习计划文本
            subject: 学习科目/方向（如 数学、英语、编程）
        返回:
            文本形式的资源推荐列表
        """
        try:
            profile = json.loads(profile_json)
        except Exception:
            profile = {"preferences": []}

        prefs = profile.get("preferences", [])
        lines: List[str] = [f"{subject} 学习资源推荐:"]

        # 根据偏好推荐资源形式
        if "偏好视频学习" in prefs:
            lines.append("- 视频课程：选择结构清晰、有配套练习的系统视频课，每天看 1~2 讲并做随堂题。")
        if "偏好刷题练习" in prefs:
            lines.append("- 题库与练习：使用线上题库/刷题 App，按知识点和难度分层刷题，并记录错题。")
        if "偏好系统化梳理" in prefs or not prefs:
            lines.append("- 教材与笔记：选择权威教材或官方教程，配合自己的知识框架做思维导图或笔记。")

        lines.append(
            "\n使用建议：\n"
            "- 与前面制定的学习计划相结合，明确每天/每周要完成的具体资源与任务。\n"
            "- 避免只“看不练”，看完视频或讲义后务必做相应练习。"
        )
        return "\n".join(lines)


class PracticeGeneratorTool:
    """练习任务生成工具：根据薄弱点自动生成练习建议"""

    def __init__(self):
        self.name = "练习任务生成"
        self.description = "根据薄弱点与学习阶段，生成可执行的练习任务清单（题型、数量、时间安排等）"

    def run(self, diagnosis_text: str, stage: str = "本周") -> str:
        """生成练习任务清单（描述级别，不实际出题）

        参数:
            diagnosis_text: 知识诊断文本
            stage: 学习阶段描述（如 本周、今天）
        返回:
            文本形式的练习任务列表
        """
        lines: List[str] = [f"{stage} 练习任务建议:"]

        # 粗略根据关键字调整任务侧重点
        if "解题能力薄弱" in diagnosis_text or "错题较多" in diagnosis_text:
            lines.append("- 每天 10~20 道针对薄弱知识点的典型题目，做完后认真订正。")
            lines.append("- 每周安排 1~2 次小测，模拟真实考试/实战的答题节奏。")
        if "基础知识不扎实" in diagnosis_text:
            lines.append("- 每天抽 15~20 分钟，按知识点快速复习并自测基本概念。")
        if "记忆与复习规划不足" in diagnosis_text:
            lines.append("- 采用间隔重复：第 1/3/7 天分别复习同一批知识点或错题。")

        if len(lines) == 1:
            lines.append("- 根据当前学习计划，自主从课本/题库中选择 10~15 道题进行巩固练习。")

        return "\n".join(lines)


class FeedbackReflectionTool:
    """学习反思总结工具：帮助学生基于完成情况做阶段性复盘"""

    def __init__(self):
        self.name = "学习反思总结"
        self.description = "根据学生对本阶段学习完成情况的自述，帮助梳理收获、问题和下阶段优化点"

    def run(self, completion_report: str) -> str:
        """根据完成情况做总结与反思提示

        参数:
            completion_report: 学生对近期学习执行情况的自我反馈
        返回:
            文本形式的反思总结与改进建议
        """
        lines: List[str] = ["本阶段学习反思与建议:"]

        if any(k in completion_report for k in ["没完成", "落下", "拖延"]):
            lines.append("- 完成度：低于预期，存在任务积压或拖延。")
            lines.append("- 建议：适当压缩单次任务量，缩短每次学习时间，先建立“每天学一点”的节奏。")
        elif any(k in completion_report for k in ["基本完成", "大部分完成", "按计划"]):
            lines.append("- 完成度：总体良好，已经形成基本学习习惯。")
            lines.append("- 建议：在保证频率的基础上，适当提升难度或增加综合性任务。")
        else:
            lines.append("- 完成度：描述不够具体，建议列出本周实际完成的资源与题量。")

        if any(k in completion_report for k in ["效率不高", "容易分心", "专注不了"]):
            lines.append("- 专注问题：可以尝试番茄钟、学习打卡等方式提升专注时间质量。")

        lines.append("- 建议记录：每周固定做一次 5~10 分钟的复盘，总结“做得好/需要改进”的两三点。")
        return "\n".join(lines)


# ===================== Agent 构建 =====================


def create_virtual_tutor_agent():
    """创建虚拟助教 Agent（支持多工具串联，个性化学习建议与答疑）"""
    # 实例化工具
    profile_tool = LearningProfileTool()
    assessment_tool = KnowledgeAssessmentTool()
    plan_tool = StudyPlanTool()
    resource_tool = ResourceRecommendationTool()
    practice_tool = PracticeGeneratorTool()
    reflection_tool = FeedbackReflectionTool()

    tools = [
        Tool(
            name=profile_tool.name,
            func=profile_tool.run,
            description=(
                "根据学生提供的背景和目标描述，构建学习画像，输出 JSON 字符串。"
                "通常应作为第一步使用，以便后续工具复用画像。"
            ),
        ),
        Tool(
            name=assessment_tool.name,
            func=assessment_tool.run,
            description=(
                "输入为：1) 学习画像 JSON（可由“学习画像分析”工具输出），2) 学生近期表现描述。"
                "输出对当前知识掌握情况的诊断总结，可作为后续学习计划和练习生成工具的输入。"
            ),
        ),
        Tool(
            name=plan_tool.name,
            func=plan_tool.run,
            description=(
                "根据学习画像 JSON 和知识诊断结果文本，生成若干周的学习计划。"
                "第一个参数是画像 JSON，第二个参数是诊断文本，第三个参数是周数（整数，可选）。"
            ),
        ),
        Tool(
            name=resource_tool.name,
            func=resource_tool.run,
            description=(
                "根据学习画像 JSON、学习计划文本以及科目名称（如 数学/英语/编程），"
                "推荐适合的资源类型和使用策略。"
            ),
        ),
        Tool(
            name=practice_tool.name,
            func=practice_tool.run,
            description=(
                "根据知识诊断结果文本和学习阶段描述（如 本周/今天），"
                "生成可执行的练习任务建议列表。"
            ),
        ),
        Tool(
            name=reflection_tool.name,
            func=reflection_tool.run,
            description=(
                "根据学生对最近一段时间学习完成情况的自述，"
                "帮助梳理收获、问题与下阶段需要优化的地方。"
            ),
        ),
    ]

    # 初始化 LLM
    llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=api_key)

    # 提示模板：说明工具间的典型串联关系
    prompt = PromptTemplate.from_template(
        """你是一名耐心专业的虚拟助教，负责为学生提供个性化学习建议和答疑。
你可以使用以下工具来辅助完成任务：
{tools}
可用工具名称: {tool_names}

工具之间可以串联使用，典型流程包括（但不限于）：
- 先用“学习画像分析”理解学生的基础与目标；
- 再用“知识诊断”结合学生近期表现做更精细的水平判断；
- 然后用“学习计划制定”为未来几周规划学习路径；
- 接着用“学习资源推荐”和“练习任务生成”给出具体可执行方案；
- 在阶段结束后，用“学习反思总结”帮助学生复盘和调整策略。

请根据学生的提问内容和当前上下文，在适当的时候调用合适的工具，可以多轮使用工具，并把前一个工具的输出作为后一个工具的输入。

使用以下 ReAct 格式进行推理：
问题: 学生的提问或需求
思考: 你当前的分析思路
行动: 要使用的工具名称，必须是 [{tool_names}] 中的一个
行动输入: 传给工具的输入
观察: 工具返回的结果
... (思考 / 行动 / 观察 可以重复多次)
思考: 我已经有了完整的个性化建议或答复
回答: 用通俗、鼓励、具体的方式回答学生，并给出后续可执行建议

现在开始与学生对话和辅导。
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


def process_student_request(message: str) -> str:
    """对学生的提问/自述进行一站式处理，返回虚拟助教的回复

    参数:
        message: 学生的自然语言输入（可以包含背景、问题、需求等）
    返回:
        助教给学生的回复文本
    """
    try:
        agent = create_virtual_tutor_agent()
        response = agent.invoke({"input": message})
        return response.get("output", str(response))
    except Exception as e:
        return f"处理学生请求时出错: {str(e)}"


if __name__ == "__main__":
    # 简单示例：学生自述与提问
    demo_message = (
        "老师好，我是计算机专业大二学生，编程基础还可以，"
        "想在两个月内系统提升算法和刷题能力，准备找实习面试。"
        "目前刷题的时候错得很多，容易忘记做过的题。"
        "可以帮我制定一个合适的学习计划，并推荐一下资源和练习方式吗？"
    )
    print("学生提问/需求:\n", demo_message)
    print("\n虚拟助教回复:\n")
    print(process_student_request(demo_message))


