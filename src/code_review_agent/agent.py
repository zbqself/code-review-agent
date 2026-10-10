"""Agent 核心模块：实现“agent 循环”。

这是整个项目最核心的文件。它实现了把“一个只会聊天的大模型”变成“能自主干活的 agent”的关键：
一个循环，反复做“问模型 -> 执行工具 -> 把结果再喂给模型”，直到模型提交最终评审。

名词解释（建议先记住）：
    - agent 循环(loop)：上面这个 while 循环。
    - 工具调用(tool call)：模型“说出”想调用的工具名和参数，由我们的程序真正执行。
    - 消息(messages)：对话历史，累积 system/user/assistant/tool 四种角色的消息。
    - 结构化输出：模型最后调用 submit_review，参数就是 ReviewReport 的结构。
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Callable

# pydantic_function_tool：openai SDK 提供的工具函数，能把一个 Pydantic 模型
# 自动转成“工具调用”所需的 JSON Schema。这里用 ReviewReport 生成 submit_review 工具。
from openai import pydantic_function_tool
from pydantic import ValidationError

from .config import LLMConfig
from .llm import LLMClient
from .prompts import fixer_prompt, system_prompt, user_prompt
from .schemas import FixReport, ReviewReport
from .tools import run_tool, set_workspace_root, tool_schemas

# submit_review 工具的描述文字（发给模型，告诉它这个工具是干嘛的）
_SUBMIT_DESCRIPTION = (
    "Submit the complete, final structured code review. Call this exactly once when done."
)


def _try_parse_json(content: str, model_cls):
    """尝试把一段文本解析成某个 Pydantic 模型；解析失败返回 None。

    用来区分两种“模型只回了文本、没调工具”的情况：
        - 模型真的把结果以 JSON 形式写在了正文里（可解析 → 返回结果对象）；
        - 模型只是在“叙述”（比如输出 “Submitting now…”）而没真正提交
          （解析失败 → 返回 None，调用方继续推它去调 submit 工具）。
    """
    text = content.strip()
    m = re.match(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```$", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        return model_cls.model_validate_json(text)
    except Exception:
        return None


class ReviewAgent:
    """代码评审 Agent：手写了一个 agent 循环。"""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.llm = LLMClient(config)  # 创建大模型客户端
        # 会话历史（这就是“记忆”）：start() 会重置它，ask() 会在它基础上继续追加
        self.messages: list[dict[str, Any]] = []
        self._final_report: ReviewReport | None = None  # 存放模型最终提交的评审报告

    @lru_cache(maxsize=1)
    def _all_tool_schemas(self) -> list[dict[str, Any]]:
        """把“普通工具”和“submit_review 工具”合在一起，作为本轮可用工具列表。

        普通工具（read_file 等）在 tools 包里注册；submit_review 比较特殊，
        它由 ReviewReport 这个 Pydantic 模型自动生成 JSON Schema。
        工具 Schema 在运行时不会变化，所以用 lru_cache 缓存结果，避免重复反射生成。
        """
        schemas = tool_schemas()
        schemas.append(
            pydantic_function_tool(
                ReviewReport,
                name="submit_review",
                description=_SUBMIT_DESCRIPTION,
            )
        )
        return schemas

    def _execute(self, name: str, arguments: str) -> str:
        """执行一次工具调用，返回“给模型看的结果字符串”。

        参数：
            name:      工具名（模型决定要调哪个）。
            arguments: 模型给出的参数（JSON 字符串）。

        特殊处理：
            如果工具是 submit_review，说明模型已经完成评审，我们把它的参数解析成
            ReviewReport 保存起来，并返回一个“已接收”的确认；
            其他工具则交给 tools 包的 run_tool 去真正执行。

        容错说明：
            工具的 arguments 是模型自由生成的字符串，完全可能不是合法 JSON，
            submit_review 的参数也可能缺字段。这里统一“把错误转成 JSON 回喂模型”，
            让模型重试，而不是让异常一路冒泡把整个评审进程弄崩。
        """
        # 先容错解析参数 JSON；失败时返回错误信息让模型重试
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return json.dumps(
                {"error": f"invalid tool arguments JSON: {exc}. 参数可能过大被截断，请精简输出后重试。"}
            )

        if name == "submit_review":
            # 解析模型提交的结构化结果；不合规时返回错误信息让模型修正
            try:
                self._final_report = ReviewReport.model_validate(args)
            except ValidationError as exc:
                return json.dumps({"error": f"invalid review payload: {exc}"})
            return json.dumps({"accepted": True})
        return run_tool(name, args)

    def start(
        self,
        target: str,
        diff_context: str | None = None,
        on_text: Callable[[str], None] | None = None,
        knowledge_context: str | None = None,
    ) -> ReviewReport:
        """开始一次全新的代码评审，返回结构化报告。

        参数：
            target:       要评审的仓库目录或单个文件。
            diff_context: 可选，一段 diff/patch 内容作为初始上下文。
            on_text:      可选，流式文本回调（用于实时打印模型输出）。
            knowledge_context: 可选，RAG 检索到的规范/背景资料，会注入到 system 提示词里。
        """
        # 每次评审开始前都重置结果，避免同一个实例被复用时返回上一次的陈旧报告
        self._final_report = None

        # ---- 第 1 步：确定工作区根目录 ----
        # 先校验目标存在，避免用户拼错路径后 agent 静默产出一份空报告
        if not os.path.exists(target):
            raise ValueError(f"目标路径不存在: {target}")

        # 目标是目录时，根目录就是它；目标是单个文件时，根目录是它所在的目录，
        # 并把“要评审的东西”改成文件名，方便模型用相对路径去读。
        if self._is_dir(target):
            root = target
            hint = target
        else:
            root = self._parent(target)
            hint = os.path.basename(target)
        set_workspace_root(root)

        # ---- 第 2 步：准备初始消息，开启一个“新会话” ----
        # system 定人设和规则；user 告诉它评审什么。
        system_msg = system_prompt(self.config.review_language)
        # 如果提供了 RAG 检索到的知识库上下文，就追加到 system 里，让模型参考着评审
        if knowledge_context:
            system_msg += (
                "\n\n请参考以下团队代码规范 / 背景资料，评审时若适用请据此判断：\n"
                + knowledge_context
            )
        self.messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt(hint, diff_context)},
        ]

        return self._run_loop(on_text)

    def ask(
        self,
        question: str,
        on_text: Callable[[str], None] | None = None,
    ) -> ReviewReport:
        """在已有会话的基础上继续追问（会话记忆的关键）。

        和 start() 的区别：start() 会重置对话历史，ask() 保留历史、只追加一条新问题。
        这样模型能“记得”刚才评审过的内容和结论，做出连贯的回答。
        """
        if not self.messages:
            raise RuntimeError("还没有开始评审，无法追问。请先调用 start()")
        self.messages.append({"role": "user", "content": question})
        self._final_report = None
        return self._run_loop(on_text)

    def _run_loop(self, on_text: Callable[[str], None] | None = None) -> ReviewReport:
        """agent 循环主体：反复“问模型 -> 执行工具 -> 结果回填”，直到拿到最终报告。"""
        tools = self._all_tool_schemas()
        # 当模型只回文字、不调工具时，下一轮强制它调用 submit_review
        force_submit = False

        # ---- agent 循环（最多 max_iterations 轮） ----
        for _ in range(self.config.max_iterations):
            # 流式调用：文本逐段回调给 on_text，最后一次性拿到完整消息
            tool_choice = (
                {"type": "function", "function": {"name": "submit_review"}}
                if force_submit
                else "auto"
            )
            msg = self.llm.chat_stream(
                self.messages, tools=tools, tool_choice=tool_choice, on_text=on_text
            )
            content = msg["content"]
            tool_calls = msg["tool_calls"] or []

            # 把模型这轮的回复追加进对话历史。
            # assistant 消息结构要和 OpenAI 接口约定一致：
            # 有工具调用时带 tool_calls（含 id/type/function），文本则放在 content。
            assistant: dict[str, Any] = {"role": "assistant"}
            if content:
                assistant["content"] = content
            if tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ]
            self.messages.append(assistant)

            # 如果模型请求调用工具，就逐个执行，并把结果以 role="tool" 回填
            if tool_calls:
                force_submit = False  # 模型真的调了工具，重置强制标记
                for tc in tool_calls:
                    # 通知界面：正在调用哪个工具
                    if on_text:
                        on_text(f"\n[调用工具] {tc['name']}\n")
                    result = self._execute(tc["name"], tc["arguments"])
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": result}
                    )
                # 如果刚才执行的是 submit_review，说明结果已经拿到，直接返回
                if self._final_report is not None:
                    return self._final_report
                continue  # 否则继续下一轮，让模型看到工具结果后继续思考

            # 模型没调用工具（只返回了文本）
            # 先尝试把文本当 JSON 解析（有的模型直接输出 JSON 而不调工具）。
            # 解析失败说明模型只是在“说话”（如 “Submitting now”）而没真正提交，
            # 这时推它去调用 submit_review，而不是把叙述当成最终结果。
            if content and content.strip():
                parsed = _try_parse_json(content, ReviewReport)
                if parsed is not None:
                    return parsed
            self.messages.append(
                {
                    "role": "user",
                    "content": "请直接调用 submit_review 工具提交评审结果，不要只输出文字说明。",
                }
            )
            force_submit = True  # 下一轮强制调用 submit_review
            continue

        # ---- 达到轮数上限的兜底：强制让模型提交 ----
        # 如果模型因为读太多文件而迟迟不 submit，不要直接崩溃，而是追加一条明确指令，
        # 再给它最后一次机会，让它基于已经读到的内容提交评审结果。
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the tool-call limit. Based on the code you have already "
                    "inspected, call submit_review NOW with your findings. Do not call any other tool."
                ),
            }
        )
        msg = self.llm.chat_stream(self.messages, tools=tools, on_text=on_text)
        content = msg["content"]
        # 只处理最后一次返回里的 submit_review，忽略其它工具调用
        for tc in msg["tool_calls"] or []:
            if tc["name"] == "submit_review":
                self._execute(tc["name"], tc["arguments"])
                if self._final_report is not None:
                    return self._final_report
        # 模型仍然没提交（比如只返回了文本），用兜底解析包装，保证程序不崩
        return self._parse_fallback(
            content or "评审未完成：已达到工具调用上限，且模型未提交结构化结果。"
        )

    @staticmethod
    def _is_dir(path: str) -> bool:
        """判断目标是不是一个目录。"""
        return os.path.isdir(path)

    @staticmethod
    def _parent(path: str) -> str:
        """返回文件所在目录（用于把单个文件的工作区根目录设成它的父目录）。"""
        return os.path.dirname(os.path.abspath(path))

    @staticmethod
    def _parse_fallback(content: str) -> ReviewReport:
        """兜底解析：如果模型没走 submit_review、而是直接输出了一段文本，尝试把文本当 JSON 解析。

        有些模型/情况下模型可能“偷懒”不调工具，而是直接在正文里输出 JSON。
        这里做最努力(best-effort)的解析：去掉可能包裹的 ``` 代码围栏，再尝试解析成 ReviewReport。
        解析失败就退化成一份“只有文字、没有结构化问题列表”的报告，保证程序不崩。
        """
        text = content.strip()
        # 用正则精确剥离最外层的 Markdown 代码围栏（形如 ```json\n...\n``` 或 ```...```）。
        # 注意：不能用 strip("`")，那会把 JSON 内容本身两端的反引号也误删。
        m = re.match(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```$", text, re.S)
        if m:
            text = m.group(1).strip()
        try:
            return ReviewReport.model_validate_json(text)
        except Exception as exc:
            # 解析/校验失败：把错误信息和原始文本都保留下来，避免“假阴性”
            #（看起来像“没有发现问题”，实际只是解析失败了）。
            return ReviewReport(
                summary="(模型未按结构化格式输出，解析失败)",
                issues=[],
                strengths=[],
                overall_assessment=(
                    f"[解析失败: {type(exc).__name__}: {exc}]\n\n原始输出:\n{content}"
                ),
            )


class FixAgent:
    """修复者 Agent：根据评审报告，生成具体的代码修复方案。

    这是“多 Agent 协作”里的第二个角色：
        - ReviewAgent（评审者）负责“找问题”，产出 ReviewReport；
        - FixAgent（修复者）负责“给修法”，拿 ReviewReport 做输入，读实际代码后产出 FixReport。

    和评审者不同，修复者的“最终提交”**不走 function calling**：
    因为 DeepSeek 对“复杂嵌套 JSON 的 function calling”不稳定（经常只回文字、不真正发起
    submit_fixes 调用）。所以这里让模型继续用 read_file/search_code 等工具读代码（这部分
    function calling 是可靠的），最后直接把 FixReport 以 **JSON 文本**输出，我们再解析。
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.llm = LLMClient(config)  # 创建大模型客户端
        self.messages: list[dict[str, Any]] = []

    def _execute(self, name: str, arguments: str) -> str:
        """执行一次“读工具”调用（修复者没有 submit 工具）。"""
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"invalid tool arguments JSON: {exc}"})
        return run_tool(name, args)

    def fix(
        self,
        target: str,
        report: ReviewReport,
        on_text: Callable[[str], None] | None = None,
    ) -> FixReport:
        """针对一份评审报告，读取目标代码并生成修复方案。"""
        # 设置和评审阶段一致的工作区根目录
        root = target if os.path.isdir(target) else os.path.dirname(os.path.abspath(target))
        set_workspace_root(root)
        self.messages = [
            {"role": "system", "content": fixer_prompt(report.model_dump_json())},
            {
                "role": "user",
                "content": f"请针对上面的评审报告，结合 {target} 里的实际代码，生成修复方案。",
            },
        ]
        return self._run_loop(on_text)

    def _run_loop(self, on_text: Callable[[str], None] | None = None) -> FixReport:
        """修复者的 agent 循环：用读工具收集信息，最后把 JSON 当文本输出。"""
        tools = tool_schemas()  # 只有 read_file / search_code / ...，没有 submit 工具
        for _ in range(self.config.max_iterations):
            # 修复者最终输出是一大段 JSON 文本，不实时流式打印它（否则终端会刷屏原始 JSON）；
            # 这里只拿回完整 content，最后统一由 CLI 渲染成 Markdown 展示。
            msg = self.llm.chat_stream(self.messages, tools=tools)
            content = msg["content"]
            tool_calls = msg["tool_calls"] or []

            assistant: dict[str, Any] = {"role": "assistant"}
            if content:
                assistant["content"] = content
            if tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ]
            self.messages.append(assistant)

            if tool_calls:
                # 有工具调用就执行并回填，继续下一轮
                for tc in tool_calls:
                    if on_text:
                        on_text(f"\n[调用工具] {tc['name']}\n")
                    result = self._execute(tc["name"], tc["arguments"])
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": result}
                    )
                continue

            # 没有工具调用：模型应该直接把 FixReport 的 JSON 当文本输出了
            if content and content.strip():
                parsed = _try_parse_json(content, FixReport)
                if parsed is not None:
                    return parsed
            # 文本不是合法 JSON：提示模型直接输出 JSON
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        "请直接输出 FixReport 的 JSON 对象本身"
                        '（形如 {"summary":...,"fixes":[...],"note":...}），不要输出其它文字。'
                    ),
                }
            )
            continue

        # 兜底：达到轮数上限仍没拿到 JSON
        return FixReport(
            summary="(修复者未输出结构化结果)",
            fixes=[],
            note="达到工具调用上限，仍未获得 JSON 输出。",
        )
