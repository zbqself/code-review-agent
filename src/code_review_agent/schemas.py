"""评审结果的数据结构定义（用 Pydantic 声明）。

为什么用 Pydantic：
    我们希望大模型最后返回一份“结构化”的评审报告，而不是一段自由文字。
    Pydantic 能帮我们：
      1) 声明每个字段的类型（字符串/整数/列表等）；
      2) 自动把这份结构转成 JSON Schema（作为工具调用的参数定义）；
      3) 拿到模型返回的 JSON 后做校验，并转成 Python 对象（ReviewReport）。

    这样程序就能可靠地解析结果，而不是用正则去抓 JSON。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# Severity 是一个“字面量类型”：问题严重程度只能取下面 5 个值之一。
# 用 Literal 会让生成的 JSON Schema 把可取值限定在这几个枚举里，避免模型乱填。
Severity = Literal["critical", "error", "warning", "info", "suggestion"]


class ReviewIssue(BaseModel):
    """一条代码问题。

    这是评审报告里最小的单元：一个问题对应某一行/一段代码、一个说明和一个建议。
    每个字段的 description 会出现在 JSON Schema 里，帮助大模型理解该填什么。
    """

    severity: Severity = Field(description="严重程度")
    category: str = Field(description="问题类别，如 correctness/security/performance/error-handling")
    file: str | None = Field(default=None, description="问题所在的文件路径（相对仓库根）")
    start_line: int | None = Field(default=None, description="起始行号")
    end_line: int | None = Field(default=None, description="结束行号")
    title: str = Field(description="一句话标题")
    description: str = Field(description="问题说明：为什么是问题、触发条件（1-2 句话，简洁）")
    suggestion: str = Field(description="具体的修复建议（简洁）")
    code_snippet: str | None = Field(default=None, description="相关代码片段（可选，尽量简短，必要时才填）")


class ReviewReport(BaseModel):
    """整份评审报告。

    这是 submit_review 工具的参数结构，也是我们最终拿到的结果对象。
    """

    summary: str = Field(description="一段话总结本次评审的整体结论")
    issues: list[ReviewIssue] = Field(default_factory=list, description="发现的问题列表")
    strengths: list[str] = Field(default_factory=list, description="代码写得好的地方")
    overall_assessment: str = Field(description="总体评价与下一步建议")


class FixSuggestion(BaseModel):
    """针对评审报告里某一条问题给出的具体修复方案。"""

    issue_index: int = Field(description="对应评审报告里第几条问题（从 1 开始）")
    file: str | None = Field(default=None, description="要修改的文件路径")
    start_line: int | None = Field(default=None, description="修改起始行号")
    end_line: int | None = Field(default=None, description="修改结束行号")
    explanation: str = Field(description="修复思路说明：为什么这样改能解决问题")
    code_before: str | None = Field(default=None, description="修改前的代码片段（只写关键 1-3 行）")
    code_after: str | None = Field(default=None, description="修改后的代码片段（只写关键 1-3 行）")


class FixReport(BaseModel):
    """修复者(FixAgent)输出的修复方案集合。

    这是 submit_fixes 工具的参数结构，与评审报告一一对应。
    """

    summary: str = Field(description="一段话总结本次修复")
    fixes: list[FixSuggestion] = Field(default_factory=list, description="修复方案列表")
    note: str = Field(default="", description="补充说明（如哪些问题未修复及原因）")
