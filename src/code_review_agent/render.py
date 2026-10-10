"""渲染模块：把 ReviewReport 对象转成 Markdown 文本，方便在终端里漂亮地展示。"""

from __future__ import annotations

from .schemas import FixReport, ReviewReport


def render_markdown(report: ReviewReport) -> str:
    """把一个结构化评审报告渲染成 Markdown 字符串。

    参数：
        report: 已经解析好的 ReviewReport 对象。
    返回：
        一段 Markdown 文本（标题、总结、亮点、问题列表、总体评价）。
    """
    # 用列表拼行，最后 join 成字符串，比反复用 + 拼接更清晰、更高效
    lines = ["# 代码评审报告", "", f"**总结**：{report.summary}", ""]

    # 亮点部分：只有存在亮点时才输出
    if report.strengths:
        lines.append("## 亮点")
        lines += [f"- {s}" for s in report.strengths]
        lines.append("")

    # 问题部分
    lines.append("## 问题")
    if not report.issues:
        lines.append("_未发现明显问题_")
    # enumerate(..., 1) 让问题序号从 1 开始
    for i, issue in enumerate(report.issues, 1):
        # 组装位置信息，形如 "src/a.py:10-15"
        loc = issue.file or "?"
        if issue.start_line:
            loc += f":{issue.start_line}"
            if issue.end_line and issue.end_line != issue.start_line:
                loc += f"-{issue.end_line}"
        lines.append(f"### {i}. [{issue.severity}] {issue.title}")
        lines.append(f"- 位置：`{loc}` · 类别：`{issue.category}`")
        lines.append(f"- 描述：{issue.description}")
        lines.append(f"- 建议：{issue.suggestion}")
        # 有代码片段就用代码块包起来展示
        if issue.code_snippet:
            lines.append("\n```")
            lines.append(issue.code_snippet.strip())
            lines.append("```")
        lines.append("")

    lines.append("## 总体评价")
    lines.append(report.overall_assessment)
    return "\n".join(lines)


def render_fix_markdown(report: FixReport) -> str:
    """把一个修复方案(FixReport)渲染成 Markdown 文本。"""
    lines = ["# 修复方案", "", f"**总结**：{report.summary}", ""]
    if not report.fixes:
        lines.append("_未生成修复方案_")
    for i, f in enumerate(report.fixes, 1):
        loc = f.file or "?"
        if f.start_line:
            loc += f":{f.start_line}"
            if f.end_line and f.end_line != f.start_line:
                loc += f"-{f.end_line}"
        lines.append(f"### {i}. 修复问题 #{f.issue_index}（`{loc}`）")
        lines.append(f"说明：{f.explanation}")
        if f.code_before:
            lines.append("\n**修改前**：\n```")
            lines.append(f.code_before.strip())
            lines.append("```")
        if f.code_after:
            lines.append("\n**修改后**：\n```")
            lines.append(f.code_after.strip())
            lines.append("```")
        lines.append("")
    if report.note:
        lines.append(f"## 说明\n{report.note}")
    return "\n".join(lines)
