"""命令行入口：用 Typer 定义命令，用 Rich 美化输出。

Typer 负责把命令行参数（比如 --format json）自动转成 Python 函数参数；
Rich 负责在终端里渲染漂亮的 Markdown / 彩色输出。

本文件新增（第一批改造）：
    - 流式输出：把模型生成的文本实时打印出来，用户不用干等；
    - --chat 追问模式：评审结束后可继续追问，体现“会话记忆”。
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.markdown import Markdown

from .agent import ReviewAgent
from .config import LLMConfig
from .render import render_markdown

# 保证 Windows 终端下中文正常输出（若 stdout 不支持 reconfigure 则忽略）。
# Windows 老终端默认编码不是 UTF-8，打印中文会乱码；这里手动改成 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

# 创建 Typer 应用对象，这就是命令行程序的“主体”
app = typer.Typer(help="LLM-powered intelligent code review agent")
console = Console()


def _print_report(report, fmt: str) -> None:
    """按指定格式输出一份评审报告（json 或 markdown）。"""
    if fmt == "json":
        console.print(report.model_dump_json(indent=2))
    else:
        console.print(Markdown(render_markdown(report)))


@app.command()
def review(
    target: str = typer.Argument(..., help="Git repo directory or a single file to review"),
    format: str = typer.Option("markdown", "--format", "-f", help="markdown | json"),
    diff: str = typer.Option(None, "--diff", help="Optional path to a diff/patch file to include"),
    chat: bool = typer.Option(False, "--chat", "-c", help="评审结束后进入追问模式"),
    kb: str = typer.Option(None, "--kb", help="RAG 知识库目录（.md/.txt 规范文档），评审时检索并注入"),
    fix: bool = typer.Option(False, "--fix", help="评审后由修复者 agent 生成修复方案"),
) -> None:
    """评审命令：`code-review <目标>`。

    参数：
        target:   必填，要评审的仓库目录或单个文件。
        --format: 输出格式，markdown 或 json。
        --diff:   可选，一个 diff/patch 文件路径，会作为初始上下文塞给模型。
        --chat:   可选，评审结束后进入交互式追问（利用会话记忆）。
        --kb:     可选，一个存放代码规范/背景资料的目录，评审时用 RAG 检索并注入提示词。
        --fix:    可选，评审结束后再让“修复者”agent 生成具体修复方案（多 Agent 协作）。
    """
    # 显式校验 --format 的取值，非法值直接报错，而不是静默回退成 markdown
    if format not in ("markdown", "json"):
        raise typer.BadParameter("只支持 markdown 或 json", param_hint="--format")

    cfg = LLMConfig.from_env()  # 从环境变量读取配置

    # 如果指定了 --diff，就先把该文件内容读出来，作为初始上下文
    diff_context = None
    if diff:
        import pathlib

        try:
            diff_context = pathlib.Path(diff).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            # 读取失败时给出清晰提示，而不是抛一堆 traceback
            raise typer.BadParameter(f"无法读取 diff 文件 {diff}: {exc}", param_hint="--diff")
        # 限制长度，避免超大 patch 撑爆上下文、产生高额 token 费用
        if len(diff_context) > 32_000:
            diff_context = diff_context[:32_000] + "\n... (diff truncated)"

    # 如果指定了 --kb，就构建 RAG 知识库并检索相关规范，注入到 system 提示词里
    knowledge_context = None
    if kb:
        from .rag import build_kb, build_query, render_context

        kb_obj = build_kb(kb)
        chunks = kb_obj.search(build_query(target), k=5)
        if chunks:
            knowledge_context = render_context(chunks)
        console.print(
            f"[dim]已加载知识库：{kb}（检索到 {len(chunks)} 个相关片段）[/dim]"
        )

    agent = ReviewAgent(cfg)

    # 流式回调：模型每吐出一段文本就实时打印。
    # json 模式下不流式，避免把过程文字混进最终的 JSON 输出里。
    def on_text(text: str) -> None:
        if format != "json":
            sys.stdout.write(text)
            sys.stdout.flush()

    # 开始评审（流式输出）
    report = agent.start(
        target,
        diff_context=diff_context,
        on_text=on_text,
        knowledge_context=knowledge_context,
    )
    if format != "json":
        sys.stdout.write("\n")  # 流式内容结束后补一个换行
    _print_report(report, format)

    # 修复模式：评审完后，由“修复者(FixAgent)”生成具体修复方案（多 Agent 协作）
    if fix:
        from .agent import FixAgent
        from .render import render_fix_markdown

        fix_report = FixAgent(cfg).fix(target, report, on_text=on_text)
        if format != "json":
            sys.stdout.write("\n")
        if format == "json":
            console.print(fix_report.model_dump_json(indent=2))
        else:
            console.print(Markdown(render_fix_markdown(fix_report)))

    # 追问模式：体现会话记忆，模型记得刚才评审的上下文
    if chat:
        console.print("[bold green]评审完成，可以继续追问（直接回车退出）。[/bold green]")
        while True:
            question = input("追问> ").strip()
            if not question:
                break
            follow = agent.ask(question, on_text=on_text)
            if format != "json":
                sys.stdout.write("\n")
            _print_report(follow, format)


pr_app = typer.Typer(help="GitHub PR 评审 bot")


@pr_app.command()
def pr_review(
    repo: str = typer.Argument(..., help="GitHub 仓库，形如 owner/repo"),
    pr_number: int = typer.Argument(..., help="PR 编号"),
    fix: bool = typer.Option(False, "--fix", help="评审后同时生成修复方案"),
) -> None:
    """评审一个 GitHub Pull Request，并把报告作为评论贴回该 PR。

    需要先在 .env 里设置 GITHUB_TOKEN（一个带 repo 权限的 Personal Access Token）。
    """
    from .github_bot import review_pr

    comment = review_pr(repo, pr_number, fix=fix)
    console.print("[bold green]已把评审报告贴到 PR。[/bold green]")
    console.print(Markdown(comment))


if __name__ == "__main__":
    # 直接 `python -m code_review_agent.cli` 运行时也走这里
    app()
