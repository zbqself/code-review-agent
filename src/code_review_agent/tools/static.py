"""静态检查工具：对代码做“不运行也能发现的问题”的检查。

两条路径：
    1. 如果系统里装了 ruff（一个流行的 Python linter），优先用它，结果更全；
    2. 否则用内置的 ast 模块做几个简单的启发式检查（兜底），保证没装 ruff 也能用。
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path

from .context import resolve_path, workspace_root
from .registry import tool


def _run_ruff(target: Path) -> str | None:
    """运行 `ruff check`；成功则返回其输出（可能为空 = 无问题），执行失败返回 None。

    区分“没发现问题”和“ruff 执行失败”很重要：如果 ruff 因为参数错误/路径不可读
    等原因失败，我们不能把它误当成“检查通过”。
    """
    ruff = shutil.which("ruff")  # 在 PATH 里找 ruff 可执行文件
    if not ruff:
        return None
    try:
        proc = subprocess.run(
            [ruff, "check", "--output-format", "concise", str(target)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    # ruff check 退出码：0=无问题，1=发现问题，2=运行出错。
    # 2 及以上表示 ruff 本身执行失败，此时返回 None 让调用方回退到内置 AST 检查。
    if proc.returncode >= 2:
        return None
    return proc.stdout.strip()


def _ast_checks(source: str) -> list[str]:
    """用 Python 内置 ast 模块做几个轻量检查（不依赖第三方库）。

    目前只查两类常见问题（作为演示/兜底）：
      1) 裸 except:（会吞掉所有异常，包括 Ctrl+C 等）；
      2) 函数默认参数是可变对象（如空列表），会导致多次调用共享同一个对象。
    """
    issues: list[str] = []
    try:
        tree = ast.parse(source)  # 把源码解析成抽象语法树(AST)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} (line {exc.lineno})"]

    # ast.walk 遍历语法树的每个节点
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            # 找“裸 except:”（既没有异常类型，也没有 as 变量）
            for handler in node.handlers:
                if handler.type is None and handler.name is None:
                    issues.append(
                        f"line {handler.lineno}: bare 'except:' swallows all exceptions "
                        "(including KeyboardInterrupt/SystemExit)"
                    )
        # async def 在 AST 里是 ast.AsyncFunctionDef，是 FunctionDef 的兄弟类而非子类，
        # 所以必须两者都判断，否则会漏掉异步函数里的可变默认参数。
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # 找“可变默认参数”（list/dict/set 作为函数默认值）
            for default in node.args.defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    issues.append(
                        f"line {default.lineno}: mutable default argument in {node.name}() "
                        "is shared across calls"
                    )
    return issues


@tool(
    "run_static_check",
    "Run static analysis on a file or directory (ruff if installed, otherwise built-in AST checks).",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File or directory relative to root."},
        },
        "required": ["path"],
    },
)
def run_static_check(path: str) -> str:
    """对文件或目录做静态检查：优先 ruff，否则用内置 AST 检查。"""
    target = resolve_path(path)
    if not target.exists():
        return json.dumps({"error": f"path does not exist: {path}"})

    # 单个 Python 文件
    if target.is_file() and target.suffix == ".py":
        ruff_out = _run_ruff(target)
        if ruff_out is not None:
            return ruff_out or "ruff: no issues"
        return "\n".join(_ast_checks(target.read_text(encoding="utf-8", errors="replace"))) or "no issues"

    # 目录：对目录跑 ruff，或对所有 .py 文件逐个做 AST 检查
    if target.is_dir():
        ruff_out = _run_ruff(target)
        if ruff_out is not None:
            return ruff_out or "ruff: no issues"
        findings: list[str] = []
        for f in sorted(target.rglob("*.py")):
            if ".git" in f.parts or "venv" in f.parts:
                continue  # 跳过 .git 和虚拟环境目录
            src = f.read_text(encoding="utf-8", errors="replace")
            for issue in _ast_checks(src):
                findings.append(f"{f.relative_to(workspace_root())}: {issue}")
            if len(findings) > 100:
                break  # 结果太多就截断
        return "\n".join(findings) or "no issues"

    return json.dumps({"error": f"unsupported path type: {path}"})
