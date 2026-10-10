"""git 工具：让 agent 能看仓库的改动（diff）和状态（status）。

这些工具内部用 subprocess 调系统里的 git 命令，把 git 的文本输出直接给模型。
"""

from __future__ import annotations

import json
import subprocess

from .context import MAX_OUTPUT_CHARS, resolve_path, workspace_root
from .registry import tool


def _git(*args: str, cwd: str) -> str | None:
    """执行一条 git 命令并返回标准输出；失败（非 git 仓库、超时等）返回 None。"""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,  # 加超时，避免命令卡死
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


@tool(
    "git_diff",
    "Get uncommitted changes (git diff) in the workspace, or the diff of a given file/dir. "
    "Returns a JSON object with a 'diff' field (and 'truncated' flag when cut off).",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Optional path relative to root to restrict the diff."},
        },
        "required": [],
    },
)
def git_diff(path: str | None = None) -> str:
    """获取工作区未提交的改动（git diff），可选限定到某个文件/目录。"""
    root = str(workspace_root())
    # 如果指定了 path，就把它解析成相对根的路径，附加到 git diff 命令后面
    target = resolve_path(path).relative_to(workspace_root()) if path else None
    args = ["diff", "--no-color", "-U3"]
    if target is not None:
        args.append(str(target))
    out = _git(*args, cwd=root)
    if out is None:
        return json.dumps({"error": "not a git repo or git failed"})
    if not out.strip():
        return json.dumps({"diff": "", "note": "working tree is clean"})
    # 统一返回 JSON，并标记是否被截断（防止超大 diff 撑爆上下文）
    truncated = len(out) > MAX_OUTPUT_CHARS
    return json.dumps({"diff": out[:MAX_OUTPUT_CHARS], "truncated": truncated})


@tool(
    "git_status",
    "Get git status (untracked/modified files) in the workspace. Returns a JSON object with a 'status' field.",
    {
        "type": "object",
        "properties": {},
        "required": [],
    },
)
def git_status() -> str:
    """获取 git status（有哪些新增/修改/未跟踪的文件）。"""
    out = _git("status", "--short", cwd=str(workspace_root()))
    if out is None:
        return json.dumps({"error": "not a git repo or git failed"})
    # 统一返回 JSON，和 read_file/list_files/search_code 等工具保持一致
    return json.dumps({"status": out.strip() or "working tree clean"})
