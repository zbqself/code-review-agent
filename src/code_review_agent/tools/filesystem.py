"""文件系统工具：让 agent 能读文件、列文件、搜索代码。

这些是 agent 最常用的“眼睛”。注意每个工具都通过 resolve_path() 做路径安全校验，
并且对输出做了长度限制，防止超长内容把上下文撑爆、把费用和延迟拉高。
"""

from __future__ import annotations

import json
from pathlib import Path
from .context import MAX_OUTPUT_CHARS, resolve_path, workspace_root
from .registry import tool

# 读文件时最多返回多少字符，避免一个超大文件塞满上下文
_MAX_FILE_CHARS = MAX_OUTPUT_CHARS
# 列文件时最多返回多少个文件
_MAX_FILES = 200
# search_code 的防护参数：限制正则长度和单行长度，降低灾难性回溯(ReDoS)风险
_MAX_PATTERN_CHARS = 500
# 单行长度上限。注意：这只能“降低”灾难性回溯的风险，并不能彻底消除；
# 彻底根治需要用 regex 库的 timeout 参数，或把匹配放进带超时的子进程。
_MAX_SEARCH_LINE_CHARS = 300
# list_files 要跳过的噪声目录（大仓库里这些目录会拖慢遍历、淹没真正的源码）
_SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", ".idea", "dist", "build", "target"}


def _read_with_lines(path: Path, start: int | None, end: int | None) -> str:
    """读取文件并给每行加上行号，可选地只读一个行区间。

    加上行号是为了让模型能准确引用“第几行”，评审时才能定位到具体位置。
    """
    # 读文件并按行拆分；errors="replace" 避免遇到非 UTF-8 编码时直接崩
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    # 规范化起止行号：start 最小为 1，end 不超过总行数
    if start is not None:
        start = max(1, start)
    if end is not None:
        end = min(len(lines), end)

    # 指定了行区间就只取区间内的行，并记住起始行号用于行号标注。
    # 注意要正确区分三种情况：
    #   只给 start -> 从 start 读到文件末尾（end 缺省为文件末尾）；
    #   只给 end   -> 从开头读到 end（start 缺省为 1）；
    #   两者都给   -> 取 [start, end] 闭区间。
    if start is not None or end is not None:
        lo = (start - 1) if start is not None else 0
        hi = end if end is not None else len(lines)
        lines = lines[lo:hi]
        lo_line = start or 1
    else:
        lo_line = 1

    # 给每行加行号（格式形如 "   12 | 内容"），按行累积长度。
    # 超预算时停在“行边界”，避免把某一行从中间切断；并告诉模型从哪里续读。
    kept: list[str] = []
    total = 0
    last_line_no = lo_line - 1  # 目前包含的最后一行号（初始为起始行之前）
    truncated = False
    for i, line in enumerate(lines):
        numbered_line = f"{lo_line + i:>6} | {line}"
        # 不是第一行、且再加这一行会超预算，就停在这里（保证至少保留一行）
        if kept and total + len(numbered_line) + 1 > _MAX_FILE_CHARS:
            truncated = True
            break
        kept.append(numbered_line)
        total += len(numbered_line) + 1
        last_line_no = lo_line + i
    text = "\n".join(kept)
    if truncated:
        text += (
            f"\n... (truncated at line {last_line_no}, "
            f"use start_line={last_line_no + 1} to continue)"
        )
    return text


@tool(
    "read_file",
    "Read a file (optionally a line range) from the workspace. Returns line-numbered content.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to the workspace root."},
            "start_line": {"type": "integer", "description": "1-based first line (optional)."},
            "end_line": {"type": "integer", "description": "1-based last line (optional)."},
        },
        "required": ["path"],
    },
)
def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    """读取一个文件（可选行区间），返回带行号的内容。"""
    target = resolve_path(path)  # 先做路径安全校验，防止越权读文件
    if not target.is_file():
        # 不是文件就返回 JSON 错误信息（而不是抛异常），让模型知道读失败了
        return json.dumps({"error": f"not a file: {path}"})
    return _read_with_lines(target, start_line, end_line)


@tool(
    "list_files",
    "List files under a directory (recursively), optionally filtered by a glob pattern.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path relative to workspace root."},
            "pattern": {"type": "string", "description": "Optional glob, e.g. '**/*.py'."},
        },
        "required": ["path"],
    },
)
def list_files(path: str, pattern: str | None = None) -> str:
    """递归列出目录下的文件，可选按 glob 模式过滤；跳过常见噪声目录。"""
    target = resolve_path(path)
    if not target.is_dir():
        return json.dumps({"error": f"not a directory: {path}"})
    # rglob 是生成器：边遍历边收集，收满 _MAX_FILES 个就提前停，避免大仓库全量物化
    collected: list[Path] = []
    for f in target.rglob(pattern or "*"):
        if not f.is_file():
            continue
        # 跳过 .git / node_modules / venv 等噪声目录，让模型看到真正的源码
        if any(part in _SKIP_DIRS for part in f.parts):
            continue
        collected.append(f)
        if len(collected) >= _MAX_FILES:
            break
    collected.sort(key=lambda p: str(p))  # 排序保证每次输出顺序一致
    rel = [str(f.relative_to(workspace_root())) for f in collected]
    truncated = len(collected) >= _MAX_FILES
    header = f"{len(collected)} files" + (
        " (reached limit, use pattern to narrow down)" if truncated else ""
    )
    return json.dumps({"count": len(collected), "files": rel, "note": header, "truncated": truncated})


@tool(
    "search_code",
    "Search file contents for a literal text or regex pattern.",
    {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Text or regex to search for."},
            "path": {"type": "string", "description": "Directory to search under (default root)."},
        },
        "required": ["pattern"],
    },
)
def search_code(pattern: str, path: str = ".") -> str:
    """在文件内容里搜索文字/正则，返回匹配到的文件和行。"""
    import re

    root = resolve_path(path)
    # 防护 1：限制正则长度，避免超长/恶意正则带来的开销
    if len(pattern) > _MAX_PATTERN_CHARS:
        return json.dumps({"error": f"pattern too long (max {_MAX_PATTERN_CHARS} chars)"})
    # 尝试把 pattern 当正则；如果正则非法，就退化成“按字面文本搜索”
    try:
        rx = re.compile(pattern)
    except re.error:
        rx = re.compile(re.escape(pattern))

    matches: list[dict[str, str]] = []
    for f in root.rglob("*"):
        if not f.is_file() or ".git" in f.parts:
            continue  # 跳过目录和 .git 内部文件
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            # 防护 2：跳过超长单行（如 minified 的 JS/资源），避免灾难性回溯
            if len(line) > _MAX_SEARCH_LINE_CHARS:
                continue
            try:
                hit = rx.search(line)
            except (RecursionError, MemoryError):
                hit = None  # 极端情况下放弃这一条，保证进程不卡死
            if hit:
                matches.append(
                    {"file": str(f.relative_to(workspace_root())), "line": i, "text": line.strip()}
                )
                # 结果过多就提前返回，防止上下文爆炸
                if len(matches) >= 200:
                    return json.dumps({"matches": matches, "truncated": True})
    return json.dumps({"matches": matches})
