"""工作区上下文模块：记录“工作区根目录”，并做路径安全校验。

为什么要“工作区根目录”：
    大模型是不可信的，它可能让我们读任意文件（比如 C:\\Windows\\... 或你电脑里的隐私文件）。
    我们规定一个“根目录”，所有工具读文件都必须限制在这个目录之内，防止越权访问。

    resolve_path() 里那句 relative_to(root) 就是关键防线：如果路径在根目录之外，
    它会抛异常，从而阻止越界读取。
"""

from __future__ import annotations

from pathlib import Path

# 模块级变量：保存当前工作区根目录（还没设置时为 None）
_workspace_root: Path | None = None

# 统一的输出长度上限，供 filesystem / git 等工具复用，避免各处各写一个魔法数字
MAX_OUTPUT_CHARS = 16_000


def set_workspace_root(path: str | Path) -> None:
    """设置工作区根目录（一次评审开始时调用一次）。"""
    global _workspace_root
    _workspace_root = Path(path).resolve()  # resolve() 转成绝对路径，避免相对路径歧义


def workspace_root() -> Path:
    """返回当前工作区根目录；如果没设置过，就退回用当前进程的工作目录。"""
    return _workspace_root or Path.cwd().resolve()


def resolve_path(path: str | Path) -> Path:
    """把一个工具给的“相对路径”解析成工作区内的绝对路径，并确保它不越界。

    参数：
        path: 工具传来的路径（通常是相对工作区根的路径）。
    返回：
        解析后的绝对路径（保证在工作区根目录之内）。
    异常：
        如果路径越出工作区根目录，会抛出带清晰语义的 ValueError。
    """
    root = workspace_root()
    candidate = (root / path).resolve()
    # 关键安全校验：candidate 必须仍在 root 之内；越界就抛出语义清晰的错误，
    # 而不是 pathlib 内部那句难懂的 “is not in the subpath of...”。
    if not candidate.is_relative_to(root):
        raise ValueError(f"路径越出工作区根目录: {path}")
    return candidate
