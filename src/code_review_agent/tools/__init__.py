"""tools 包：存放 agent 能用的各种“工具”（函数），并对外暴露注册表和常用函数。

关键点：
    导入 filesystem / git / static 这三个模块时，它们内部的 @tool 装饰器会
    自动把工具注册进 REGISTRY。所以这里“只 import 不直接用”，目的就是触发注册。
    然后再对外导出 set_workspace_root / workspace_root / run_tool / tool_schemas 等，
    方便 agent.py 使用。
"""

from __future__ import annotations

# 导入这三个模块，目的就是让它们的 @tool 装饰器把工具注册进 REGISTRY
from . import filesystem, git, static  # noqa: F401  (imports register tools)
from .context import set_workspace_root, workspace_root  # noqa: F401
from .registry import REGISTRY, run_tool, tool, tool_schemas  # noqa: F401
