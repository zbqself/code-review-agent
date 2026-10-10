"""工具注册表：负责“注册工具”和“执行工具”。

设计思路：
    每个工具函数用一个 @tool 装饰器声明它的名称、描述和参数 JSON Schema，
    装饰器自动把它登记到全局 REGISTRY 字典里。之后：
      - tool_schemas() 把注册表转成 OpenAI function calling 需要的 JSON Schema 列表；
      - run_tool() 根据模型给的工具名 + 参数，真正调用对应的 Python 函数。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    """一个工具的描述：名字、描述、参数 Schema、以及真正干活的函数。"""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]


# 全局注册表：工具名 -> Tool 对象
REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, parameters: dict[str, Any]) -> Callable[..., Any]:
    """装饰器：把下面的函数注册成一个工具。

    用法示例：
        @tool("read_file", "读文件", {...参数Schema...})
        def read_file(path: str): ...

    参数：
        name/description/parameters：会原样作为工具的元信息发给大模型。
    """

    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        REGISTRY[name] = Tool(name=name, description=description, parameters=parameters, func=func)
        return func  # 原样返回函数本身，不影响它被正常调用

    return decorator


def tool_schemas() -> list[dict[str, Any]]:
    """把注册表里的所有工具转成 OpenAI function calling 的 JSON Schema 列表。"""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in REGISTRY.values()
    ]


def run_tool(name: str, arguments: dict[str, Any]) -> str:
    """根据工具名和参数，真正执行一个工具，并把结果序列化成字符串返回。

    设计约定：工具出错了不要抛异常把整个程序弄崩，而是把错误信息以 JSON 形式
    返回给模型，让模型知道“这个工具执行失败了”，模型可以据此调整策略。
    """
    if name not in REGISTRY:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        return REGISTRY[name].func(**arguments)
    except Exception as exc:
        # 把异常转成 JSON 返回给模型，而不是向上抛出
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
