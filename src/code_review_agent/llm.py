"""LLM 客户端封装：把调用大模型的细节包起来，别的地方只用一个 .chat() 就够了。

为什么需要这层封装：
    - 统一入口：整个项目调用模型只走 LLMClient.chat()，以后换模型/换库只改这里。
    - 屏蔽细节：OpenAI 官方 SDK 既能调 OpenAI，也能调任何“兼容 OpenAI 接口”的服务
      （DeepSeek、Qwen 等都支持），我们只要传入不同的 base_url 和 model 即可。
"""

from __future__ import annotations

from typing import Any, Callable

from openai import OpenAI

from .config import LLMConfig


class LLMClient:
    """对大模型 API 的一个薄封装。"""

    def __init__(self, config: LLMConfig) -> None:
        # 保存配置，方便后面 chat() 时取 model、temperature 等
        self.config = config
        # 创建 OpenAI 客户端。
        # base_url 为 None 时连接 OpenAI 官方；否则连接指定的兼容服务（如 DeepSeek）。
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> Any:
        """调用大模型进行一轮对话（可以带工具）。

        参数：
            messages:    消息列表，包含 system/user/assistant/tool 等角色的消息。
            tools:       可选，工具列表（OpenAI function calling 的 JSON Schema 列表）。
            tool_choice: 工具选择策略，默认 "auto"（模型自己决定要不要调工具）。

        返回：
            模型返回的原始响应对象（实际使用时取 resp.choices[0].message）。
        """
        # 构造请求参数。temperature 控制回答的随机性。
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        # 只有真正传了工具时，才带上 tools / tool_choice 参数
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        return self.client.chat.completions.create(**kwargs)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        on_text: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """流式调用大模型：模型边生成、我们边拿到文本碎片，最后返回完整消息。

        和 chat() 的区别：
            chat() 要等模型把整段回复一次性生成完才返回；
            chat_stream() 是“挤牙膏”式地一段一段返回，配合 on_text 回调就能实时打印，
            用户不用干等。

        参数：
            messages:    消息列表（同 chat()）。
            tools:       可选工具列表。
            tool_choice: 工具选择策略，默认 "auto"。
            on_text:     可选回调；模型每吐出一段文本就调用一次（用于实时显示）。

        返回：
            一个与 chat() 的 message 结构一致的字典：
            {"content": str | None,
             "tool_calls": [{"id": ..., "name": ..., "arguments": ...}, ...] | None}
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": True,  # 关键开关：开启流式返回
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        stream = self.client.chat.completions.create(**kwargs)

        # 累积流式返回的碎片
        content_parts: list[str] = []
        # 工具调用是“按 index 分片”返回的（同一个工具的 arguments 可能分成多段），
        # 这里用字典按 index 归并，最后拼成一个完整的工具调用。
        tool_calls: dict[int, dict[str, str]] = {}

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 1) 文本碎片：累加，并实时回调给 on_text
            if delta.content:
                content_parts.append(delta.content)
                if on_text:
                    on_text(delta.content)

            # 2) 工具调用碎片：按 index 归并 id / name / arguments
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index if tc.index is not None else 0
                    acc = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        acc["id"] = tc.id
                    if tc.function and tc.function.name:
                        acc["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        acc["arguments"] += tc.function.arguments

        # 按 index 顺序组装工具调用列表，保证和模型意图一致
        ordered_calls = [tool_calls[i] for i in sorted(tool_calls)]
        return {
            "content": "".join(content_parts) or None,
            "tool_calls": ordered_calls or None,
        }
