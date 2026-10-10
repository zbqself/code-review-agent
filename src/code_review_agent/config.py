"""配置模块：负责读取环境变量和 .env 文件，组装成 LLMConfig 配置对象。

为什么需要这个文件：
    项目要用哪个大模型、key 是多少、温度多少、最多跑几轮……这些“配置项”如果写死在代码里，
    换一个模型或换一台机器就得改代码。把它们集中在这里、通过环境变量读取，改配置就不用动代码了。

两个核心内容：
    1. _load_dotenv()：一个极简的 .env 文件加载器（不依赖第三方库）。
    2. LLMConfig：一个数据类，把零散的环境变量打包成一个清晰的对象。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _strip_inline_comment(value: str) -> str:
    """去掉引号之外的“行内注释”（# 前面有空白），保留引号里的 #。"""
    in_quote: str | None = None
    for i, ch in enumerate(value):
        if ch in ('"', "'"):
            if in_quote == ch:
                in_quote = None  # 遇到配对的引号，退出引号状态
            elif in_quote is None:
                in_quote = ch  # 进入引号状态
        elif ch == "#" and in_quote is None and (i == 0 or value[i - 1].isspace()):
            return value[:i].rstrip()  # 注释从 # 开始，截断并去掉尾部空格
    return value


def _load_dotenv(path: Path | None = None) -> None:
    """加载 .env 文件里的配置到环境变量。

    注意：这里用 setdefault，意思是“如果环境变量已经存在，就不覆盖”。
    所以优先级是：真正设置过的环境变量 > .env 文件里的值。

    支持常见的 .env 写法：export 前缀、行内 # 注释、单/双引号包裹的值。

    参数：
        path: .env 文件的路径；不传时默认找当前工作目录下的 .env。
    """
    dotenv = path or Path.cwd() / ".env"
    if not dotenv.exists():
        return  # 没有 .env 文件就直接返回，不报错
    for raw in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        # 跳过空行、注释行（# 开头）
        if not line or line.startswith("#"):
            continue
        # 处理 "export KEY=VALUE" 写法
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        # 用 partition 只按第一个 “=” 拆成 key 和 value
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # 去掉行内注释（引号外的 #）
        value = _strip_inline_comment(value)
        # 去掉包裹值的一对引号（如 KEY="a b" 或 KEY='a#b'）
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _env_float(name: str, default: float) -> float:
    """读取一个“浮点数”环境变量；值非法时抛出带变量名的清晰错误。

    这样用户在把 LLM_TEMPERATURE 写成 0,2 或 abc 时，能立刻看到是哪个变量、错在哪，
    而不是一句毫无上下文的 ValueError。
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(f"环境变量 {name} 必须是数字，当前值：{raw!r}") from None


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """读取一个“整数”环境变量，并校验下限；值非法时抛出清晰错误。"""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            raise RuntimeError(f"环境变量 {name} 必须是整数，当前值：{raw!r}") from None
    # max_iterations 等不允许小于 minimum，否则 agent 循环可能一轮都不跑
    if value < minimum:
        raise RuntimeError(f"环境变量 {name} 不能小于 {minimum}，当前值：{value}")
    return value


@dataclass
class LLMConfig:
    """一个大模型相关的配置对象。

    @dataclass 会自动帮我们生成 __init__ 等方法，省去手写样板代码。
    每个字段的含义见下方 from_env() 里的读取逻辑。
    """

    api_key: str            # 大模型的密钥（必填，否则没法调用）
    base_url: str | None    # 接口地址；None 表示用默认地址（如 OpenAI 官方）
    model: str              # 模型名，如 deepseek-chat / qwen-plus / gpt-4o
    temperature: float      # 采样温度：越大回答越“随机/发散”，越小越“稳定”
    max_iterations: int     # agent 循环最多跑几轮，防止死循环
    review_language: str    # 评审报告用什么语言写，默认中文 zh

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量（含 .env）读取配置，返回一个 LLMConfig 对象。"""
        _load_dotenv()  # 先加载 .env，保证 .env 里的变量也能被读到

        # api_key 兼容两种变量名：OPENAI_API_KEY 或 LLM_API_KEY（和下面 base_url 的双变量写法一致）
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            # key 缺失就立刻报错并给出清晰提示，而不是等到调用时才莫名失败
            raise RuntimeError(
                "缺少 API key：请在环境变量或 .env 中设置 OPENAI_API_KEY（或 LLM_API_KEY）"
            )

        return cls(
            api_key=api_key,
            base_url=os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
            model=os.environ.get("LLM_MODEL", "deepseek-chat"),
            temperature=_env_float("LLM_TEMPERATURE", 0.2),
            max_iterations=_env_int("LLM_MAX_ITERATIONS", 30, minimum=1),
            review_language=os.environ.get("LLM_REVIEW_LANGUAGE", "zh"),
        )
