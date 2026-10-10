"""让这个包可以像脚本一样直接运行。

用法：
    python -m code_review_agent

原理：
    执行 `python -m code_review_agent` 时，Python 会找到这个 `__main__.py` 并运行它。
    这里只是简单地把命令行入口 `app()` 启动起来，真正的逻辑都在 cli.py 里。
"""

from .cli import app

# 启动命令行程序（入口逻辑见 cli.py）
app()
