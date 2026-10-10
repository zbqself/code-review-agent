"""code_review_agent 包的入口文件。

这个文件的作用是让 `src/code_review_agent/` 这个目录被 Python 识别为一个“包(package)”。
只要目录里有这个 `__init__.py`，项目里就能用
`from code_review_agent import ...` 来导入里面的模块。

`__version__` 记录当前项目的版本号，方便以后发布或排查问题时知道用的是哪个版本。
"""

__version__ = "0.1.0"
