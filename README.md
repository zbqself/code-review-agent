# Code Review Agent

一个基于 LLM 的智能代码评审 Agent。核心是一个手写的 **agent 循环**：
模型通过**工具调用**（读文件、搜索、git diff、静态检查）主动收集证据，
最后调用 `submit_review` 工具输出结构化评审报告。

## 架构与工作流

同一套 agent 核心引擎，支持三种入口：

- **CLI**：`code-review`（本地评审）/ `code-review-pr`（评审 GitHub PR）
- **Web**：FastAPI + 单页前端（`code-review-web`）
- **GitHub PR bot**：webhook 自动评审并回帖

```
          用户 / 网页 / GitHub PR 事件
                      │
                      ▼
          ┌─── Agent 循环（最多 N 轮）───┐
          │ 问模型 → 要调工具吗？          │
          │   ├─ 是 → 执行工具 → 结果回填 ─┼──► 回到循环
          │   └─ 否 → 提交结构化评审        │
          └──────────────┬───────────────┘
                         ▼
           ReviewReport（结构化评审结果）
                         │
                         ├─（可选）FixAgent → FixReport（修复方案）
                         ▼
        渲染成 Markdown / JSON / 网页卡片 / PR 评论
```

## 特性

- 手写 agent 循环 + OpenAI 兼容的工具调用协议
- 内置工具：`read_file`、`list_files`、`search_code`、`git_diff`、`run_static_check`
- 结构化输出（Pydantic `ReviewReport`），可导出 Markdown / JSON
- 支持 DeepSeek / Qwen / OpenAI 等任何 OpenAI 兼容接口
- **流式输出**：模型边生成边显示，不用干等
- **会话记忆**：评审结束后可以继续追问（`--chat`），模型记得刚才的上下文
- **RAG 知识库**：把团队代码规范做成可检索知识库，评审时检索并注入提示词（`--kb`）
- **多 Agent 协作**：评审者 + 修复者，`--fix` 一键生成「问题 + 修复方案」
- **Web 界面**：浏览器里输入路径、点按钮评审，结果以卡片展示（FastAPI + 单页前端）
- **GitHub PR bot**：自动评审 Pull Request，并把报告作为评论贴回 PR

## 安装

```bash
uv venv --python 3.12
uv pip install -e .
# 或 pip install -e .

# 需要 RAG 用阿里 embedding + Chroma 向量库时，额外装（不装也能用，会自动回退到 TF-IDF）：
pip install -e ".[rag]"

# 需要 Web 界面时，额外装：
pip install -e ".[web]"
```

## 配置

```bash
cp .env.example .env
# 编辑 .env 填入 LLM_API_KEY 等
# 用 RAG 的话再填 EMBEDDING_API_KEY（阿里 DashScope）
```

## 使用

```bash
# 评审一个 Git 仓库目录（agent 会自己读文件、跑 git diff 和静态检查）
code-review /path/to/repo

# 评审单个文件
code-review src/app.py

# 输出 JSON
code-review /path/to/repo --format json

# 评审后进入追问模式（会话记忆）
code-review /path/to/repo --chat

# 用 RAG 知识库评审（先准备一个放 .md/.txt 规范文档的目录）
code-review /path/to/repo --kb examples/kb

# 评审后自动生成修复方案（多 Agent：评审者 -> 修复者）
code-review /path/to/repo --fix

# 评审一个 GitHub PR，并把报告作为评论贴回（需在 .env 配置 GITHUB_TOKEN）
code-review-pr owner/repo 123 --fix
```

评审过程中会**实时流式**打印模型输出和工具调用过程（`--format json` 时会关闭流式，
保证输出是干净的 JSON）。

`examples/kb/` 里放了 Python / Java / 安全 / 数据库 / 通用评审清单等示例规范文档，
可以直接拿来试 RAG 效果。

RAG 检索（见 `src/code_review_agent/rag.py`）默认用**阿里 DashScope embedding
（text-embedding-v3）+ Chroma 向量库**：在 `.env` 里配 `EMBEDDING_API_KEY`，并
`pip install -e ".[rag]"` 装上 chromadb。向量持久化到项目根目录的 `chroma_db/`。
如果没装 chromadb、没配 embedding key 或调用失败，会自动回退到内置的纯 Python
TF-IDF 检索（无需额外依赖）。

## Web 界面

```bash
# 启动 Web 服务
code-review-web

# 或
uvicorn code_review_agent.web:app --host 127.0.0.1 --port 8000
```

然后浏览器打开 http://127.0.0.1:8000 。页面上输入目标路径，可勾选「生成修复方案」
和「使用 RAG 知识库」，点「开始评审」即可看到结构化结果卡片。

## GitHub PR bot

在 `.env` 里配置 `GITHUB_TOKEN`（GitHub Personal Access Token，需 `repo` 权限），然后：

```bash
# 评审 owner/repo 的第 123 个 PR，并把报告作为评论贴回
code-review-pr owner/repo 123

# 同时生成修复方案
code-review-pr owner/repo 123 --fix
```

也可以把 `/webhook/github` 配置成 GitHub 仓库的 webhook（事件选 pull_request），
这样每次有人开/更新 PR，服务就会在后台自动评审并回帖。本地调试需要把服务暴露到公网
（如 ngrok），再在 GitHub 仓库 Settings → Webhooks 里填公网地址。

安全：在 `.env` 里配 `GITHUB_WEBHOOK_SECRET`（任意随机字符串），并在 GitHub 配置
webhook 时填同样的 Secret。服务会校验每个请求的 `X-Hub-Signature-256` 签名，
防止别人伪造请求。

## 核心 API（供二次开发）

```python
from code_review_agent.agent import ReviewAgent
from code_review_agent.config import LLMConfig

agent = ReviewAgent(LLMConfig.from_env())

# 开始一次新评审（会重置对话历史）
report = agent.start("/path/to/repo")

# 在已有会话上继续追问（会话记忆）
followup = agent.ask("第三个问题具体在哪一行？")
```

## 目录结构

```
src/code_review_agent/
  agent.py     # agent 循环：工具调用调度与结构化结果解析
  llm.py       # LLM 客户端封装
  schemas.py   # 评审结果的数据模型
  rag.py       # RAG：embedding + Chroma 向量库（TF-IDF 回退）
  tools/       # 各工具实现与注册表
  cli.py       # 命令行入口
  web.py       # Web 界面（FastAPI + 单页前端）
  github_bot.py # GitHub PR bot
```
