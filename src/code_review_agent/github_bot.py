"""GitHub PR bot：自动评审 Pull Request，并把评审报告作为评论贴回 PR。

流程：
    1) 用 GITHUB_TOKEN 鉴权，调用 GitHub REST API 拉取指定 PR 的 diff；
    2) 把 diff 交给评审者(ReviewAgent)评审，可选再让修复者(FixAgent)生成修复方案；
    3) 把评审报告（Markdown）作为评论发到该 PR 上。

用法：
    code-review-pr owner/repo 123 --fix
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import httpx

from .agent import FixAgent, ReviewAgent
from .config import LLMConfig
from .render import render_fix_markdown, render_markdown

_GITHUB_API = "https://api.github.com"

# 有些 Windows / 公司网络环境无法验证 GitHub 的 SSL 证书
#（报错：unable to get local issuer certificate）。
# 这里先设为 False 跳过证书校验，方便本地测试；更稳妥的做法是：
#   pip install --upgrade certifi
# 修复 CA 证书包后，再把这行改回 True。
_VERIFY_SSL = False


class GitHubClient:
    """封装 GitHub REST API 的最小客户端（拉取 PR diff + 发评论）。"""

    def __init__(self, token: str) -> None:
        # Bearer token 是 GitHub 推荐的鉴权方式；Accept/X-GitHub-Api-Version 保证返回结构稳定
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _get(self, path: str):
        resp = httpx.get(_GITHUB_API + path, headers=self.headers, timeout=30, verify=_VERIFY_SSL)
        resp.raise_for_status()  # 非 2xx 直接抛异常
        return resp.json()

    def get_pr_diff(self, repo: str, pr_number: int) -> str:
        """拉取一个 PR 的所有改动文件及其 diff，拼成一段文本。"""
        files = self._get(f"/repos/{repo}/pulls/{pr_number}/files")
        parts: list[str] = []
        for f in files:
            patch = f.get("patch") or "（二进制或无文本 diff）"
            parts.append(f"文件: {f['filename']}（状态: {f['status']}）\n{patch}")
        return "\n\n".join(parts)

    def post_comment(self, repo: str, pr_number: int, body: str) -> dict:
        """在 PR 下发一条评论（PR 的评论走 /issues/{number}/comments 接口）。"""
        resp = httpx.post(
            _GITHUB_API + f"/repos/{repo}/issues/{pr_number}/comments",
            headers=self.headers,
            json={"body": body},
            timeout=30,
            verify=_VERIFY_SSL,
        )
        resp.raise_for_status()
        return resp.json()


def review_pr(repo: str, pr_number: int, fix: bool = False) -> str:
    """拉取 PR diff → 评审（可选修复）→ 把报告作为评论贴回 PR，返回评论内容。"""
    # 先加载 .env（否则读不到 .env 里的 GITHUB_TOKEN）
    cfg = LLMConfig.from_env()

    # 直接从 .env 文件读 GITHUB_TOKEN，优先级高于终端里可能残留的旧环境变量
    token = _read_token_from_env_file() or os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_API_TOKEN", "")
    if not token:
        raise RuntimeError(
            "缺少 GITHUB_TOKEN。请确认 .env 里有 GITHUB_TOKEN，"
            "且终端没有同名旧环境变量覆盖（可运行 echo $env:GITHUB_TOKEN 检查）"
        )

    client = GitHubClient(token)
    diff = client.get_pr_diff(repo, pr_number)

    # PR bot 主要评审“改动”，所以把 diff 作为输入；同时写进临时目录，
    # 这样 agent 的工具（read_file 等）也能读到一个真实存在的工作区。
    tmp = tempfile.mkdtemp(prefix="pr-review-")
    Path(tmp, "pr.diff").write_text(diff, encoding="utf-8")

    report = ReviewAgent(cfg).start(tmp, diff_context=diff)
    comment = render_markdown(report)

    if fix:
        fix_report = FixAgent(cfg).fix(tmp, report)
        comment += "\n\n---\n\n" + render_fix_markdown(fix_report)

    client.post_comment(repo, pr_number, comment)
    return comment


def _read_token_from_env_file() -> str | None:
    """从项目 .env 文件里直接读取 GITHUB_TOKEN（优先于 os.environ）。"""
    dotenv = Path.cwd() / ".env"
    if not dotenv.exists():
        return None
    for raw in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() in ("GITHUB_TOKEN", "GITHUB_API_TOKEN"):
            return value.strip().strip('"').strip("'")
    return None
