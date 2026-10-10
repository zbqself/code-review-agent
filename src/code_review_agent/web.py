"""Web 界面：用 FastAPI 提供 REST 接口 + 一个单页前端（无构建步骤）。

运行方式：
    code-review-web
    或
    uvicorn code_review_agent.web:app --host 127.0.0.1 --port 8000

然后在浏览器打开 http://127.0.0.1:8000 即可。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .agent import FixAgent, ReviewAgent
from .config import LLMConfig, _load_dotenv
from .github_bot import review_pr
from .rag import build_kb, build_query, render_context

# 启动 Web 服务时就加载 .env，这样 webhook 校验能读到 GITHUB_WEBHOOK_SECRET 等配置
_load_dotenv()

app = FastAPI(title="Code Review Agent")
logger = logging.getLogger("code_review_agent.web")
# 默认 logger 没有 handler，INFO 会被丢弃；这里手动加一个，让日志真正打印到终端
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_handler)
logger.propagate = False


class ReviewRequest(BaseModel):
    """前端发来的评审请求。"""

    target: str  # 要评审的目录或文件路径
    fix: bool = False  # 是否同时生成修复方案
    kb_dir: str | None = None  # 可选 RAG 知识库目录


INDEX_HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>代码评审 Agent</title>
<style>
  :root { --bg:#f6f8fa; --card:#fff; --line:#e2e5e9; --text:#1f2328; --muted:#6b7280; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--text); }
  .wrap { max-width: 920px; margin: 0 auto; padding: 28px 20px 60px; }
  h1 { font-size: 24px; margin: 0 0 6px; }
  .sub { color: var(--muted); margin: 0 0 22px; }
  .panel { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px; margin-bottom:20px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
  label { display:block; font-size:13px; color:var(--muted); margin:10px 0 4px; }
  input[type=text] { width:100%; padding:9px 10px; border:1px solid var(--line); border-radius:8px; font-size:14px; }
  .row { display:flex; gap:16px; align-items:flex-end; flex-wrap:wrap; }
  .check { display:flex; align-items:center; gap:6px; font-size:14px; color:var(--text); margin:12px 0 0; }
  button { margin-top:16px; padding:10px 22px; background:#1668dc; color:#fff; border:none; border-radius:8px; font-size:15px; cursor:pointer; }
  button:disabled { background:#9bb7e6; cursor:not-allowed; }
  .spin { display:none; margin:16px 0; color:var(--muted); font-size:14px; }
  .err { background:#fdecea; color:#b02a1a; border:1px solid #f5c6c0; border-radius:8px; padding:12px; white-space:pre-wrap; }
  h2 { font-size:18px; margin:22px 0 10px; border-bottom:1px solid var(--line); padding-bottom:6px; }
  .issue, .fix { border:1px solid var(--line); border-left:4px solid #1668dc; border-radius:8px; padding:12px 14px; margin:10px 0; background:#fff; }
  .tag { display:inline-block; font-size:12px; font-weight:600; padding:2px 8px; border-radius:10px; margin-right:6px; }
  .critical { background:#fdecea; color:#b02a1a; } .error { background:#fff0e6; color:#c05621; }
  .warning { background:#fff8e1; color:#9a6700; } .info { background:#e8f0fe; color:#1a56c4; } .suggestion { background:#eef1f4; color:#57606a; }
  .loc { font-family:Consolas,monospace; font-size:12px; color:var(--muted); }
  pre { background:#0d1117; color:#e6edf3; padding:12px; border-radius:8px; overflow:auto; font-size:12.5px; line-height:1.5; }
  code { font-family:Consolas,monospace; }
  .strengths { margin:6px 0 0; padding-left:20px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>智能代码评审 Agent</h1>
  <p class="sub">输入仓库或文件路径，agent 会自动读代码、检索规范、给出评审与修复方案。</p>

  <div class="panel">
    <label>目标路径（目录或文件）</label>
    <input type="text" id="target" placeholder="例如 D:\Java-project\hm-dianping\src\main\java">
    <div class="row">
      <label class="check"><input type="checkbox" id="fix"> 同时生成修复方案（多 Agent）</label>
      <label class="check"><input type="checkbox" id="kb"> 使用 RAG 知识库</label>
    </div>
    <label id="kbLabel" style="display:none;">知识库目录（.md/.txt 规范文档）</label>
    <input type="text" id="kbDir" style="display:none;" placeholder="例如 examples\kb">
    <br>
    <button id="runBtn" onclick="runReview()">开始评审</button>
  </div>

  <div class="spin" id="spinner">评审中，请稍候（大项目可能需要一两分钟）……</div>
  <div id="result"></div>
</div>

<script>
function escapeHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function tag(name) {
  const m = { critical:'critical', error:'error', warning:'warning', info:'info', suggestion:'suggestion' };
  const c = m[name] || 'info';
  return '<span class="tag ' + c + '">' + escapeHtml(name) + '</span>';
}
function loc(file, s, e) {
  if (!file && !s) return '';
  let t = escapeHtml(file || '?');
  if (s) { t += ':' + s; if (e && e !== s) t += '-' + e; }
  return '<div class="loc">' + t + '</div>';
}
function codeBlock(c) {
  if (!c) return '';
  return '<pre><code>' + escapeHtml(c) + '</code></pre>';
}

function renderReview(r) {
  let html = '<h2>代码评审报告</h2>';
  html += '<p><b>总结</b>：' + escapeHtml(r.summary) + '</p>';
  if (r.strengths && r.strengths.length) {
    html += '<b>亮点</b><ul class="strengths">';
    r.strengths.forEach(s => html += '<li>' + escapeHtml(s) + '</li>');
    html += '</ul>';
  }
  html += '<b>问题</b>';
  if (!r.issues || !r.issues.length) html += '<p>未发现明显问题</p>';
  (r.issues || []).forEach((it, i) => {
    html += '<div class="issue"><div>' + tag(it.severity) + '<b>' + escapeHtml(it.title) + '</b></div>';
    html += loc(it.file, it.start_line, it.end_line);
    html += '<div style="margin-top:6px;color:#57606a;">类别：' + escapeHtml(it.category) + '</div>';
    html += '<div style="margin-top:6px;">' + escapeHtml(it.description) + '</div>';
    html += '<div style="margin-top:6px;"><b>建议</b>：' + escapeHtml(it.suggestion) + '</div>';
    html += codeBlock(it.code_snippet);
    html += '</div>';
  });
  if (r.overall_assessment) html += '<b>总体评价</b><p>' + escapeHtml(r.overall_assessment) + '</p>';
  return html;
}

function renderFix(f) {
  let html = '<h2>修复方案</h2>';
  html += '<p><b>总结</b>：' + escapeHtml(f.summary) + '</p>';
  if (!f.fixes || !f.fixes.length) html += '<p>未生成修复方案</p>';
  (f.fixes || []).forEach(fx => {
    html += '<div class="fix"><b>修复问题 #' + escapeHtml(fx.issue_index) + '</b>';
    html += loc(fx.file, fx.start_line, fx.end_line);
    html += '<div style="margin-top:6px;">' + escapeHtml(fx.explanation) + '</div>';
    if (fx.code_before) { html += '<div style="margin-top:8px;color:#57606a;">修改前：</div>' + codeBlock(fx.code_before); }
    if (fx.code_after) { html += '<div style="margin-top:8px;color:#57606a;">修改后：</div>' + codeBlock(fx.code_after); }
    html += '</div>';
  });
  if (f.note) html += '<b>说明</b><p>' + escapeHtml(f.note) + '</p>';
  return html;
}

async function runReview() {
  const target = document.getElementById('target').value.trim();
  const fix = document.getElementById('fix').checked;
  const kb = document.getElementById('kb').checked;
  const kbDir = document.getElementById('kbDir').value.trim();
  if (!target) { alert('请输入目标路径'); return; }
  const body = { target: target, fix: fix, kb_dir: (kb && kbDir) ? kbDir : null };
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('result').innerHTML = '';
  try {
    const resp = await fetch('/api/review', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || '请求失败');
    let html = renderReview(data.review || {});
    if (data.fix) html += renderFix(data.fix);
    document.getElementById('result').innerHTML = html;
  } catch (e) {
    document.getElementById('result').innerHTML = '<div class="err">错误：' + escapeHtml(e.message) + '</div>';
  } finally {
    btn.disabled = false;
    document.getElementById('spinner').style.display = 'none';
  }
}

document.getElementById('kb').addEventListener('change', e => {
  document.getElementById('kbDir').style.display = e.target.checked ? 'block' : 'none';
  document.getElementById('kbLabel').style.display = e.target.checked ? 'block' : 'none';
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """返回前端页面。"""
    return HTMLResponse(INDEX_HTML)


@app.post("/api/review")
def review(req: ReviewRequest) -> dict:
    """执行一次评审（可选修复 / RAG），返回结构化结果。"""
    try:
        cfg = LLMConfig.from_env()

        # 可选 RAG：构建知识库并检索相关规范
        knowledge_context = None
        if req.kb_dir:
            kb = build_kb(req.kb_dir)
            chunks = kb.search(build_query(req.target), k=5)
            if chunks:
                knowledge_context = render_context(chunks)

        # 评审者
        agent = ReviewAgent(cfg)
        report = agent.start(req.target, knowledge_context=knowledge_context)

        # 修复者（多 Agent）
        fix_report = None
        if req.fix:
            fixer = FixAgent(cfg)
            fix_report = fixer.fix(req.target, report)

        return {
            "review": report.model_dump(),
            "fix": fix_report.model_dump() if fix_report else None,
        }
    except Exception as exc:
        # 把错误以可读形式返回给前端
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/webhook/github")
async def github_webhook(request: Request, background: BackgroundTasks) -> dict:
    """接收 GitHub webhook：校验签名后，PR 开/更新时在后台自动评审并回帖。"""
    event = request.headers.get("x-github-event", "")
    logger.info(f"收到 webhook 事件: {event}")
    if event != "pull_request":
        return {"ok": True, "ignored": event}

    # 1) 读原始请求体（HMAC 必须用原始 bytes 计算，不能用解析后的 JSON）
    raw_body = await request.body()

    # 2) 校验签名，防止别人伪造请求乱发评论
    _verify_github_signature(request.headers, raw_body)

    # 3) 解析 JSON，处理 pull_request 事件
    payload = json.loads(raw_body)
    action = payload.get("action")
    repo = payload["repository"]["full_name"]
    pr_number = payload["pull_request"]["number"]
    logger.info(f"pull_request action={action} repo={repo} pr=#{pr_number}")
    if action not in ("opened", "reopened", "synchronize"):
        return {"ok": True, "ignored": action}
    # 后台执行评审（先返回 200，避免阻塞 webhook）
    background.add_task(_safe_review_pr, repo, pr_number)
    return {"ok": True, "queued": f"{repo}#{pr_number}"}


def _safe_review_pr(repo: str, pr_number: int) -> None:
    """后台评审的包装：捕获并打印异常，避免失败静默。"""
    try:
        logger.info(f"开始评审 {repo}#{pr_number} ...")
        review_pr(repo, pr_number)
        logger.info(f"{repo}#{pr_number} 评审完成，已回帖")
    except Exception as exc:
        logger.exception(f"{repo}#{pr_number} 评审失败: {exc}")


def _verify_github_signature(headers, raw_body: bytes) -> None:
    """校验 GitHub webhook 的 HMAC-SHA256 签名。

    GitHub 会在请求头 X-Hub-Signature-256 里带上 sha256=HMAC(secret, body)。
    这里用自己的 secret 对原始 body 再算一遍，并用常量时间比较（hmac.compare_digest）
    比对，防止时序攻击。
    """
    secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
    if not secret:
        # 没配 secret 时跳过校验（仅限本地测试；对外暴露前务必配置）
        return
    signature = headers.get("x-hub-signature-256", "")
    expected = "sha256=" + hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=403, detail="无效的 webhook 签名")


def main() -> None:
    """命令行入口：启动 uvicorn。"""
    import uvicorn

    uvicorn.run("code_review_agent.web:app", host="127.0.0.1", port=8000)
