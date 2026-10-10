"""RAG（检索增强生成）模块：把团队代码规范 / 历史结论做成可检索的知识库。

什么是 RAG，为什么要它：
    大模型的知识是“冻结”在训练时刻的，它不知道你团队的特定规范（比如“我们禁止裸 except”、
    “接口必须统一返回 Result 包装”）。RAG 的思路是：
        1) 先把你的规范文档切成小块（chunk），建立索引；
        2) 评审时先“检索”出和当前项目最相关的几块；
        3) 把这些块塞进提示词，让模型“参考”着这些内容来评审。

本模块用**纯 Python 的 TF-IDF + 余弦相似度**实现检索：
    - 不依赖任何外部向量库、也不调用 embedding 模型，开箱即用；
    - 适合教学（能讲清楚“切块、分词、IDF 加权、余弦相似度”每一步）；
    - 后续想升级，可以把这里的检索替换成真正的 embedding + 向量库（Chroma/FAISS 等）。
"""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    """知识库里的一个文本片段。"""

    text: str
    source: str  # 来源文件路径
    index: int   # 在来源文件里的块序号


def _tokenize(text: str) -> list[str]:
    """把文本切成 token：英文单词 + 中文双字（bigram）。

    中文按“双字”切，是因为单个汉字的语义太弱、双字能保留更多上下文，
    而且实现简单、不需要引入分词库。
    """
    text = text.lower()
    # [a-z0-9_]+ 匹配英文单词；[\u4e00-\u9fff]{2} 匹配两个连续中文汉字（bigram）
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2}", text)


def chunk_text(text: str, source: str, max_chars: int = 600, overlap: int = 80) -> list[Chunk]:
    """把一篇文档按长度切成若干有重叠的小块。

    为什么要切块：
        1) 检索通常在“块”粒度上做，块越小越精准；
        2) 一次塞太多内容会浪费上下文、也稀释相关性。
    重叠(overlap)让相邻块之间有上下文衔接，避免把一句话从中间硬生生切断。
    """
    text = text.strip()
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, index=idx))
            idx += 1
        if end >= len(text):
            break
        start = end - overlap
    return chunks


class TFIDFKnowledgeBase:
    """基于 TF-IDF + 余弦相似度的知识库（兜底方案，不需要 embedding 或向量库）。"""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._vocab: list[str] = []
        self._idf: dict[str, float] = {}
        self._vectors: list[list[float]] = []

    def ingest_dir(self, directory: str | Path) -> int:
        """读取目录下所有 .md / .txt 文件并切块，返回块的数量。"""
        root = Path(directory)
        count = 0
        for f in sorted(root.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in (".md", ".txt"):
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
            for chunk in chunk_text(text, source=str(f)):
                self.chunks.append(chunk)
                count += 1
        return count

    def build_index(self) -> None:
        """基于当前所有块构建 TF-IDF 索引。

        步骤：
            1) 统计每个 token 出现在多少个块里（文档频率 DF）；
            2) 计算 IDF（逆文档频率）：出现得越少的词，权重越高（更有区分度）；
            3) 给每个块算一个 TF-IDF 向量。
        """
        n = len(self.chunks)
        df: dict[str, int] = {}
        tokenized_chunks: list[list[str]] = []
        for c in self.chunks:
            tokens = _tokenize(c.text)
            tokenized_chunks.append(tokens)
            # DF 只关心“这个词有没有出现在这个块里”，所以用 set 去重
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        self._vocab = sorted(df.keys())
        # 平滑版 IDF：ln((N+1)/(DF+1)) + 1，避免除零、也避免极端值
        self._idf = {t: math.log((n + 1) / (df[t] + 1)) + 1 for t in self._vocab}

        self._vectors = []
        for tokens in tokenized_chunks:
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._vectors.append([tf.get(t, 0) * self._idf[t] for t in self._vocab])

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """检索与 query 最相关的 k 个块（按余弦相似度降序）。"""
        if not self.chunks:
            return []
        qv = self._query_vector(query)
        scores = [self._cosine(qv, v) for v in self._vectors]
        top_idx = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in top_idx]

    def _query_vector(self, query: str) -> list[float]:
        """把查询文本也转成同一套 TF-IDF 向量，才能和块向量做相似度。"""
        tf: dict[str, int] = {}
        for t in _tokenize(query):
            tf[t] = tf.get(t, 0) + 1
        return [tf.get(t, 0) * self._idf.get(t, 0.0) for t in self._vocab]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """两个向量的余弦相似度（-1 到 1，越大越相似）。"""
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


def build_query(target: str) -> str:
    """根据评审目标构造检索 query：用路径名 + 顶层文件/目录名。

    这样如果评审一个 Java 项目（顶层有 pom.xml、.java），就能更大概率匹配到
    知识库里讲 Java 的规范；评审 Python 项目则匹配 Python 规范。
    """
    parts = [os.path.basename(os.path.abspath(target))]
    if os.path.isdir(target):
        try:
            for name in os.listdir(target):
                parts.append(name)
                if len(parts) >= 20:
                    break
        except OSError:
            pass
    return " ".join(parts)


def render_context(chunks: list[Chunk]) -> str:
    """把检索到的块拼成一段上下文文本，用于塞进提示词。"""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] 来源: {c.source}\n{c.text}")
    return "\n\n".join(parts)


def default_persist_dir() -> str:
    """返回向量库默认持久化目录（基于项目根目录的绝对路径）。

    用绝对路径而不是相对路径 "./chroma_db"，是为了无论从哪个目录启动 `code-review`，
    都读写同一个向量库目录，避免“换个工作目录就找不到向量数据”的坑。
    rag.py 位于 <项目根>/src/code_review_agent/rag.py，往上三级就是项目根。
    """
    return str(Path(__file__).resolve().parent.parent.parent / "chroma_db")


# ========================================================================
# 下面是“真正的 embedding + 向量库”路径（首选）：
#   - embedding 用阿里 DashScope 的 text-embedding-v3（OpenAI 兼容接口）；
#   - 向量库用 Chroma（本地持久化向量数据库，按余弦相似度检索）。
# 如果没配 embedding key、没装 chromadb 或调用失败，build_kb() 会自动回退到上面的 TFIDFKnowledgeBase。
# ========================================================================


class EmbeddingClient:
    """用 OpenAI 兼容接口调用 embedding 模型（默认阿里 DashScope text-embedding-v3）。

    阿里 DashScope 提供了 OpenAI 兼容的 /embeddings 接口，所以直接用 openai SDK 即可，
    只需把 base_url 指向 https://dashscope.aliyuncs.com/compatible-mode/v1。
    """

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把一批文本转成向量，返回与输入顺序一致的向量列表。"""
        resp = self.client.embeddings.create(model=self.model, input=texts)
        # 按返回里的 index 排序，确保和输入顺序一致
        data = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in data]


class VectorKnowledgeBase:
    """基于 embedding + Chroma 向量库的知识库（真正的本地向量数据库）。

    工作流程：
        1) 读取规范文档并切块；
        2) 用 EmbeddingClient 把所有块转成向量；
        3) 存进 Chroma（本地持久化向量数据库，按余弦相似度检索）；
        4) 查询时把 query 也转成向量，用 Chroma 找最相似的块。

    持久化：向量数据存在 <项目根>/chroma_db/ 目录（绝对路径，不随工作目录变化），
    里面有 chroma.sqlite3 等文件，和常见 Chroma 向量库目录结构一致。
    """

    _BATCH = 10  # DashScope text-embedding-v3/v4 单次请求最多 10 条

    def __init__(self, embedder: EmbeddingClient, persist_dir: str | None = None) -> None:
        self.embedder = embedder
        self.persist_dir = persist_dir or default_persist_dir()
        self._collection = None

    def ingest_dir(self, directory: str | Path) -> int:
        """读取文档、切块、embedding、写入 Chroma，返回块数量。"""
        import chromadb

        chunks: list[Chunk] = []
        root = Path(directory)
        for f in sorted(root.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in (".md", ".txt"):
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
            chunks.extend(chunk_text(text, source=str(f)))

        if not chunks:
            return 0

        client = chromadb.PersistentClient(path=self.persist_dir)
        # cosine 距离度量；用 upsert 保证重复运行不会残留旧数据
        self._collection = client.get_or_create_collection(
            name="guidelines", metadata={"hnsw:space": "cosine"}
        )

        texts = [c.text for c in chunks]
        vectors = self._embed_batched(texts)
        ids = [str(i) for i in range(len(chunks))]
        metadatas = [{"source": c.source, "index": c.index} for c in chunks]

        self._collection.upsert(
            ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas
        )
        return len(chunks)

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """把 query 转成向量，用 Chroma 检索最相似的 k 个块。"""
        if self._collection is None:
            return []
        qv = self.embedder.embed([query])[0]
        res = self._collection.query(
            query_embeddings=[qv], n_results=k, include=["documents", "metadatas"]
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        return [
            Chunk(text=d, source=m["source"], index=m.get("index", 0))
            for d, m in zip(docs, metas)
        ]

    def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        """分批次调用 embedding，避免一次传太多文本。"""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH):
            vectors.extend(self.embedder.embed(texts[i : i + self._BATCH]))
        return vectors


def build_kb(
    directory: str | Path,
    embedding_api_key: str | None = None,
    embedding_base_url: str | None = None,
    embedding_model: str | None = None,
    persist_dir: str | None = None,
) -> TFIDFKnowledgeBase | VectorKnowledgeBase:
    """构建知识库：优先用 embedding + Chroma，缺配置/失败时自动回退到 TF-IDF。"""
    api_key = (
        embedding_api_key
        or os.environ.get("EMBEDDING_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY", "")
    )
    if api_key:
        try:
            import chromadb  # noqa: F401  # 确保装了 chromadb

            embedder = EmbeddingClient(
                api_key=api_key,
                base_url=embedding_base_url
                or os.environ.get(
                    "EMBEDDING_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                model=embedding_model
                or os.environ.get("EMBEDDING_MODEL", "text-embedding-v3"),
            )
            kb: VectorKnowledgeBase = VectorKnowledgeBase(
                embedder, persist_dir=persist_dir
            )
            kb.ingest_dir(directory)
            return kb
        except Exception as exc:
            print(f"[警告] 向量库初始化失败，已回退到 TF-IDF 检索：{exc}", file=sys.stderr)
    kb = TFIDFKnowledgeBase()
    kb.ingest_dir(directory)
    kb.build_index()
    return kb
