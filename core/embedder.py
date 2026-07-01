"""
embedder.py
-----------
把文本(chunk 或 query)转成向量(embedding)。

embedding 是一串浮点数(text-embedding-3-small 是 1536 维),
向量空间里距离近 ≈ 语义相近。检索阶段就是在这个向量空间里找邻居。

本模块对外提供两层 API:
- embed_texts(texts):底层,纯字符串 list -> 向量 list。
  query 阶段也用它,因为 query 是普通字符串,不是 Chunk。
- embed_chunks(chunks):上层,把 chunker 输出的 Chunk 贴上向量,
  得到 EmbeddedChunk,后续可以直接写入 vector DB。
"""

from __future__ import annotations

from typing import Any, TypedDict

from openai import OpenAI, OpenAIError

from config import EMBEDDING_MODEL, OPENAI_API_KEY
from core.chunker_handwritten import Chunk


# OpenAI 单条 input 的 token 上限是 8192。
# 我们按字符近似保护:中文 1 字 ≈ 1.5 token,英文不会更糟,
# 所以 6000 字符是相对安全的"截断阈值"。
# 学习阶段够用了,真要精确就上 tiktoken 算 token 数。
MAX_INPUT_CHARS = 6000

# 一次 API 调用最多打包多少条文本。
# OpenAI 限制 2048,但总 token 数也有限制,小一点更稳。
DEFAULT_BATCH_SIZE = 100


class EmbeddedChunk(TypedDict):
    """Chunk + 它的向量。下游(retriever/写库)消费这个结构。"""

    content: str
    metadata: dict[str, Any]
    embedding: list[float]


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """
    懒加载 OpenAI 客户端,且整个进程复用一个实例。

    为什么用模块级单例?
    - openai SDK 的 Client 内部维护 HTTP 连接池,复用更省资源
    - 但放在模块顶层 `client = OpenAI()` 会让 import 时立刻校验 API Key,
      不方便单元测试和模块加载

    > Java 类比:这就是 lazy-init singleton,和 double-checked locking 同思路,
      只不过 Python 模块 import 本身是线程安全的,不需要锁。
    """

    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY 没有设置。"
                "请在环境变量或 .env 文件里配置后再运行。"
            )
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _truncate(text: str) -> str:
    """
    超过安全长度时截断并打印警告。

    为什么不直接报错?RAG 系统的容错性比 100% 完整更重要 —
    用户上传一份 100 页 PDF,不希望因为某一段过长就整个失败。
    """

    if len(text) <= MAX_INPUT_CHARS:
        return text
    print(
        f"[embedder] 警告:某条文本长度 {len(text)} 超过 {MAX_INPUT_CHARS},已截断。"
        " 建议在 chunker 阶段使用更小的 chunk_size。"
    )
    return text[:MAX_INPUT_CHARS]


def embed_texts(
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """
    把一批文本转成向量,返回的向量顺序和入参顺序一致。

    分批策略:把 texts 切成多个 batch,每个 batch 内一次 API 调用。
    OpenAI 接口会返回 data[i].index,理论上要按 index 还原顺序;
    但官方 SDK 已经保证按入参顺序返回,所以这里直接按下标取。
    """

    if not texts:
        return []

    client = _get_client()
    safe_texts = [_truncate(text) for text in texts]

    all_vectors: list[list[float]] = []

    # 经典批处理循环。range(0, n, step) 在 Java 里相当于 for(i=0; i<n; i+=step)
    for start in range(0, len(safe_texts), batch_size):
        batch = safe_texts[start : start + batch_size]
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        except OpenAIError as exc:
            raise RuntimeError(
                f"调用 embedding 接口失败(batch 起始位置 {start}): {exc}"
            ) from exc

        # response.data 是 list,长度等于这一批的输入数量
        all_vectors.extend(item.embedding for item in response.data)

    return all_vectors


def embed_chunks(
    chunks: list[Chunk],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[EmbeddedChunk]:
    """
    给 chunker 输出的每个 Chunk 算向量,生成 EmbeddedChunk。

    本质就是 embed_texts 的一个语义包装层:
    chunker 关心"切",embedder 关心"算向量",
    本函数负责把两者的输出黏起来交给下游。
    """

    if not chunks:
        return []

    texts = [chunk["content"] for chunk in chunks]
    vectors = embed_texts(texts, batch_size=batch_size)

    return [
        {
            "content": chunk["content"],
            "metadata": chunk["metadata"],
            "embedding": vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]


def embed_query(query: str) -> list[float]:
    """
    给检索阶段的查询字符串生成向量。

    为什么单独拎一个函数,而不是直接 embed_texts([query])[0]?
    - 语义更清晰:retriever 调用方一看 embed_query 就知道意图
    - 后续可能在这里加 query 专属的预处理(改写、扩展、去停用词等)
      ,把改动隔离在一个函数里
    """

    if not query.strip():
        raise ValueError("query 不能为空字符串")
    return embed_texts([query])[0]
