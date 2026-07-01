"""
retriever.py
------------
负责向量数据库的"写"和"查"。

写(建库阶段):  index_chunks(embedded_chunks) -> 把 chunk + 向量塞进 ChromaDB
查(查询阶段):  search(query)               -> 返回 top-k 最相关的 chunk

设计上 retriever 是 vector DB 的唯一出入口,
其他模块不直接碰 chromadb,以后换数据库只改这一个文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import chromadb

from config import COLLECTION_NAME, DB_PATH, TOP_K
from core.chunker_handwritten import Chunk
from core.embedder import EmbeddedChunk, embed_query


# Chroma metadata 只允许 str/int/float/bool/None,
# loader 可能塞进 list(如 excel 的 sheets),这里要转一手。
_SCALAR_TYPES = (str, int, float, bool, type(None))


class RetrievalResult(TypedDict):
    """检索结果的对外统一结构,屏蔽 Chroma 的原生格式。"""

    content: str
    metadata: dict[str, Any]
    distance: float  # 距离,越小越相关(Chroma 默认 L2 距离平方)


_collection = None  # 模块级单例,懒加载


def _get_collection():
    """
    懒加载并复用同一个 collection 句柄。

    PersistentClient(path=...) 会自动在该目录下建库,
    如果目录已存在就直接复用 —— 这就是"持久化"的含义。
    """

    global _collection
    if _collection is None:
        # 确保数据目录存在(Chroma 自己也会建,这里显式一下更清晰)
        Path(DB_PATH).mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=DB_PATH)
        _collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    把 metadata 转成 Chroma 能吃的形状。

    - 标量(str/int/float/bool/None)原样保留
    - list / tuple 转成逗号分隔的字符串
    - 其他类型粗暴转 str

    > Java 类比:就像存进关系型数据库前把对象拍平,
      复杂字段序列化成字符串列存放。
    """

    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, _SCALAR_TYPES):
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            clean[key] = ",".join(str(v) for v in value)
        else:
            clean[key] = str(value)
    return clean


def _build_id(metadata: dict[str, Any]) -> str:
    """
    用 filename + chunk_index 作 ID,保证同文件重新入库时会覆盖旧数据。

    避免使用随机 UUID:那样会让"二次上传"变成"重复入库"。
    """

    filename = metadata.get("filename", "unknown")
    chunk_index = metadata.get("chunk_index", 0)
    return f"{filename}#{chunk_index}"


def index_chunks(embedded_chunks: list[EmbeddedChunk]) -> int:
    """
    把一批 EmbeddedChunk 写入向量数据库,返回写入条数。

    用 upsert 而不是 add:
    - add: 重复 id 会报错
    - upsert: 重复 id 会覆盖,适合反复跑脚本和增量更新
    """

    if not embedded_chunks:
        return 0

    collection = _get_collection()

    ids = [_build_id(ec["metadata"]) for ec in embedded_chunks]
    embeddings = [ec["embedding"] for ec in embedded_chunks]
    documents = [ec["content"] for ec in embedded_chunks]
    metadatas = [_sanitize_metadata(ec["metadata"]) for ec in embedded_chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return len(embedded_chunks)


def search(query: str, top_k: int = TOP_K) -> list[RetrievalResult]:
    """
    用 query 字符串检索 top-k 最相关的 chunk。

    完整流程:
        query 文本 -> embed_query -> 向量
        -> Chroma 用向量算距离,找出最近的 top_k 条
        -> 解析 Chroma 返回 -> RetrievalResult list
    """

    if not query.strip():
        raise ValueError("query 不能为空")

    collection = _get_collection()
    query_vector = embed_query(query)

    # Chroma 支持一次查多个 query,所以入参用 list 包一层
    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )

    # raw 的形状:每个字段都是 list[list[...]],外层是 batch 维度
    # 我们只查了一个 query,所以全取 [0]
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    results: list[RetrievalResult] = []
    for content, metadata, distance in zip(documents, metadatas, distances):
        results.append(
            {
                "content": content,
                "metadata": dict(metadata),  # Chroma 返回的可能是只读视图,拷贝一份
                "distance": float(distance),
            }
        )
    return results


def reset_collection() -> None:
    """
    清空 collection,方便重新建库。

    学习阶段反复调试时很有用 —— 否则旧数据会一直堆积。
    生产环境慎用。
    """

    global _collection
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        # 不存在就忽略,delete 不存在的 collection 在不同版本行为不一致
        pass
    _collection = None
