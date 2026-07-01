"""
chunker.py
----------
这个模块负责把长文档切成多个"小块"(chunk)。

为什么要切块?
1. embedding 模型更适合处理较短文本(过长会被截断或语义被平均化)
2. 检索时小块比整篇文档更精准:用户问"年假怎么申请",
   返回"那一小段流程"比返回"整本员工手册"有用得多
3. 把相关片段(而不是整篇)喂给 LLM,更省 token、更便宜、更快

本模块输入是 loader.py 输出的 LoadedDocument(整篇文本 + metadata),
输出是一组 Chunk(每小段文本 + 继承并扩展的 metadata)。
"""

from __future__ import annotations

from typing import Any, TypedDict

from core.loader import LoadedDocument
from config import CHUNK_SIZE, CHUNK_OVERLAP


class Chunk(TypedDict):
    """
    切完之后每一块的统一结构。

    刻意和 LoadedDocument 长得几乎一样(content + metadata),
    这样下游 embedder 只关心"文本 + metadata",不关心来源粒度。
    """

    content: str
    metadata: dict[str, Any]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    把一段长文本按"定长 + 重叠"的方式切成多段。

    算法本质是一个滑动窗口:
    - 窗口宽度 = chunk_size
    - 每次窗口向右滑动 (chunk_size - chunk_overlap) 个字符
    - 直到窗口右边超出文本结尾

    例:chunk_size=500, chunk_overlap=50 时
        块1: [0, 500)
        块2: [450, 950)   <- 起点 = 上一块起点 + (500 - 50)
        块3: [900, 1400)
        ...

    参数校验是必要的:overlap >= chunk_size 会导致窗口不前进(死循环)。
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size (否则窗口不会前进)")

    if not text:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # 如果这一块已经吃到文本结尾,就停;否则窗口右移 step 个字符
        if end >= len(text):
            break
        start += step

    return chunks


def chunk_document(
    document: LoadedDocument,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    把一篇 LoadedDocument 切成多个 Chunk。

    关键点:每个 chunk 继承原文档的 metadata,再加上 chunk_index / chunk_total,
    方便后续:
    - 调试时看出某条检索结果来自哪个文件的第几块
    - UI 上把答案来源标注成 "report.docx (2/5)"
    """

    text_pieces = chunk_text(document["content"], chunk_size, chunk_overlap)
    total = len(text_pieces)

    chunks: list[Chunk] = []
    for index, piece in enumerate(text_pieces):
        # 浅拷贝原 metadata 后再扩展,避免修改到上游的 dict
        new_metadata = dict(document["metadata"])
        new_metadata["chunk_index"] = index
        new_metadata["chunk_total"] = total

        chunks.append({"content": piece, "metadata": new_metadata})

    return chunks


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    批量切多篇文档,把所有 chunk 拼成一个扁平 list。

    返回扁平 list 是因为下游 embedder 通常按"一条文本一个向量"批量处理,
    不需要知道哪个 chunk 来自哪篇文档(metadata 里已经写了)。
    """

    all_chunks: list[Chunk] = []
    for document in documents:
        all_chunks.extend(chunk_document(document, chunk_size, chunk_overlap))
    return all_chunks
