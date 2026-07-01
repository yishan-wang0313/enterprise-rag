"""
retriever.py  (langchain 版,企业级第 4 段)
-------------------------------------------
混合检索 + 重排序。三件套:

  1) 向量检索 (vector search) — 语义相似,但可能错过精确词
  2) 关键词检索 (BM25)          — 精确词命中,但不懂同义
  3) RRF 融合 + CrossEncoder 重排序 — 把两路结果合并并精挑细选

最终对外只一个 search() 函数,返回 RERANK_TOP_K 条精排结果。

CLI:
    python core/retriever.py "你的问题"
"""

from __future__ import annotations

# ---- bootstrap: 允许 `python core/retriever.py ...` 直接跑 ----
import sys
from pathlib import Path

_ENTERPRISE_RAG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ENTERPRISE_RAG_DIR))
# -------------------------------------------------------------

from typing import TypedDict

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder

from config import (
    COLLECTION_NAME,
    DB_PATH,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    RERANK_TOP_K,
    TOP_K,
)


# CrossEncoder 模型 —— 多语言重排序,首选 BAAI 系列(支持中文)。
# 第一次运行会从 HuggingFace 下载约 280 MB,后续走本地缓存。
_RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# RRF 的 k 常量,论文经验值 60。控制"排名靠前优势"的衰减速度。
_RRF_K = 60


class SearchResult(TypedDict):
    """对外统一的结果结构。"""

    content: str
    metadata: dict
    score: float          # 重排序后的分数,越大越相关


# ---------------------------------------------------------------------------
# 1) 懒加载:向量库、BM25 索引、CrossEncoder 模型
# ---------------------------------------------------------------------------
_vectorstore: Chroma | None = None
_bm25_retriever: BM25Retriever | None = None
_cross_encoder: CrossEncoder | None = None


def _get_vectorstore() -> Chroma:
    """复用 chunker.py 写入时用的同一个 collection。"""

    global _vectorstore
    if _vectorstore is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 未设置")
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=DB_PATH,
        )
    return _vectorstore


def _get_bm25_retriever() -> BM25Retriever:
    """
    BM25 是"内存索引",需要把所有文档预先加载进来。
    我们从 Chroma 一次性拉出全部 chunk,基于它构建 BM25。

    注意:这意味着如果知识库**变更了**(新增/删除文档),
    本进程内的 BM25 索引不会自动同步 —— 学习阶段可接受;
    生产环境通常用 OpenSearch / Elastic 这种支持热更新的 BM25 引擎。
    """

    global _bm25_retriever
    if _bm25_retriever is None:
        vectorstore = _get_vectorstore()
        raw = vectorstore._collection.get()  # 拿底层 chromadb collection
        documents: list[Document] = [
            Document(page_content=content, metadata=meta or {})
            for content, meta in zip(raw["documents"], raw["metadatas"])
        ]
        if not documents:
            # 用领域异常代替 RuntimeError,让 app.py 能友好展示
            from core.exceptions import EmptyKnowledgeBaseError
            raise EmptyKnowledgeBaseError(
                "collection is empty; import documents first via chunker.py"
            )
        # 默认 k=4,这里我们要每次手动指定
        _bm25_retriever = BM25Retriever.from_documents(documents)
    return _bm25_retriever


def reset_bm25_cache() -> None:
    """
    清空 BM25 内存索引,下次 keyword_search 时会从 Chroma 重新拉数据重建。

    必须在以下时机调用:
    - process_document() 写入新文档之后
    - delete_document() 删除文档之后
    否则同进程内 BM25 看到的还是旧快照(参见第 4 段坑笔记)。
    """

    global _bm25_retriever
    _bm25_retriever = None


def _get_cross_encoder() -> CrossEncoder:
    """加载重排序模型(第一次会下载 ~280 MB 到本地缓存)。"""

    global _cross_encoder
    if _cross_encoder is None:
        print(f"📥 加载 CrossEncoder 模型: {_RERANKER_MODEL_NAME}(首次会下载)")
        _cross_encoder = CrossEncoder(_RERANKER_MODEL_NAME)
    return _cross_encoder


# ---------------------------------------------------------------------------
# 2) 向量检索
# ---------------------------------------------------------------------------
def vector_search(query: str, top_k: int = TOP_K) -> list[Document]:
    """语义检索:把 query 转向量,在 Chroma 里找最近邻 top_k 条。"""

    return _get_vectorstore().similarity_search(query, k=top_k)


# ---------------------------------------------------------------------------
# 3) 关键词检索 (BM25)
# ---------------------------------------------------------------------------
def keyword_search(query: str, top_k: int = TOP_K) -> list[Document]:
    """
    BM25 检索:经典的"词频 + 逆文档频率"加权,对精确词命中很敏感。
    适合检索"型号、人名、专有名词"这种向量容易丢的细节。
    """

    bm25 = _get_bm25_retriever()
    bm25.k = top_k  # 临时改 top_k
    return bm25.invoke(query)


# ---------------------------------------------------------------------------
# 4) RRF 融合 —— 把多路检索结果合并成一份带分数的列表
# ---------------------------------------------------------------------------
def rrf_fuse(
    rank_lists: list[list[Document]],
    k: int = _RRF_K,
) -> list[tuple[Document, float]]:
    """
    Reciprocal Rank Fusion(倒数排名融合)。

    每个文档在每路检索里的得分 = 1 / (k + rank),最后求和。
    rank 从 1 开始,所以排第 1 的得分最高 = 1/61 ≈ 0.0164。

    优点:不需要不同路结果分数可比(向量距离 vs BM25 分数完全不同尺度),
          只用"名次"参与运算,天然鲁棒。
    """

    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}

    for rank_list in rank_lists:
        for rank, doc in enumerate(rank_list, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            # 同 key 文档以第一次见到的为准
            docs_by_key.setdefault(key, doc)

    # 按融合分数降序
    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    return [(docs_by_key[key], scores[key]) for key in sorted_keys]


def _doc_key(doc: Document) -> str:
    """
    用 filename + chunk_index 作"去重 key",和我们 chunker 里的 id 策略一致。
    没 metadata 时退化成用内容前 100 字。
    """

    meta = doc.metadata or {}
    if "filename" in meta and "chunk_index" in meta:
        return f"{meta['filename']}#{meta['chunk_index']}"
    return doc.page_content[:100]


# ---------------------------------------------------------------------------
# 5) CrossEncoder 重排序
# ---------------------------------------------------------------------------
def rerank(
    query: str,
    candidates: list[Document],
    top_n: int = RERANK_TOP_K,
) -> list[tuple[Document, float]]:
    """
    用 CrossEncoder 给每个候选文档重新打分,挑出 top_n。

    和 embedding 检索的本质区别:
    - embedding: query 和 doc 分别向量化,再算距离 —— 模型从没"同时见过"它俩
    - cross encoder: query 和 doc 一起喂进 transformer,模型直接判断"匹配度"
    精度高很多,但慢得多 —— 所以只用来精排少量候选(几十条),不能粗召。
    """

    if not candidates:
        return []

    model = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = model.predict(pairs)  # 返回 list[float]

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(doc, float(score)) for doc, score in scored[:top_n]]


# ---------------------------------------------------------------------------
# 6) 主入口:三步组合
# ---------------------------------------------------------------------------
def search(
    query: str,
    top_k: int = TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    verbose: bool = True,
) -> list[SearchResult]:
    """
    完整混合检索:vector + bm25 → RRF 融合 → CrossEncoder 重排序 → 返回精排。
    """

    if verbose:
        print(f"🔍 检索中:{query}")

    # 1) 两路粗召
    vec_results = vector_search(query, top_k)
    kw_results = keyword_search(query, top_k)

    if verbose:
        print(f"   向量检索:{len(vec_results)} 条 | 关键词检索:{len(kw_results)} 条")

    # 2) RRF 融合
    fused = rrf_fuse([vec_results, kw_results])

    if verbose:
        print(f"   融合后:{len(fused)} 条")

    # 3) 重排序
    candidates = [doc for doc, _ in fused]
    reranked = rerank(query, candidates, rerank_top_n := rerank_top_k)

    if verbose:
        print(f"   重排序后:{len(reranked)} 条")

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata or {},
            "score": score,
        }
        for doc, score in reranked
    ]


# ---------------------------------------------------------------------------
# 7) CLI
# ---------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print('用法: python core/retriever.py "你的问题"')
        return

    query = sys.argv[1]
    results = search(query)

    print()
    print(f"📋 最终检索结果(共 {len(results)} 条):")
    print()
    for i, r in enumerate(results, start=1):
        filename = r["metadata"].get("filename", "unknown")
        print(f"[{i}] 来源:{filename}")
        print(f"    重排序分:{r['score']:.4f}")
        # 内容只看前 100 字,避免太长
        snippet = r["content"][:100].replace("\n", " ")
        print(f"    内容:{snippet}")
        print()


if __name__ == "__main__":
    main()
