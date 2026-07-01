"""
chunker.py  (langchain 版)
----------------------------
企业级 RAG 第 3 段:智能切块 + 构建知识库。

和我们之前 core/chunker_handwritten.py(手写"硬切")的本质区别:

  手写版:    定长 N 个字符切一刀,不管语义
  这一版:    优先按段落 \\n\\n 切,其次按 \\n,再按 。!? 等标点,
             最后才按字符切。尽量保住完整句子。
             这就是 langchain RecursiveCharacterTextSplitter 的核心思路。

同时这个模块还做了"建库"的活,对外暴露:
  - process_document(doc):  切块 + 写入 ChromaDB
  - delete_document(name):  按 filename 删除某个文件的所有 chunk
  - get_status():           查看知识库总块数 / 来源文件列表
  - 命令行:python core/chunker.py 文件路径    一步建库
"""

from __future__ import annotations

# bootstrap: 让命令行 `python core/chunker.py` 直接跑也能找到 config / core
import sys
from pathlib import Path

_ENTERPRISE_RAG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ENTERPRISE_RAG_DIR))

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DB_PATH,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)
from core.loader import LoadedDocument, load_document


# ---------------------------------------------------------------------------
# 1) 切块器:RecursiveCharacterTextSplitter
# ---------------------------------------------------------------------------
# separators 是"分隔符优先级表":从前往后依次尝试,直到能切到 chunk_size 以下。
# 中英文混合场景,我们把中文标点放在前面,优先级高于英文。
_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,                    # 按"字符数"度量(简单,够用)
    separators=[
        "\n\n",   # 段落分隔(最优先)
        "\n",     # 行内换行
        "。", "!", "?",   # 中文句末
        ".", "!", "?",    # 英文句末
        ";", ";",          # 分号
        " ",                # 空格
        "",                 # 最后兜底:硬切
    ],
)


# ---------------------------------------------------------------------------
# 2) 向量库句柄(langchain 的 Chroma vectorstore,懒加载单例)
# ---------------------------------------------------------------------------
_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma:
    """
    懒加载 langchain 的 Chroma vectorstore。

    langchain 的 Chroma 内部仍然是用 chromadb 库,只是包了一层 LangChain 通用接口。
    所以这里写入的数据,我们手写版 retriever.py 也能读到(同一个 collection)。
    """

    global _vectorstore
    if _vectorstore is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 未设置")

        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
        )
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=DB_PATH,
        )
    return _vectorstore


# ---------------------------------------------------------------------------
# 3) 切块:LoadedDocument -> list[Document]
# ---------------------------------------------------------------------------
def split_document(document: LoadedDocument) -> list[Document]:
    """
    把一篇 LoadedDocument 切成 langchain Document 列表。

    每个 Document 携带继承自原文件的 metadata,再加 chunk_index / chunk_total,
    方便后续 UI 上标注"来自 report.docx 第 3/12 块"。
    """

    text_pieces: list[str] = _TEXT_SPLITTER.split_text(document["content"])
    total = len(text_pieces)

    docs: list[Document] = []
    for index, piece in enumerate(text_pieces):
        # 浅拷贝 metadata 后扩展,避免污染上游
        meta: dict[str, Any] = dict(document["metadata"])
        meta["chunk_index"] = index
        meta["chunk_total"] = total

        # Chroma metadata 不接受 list/dict,把 sheets 这种序列化成字符串
        if isinstance(meta.get("sheets"), (list, tuple)):
            meta["sheets"] = ",".join(str(s) for s in meta["sheets"])

        docs.append(Document(page_content=piece, metadata=meta))

    return docs


# ---------------------------------------------------------------------------
# 4) 入库:upsert 写入 ChromaDB
# ---------------------------------------------------------------------------
def process_document(document: LoadedDocument) -> int:
    """
    完整流程:解析后的 LoadedDocument -> 切块 -> 写入向量库。
    返回写入的 chunk 数量。

    ID 策略:"{filename}#{chunk_index}"。同一文件二次入库会"先删后加",
    保证不重复(标准 upsert)。
    """

    filename = document["metadata"].get("filename", "unknown")
    char_count = len(document["content"])
    print(f"✅ 解析完成:{filename}({char_count} 字符)")

    # 切块
    docs = split_document(document)
    print(f"✂️  切块完成:{filename} → {len(docs)} 块")

    if not docs:
        print("⚠️  没有可写入的块(文档为空?)")
        return 0

    # 构造稳定 ID
    ids = [f"{filename}#{i}" for i in range(len(docs))]

    # langchain 的 add_documents 没有 upsert 选项,我们自己模拟:先删后加。
    # 不存在的 id delete 会被 chromadb 静默忽略,所以可以放心调。
    vectorstore = _get_vectorstore()
    try:
        vectorstore.delete(ids=ids)
    except Exception:
        # 不同版本的 langchain-chroma 在 id 不存在时行为不一致,忽略即可
        pass

    vectorstore.add_documents(documents=docs, ids=ids)
    print(f"  📦 已存入 {len(docs)}/{len(docs)} 块...")
    print(f"✅ 知识库构建完成,共存入 {len(docs)} 块")

    # 重置 BM25 内存索引,让下次检索能看到新数据
    _invalidate_bm25_cache()
    return len(docs)


# ---------------------------------------------------------------------------
# 5) 删除某个文件的全部 chunk
# ---------------------------------------------------------------------------
def delete_document(filename: str) -> int:
    """
    按 filename 删除该文件在向量库里的所有 chunk。

    实现:借助 chromadb 的 metadata where filter 找到所有 id,再批量 delete。
    langchain 的 Chroma vectorstore 内部仍是 chromadb,所以通过 ._collection
    访问底层 collection 做更精确的操作。
    """

    vectorstore = _get_vectorstore()
    collection = vectorstore._collection  # 拿到底层 chromadb collection

    result = collection.get(where={"filename": filename})
    ids: list[str] = result.get("ids", [])

    if not ids:
        print(f"⚠️  知识库里没有找到 {filename} 的 chunk")
        return 0

    collection.delete(ids=ids)
    print(f"🗑️  已从知识库删除 {filename}({len(ids)} 块)")

    # 重置 BM25 内存索引,让下次检索看不到已删除数据
    _invalidate_bm25_cache()
    return len(ids)


def _invalidate_bm25_cache() -> None:
    """
    通知 retriever 模块清空 BM25 缓存。
    单独抽函数是为了延迟 import,避免循环依赖。
    """

    try:
        from core.retriever import reset_bm25_cache
        reset_bm25_cache()
    except ImportError:
        # retriever 还没加载就无所谓,反正用的时候才会建索引
        pass


# ---------------------------------------------------------------------------
# 6) 知识库状态:总块数 + 来源文件列表
# ---------------------------------------------------------------------------
def get_status() -> dict[str, Any]:
    """
    返回知识库总块数和所有来源文件(去重)。
    """

    vectorstore = _get_vectorstore()
    collection = vectorstore._collection

    total = collection.count()

    # 把所有 metadata 拉出来,提取 filename 去重
    all_metadatas = collection.get().get("metadatas") or []
    files = sorted(
        {meta["filename"] for meta in all_metadatas if meta and "filename" in meta}
    )

    return {"total_chunks": total, "source_files": files}


def print_status() -> None:
    """打印知识库状态(给 CLI 用)。"""

    status = get_status()
    print()
    print("📊 知识库状态:")
    print(f"   总块数:{status['total_chunks']}")
    print(f"   来源文件:{status['source_files']}")


# ---------------------------------------------------------------------------
# 7) CLI 入口:python core/chunker.py 文件路径
# ---------------------------------------------------------------------------
def main() -> None:
    """
    用法:
        python core/chunker.py 文件.docx       # 解析 + 切块 + 入库 + 打印状态
        python core/chunker.py                 # 仅打印当前知识库状态
    """

    if len(sys.argv) < 2:
        print_status()
        return

    file_path = sys.argv[1]
    document = load_document(file_path)
    process_document(document)
    print_status()


if __name__ == "__main__":
    main()
