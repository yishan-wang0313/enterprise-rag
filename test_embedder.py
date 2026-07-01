"""
test_embedder.py
----------------
串起来跑 loader -> chunker -> embedder,看看每步产物。

注意:这个脚本会真的调用 OpenAI API,会消耗少量额度。
可以从任意目录运行,也可以在 IDE 里直接 Run/Debug。
"""

# bootstrap: 让脚本不依赖于"从 enterprise_rag/ 启动"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.chunker import chunk_documents
from core.embedder import embed_chunks, embed_query
from core.loader import load_documents
from test_loader import create_sample_files


def main() -> None:
    # ---- 1. loader ----
    sample_files = create_sample_files()
    documents = load_documents(sample_files)
    print(f"loader  : {len(sample_files)} 个文件 -> {len(documents)} 个 LoadedDocument")

    # ---- 2. chunker ----
    chunks = chunk_documents(documents, chunk_size=80, chunk_overlap=10)
    print(f"chunker : {len(documents)} 个文档 -> {len(chunks)} 个 Chunk")

    # ---- 3. embedder ----
    embedded = embed_chunks(chunks)
    print(f"embedder: {len(chunks)} 个 Chunk -> {len(embedded)} 个 EmbeddedChunk")

    # 看一下第一条产物的形状
    first = embedded[0]
    print()
    print("第一条 EmbeddedChunk 结构:")
    print(f"  content (前 50 字): {first['content'][:50]!r}")
    print(f"  metadata          : {first['metadata']}")
    print(f"  embedding 维度    : {len(first['embedding'])}")
    print(f"  embedding 前 5 维 : {first['embedding'][:5]}")

    # ---- 4. query 也要 embed ----
    query_vec = embed_query("员工有哪些部门?")
    print()
    print(f"query embedding 维度: {len(query_vec)}")
    # 注意:query 向量和 chunk 向量维度必须一致,
    # 否则后面没法在同一个向量空间里算距离


if __name__ == "__main__":
    main()
