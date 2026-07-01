"""
test_retriever.py
-----------------
跑通 loader -> chunker -> embedder -> retriever 全链路。
最后一步用一个自然语言 query 去检索,看返回结果对不对。

可以从任意目录运行,也可以在 IDE 里直接 Run/Debug。
"""

# bootstrap: 让脚本不依赖于"从 enterprise_rag/ 启动"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.chunker import chunk_documents
from core.embedder import embed_chunks
from core.loader import load_documents
from core.retriever_handwritten import index_chunks, reset_collection, search
from test_loader import create_sample_files


def main() -> None:
    # 学习阶段每次重建,避免旧数据干扰
    reset_collection()

    # ---- 建库阶段 ----
    sample_files = create_sample_files()
    documents = load_documents(sample_files)
    chunks = chunk_documents(documents, chunk_size=80, chunk_overlap=10)
    embedded = embed_chunks(chunks)
    count = index_chunks(embedded)
    print(f"已写入 {count} 个 chunk 到向量库")

    # ---- 查询阶段 ----
    queries = [
        "公司里有哪些部门?",     # 应该命中 sample.xlsx(里面有 Department)
        "什么是 Markdown?",       # 应该命中 sample.md
        "Word 文档里写了什么?",   # 应该命中 sample.docx
    ]

    for query in queries:
        print()
        print("=" * 60)
        print(f"Query: {query}")
        results = search(query, top_k=3)
        for i, result in enumerate(results, start=1):
            print(
                f"  Top {i}  "
                f"distance={result['distance']:.4f}  "
                f"from={result['metadata'].get('filename')}"
            )
            print(f"         {result['content'][:80]!r}")


if __name__ == "__main__":
    main()
