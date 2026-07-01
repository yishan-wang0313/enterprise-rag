"""
test_generator.py
-----------------
完整端到端跑一遍 RAG:
loader -> chunker -> embedder -> retriever -> generator

可以从任意目录运行,也可以在 IDE 里直接 Run/Debug:
    python test_generator.py
"""

# ---- bootstrap: 把脚本所在目录(enterprise_rag/)加到 sys.path ----
# 这样无论 cwd 在哪里(终端 cd 进来 / IDE 直接 Run),
# `from config import ...` 和 `from core.xxx import ...` 都能解析。
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# ------------------------------------------------------------------

from core.chunker import chunk_documents
from core.embedder import embed_chunks
from core.generator_handwritten import generate_answer
from core.loader import load_documents
from core.retriever_handwritten import index_chunks, reset_collection
from test_loader import create_sample_files


def build_knowledge_base() -> None:
    """一次性把 sample 文件灌进 vector DB。"""

    reset_collection()
    sample_files = create_sample_files()
    documents = load_documents(sample_files)
    chunks = chunk_documents(documents, chunk_size=80, chunk_overlap=10)
    embedded = embed_chunks(chunks)
    count = index_chunks(embedded)
    print(f"知识库构建完成,共 {count} 个 chunk\n")


def ask(query: str, debug: bool = False) -> None:
    """问一个问题,打印答案和来源。"""

    print("=" * 60)
    print(f"Q: {query}")
    result = generate_answer(query, top_k=3, debug=debug)
    print(f"A: {result['answer']}")
    print("\n依据的资料:")
    for i, source in enumerate(result["sources"], start=1):
        meta = source["metadata"]
        print(
            f"  [{i}] {meta.get('filename')} "
            f"(distance={source['distance']:.3f}): "
            f"{source['content'][:60]!r}"
        )
    print()


def main() -> None:
    build_knowledge_base()

    # 三种典型场景:
    # 第一题开 debug,亲眼看一下发给 LLM 的真实 prompt 长什么样
    ask("公司里有哪些部门?", debug=True)   # 应该能答出 Engineering / Sales
    ask("Alice 的考试得了多少分?")          # 应该能答出 98
    ask("CEO 是谁?")                        # 知识库里没有 — 应该回答"没找到"


if __name__ == "__main__":
    main()
