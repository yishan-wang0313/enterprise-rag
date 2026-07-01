"""
compare_chunkers.py
-------------------
把同一个文件分别用"手写硬切"和"langchain 递归切"跑一遍,并排打印结果。
观察两版切法对"句子完整性"和"chunk 数量"的影响。

用法(从任意目录):
    python compare_chunkers.py uploads/big_demo.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.chunker import split_document as split_lc            # langchain 版
from core.chunker_handwritten import chunk_document as split_hw  # 手写版
from core.loader import load_document


def show(title: str, chunks: list) -> None:
    """统一打印格式 —— 不论是 langchain Document 还是手写 Chunk(dict)。"""
    print(f"\n{'=' * 70}")
    print(f"{title}  →  {len(chunks)} 块")
    print('=' * 70)
    for i, c in enumerate(chunks):
        text = c.page_content if hasattr(c, "page_content") else c["content"]
        # 只看开头和结尾各 40 字,直观感受切口在哪
        head = text[:40].replace("\n", "⏎")
        tail = text[-40:].replace("\n", "⏎")
        print(f"  [{i+1}/{len(chunks)}] len={len(text):>4}")
        print(f"       开头: {head!r}")
        print(f"       结尾: {tail!r}")


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python compare_chunkers.py <文件路径>")
        return

    doc = load_document(sys.argv[1])
    print(f"原文长度: {len(doc['content'])} 字符")

    # 用相同的 chunk_size / overlap,公平比较
    hw_chunks = split_hw(doc, chunk_size=500, chunk_overlap=50)
    lc_chunks = split_lc(doc)  # 用 config 里的默认值

    show("手写版(定长硬切 500/50)", hw_chunks)
    show("langchain 版(递归智能切 500/50)", lc_chunks)


if __name__ == "__main__":
    main()
