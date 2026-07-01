"""
test_chunker.py
---------------
跑一下 chunker,看看切出来的块长什么样。

可以从任意目录运行,也可以在 IDE 里直接 Run/Debug。
"""

# bootstrap: 让脚本不依赖于"从 enterprise_rag/ 启动"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.chunker_handwritten import chunk_text, chunk_document, chunk_documents
from core.loader import load_documents
from test_loader import create_sample_files


def demo_chunk_text() -> None:
    """直接对一段字符串做切块,最容易看出滑动窗口的效果。"""

    print("=" * 60)
    print("demo 1: chunk_text 滑动窗口")
    print("=" * 60)

    # 一段刻意构造的长文本,方便观察重叠
    text = "".join(f"[{i:03d}]" for i in range(200))  # "[000][001][002]..."
    pieces = chunk_text(text, chunk_size=50, chunk_overlap=10)

    print(f"原文本长度: {len(text)}")
    print(f"切成 {len(pieces)} 块,每块 50 字符,重叠 10 字符")
    for i, piece in enumerate(pieces[:4]):  # 只看前 4 块
        print(f"  块 {i}: {piece}")
    print("  ...")


def demo_chunk_document() -> None:
    """对 loader 加载出来的真实文档做切块。"""

    print()
    print("=" * 60)
    print("demo 2: 对 loader 输出做切块")
    print("=" * 60)

    sample_files = create_sample_files()
    documents = load_documents(sample_files)

    # 用小一点的 chunk_size 才能在 sample 文件上看出多块效果
    chunks = chunk_documents(documents, chunk_size=40, chunk_overlap=8)

    print(f"4 个 sample 文件 -> {len(chunks)} 个 chunk")
    for chunk in chunks:
        meta = chunk["metadata"]
        print(
            f"  {meta['filename']} "
            f"({meta['chunk_index'] + 1}/{meta['chunk_total']}): "
            f"{chunk['content'][:40]!r}"
        )


if __name__ == "__main__":
    demo_chunk_text()
    demo_chunk_document()
