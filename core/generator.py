"""
generator.py  (企业级第 5 段)
-----------------------------
端到端 RAG 答案生成。

对外两个主要函数:
  - generate_answer(query)           一次性返回完整答案 + sources(CLI / 测试用)
  - generate_answer_stream(query)    yield 流式 token + sources(给 Streamlit UI 用)

底层检索走 core.retriever.search()(vector + BM25 + RRF + CrossEncoder 重排序)。
"""

from __future__ import annotations

# bootstrap
import sys
from pathlib import Path

_ENTERPRISE_RAG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ENTERPRISE_RAG_DIR))

from typing import Generator, Iterable, TypedDict

from openai import OpenAI, OpenAIError

from config import CHAT_MODEL, OPENAI_API_KEY
from core.retriever import SearchResult, search


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
你是一个企业知识库问答助手。

工作方式:
1. 优先使用下方【参考资料】中的内容来回答用户问题。
   参考资料可能是表格、列表或纯文本,只要里面有相关信息,就提取出来回答。
2. 即使参考资料的格式看起来零碎(例如 tab/管道符分隔的表格),
   也请尝试理解其结构并据此作答。
3. 只有当参考资料里确实**完全没有**与问题相关的信息时,
   才回答:"我在知识库里没有找到相关信息,无法回答这个问题。"
4. 回答时尽量在末尾用 [编号] 的形式标注引用,例如 "...是工程部 [1]"。
5. 用简洁、自然的中文回答。
"""


class RagAnswer(TypedDict):
    """非流式版的返回结构。"""

    answer: str
    sources: list[SearchResult]


# 流式版每次 yield 的事件结构(类似前端的 SSE):
#   {"type": "sources",       "data": [...]}     - 检索阶段完成,先把 sources 推过去
#   {"type": "answer_chunk",  "content": "..."}  - LLM 流式吐出来的 token
class StreamEvent(TypedDict, total=False):
    type: str          # "sources" | "answer_chunk"
    data: list         # 当 type == "sources" 时存在
    content: str       # 当 type == "answer_chunk" 时存在


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 未设置")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Prompt 组装
# ---------------------------------------------------------------------------
def build_context(results: list[SearchResult]) -> str:
    """把检索结果拼成"参考资料"块,每段编号 + 来源标注。"""

    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        meta = result["metadata"] or {}
        filename = meta.get("filename", "unknown")
        chunk_idx = meta.get("chunk_index", 0)
        chunk_total = meta.get("chunk_total", 1)

        header = f"[{index}] (来源: {filename}, 第 {chunk_idx + 1}/{chunk_total} 块)"
        blocks.append(f"{header}\n{result['content']}")
    return "\n---\n".join(blocks)


def build_user_prompt(query: str, context: str) -> str:
    return f"""【参考资料】
{context}

【用户问题】
{query}
"""


# ---------------------------------------------------------------------------
# 非流式:返回完整答案(给 CLI / 测试用)
# ---------------------------------------------------------------------------
def generate_answer(query: str) -> RagAnswer:
    if not query.strip():
        raise ValueError("query 不能为空")

    sources = search(query, verbose=False)
    if not sources:
        return {
            "answer": "我在知识库里没有找到相关信息,无法回答这个问题。",
            "sources": [],
        }

    user_prompt = build_user_prompt(query, build_context(sources))
    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
    except OpenAIError as exc:
        raise RuntimeError(f"调用 Chat 模型失败: {exc}") from exc

    answer = (response.choices[0].message.content or "").strip()
    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# 流式:逐 token 推送(给 Streamlit UI 用)
# ---------------------------------------------------------------------------
def generate_answer_stream(query: str) -> Generator[StreamEvent, None, None]:
    """
    生成器函数。先 yield 一个 sources 事件,再 yield 一连串 answer_chunk。

    为什么用"事件流"模式?
    - UI 可以"边检索边显示" —— 用户先看到来源,再看到答案逐字浮现
    - 体验大幅好于"转圈等十秒"
    - 接口形态和 OpenAI 的 SSE 流类似,你以后接其他 LLM 也容易迁移
    """

    if not query.strip():
        raise ValueError("query 不能为空")

    sources = search(query, verbose=False)

    # 先把 sources 推给 UI,让用户立刻看到"检索到了什么"
    yield {"type": "sources", "data": sources}

    if not sources:
        yield {
            "type": "answer_chunk",
            "content": "我在知识库里没有找到相关信息,无法回答这个问题。",
        }
        return

    user_prompt = build_user_prompt(query, build_context(sources))
    client = _get_client()

    try:
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            stream=True,   # 👈 关键:开启流式
        )
    except OpenAIError as exc:
        yield {"type": "answer_chunk", "content": f"\n[调用 Chat 模型失败: {exc}]"}
        return

    # OpenAI 流式接口返回一个迭代器,每个 chunk 是一小段 delta
    for chunk in stream:
        # 注意:首个 chunk 的 delta.content 可能是 None(role 那段)
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield {"type": "answer_chunk", "content": delta}


# ---------------------------------------------------------------------------
# CLI(主要用于调试,真实使用走 app.py)
# ---------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print('用法: python core/generator.py "你的问题"')
        return

    query = sys.argv[1]
    print(f"Q: {query}\n")

    # 跑流式版,展示"边写边出"的效果
    print("A: ", end="", flush=True)
    sources: list[SearchResult] = []
    for event in generate_answer_stream(query):
        if event["type"] == "sources":
            sources = event["data"]
        elif event["type"] == "answer_chunk":
            print(event["content"], end="", flush=True)
    print("\n")

    print("依据的资料:")
    for i, src in enumerate(sources, start=1):
        meta = src["metadata"] or {}
        fn = meta.get("filename", "?")
        snippet = src["content"][:80].replace("\n", " ")
        print(f"  [{i}] {fn} (score={src['score']:.4f}): {snippet}")


if __name__ == "__main__":
    main()
