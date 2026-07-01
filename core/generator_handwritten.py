"""
generator.py
------------
RAG 的最后一步:把"用户问题 + 检索到的片段"组装成 prompt,
丢给 Chat 模型,得到最终答案。

设计要点:
1. 用 system prompt 锁定角色和"不要瞎编"的边界
2. 把检索结果作为"参考资料 [1][2]..."插入,鼓励模型标注引用
3. 检索结果为空时单独处理(否则 LLM 会自由发挥)
4. 同时返回答案文本 + 它依赖的 sources,方便上层 UI 展示
"""

from __future__ import annotations

from typing import TypedDict

from openai import OpenAI, OpenAIError

from config import CHAT_MODEL, OPENAI_API_KEY
from core.retriever_handwritten import RetrievalResult, search


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
    """generator 的最终对外结构。"""

    answer: str                       # LLM 生成的答案文本
    sources: list[RetrievalResult]    # 依据的检索结果,带 metadata 和 distance


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """和 embedder 同款的 lazy singleton。两个模块各自持有客户端是有意的:
    解耦,以后想给 chat 和 embedding 配不同的 base_url 也容易。"""

    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 未设置")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def build_context(results: list[RetrievalResult]) -> str:
    """
    把检索结果拼成 LLM 容易读的"参考资料"块。

    格式刻意做成:
        [1] (来源: filename, 第 0/3 块)
        正文...
        ---
        [2] (来源: ...)
        正文...

    编号和分隔线让 LLM 容易区分多段资料,
    metadata 信息(filename + chunk_index)让答案能溯源。
    """

    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        meta = result["metadata"]
        filename = meta.get("filename", "unknown")
        chunk_idx = meta.get("chunk_index", 0)
        chunk_total = meta.get("chunk_total", 1)

        header = f"[{index}] (来源: {filename}, 第 {chunk_idx + 1}/{chunk_total} 块)"
        blocks.append(f"{header}\n{result['content']}")

    return "\n---\n".join(blocks)


def build_user_prompt(query: str, context: str) -> str:
    """组合用户 prompt:参考资料在前,问题在后,顺序符合人类阅读习惯。"""

    return f"""【参考资料】
{context}

【用户问题】
{query}
"""


def generate_answer(
    query: str,
    top_k: int | None = None,
    debug: bool = False,
) -> RagAnswer:
    """
    端到端:接收 query -> 检索 -> 拼 prompt -> 调 LLM -> 返回答案 + 来源。

    这是整个 RAG 系统对外的主入口。
    Streamlit 页面拿到用户输入后只调这一个函数。
    """

    if not query.strip():
        raise ValueError("query 不能为空")

    # 1. 检索
    sources = search(query) if top_k is None else search(query, top_k=top_k)

    # 2. 空检索结果的兜底:直接告诉用户没找到,不浪费 LLM 调用
    if not sources:
        return {
            "answer": "我在知识库里没有找到相关信息,无法回答这个问题。",
            "sources": [],
        }

    # 3. 拼 prompt
    context = build_context(sources)
    user_prompt = build_user_prompt(query, context)

    if debug:
        # 调试模式:把真实发给 LLM 的两段 prompt 打印出来。
        # RAG 调优最重要的一步 —— 你必须能"看见"模型看到了什么。
        print("\n----- [DEBUG] SYSTEM PROMPT -----")
        print(SYSTEM_PROMPT)
        print("----- [DEBUG] USER PROMPT -----")
        print(user_prompt)
        print("----- [DEBUG] END -----\n")

    # 4. 调 Chat 模型
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            # temperature 低一点,让回答更"贴着资料",减少发挥
            temperature=0.2,
        )
    except OpenAIError as exc:
        raise RuntimeError(f"调用 Chat 模型失败: {exc}") from exc

    answer = response.choices[0].message.content or ""

    return {"answer": answer.strip(), "sources": sources}
