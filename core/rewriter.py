"""
rewriter.py (第 6 段优化 1)
---------------------------
查询改写:把带代词/省略的追问改成独立问题。

场景:
    用户: RAG 是什么?
    AI:   RAG 是检索增强生成...
    用户: 它有什么缺点?          ← "它"指 RAG,但 retriever 不知道

改写后:
    "RAG 有什么缺点?"           ← 独立、可检索

设计要点:
- 只在有对话历史时改写,首问直接透传
- 只带最近 3 轮上下文,prompt 短、便宜
- temperature=0,改写不是创造性任务
- 改写失败 → 兜底返回原 query(不能因为改写挂了就把主流程堵住)
"""

from __future__ import annotations

# bootstrap
import sys
from pathlib import Path

_ENTERPRISE_RAG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ENTERPRISE_RAG_DIR))

from openai import OpenAI, OpenAIError

from config import CHAT_MODEL, OPENAI_API_KEY


# 最多带几轮对话进 prompt(1 轮 = 一个 user + 一个 assistant)
_MAX_HISTORY_TURNS = 3


REWRITE_SYSTEM_PROMPT = """\
You are a query-rewriting assistant. Given a multi-turn conversation history between
a user and an AI, and the user's latest question, rewrite that latest question into a
standalone question that can be understood without the conversation context.

Rules:
- Output only the rewritten question. No explanation, no quotes, no extra text.
- If the latest question is already self-contained (no pronouns, no ellipsis), return it as-is.
- Preserve the user's original tone and question type (question stays a question).
- IMPORTANT: Keep the rewritten question in the SAME LANGUAGE as the user's original question.

Example 1 (English):
  History:
    user: What is RAG?
    AI: RAG stands for Retrieval-Augmented Generation...
  Latest question: What are its downsides?
  Rewritten: What are the downsides of RAG?

Example 2 (Chinese):
  历史:
    用户: 公司的年假政策是什么?
    AI: 年假 15 天...
  最新提问: 那病假呢?
  改写为: 公司的病假政策是什么?

Example 3 (unrelated topic — do NOT force it back on topic):
  History:
    user: What is RAG?
    AI: ...
  Latest question: What's the weather today?
  Rewritten: What's the weather today?
"""


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def rewrite_query(query: str, history: list[dict]) -> str:
    """
    把用户当前 query 结合历史改写成独立问题。

    history 格式:[{"role": "user"|"assistant", "content": str}, ...]
    返回:改写后的 query(或原 query,视情况)
    """

    query = query.strip()
    if not query:
        return query

    # 首问 or 没历史 → 不需要改写
    if not history:
        return query

    # 只带最近 N 轮
    recent = history[-(_MAX_HISTORY_TURNS * 2):]

    history_text = "\n".join(
        f"  {'用户' if m['role'] == 'user' else 'AI'}: {m['content']}"
        for m in recent
    )
    user_prompt = f"对话历史:\n{history_text}\n最新提问: {query}\n改写为:"

    try:
        response = _get_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,          # 改写要确定性
            max_tokens=200,         # 防止 LLM 啰嗦
        )
    except OpenAIError:
        # 改写失败 → 兜底原 query,不阻塞主流程
        return query

    rewritten = (response.choices[0].message.content or "").strip()

    # 防御:LLM 偶尔会返回空 / 巨长 / 带引号 —— 都视为改写失败
    if not rewritten or len(rewritten) > 500:
        return query
    # 去掉可能的引号包裹
    rewritten = rewritten.strip("\"'「」“”")

    return rewritten
