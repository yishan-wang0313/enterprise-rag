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
你是一个查询改写助手。给定用户和 AI 的多轮对话历史,以及用户的最新提问,
把最新提问改写成"完全脱离对话上下文也能理解"的独立问题。

规则:
- 只输出改写后的问题,不要解释、不要加引号、不要任何多余的话
- 如果最新提问本来就完全独立(没有代词、没有省略),原样返回即可
- 保留用户原本的语气和问题类型(问句还是问句,陈述还是陈述)

例 1:
  历史:
    用户: RAG 是什么?
    AI: RAG 是检索增强生成技术...
  最新提问: 它有什么缺点?
  改写为: RAG 有什么缺点?

例 2:
  历史:
    用户: 公司的年假政策是什么?
    AI: 年假 15 天...
  最新提问: 那病假呢?
  改写为: 公司的病假政策是什么?

例 3:
  历史:
    用户: RAG 是什么?
    AI: ...
  最新提问: 今天天气怎么样?
  改写为: 今天天气怎么样?
"""


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 未设置")
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
