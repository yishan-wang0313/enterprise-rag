"""
agent.py  (第 6 段之后 · Week 3 独立升级)
------------------------------------------
LangGraph 版 ReAct Agent。让 LLM 自己决定要不要检索 KB、要不要算数、要不要看时间。

对外只关心两个函数:
    run_agent(query, thread_id)     -> 一次跑完,返回完整最终 state
    stream_agent(query, thread_id)  -> yield 结构化事件流(tool_call / tool_result / answer)

设计上和 generator.py 平级 —— app.py 里加一个"Agent Mode"开关,
不勾就走原来的 generate_answer_stream (纯 RAG),勾了就走 stream_agent。

图结构:
    START -> agent_node -> [conditional: has tool_calls?]
                          |-yes-> tool_node -> agent_node (循环)
                          |-no-->  END
"""

from __future__ import annotations

# ---- bootstrap: 支持 `python core/agent.py "..."` 直接跑 ----
import sys
from pathlib import Path

_ENTERPRISE_RAG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ENTERPRISE_RAG_DIR))
# ---------------------------------------------------------------

import ast
import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Generator, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import CHAT_MODEL, OPENAI_API_KEY
from core.retriever import search as _rag_search


# ---------------------------------------------------------------------------
# 1) Tools —— agent 可调用的原子能力
# ---------------------------------------------------------------------------
# 关键:tool 的 docstring 就是给 LLM 看的"使用说明书",要写清楚
# - 干什么用
# - 什么场景下调用
# - 参数含义
# LLM 靠这段文字决定要不要调、怎么调。


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal document knowledge base (hybrid RAG) for information.

    Use this tool when the user asks about facts, policies, definitions, or any
    content that is likely stored in the uploaded documents. Do NOT use it for
    general world knowledge, math, or current time.

    Args:
        query: A focused natural-language question to search for.
    """
    results = _rag_search(query, verbose=False)
    if not results:
        return "No relevant information found in the knowledge base for this query."

    blocks: list[str] = []
    for i, r in enumerate(results, start=1):
        meta = r.get("metadata") or {}
        source = meta.get("filename", "unknown")
        snippet = r["content"][:400].replace("\n", " ")
        blocks.append(f"[{i}] (source: {source}) {snippet}")
    return "\n---\n".join(blocks)


# ---- calculator: AST 安全 eval ----
# eval 直接跑用户字符串是安全灾难 —— 我们只允许算术节点,拒绝任何 name/call。
_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Walk an AST allowing only numeric literals and arithmetic operators."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        return _ALLOWED_BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED_UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a pure arithmetic expression.

    Supports +, -, *, /, //, **, % on integers and floats. NO variables,
    NO function calls, NO names. Use for straightforward math like "2 + 3 * 4"
    or "(150 + 200) / 2".

    Args:
        expression: A plain arithmetic expression as a string.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"Result: {result}"
    except Exception as exc:
        return f"Error: {exc}"


@tool
def current_time() -> str:
    """Return the current UTC date and time.

    Use this only when the user explicitly asks about the current time / date,
    or when time is needed to answer their question.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


TOOLS = [search_knowledge_base, calculator, current_time]


# ---------------------------------------------------------------------------
# 2) State schema
# ---------------------------------------------------------------------------
# `add_messages` 是 LangGraph 提供的 reducer:每个节点返回的 messages 会被追加,
# 而不是覆盖整个 list。这是多轮/多节点的正确模式。


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ---------------------------------------------------------------------------
# 3) LLM —— 绑定 tools,让 LLM 知道自己有哪些工具可用
# ---------------------------------------------------------------------------
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.2,
        ).bind_tools(TOOLS)
    return _llm


AGENT_SYSTEM_PROMPT = """You are a helpful assistant with access to three tools:
- search_knowledge_base(query): retrieve information from the user's uploaded documents
- calculator(expression): evaluate arithmetic
- current_time(): get the current UTC date/time

Guidelines:
1. Decide whether you need a tool. Simple greetings / general knowledge → answer directly.
   Questions about the user's documents → call search_knowledge_base.
   Math → call calculator. Time / date → call current_time.
2. You may chain tool calls (e.g. search, then calculate on the result).
3. When search_knowledge_base returns "No relevant information found", tell the user
   honestly that the knowledge base doesn't cover it. Do not fabricate.
4. Cite sources when using knowledge base results, using the bracket numbers like [1].
5. Match the user's language — English question → English answer, Chinese → Chinese.
"""


# ---------------------------------------------------------------------------
# 4) Nodes
# ---------------------------------------------------------------------------
def agent_node(state: AgentState) -> dict[str, Any]:
    """LLM step: decide next tool call OR produce final answer."""
    messages = list(state["messages"])
    # 保证第一条是 SystemMessage(只在首次调用时插入)
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + messages

    response = _get_llm().invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Router: if the last LLM message has tool_calls, run tools; else END."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# 5) Compile the graph (lazy singleton)
# ---------------------------------------------------------------------------
_graph = None
_checkpointer = InMemorySaver()  # 进程内多轮记忆;真实项目换 SqliteSaver 或 RedisSaver


def _build_graph():
    """
    Compile once. Structure:
      START -> agent -> [tools if any tool_calls, else END]
      tools -> agent  (loop back — the model may want another tool call)
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=_checkpointer)


def get_agent():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ---------------------------------------------------------------------------
# 6) Public API
# ---------------------------------------------------------------------------
def run_agent(query: str, thread_id: str = "default") -> dict:
    """Blocking: run agent to completion, return final state (with all messages)."""
    if not query.strip():
        raise ValueError("query must not be empty")
    config = {"configurable": {"thread_id": thread_id}}
    return get_agent().invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )


class AgentEvent(TypedDict, total=False):
    """Structured event yielded by stream_agent for UI consumption.

    types:
      - "tool_call":    {name, args}          -- LLM decided to call a tool
      - "tool_result":  {name, content}       -- tool returned a value
      - "answer":       {content}             -- final answer (one shot, not token-stream)
      - "done":         {}                    -- graph finished
    """

    type: str
    name: str
    args: dict
    content: str


def stream_agent(query: str, thread_id: str = "default") -> Generator[AgentEvent, None, None]:
    """Yield UI-friendly structured events as the agent runs.

    Not token-level streaming (yet) — each event corresponds to one graph node
    completing. This is enough for UX: user sees "🔧 calling tool X" then
    "📤 got result" then final answer.
    """

    if not query.strip():
        raise ValueError("query must not be empty")
    config = {"configurable": {"thread_id": thread_id}}

    for update in get_agent().stream(
        {"messages": [HumanMessage(content=query)]},
        config=config,
        stream_mode="updates",
    ):
        # `update` is a dict: {node_name: {"messages": [...]}}
        for node_name, node_output in update.items():
            for msg in node_output.get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        yield {
                            "type": "tool_call",
                            "name": tc["name"],
                            "args": tc.get("args", {}),
                        }
                elif isinstance(msg, ToolMessage):
                    yield {
                        "type": "tool_result",
                        "name": msg.name or "",
                        "content": str(msg.content),
                    }
                elif isinstance(msg, AIMessage) and msg.content:
                    # 无 tool_call 的 AIMessage = 最终答案
                    yield {"type": "answer", "content": msg.content}

    yield {"type": "done"}


# ---------------------------------------------------------------------------
# 7) CLI
# ---------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python core/agent.py "your question"')
        return

    query = sys.argv[1]
    print(f"❓ Query: {query}\n")

    for event in stream_agent(query):
        t = event["type"]
        if t == "tool_call":
            print(f"🔧 tool_call: {event['name']}({event['args']})")
        elif t == "tool_result":
            snippet = event["content"][:300].replace("\n", " ")
            print(f"📤 tool_result [{event['name']}]: {snippet}")
        elif t == "answer":
            print(f"\n💬 Answer:\n{event['content']}\n")
        elif t == "done":
            print("✅ done")


if __name__ == "__main__":
    main()
