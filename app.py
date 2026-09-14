"""
app.py  (企业级第 5 段)
-----------------------
Streamlit Web 界面,把 loader / chunker / retriever / generator 串成产品。

启动:
    streamlit run app.py
浏览器访问:
    http://localhost:8501

布局:
  左侧栏:文档上传、导入、状态、删除
  主区域:对话(用户提问 + 助手回答 + 来源 expander)
"""

# bootstrap
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import streamlit as st

from core.chunker import delete_document, get_status, process_document
from core.exceptions import RagError, friendly_message
from core.generator import generate_answer_stream
from core.loader import SUPPORTED_EXTENSIONS, load_document
from core.rewriter import rewrite_query


# ---------------------------------------------------------------------------
# 页面基础设置(必须放在所有 st.xxx 之前)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise RAG Q&A",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state 初始化
# ---------------------------------------------------------------------------
# Streamlit 每次交互都会重跑整个脚本,要把"跨次需要保留的状态"放进 session_state。
# 这里我们要保留对话历史(消息列表)。
if "messages" not in st.session_state:
    st.session_state.messages = []   # list[{"role": "user"|"assistant", "content": str, "sources": [...]}]


# ---------------------------------------------------------------------------
# 工具函数:把上传的文件保存到本地 uploads/ 目录
# ---------------------------------------------------------------------------
def save_uploaded_file(uploaded_file) -> Path:
    """Streamlit 给的 uploaded_file 是内存对象,我们要写到磁盘上,
    因为 loader.py 接收的是文件路径。"""

    uploads_dir = _HERE / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    target = uploads_dir / uploaded_file.name
    target.write_bytes(uploaded_file.getbuffer())
    return target


# ---------------------------------------------------------------------------
# 侧边栏:文档管理
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Document Management")

    uploaded_files = st.file_uploader(
        "Upload documents (docx / xlsx / txt / md)",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )

    # "导入到知识库" 按钮
    if st.button("⬆️ Import to Knowledge Base", disabled=not uploaded_files, use_container_width=True):
        any_success = False
        for f in uploaded_files:
            try:
                with st.spinner(f"Processing {f.name} ..."):
                    # 1) 落盘 2) loader 解析 3) chunker 切块 + 入库
                    local_path = save_uploaded_file(f)
                    document = load_document(str(local_path))
                    count = process_document(document)
                st.success(f"✅ {f.name} imported ({count} chunks)")
                any_success = True
            except Exception as e:
                # 一个文件失败不该拖累其他文件
                st.error(f"❌ {f.name} failed: {friendly_message(e)}")
        if any_success:
            # 只要有一个成功,就刷新一下让"知识库状态"更新
            st.rerun()

    st.divider()
    st.header("📊 Knowledge Base")

    status = get_status()
    st.metric("Total chunks", status["total_chunks"])

    if status["source_files"]:
        st.write("**📄 Imported files:**")
        for filename in status["source_files"]:
            col1, col2 = st.columns([4, 1])
            col1.text(filename)
            # 每个文件配一个删除按钮。key 必须唯一,否则 Streamlit 会报错
            if col2.button("🗑️", key=f"del_{filename}", help=f"Delete {filename}"):
                delete_document(filename)
                st.rerun()
    else:
        st.info("Knowledge base is empty. Upload some files above to get started.")


# ---------------------------------------------------------------------------
# 主区域:标题 + 对话
# ---------------------------------------------------------------------------
st.title("📚 Enterprise Knowledge Base Q&A")
st.caption("Hybrid retrieval (Vector + BM25) · RRF fusion · CrossEncoder reranking · Streaming GPT")


def render_sources(sources: list, expanded: bool = False) -> None:
    """把检索来源渲染成一个可折叠的 expander。"""

    if not sources:
        return
    with st.expander(f"📌 View sources ({len(sources)})", expanded=expanded):
        for i, src in enumerate(sources, start=1):
            meta = src.get("metadata") or {}
            filename = meta.get("filename", "unknown")
            chunk_idx = meta.get("chunk_index", 0)
            chunk_total = meta.get("chunk_total", 1)
            score = src.get("score", 0.0)

            st.markdown(
                f"**[{i}] {filename}** "
                f"_(chunk {chunk_idx + 1}/{chunk_total}, rerank score: {score:.4f})_"
            )
            # 用 st.text 而不是 st.markdown,避免内容里的 markdown 字符乱解析
            st.text(src["content"][:400])
            st.divider()


# 1) 先回放历史消息(Streamlit 每次重跑都要这样还原)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources") or [])


# 2) 底部输入框
query = st.chat_input("Ask a question...")

if query:
    # 2.1 用户消息上屏 & 存进 session
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2.2 查询改写(如果有对话历史)
    # 注意:传入的 history 是"截止到用户本次提问之前"的,
    #      所以我们从 session 里排除刚 append 的这条 user message
    history_for_rewrite = st.session_state.messages[:-1]
    with st.spinner("Understanding your question..."):
        rewritten_query = rewrite_query(query, history_for_rewrite)

    # 2.3 助手消息:流式渲染
    with st.chat_message("assistant"):
        # 如果改写了,先给用户一个透明的提示
        if rewritten_query != query:
            st.caption(f"🔄 Interpreted as: {rewritten_query}")

        sources_slot = st.empty()
        answer_slot = st.empty()

        full_answer = ""
        sources_data: list = []

        try:
            with st.spinner("Retrieving + thinking..."):
                for event in generate_answer_stream(rewritten_query):
                    if event["type"] == "sources":
                        sources_data = event["data"]
                        with sources_slot.container():
                            render_sources(sources_data, expanded=False)
                    elif event["type"] == "answer_chunk":
                        full_answer += event["content"]
                        answer_slot.markdown(full_answer + "▌")
            # 流正常结束,去掉打字机光标
            answer_slot.markdown(full_answer)

        except RagError as e:
            # 业务异常 —— 走友好错误提示
            answer_slot.error(friendly_message(e))
            full_answer = f"[Error] {e.user_message}"

        except Exception as e:
            # 未预期错误 —— 别崩,给个兜底提示
            answer_slot.error(friendly_message(e))
            full_answer = f"[System Error] {type(e).__name__}"

    # 2.4 把这轮助手回复存进历史
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_answer,
            "sources": sources_data,
        }
    )
