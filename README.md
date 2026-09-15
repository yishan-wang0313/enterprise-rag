# Enterprise RAG + Agent — Hybrid Retrieval, CrossEncoder Rerank, LangGraph Tools

An end-to-end **Retrieval-Augmented Generation** system built from scratch to production-shape:
hybrid retrieval (vector + BM25), CrossEncoder rerank, streaming generation with source citations,
multi-turn query rewriting, a **LangGraph ReAct agent** that routes between the RAG search and
other tools, and a Streamlit UI.

Built as a learning-by-shipping project. Every stage is exposed and pluggable — no framework magic.

---

## Live Demo

**👉 Try it here: [yishan-enterprise-rag.streamlit.app](https://yishan-enterprise-rag.streamlit.app)**

Upload any `.docx` / `.xlsx` / `.txt` / `.md` from the sidebar, then ask a question.
Responds in English or Chinese to match your query language.

> ℹ️ The live demo runs in "lite mode" (`ENABLE_RERANK=false`) to fit Streamlit Cloud's
> memory tier — vector + BM25 + RRF is enabled, but the CrossEncoder rerank step is
> skipped. Run locally to see the full pipeline.

## Screenshot

<!-- Add a screenshot of the UI at docs/screenshot.png after taking one -->
<!-- ![Screenshot](docs/screenshot.png) -->

---

## Architecture

```
                ┌──────────────┐
uploaded file ──▶│    loader    │──▶ LoadedDocument {content, metadata}
                └──────────────┘         (docx / xlsx / txt / md)
                        │
                        ▼
                ┌──────────────┐
                │   chunker    │──▶ Recursive splitter (paragraph → line → punctuation → char)
                └──────────────┘         upserts into ChromaDB (langchain vectorstore)
                        │
                        ▼
                ┌──────────────────────────────────────────┐
     query ───▶ │             retriever                     │
                │  ┌───────────────┐   ┌──────────────┐    │
                │  │ Vector (Chroma│   │  BM25 (rank_ │    │
                │  │  + OpenAI     │   │  bm25 in-mem │    │
                │  │  embedding)   │   │  index)      │    │
                │  └───────┬───────┘   └──────┬───────┘    │
                │          └──────┬───────────┘             │
                │                 ▼                         │
                │        RRF fusion (k=60)                  │
                │                 ▼                         │
                │  CrossEncoder rerank                      │
                │  (BAAI/bge-reranker-base, local)          │
                │                 ▼                         │
                │        top RERANK_TOP_K results           │
                └────────────────┬─────────────────────────┘
                                 ▼
                        ┌────────────────┐
                        │   rewriter     │  ← multi-turn query rewriting
                        │  (resolves     │    (LLM, temp=0, cheap)
                        │   pronouns)    │
                        └────────┬───────┘
                                 ▼
                        ┌────────────────┐
                        │   generator    │  ← GPT-4o-mini, streaming
                        │  (system       │    grounding constraints,
                        │   prompt +     │    [1][2] citation markers
                        │   sources)     │
                        └────────┬───────┘
                                 ▼
                          streamed answer
                          + source list
                                 ▼
                        ┌────────────────┐
                        │  Streamlit UI  │
                        └────────────────┘
```

---

## Agent layer (LangGraph ReAct)

Beyond pure RAG, the system includes a **ReAct-style agent** ([`core/agent.py`](core/agent.py))
that lets the LLM decide, per turn, which tool to invoke:

- `search_knowledge_base(query)` — the full hybrid RAG pipeline above
- `calculator(expression)` — AST-based safe arithmetic (rejects any name/call node — no `eval` risk)
- `current_time()` — UTC timestamp

The graph is a standard three-node loop:

```
START → agent_node → conditional
                     ├─ tool_calls present → tool_node → agent_node (loop)
                     └─ final answer       → END
```

State uses LangGraph's `add_messages` reducer; conversation memory is persisted via
`InMemorySaver` keyed on `thread_id`. `stream_agent()` yields structured events
(`tool_call` / `tool_result` / `answer`) suitable for driving a live UI.

**Example: self-correcting error recovery in action**

```
❓ Query: What time is it right now, and how many hours until midnight UTC?
🔧 tool_call: current_time({})
🔧 tool_call: calculator({'expression': '24 - current_hour'})
📤 tool_result [current_time]: 2026-09-15 06:32:27 UTC
📤 tool_result [calculator]: Error: Unsupported expression node: Name
🔧 tool_call: current_time({})                  ← agent re-plans after seeing the error
📤 tool_result [current_time]: 2026-09-15 06:32:30 UTC
🔧 tool_call: calculator({'expression': '24 - 6'})  ← now with a literal number
📤 tool_result [calculator]: Result: 18
💬 Answer: The current time is 06:32 UTC. There are 18 hours until midnight UTC.
```

Try it: `python core/agent.py "your question"`.

## Why this design

| Choice | Reason |
|---|---|
| **Hybrid retrieval** (vector + BM25) | Vector catches semantics; BM25 nails exact terms like model names or IDs. Neither alone is enough. |
| **RRF fusion** | Merges two ranked lists using *rank only*, so vector-distance and BM25 scores don't need normalization. Simple, robust, industry-standard. |
| **CrossEncoder rerank** | Bi-encoder embeddings never let query and doc "meet". A cross-encoder joint-scores them with full attention, catching answering-quality that pure embedding retrieval misses. |
| **Query rewriting** | Users ask follow-ups with pronouns ("What are its downsides?"). Rewriter uses chat history + a cheap LLM call to make each query self-contained. |
| **Streaming + citations** | Time-to-first-token matters for UX. Citations make answers verifiable. |
| **Domain exceptions** | `RagError` hierarchy separates user-facing messages from raw tracebacks. Empty-KB / config / retrieval / generation failures each render friendly UI. |
| **Persistent Chroma + stable IDs** | `{filename}#{chunk_index}` IDs make re-imports upsert (no duplicates), and support metadata-filtered deletion. |
| **ReAct agent over RAG** | The LLM decides *when* to search vs answer directly, chains tool calls, and self-corrects on tool errors — a much richer UX than always-search RAG. |
| **AST-based calculator (no `eval`)** | Sandboxing user math via `ast.parse` + a whitelist of `BinOp` / `UnaryOp` nodes; any `Name` / `Call` / `Attribute` node is rejected before evaluation. Safe to expose to arbitrary LLM output. |

---

## Tech Stack

- **Language**: Python 3.11+
- **LLM**: OpenAI GPT-4o-mini (chat) + text-embedding-3-small (embeddings)
- **Vector DB**: ChromaDB (persistent, local)
- **Retrieval**: langchain `Chroma` + `rank_bm25` + custom RRF
- **Rerank**: sentence-transformers CrossEncoder (`BAAI/bge-reranker-base`)
- **Agent**: LangGraph 1.x `StateGraph` + `ToolNode` + `InMemorySaver`
- **Chunker**: langchain `RecursiveCharacterTextSplitter`
- **UI**: Streamlit
- **Doc parsing**: python-docx, openpyxl, markdown

---

## Getting Started

### 1. Prerequisites
- Python 3.11+ (tested on 3.14)
- An OpenAI API key

### 2. Install

```bash
git clone https://github.com/<your-username>/enterprise-rag.git
cd enterprise-rag
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

### 4. Run the UI

```bash
python -m streamlit run app.py
```

Open http://localhost:8501, upload a document from the sidebar, and start asking questions.

### 5. Or use the CLI

```bash
# Index a document into the KB
python core/chunker.py path/to/your/file.docx

# Query
python core/retriever.py "your question"

# End-to-end (retrieval + generation)
python core/generator.py "your question"
```

---

## Project Layout

```
enterprise_rag/
├── app.py                       # Streamlit UI (upload / chat / delete)
├── config.py                    # Central config (chunk_size, top_k, models)
├── .streamlit/config.toml       # Streamlit runtime settings
│
├── core/
│   ├── loader.py                # Unified LoadedDocument for 4 formats
│   ├── chunker.py               # Recursive splitter + Chroma upsert
│   ├── retriever.py             # Vector + BM25 + RRF + CrossEncoder
│   ├── generator.py             # Streaming GPT with grounding + citations
│   ├── rewriter.py              # Multi-turn query rewriting
│   ├── agent.py                 # LangGraph ReAct agent + safe tools
│   └── exceptions.py            # Domain exception hierarchy
│
├── database/chroma_db/          # Persistent vector store (gitignored)
└── uploads/                     # User-uploaded files (gitignored)
```

---

## Configuration

All tunables live in [`config.py`](config.py):

| Setting | Default | Notes |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk; upper bound |
| `CHUNK_OVERLAP` | 50 | Overlap between adjacent chunks |
| `TOP_K` | 20 | Candidates recalled per retrieval path |
| `RERANK_TOP_K` | 4 | Final results sent to the LLM |
| `EMBEDDING_MODEL` | text-embedding-3-small | 1536-dim |
| `CHAT_MODEL` | gpt-4o-mini | |
| `ENABLE_RERANK` env var | `true` | Set to `false` to skip loading the 1.1GB CrossEncoder (lite mode for constrained cloud tiers) |

---

## What's inside the folder that isn't in the docs

- **`_handwritten` counterparts** — every module (`chunker_handwritten.py`, `retriever_handwritten.py`, `generator_handwritten.py`) has a from-scratch version that predates the langchain migration. Kept for pedagogical comparison; not on the runtime path.
- **`compare_chunkers.py`** — side-by-side diff of hand-written fixed-size chunking vs. langchain recursive chunking on the same input.
- **`test_*.py`** — smoke tests for each stage of the pipeline.

---

## Known Limitations & Next Steps

- [ ] BM25 is an in-memory snapshot rebuilt on invalidation; not suitable beyond ~100k chunks. Swap to Elasticsearch/OpenSearch for scale.
- [ ] No evaluation harness yet — a QA test set with automated scoring is the top priority.
- [ ] No embedding cache — repeated identical text is re-embedded.
- [ ] No metadata filtering exposed in the UI (e.g. "only search Word files").
- [ ] Single-user only. No concurrent-user isolation.

---

## License

MIT
