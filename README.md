# Enterprise RAG — Hybrid Retrieval + CrossEncoder Rerank

An end-to-end **Retrieval-Augmented Generation** system built from scratch to production-shape:
hybrid retrieval (vector + BM25), CrossEncoder rerank, streaming generation with source citations,
multi-turn query rewriting, and a Streamlit UI.

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

---

## Tech Stack

- **Language**: Python 3.11+
- **LLM**: OpenAI GPT-4o-mini (chat) + text-embedding-3-small (embeddings)
- **Vector DB**: ChromaDB (persistent, local)
- **Retrieval**: langchain `Chroma` + `rank_bm25` + custom RRF
- **Rerank**: sentence-transformers CrossEncoder (`BAAI/bge-reranker-base`)
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
