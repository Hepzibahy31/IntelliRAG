---
title: IntelliRAG
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🧠 IntelliRAG
**Adaptive Knowledge Intelligence Platform using Advanced Retrieval-Augmented Generation (RAG) and LLMs**

Upload PDF/TXT documents, ask questions in plain language, and get answers with the source file, page number and a confidence score.
Back end: 100% Python. Front end: a single HTML/CSS/JavaScript page served by the same Python server.

## Run on your PC (Windows, VS Code terminal)
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
notepad .env          # paste a free Groq key (https://console.groq.com/keys) and save
python server.py
```
Open **http://localhost:8000**. The first start indexes the documents in `data/` (a few minutes for the 208-page HR policy; it is saved in `storage/`, so the next start is fast).
No key? It still works in *retrieval-only mode* (shows the best matching passage). A key switches on AI answers, query rewriting, multi-query and multi-hop.

Terminal version (same engine): `python main.py`

## What is implemented
| Level | Concept | File |
|---|---|---|
| Basic | Document loading, chunking, embeddings, FAISS vector DB, similarity search + Top-K | `rag/documents.py`, `chunker.py`, `models.py`, `vector_store.py` |
| Intermediate | Prompt template, LLM integration (Groq / OpenAI / Ollama), context-based generation | `rag/answer.py`, `llm.py` |
| Advanced | Query rewriting, multi-query, hybrid search (BM25 + vectors, RRF), reranking, metadata filter (file + page range), multi-hop | `rag/answer.py`, `retrieval.py`, `bm25.py` |
| Evaluation | Retrieval benchmark (Hit@1/3/5, MRR) comparing strategies, LLM answer judge | `rag/evaluation.py`, `data/eval_set.json` |
| Reliability | Citations, confidence score, "not found" fallback, saved index, rate limiting | `rag/answer.py`, `vector_store.py`, `server.py` |

**Not built:** Agentic RAG (listed as future work).
Features marked "needs LLM key" (rewriting, multi-query, multi-hop, answer writing, answer judge) are off until a key is set.

## Put your own questions in the benchmark
Edit `data/eval_set.json`: each item has a `question`, the `source` file and `keywords` that must appear in the correct passage.

## Host it online (Hugging Face Spaces, free)
1. Create an account at huggingface.co → **New Space** → name `IntelliRAG` → SDK **Docker** → Blank → **Public**.
2. Upload all files of this folder (Files tab → Add file → Upload files). Keep `README.md` as it is.
3. Space **Settings → Variables and secrets → New secret**: name `GROQ_API_KEY`, value = your key.
4. Wait for the build (about 10–15 minutes the first time). Your link: `https://<username>-intellirag.hf.space`.

Notes: uploaded files on a hosted Space are shared by everyone who opens the link and disappear when the Space restarts; the documents in `data/` are always available. Free Spaces sleep when unused, so open the link once before a demo.
