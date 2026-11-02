"""Step 6 - Evaluation.
(a) Retrieval benchmark: compares retrieval strategies on a test set (Hit@1/3/5, MRR, speed).
(b) LLM judge: scores one answer for groundedness and relevance."""
import json
import re
import time

from . import config, llm
from .answer import multi_queries
from .retrieval import retrieve


def load_eval_set():
    path = config.DATA_DIR / "eval_set.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(text):
    return re.sub(r"\s+", " ", text.lower())


def _is_relevant(chunk, item):
    if item.get("source") and chunk["source"] != item["source"]:
        return False
    text = _norm(chunk["text"])
    return any(_norm(k) in text for k in item.get("keywords", []))


def _first_rank(chunks, item):
    for rank, c in enumerate(chunks, start=1):
        if _is_relevant(c, item):
            return rank
    return None


CONFIGS = [
    ("Dense only (baseline)", dict(hybrid=False, rerank=False, multi=False)),
    ("Hybrid: BM25 + vectors", dict(hybrid=True, rerank=False, multi=False)),
    ("Hybrid + Rerank", dict(hybrid=True, rerank=True, multi=False)),
]
FULL = ("Full: Multi-query + Hybrid + Rerank", dict(hybrid=True, rerank=True, multi=True))


def run_benchmark(kb, top_k=5):
    items = [i for i in load_eval_set() if not i.get("source") or i["source"] in kb.sources()]
    if not items:
        return {"error": "No test questions match the indexed documents. Add data/eval_set.json."}
    configs = list(CONFIGS) + ([FULL] if llm.available() else [])
    extra_queries = {}
    if llm.available():
        for item in items:
            try:
                extra_queries[item["question"]] = multi_queries(item["question"])
            except llm.LLMError:
                extra_queries[item["question"]] = []

    results, details = [], [{"question": i["question"], "source": i.get("source"), "ranks": []} for i in items]
    for name, cfg in configs:
        ranks, spent = [], 0.0
        for item, det in zip(items, details):
            queries = [item["question"]] + (extra_queries.get(item["question"], []) if cfg["multi"] else [])
            t = time.perf_counter()
            chunks, _ = retrieve(kb, queries, item["question"], top_k=top_k, candidates=20,
                                 hybrid=cfg["hybrid"], rerank=cfg["rerank"])
            spent += time.perf_counter() - t
            rank = _first_rank(chunks, item)
            ranks.append(rank)
            det["ranks"].append(rank)
        n = len(items)
        hit = lambda k: round(100 * sum(1 for r in ranks if r and r <= k) / n, 1)
        mrr = round(sum(1 / r for r in ranks if r) / n, 3)
        results.append({"name": name, "hit1": hit(1), "hit3": hit(3), "hit5": hit(5),
                        "mrr": mrr, "ms": round(1000 * spent / n)})
    return {"questions": len(items), "configs": results, "config_names": [c[0] for c in configs],
            "details": details}


def judge(question, answer, context):
    if not llm.available():
        raise llm.LLMError("Rating needs an LLM key (add GROQ_API_KEY).")
    prompt = f"""You are a strict evaluator of a RAG system.
Question: {question}
Context: {context[:4000]}
Answer: {answer}

Score from 1 to 5:
- groundedness: is every claim supported by the context?
- relevance: does the answer address the question?
Reply ONLY with JSON: {{"groundedness": n, "relevance": n, "comment": "one short sentence"}}"""
    out = llm.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=150)
    match = re.search(r"\{.*\}", out, re.S)
    try:
        return json.loads(match.group(0))
    except Exception:
        return {"groundedness": None, "relevance": None, "comment": out[:200]}
