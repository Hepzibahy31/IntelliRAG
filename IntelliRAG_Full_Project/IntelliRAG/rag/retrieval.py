"""Step 4 - Advanced retrieval: multi-query + hybrid (BM25 + FAISS, fused with RRF)
+ metadata filter + cross-encoder reranking."""
import math

from .models import embed_query, rerank_scores


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0)))


def rrf(rankings, k=60):
    """Reciprocal Rank Fusion: merges several ranked lists into one score per chunk."""
    scores = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return scores


def retrieve(kb, queries, question, top_k=5, candidates=20, hybrid=True, rerank=True, flt=None):
    """queries[0] must be the (standalone) question itself.
    Returns (results, info). Each result is a chunk dict plus similarity / rerank / confidence."""
    info = {"rerank_used": False}
    with kb.lock:
        if kb.size() == 0:
            return [], info
        q_vecs = [embed_query(q) for q in queries]
        rankings = []
        for q, vec in zip(queries, q_vecs):
            rankings.append([i for i, _ in kb.dense_search(vec, candidates, flt)])
            if hybrid:
                rankings.append([i for i, _ in kb.keyword_search(q, candidates, flt)])
        fused = rrf([r for r in rankings if r])
        order = sorted(fused, key=fused.get, reverse=True)[:candidates]
        if not order:
            return [], info
        base = q_vecs[0]
        results = []
        for idx in order:
            chunk = dict(kb.chunks[idx])
            chunk["similarity"] = float(kb.emb[idx] @ base)
            results.append(chunk)

    if rerank:
        try:
            scores = rerank_scores(question, [c["text"] for c in results])
            for c, s in zip(results, scores):
                c["rerank"] = s
            results.sort(key=lambda c: c["rerank"], reverse=True)
            info["rerank_used"] = True
        except Exception as error:          # e.g. model could not be downloaded
            info["rerank_error"] = str(error)[:150]

    for c in results:
        c["confidence"] = sigmoid(c["rerank"]) if info["rerank_used"] else max(0.0, c["similarity"])
    return results[:top_k], info
