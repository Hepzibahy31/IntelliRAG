"""Embedding model + reranker (loaded once, on first use)."""
import threading
import numpy as np

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_load_lock = threading.Lock()
_infer_lock = threading.Lock()
_embedder = None
_reranker = None


def get_embedder():
    global _embedder
    with _load_lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(EMBED_MODEL)
        return _embedder


def get_reranker():
    global _reranker
    with _load_lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(RERANK_MODEL)
        return _reranker


def embed_texts(texts, batch_size=64):
    model = get_embedder()
    with _infer_lock:
        vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=len(texts) > 300)
    return np.asarray(vectors, dtype="float32")


def embed_query(query: str):
    return embed_texts([query])[0]


def rerank_scores(question, passages):
    model = get_reranker()
    with _infer_lock:
        scores = model.predict([(question, p) for p in passages], batch_size=32)
    return [float(s) for s in scores]
