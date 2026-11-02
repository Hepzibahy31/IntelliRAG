"""Step 3 - Embeddings + FAISS vector index + BM25 keyword index + metadata.

The knowledge base is saved to disk (storage/) so restarts are fast.
"""
import json
import threading
from pathlib import Path

import numpy as np

from . import config
from .bm25 import BM25
from .chunker import create_chunks
from .documents import list_files, load_file
from .models import embed_texts


class KnowledgeBase:
    def __init__(self, storage_dir=None):
        self.dir = Path(storage_dir or config.STORAGE_DIR)
        self.lock = threading.RLock()
        self.chunks = []
        self.emb = None            # float32 matrix (n, dim)
        self.index = None          # FAISS inner-product index (cosine, vectors are normalised)
        self.bm25 = None

    # ---------- info ----------
    def size(self):
        return len(self.chunks)

    def sources(self):
        return {c["source"] for c in self.chunks}

    def documents(self):
        info = {}
        for c in self.chunks:
            d = info.setdefault(c["source"], {"name": c["source"], "chunks": 0, "pages": set()})
            d["chunks"] += 1
            if c["page"] is not None:
                d["pages"].add(c["page"])
        return [{"name": d["name"], "chunks": d["chunks"], "pages": max(d["pages"]) if d["pages"] else None}
                for d in sorted(info.values(), key=lambda x: x["name"].lower())]

    # ---------- build ----------
    def _rebuild(self):
        import faiss
        if not self.chunks:
            self.emb, self.index, self.bm25 = None, None, None
            return
        self.index = faiss.IndexFlatIP(self.emb.shape[1])
        self.index.add(self.emb)
        self.bm25 = BM25([c["text"] for c in self.chunks])

    def add_documents(self, docs):
        """docs: list of {text, source, page}. Returns number of new chunks."""
        with self.lock:
            existing = self.sources()
        docs = [d for d in docs if d["source"] not in existing]
        new = create_chunks(docs)
        if not new:
            return 0
        vectors = embed_texts([c["text"] for c in new])          # heavy part: outside the lock
        with self.lock:
            self.chunks.extend(new)
            self.emb = vectors if self.emb is None else np.vstack([self.emb, vectors])
            self._rebuild()
            self.save()
        return len(new)

    def remove_document(self, name):
        with self.lock:
            keep = [i for i, c in enumerate(self.chunks) if c["source"] != name]
            if len(keep) == len(self.chunks):
                return False
            self.chunks = [self.chunks[i] for i in keep]
            self.emb = self.emb[keep] if keep else None
            self._rebuild()
            self.save()
            return True

    # ---------- persistence ----------
    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")
        if self.emb is not None:
            np.save(self.dir / "embeddings.npy", self.emb)
        elif (self.dir / "embeddings.npy").exists():
            (self.dir / "embeddings.npy").unlink()

    def load(self):
        cj, en = self.dir / "chunks.json", self.dir / "embeddings.npy"
        if not (cj.exists() and en.exists()):
            return False
        try:
            chunks = json.loads(cj.read_text(encoding="utf-8"))
            emb = np.load(en).astype("float32")
            if len(chunks) != len(emb):
                return False
            with self.lock:
                self.chunks, self.emb = chunks, emb
                self._rebuild()
            return True
        except Exception as error:
            print("Could not load saved index:", error)
            return False

    # ---------- search ----------
    @staticmethod
    def _allowed(chunk, flt):
        if not flt:
            return True
        if flt.get("sources") and chunk["source"] not in flt["sources"]:
            return False
        page = chunk["page"]
        if page is not None:
            if flt.get("page_min") is not None and page < flt["page_min"]:
                return False
            if flt.get("page_max") is not None and page > flt["page_max"]:
                return False
        return True

    def dense_search(self, query_vec, k, flt=None):
        """Semantic search with FAISS. Returns [(chunk_index, cosine)]."""
        with self.lock:
            if self.index is None:
                return []
            n = len(self.chunks)
            search_k = n if flt else min(k, n)
            scores, ids = self.index.search(np.asarray([query_vec], dtype="float32"), search_k)
            out = []
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0 or not self._allowed(self.chunks[int(idx)], flt):
                    continue
                out.append((int(idx), float(score)))
                if len(out) >= k:
                    break
            return out

    def keyword_search(self, query, k, flt=None):
        """BM25 keyword search. Returns [(chunk_index, score)]."""
        with self.lock:
            if self.bm25 is None:
                return []
            scores = self.bm25.scores(query)
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            out = []
            for i in order:
                if scores[i] <= 0:
                    break
                if not self._allowed(self.chunks[i], flt):
                    continue
                out.append((i, scores[i]))
                if len(out) >= k:
                    break
            return out


def ingest_folder(kb: KnowledgeBase, folder=None, log=print):
    """Index every PDF/TXT/MD file in the data folder that is not indexed yet."""
    total = 0
    for path in list_files(folder):
        if path.name in kb.sources():
            continue
        try:
            log(f"Indexing {path.name} ...")
            total += kb.add_documents(load_file(path))
        except Exception as error:
            log(f"Could not index {path.name}: {error}")
    return total
