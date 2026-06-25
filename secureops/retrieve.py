"""Retrieval: dense + BM25 hybrid (Reciprocal Rank Fusion) + cross-encoder rerank.

Why each piece:
  - Dense (MiniLM) captures paraphrase / semantic similarity.
  - BM25 captures exact tokens dense models miss — CVE ids, product names, T-codes.
  - RRF fuses the two rank lists without tuning score scales.
  - A cross-encoder rerank then reorders the candidate pool by true relevance.

The ranking math (`reciprocal_rank_fusion`, `tokenize`) is pure and unit-tested;
the `Retriever` class is the thin integration layer over Chroma + rank_bm25.
"""
from __future__ import annotations

import re

from .config import Config

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer used for BM25."""
    return _TOKEN.findall(text.lower())


def reciprocal_rank_fusion(result_lists: list[list[str]], rrf_k: int = 60) -> dict[str, float]:
    """Fuse several ranked id-lists into a single id -> score map.

    score(id) = sum over lists of 1 / (rrf_k + rank), rank starting at 1.
    """
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, doc_id in enumerate(results, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return scores


class Retriever:
    """Loads a persisted index and answers queries with hybrid + rerank retrieval."""

    def __init__(self, cfg: Config, embedder=None, collection=None, reranker=None):
        self.cfg = cfg
        self._embedder = embedder
        self._collection = collection
        self._reranker = reranker
        self._bm25 = None
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._metas: list[dict] = []

    # -- lazy resource loading -------------------------------------------------
    def _ensure(self):
        from .index import get_embedder, load_collection

        if self._collection is None:
            self._collection = load_collection(self.cfg)
        if self._embedder is None:
            self._embedder = get_embedder(self.cfg.index.embed_model)
        if self.cfg.retrieval.hybrid and self._bm25 is None:
            data = self._collection.get()  # pull all docs to build the BM25 index
            self._ids = data["ids"]
            self._docs = data["documents"]
            self._metas = data["metadatas"]
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([tokenize(d) for d in self._docs])

    def _dense(self, question: str, n: int, where=None):
        q_emb = self._embedder.encode([question]).tolist()
        res = self._collection.query(query_embeddings=q_emb, n_results=n, where=where)
        ids = res["ids"][0]
        lookup = {i: (d, m) for i, d, m in zip(ids, res["documents"][0], res["metadatas"][0])}
        return ids, lookup

    def _bm25_order(self, question: str, n: int) -> list[str]:
        scores = self._bm25.get_scores(tokenize(question))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self._ids[i] for i in ranked]

    def _rerank(self, question, candidates):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.cfg.retrieval.reranker_model)
        pairs = [(question, doc) for _, doc, _ in candidates]
        scores = self._reranker.predict(pairs)
        order = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in order]

    # -- public API ------------------------------------------------------------
    def retrieve(self, question: str, k: int | None = None, where=None):
        """Return up to k results as (text, metadata, score) tuples."""
        self._ensure()
        rc = self.cfg.retrieval
        k = k or rc.top_k
        n = rc.candidate_k

        dense_ids, lookup = self._dense(question, n, where)

        if rc.hybrid:
            bm25_ids = self._bm25_order(question, n)
            fused = reciprocal_rank_fusion([dense_ids, bm25_ids], rc.rrf_k)
            order = sorted(fused, key=fused.get, reverse=True)
            # BM25-only ids won't be in the dense lookup — backfill from the full corpus.
            for cid in order:
                if cid not in lookup:
                    j = self._ids.index(cid)
                    lookup[cid] = (self._docs[j], self._metas[j])
        else:
            order = list(dense_ids)

        pool = order[: rc.candidate_k] if rc.rerank else order[:k]
        candidates = [(cid, lookup[cid][0], lookup[cid][1]) for cid in pool]
        if rc.rerank:
            candidates = self._rerank(question, candidates)
        return [(doc, meta, 0.0) for _, doc, meta in candidates[:k]]
