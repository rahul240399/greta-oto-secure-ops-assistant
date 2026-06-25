from secureops import retrieve


def test_tokenize():
    assert retrieve.tokenize("CVE-2023-1234, Rockwell!") == ["cve", "2023", "1234", "rockwell"]


def test_rrf_rewards_agreement():
    # 'b' is ranked highly by both lists, so it should win after fusion.
    dense = ["a", "b", "c"]
    bm25 = ["b", "d", "a"]
    scores = retrieve.reciprocal_rank_fusion([dense, bm25], rrf_k=60)
    order = sorted(scores, key=scores.get, reverse=True)
    assert order[0] == "b"
    # an id present in only one list still appears
    assert "c" in scores and "d" in scores


def test_rrf_score_formula():
    scores = retrieve.reciprocal_rank_fusion([["x"]], rrf_k=60)
    assert abs(scores["x"] - (1.0 / 61)) < 1e-9


class _FakeCollection:
    """Minimal Chroma-like stub for exercising Retriever without real deps."""

    def __init__(self, docs, metas):
        self._ids = [f"chunk-{i}" for i in range(len(docs))]
        self._docs = docs
        self._metas = metas

    def get(self):
        return {"ids": self._ids, "documents": self._docs, "metadatas": self._metas}

    def query(self, query_embeddings, n_results, where=None):
        # pretend the first n docs are the nearest neighbours
        n = min(n_results, len(self._ids))
        return {"ids": [self._ids[:n]], "documents": [self._docs[:n]],
                "metadatas": [self._metas[:n]]}


class _FakeEmbedder:
    def encode(self, texts, show_progress_bar=False):
        import numpy as np

        return np.zeros((len(texts), 1))  # flat scores -> BM25 decides the order


def test_bm25_ranks_exact_token_match_first():
    """BM25 is the half of hybrid retrieval that catches exact ids dense models miss."""
    from rank_bm25 import BM25Okapi

    docs = [
        "General guidance about operational technology security and networks.",
        "Advisory ICSA-24-001 affects the Rockwell CompactLogix controller, CVE-2024-9999.",
        "Background notes on industrial protocols and segmentation.",
    ]
    bm25 = BM25Okapi([retrieve.tokenize(d) for d in docs])
    scores = bm25.get_scores(retrieve.tokenize("CVE-2024-9999 Rockwell CompactLogix"))
    assert scores.argmax() == 1  # the exact-token advisory wins


def test_retriever_dense_only_wiring():
    """With hybrid+rerank off, retrieve() returns the dense neighbours, top-k honoured."""
    from secureops.config import Config

    docs = ["doc zero text", "doc one text", "doc two text"]
    metas = [{"source": f"s{i}", "page": 1} for i in range(3)]

    cfg = Config()
    cfg.retrieval.hybrid = False
    cfg.retrieval.rerank = False
    cfg.retrieval.candidate_k = 3

    r = retrieve.Retriever(cfg, embedder=_FakeEmbedder(),
                           collection=_FakeCollection(docs, metas))
    hits = r.retrieve("anything", k=2)
    assert len(hits) == 2
    assert hits[0][0] == "doc zero text"
    assert hits[0][1]["source"] == "s0"
