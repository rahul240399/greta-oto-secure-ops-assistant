"""Embedding + ChromaDB indexing.

Heavy deps (chromadb, sentence-transformers) are imported lazily so importing
this module is cheap and offline-safe.
"""
from __future__ import annotations

from .config import Config


def get_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def get_chroma_client(persist_dir: str):
    import chromadb

    return chromadb.PersistentClient(path=persist_dir)


def load_collection(cfg: Config):
    client = get_chroma_client(cfg.index.persist_dir)
    return client.get_or_create_collection(cfg.index.collection,
                                            metadata={"hnsw:space": "cosine"})


def build_index(chunks: list[str], metadatas: list[dict], cfg: Config,
                embedder=None, log=print):
    """Embed chunks in batches and add them to a persistent Chroma collection.

    Skips work if the collection already holds at least as many chunks.
    """
    collection = load_collection(cfg)
    if collection.count() >= len(chunks):
        log(f"index already built ({collection.count()} chunks)")
        return collection

    embedder = embedder or get_embedder(cfg.index.embed_model)
    batch = cfg.index.batch_size
    for i in range(0, len(chunks), batch):
        docs = chunks[i:i + batch]
        embs = embedder.encode(docs, show_progress_bar=False).tolist()
        collection.add(
            ids=[f"chunk-{j}" for j in range(i, i + len(docs))],
            documents=docs,
            embeddings=embs,
            metadatas=metadatas[i:i + batch],
        )
        log(f"indexed {min(i + batch, len(chunks))}/{len(chunks)}")
    log(f"index built: {collection.count()} chunks")
    return collection
