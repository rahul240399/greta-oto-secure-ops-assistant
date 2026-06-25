"""Build the corpus and vector index from a clean clone.

    python -m scripts.build_index

Steps: download PDFs -> fetch CISA advisories -> parse -> chunk -> embed -> index.
"""
from __future__ import annotations

from secureops import chunk, index, ingest
from secureops.config import load_config


def main() -> None:
    cfg = load_config()
    print("== SecureOps index build ==")
    ingest.download_pdfs(cfg)
    advisories = ingest.fetch_cisa(cfg)
    print(f"corpus: {len(cfg.corpus.pdfs)} PDFs + {len(advisories)} advisories")

    documents = ingest.load_documents(cfg)
    chunks, metadatas = chunk.chunk_documents(documents, cfg.chunking)
    print(f"chunked into {len(chunks)} chunks (strategy={cfg.chunking.strategy})")

    collection = index.build_index(chunks, metadatas, cfg)
    print(f"done — {collection.count()} chunks indexed in {cfg.index.persist_dir}")


if __name__ == "__main__":
    main()
