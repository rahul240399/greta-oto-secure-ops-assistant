"""Chunking strategies.

`naive` is the Tier-1 fixed-size character chunker (kept for A/B comparison).
`structure_aware` is the Tier-2 upgrade:
  - NIST pages -> sentence-aware packing (never split mid-sentence)
  - CISA advisories -> split on their consistent section headers, then pack

`gold_phrase`-based evaluation is content-based, so it stays comparable across
strategies — which is the whole point of keeping both.
"""
from __future__ import annotations

import re

from .config import ChunkingCfg

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Section headings that recur in CISA ICS advisories.
_ADVISORY_SECTIONS = [
    "executive summary", "risk evaluation", "technical details", "vulnerability overview",
    "affected products", "background", "mitigations", "summary",
]


def naive_chunk(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Fixed-size character chunking with overlap."""
    if size <= 0:
        raise ValueError("size must be > 0")
    step = max(1, size - overlap)
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks


def sentence_pack(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Pack whole sentences up to ~size chars, carrying ~overlap chars between chunks."""
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        if cur and len(cur) + 1 + len(s) > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = (tail + " " + s).strip() if tail else s
        else:
            cur = (cur + " " + s).strip() if cur else s
    if cur:
        chunks.append(cur)
    return chunks


def split_advisory_sections(text: str) -> list[tuple[str, str]]:
    """Split advisory text into (section_name, section_text) on known headers.

    Falls back to a single ('', text) when no headers are found.
    """
    low = text.lower()
    hits = []
    for name in _ADVISORY_SECTIONS:
        idx = low.find(name)
        if idx != -1:
            hits.append((idx, name))
    if not hits:
        return [("", text)]
    hits.sort()
    sections = []
    for i, (start, name) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        sections.append((name, text[start:end].strip()))
    return sections


def chunk_document(doc: dict, cfg: ChunkingCfg) -> list[tuple[str, dict]]:
    """Chunk one document unit -> list of (chunk_text, extra_metadata)."""
    text = doc["text"]
    if cfg.strategy == "naive":
        return [(p, {}) for p in naive_chunk(text, cfg.size, cfg.overlap)]

    # structure_aware
    if doc.get("doc_type") == "advisory":
        out: list[tuple[str, dict]] = []
        for section, sec_text in split_advisory_sections(text):
            for p in sentence_pack(sec_text, cfg.size, cfg.overlap):
                out.append((p, {"section": section} if section else {}))
        return out
    return [(p, {}) for p in sentence_pack(text, cfg.size, cfg.overlap)]


def chunk_documents(documents: list[dict], cfg: ChunkingCfg) -> tuple[list[str], list[dict]]:
    """Chunk every document, returning parallel (chunks, metadatas) lists."""
    chunks: list[str] = []
    metadatas: list[dict] = []
    for doc in documents:
        for piece, extra in chunk_document(doc, cfg):
            chunks.append(piece)
            md = {"source": doc["source"], "page": doc.get("page", 1),
                  "doc_type": doc.get("doc_type", "")}
            md.update(extra)
            metadatas.append(md)
    return chunks, metadatas
