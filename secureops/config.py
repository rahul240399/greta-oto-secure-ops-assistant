"""Typed configuration loaded from config.yaml, with environment overrides.

Keeping every tunable in one validated object means experiments (chunk size,
embedding model, hybrid on/off, rerank on/off) are reproducible and diffable.
"""
from __future__ import annotations

import os
import pathlib
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class CorpusCfg(BaseModel):
    dir: str = "corpus"
    pdfs: dict[str, str] = Field(default_factory=dict)
    cisa_feed_url: str = "https://www.cisa.gov/cybersecurity-advisories/ics-advisories.xml"
    cisa_n_advisories: int = 50


class ChunkingCfg(BaseModel):
    strategy: str = "structure_aware"  # naive | structure_aware
    size: int = 1000
    overlap: int = 150


class IndexCfg(BaseModel):
    persist_dir: str = "./chroma_db"
    collection: str = "secureops"
    embed_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 256


class RetrievalCfg(BaseModel):
    top_k: int = 5
    candidate_k: int = 20
    hybrid: bool = True
    rrf_k: int = 60
    rerank: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class GenerationCfg(BaseModel):
    model: str = "gemini-2.5-flash"
    max_retries: int = 3


class GuardrailsCfg(BaseModel):
    max_question_chars: int = 2000
    sanitize_context: bool = True
    enforce_citations: bool = True


class Config(BaseModel):
    corpus: CorpusCfg = Field(default_factory=CorpusCfg)
    chunking: ChunkingCfg = Field(default_factory=ChunkingCfg)
    index: IndexCfg = Field(default_factory=IndexCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    generation: GenerationCfg = Field(default_factory=GenerationCfg)
    guardrails: GuardrailsCfg = Field(default_factory=GuardrailsCfg)
    # Secret: never written to YAML — read from the environment at load time.
    google_api_key: Optional[str] = None


def load_config(path: str | os.PathLike = "config.yaml") -> Config:
    """Load config.yaml (if present) and overlay GOOGLE_API_KEY from the env."""
    data: dict = {}
    p = pathlib.Path(path)
    if p.exists():
        data = yaml.safe_load(p.read_text()) or {}
    cfg = Config(**data)
    cfg.google_api_key = os.environ.get("GOOGLE_API_KEY", cfg.google_api_key)
    return cfg
