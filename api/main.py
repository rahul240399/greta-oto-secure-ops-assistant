"""FastAPI backend for SecureOps Assistant.

Run:  uvicorn api.main:app --reload
Then: curl -s localhost:8000/ask -H 'content-type: application/json' \
           -d '{"question": "What does NIST recommend for remote access to OT?"}'

The pipeline is built lazily on first request (so importing this module — e.g.
in tests — needs no index or API key). Tests override `get_pipeline`.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    k: Optional[int] = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    n: int
    source: Optional[str] = None
    page: Optional[int] = None


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    citations: list[int]
    invalid_citations: list[int]
    input_flags: list[str]
    sources: list[Source]


@lru_cache
def get_pipeline():
    """Build the pipeline once and cache it. Overridden in tests."""
    from secureops.pipeline import SecureOpsPipeline

    return SecureOpsPipeline()


app = FastAPI(title="SecureOps Assistant API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline=Depends(get_pipeline)):
    try:
        return pipeline.ask(req.question, k=req.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pipeline error: {e}")
