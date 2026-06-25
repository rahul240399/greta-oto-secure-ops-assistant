"""Security guardrails for the RAG pipeline.

Three defended surfaces (see docs/SECURITY.md):
  1. Malicious user input  -> check_input() flags prompt-injection / oversized queries.
  2. Poisoned corpus       -> sanitize_context() neutralises instruction-like text
                              hidden in retrieved chunks (indirect prompt injection).
  3. Untrustworthy output  -> verify_citations() catches fabricated [n] references;
                              is_refusal() confirms the honest-refusal contract.

All functions are pure and unit-tested.
"""
from __future__ import annotations

import re

from .config import GuardrailsCfg

# Patterns that signal an attempt to override instructions or exfiltrate the prompt.
INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|prompts|rules)",
    r"disregard (all |any |the )?(previous|prior|above)",
    r"forget (all |everything |your )",
    r"reveal (your |the )?(system )?(prompt|instructions)",
    r"print (your |the )?(system )?prompt",
    r"you are now",
    r"act as (if|an|a)\b",
    r"developer mode",
    r"jailbreak",
    r"override (the |your )?(safety|security|guardrails)",
    r"exfiltrate",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

REFUSAL_MARK = "i don't have enough information"


def check_input(question: str, cfg: GuardrailsCfg) -> list[str]:
    """Return a list of issue tags for a user question ([] means clean)."""
    issues: list[str] = []
    if len(question) > cfg.max_question_chars:
        issues.append("too_long")
    for rx in _INJECTION_RE:
        if rx.search(question):
            issues.append(f"injection:{rx.pattern}")
    return issues


def sanitize_context(text: str) -> str:
    """Redact instruction-like lines from retrieved corpus text.

    Defends against indirect prompt injection: a poisoned document that contains
    "ignore previous instructions and ..." should not reach the LLM verbatim.
    """
    cleaned = []
    for line in text.splitlines() or [text]:
        if any(rx.search(line) for rx in _INJECTION_RE):
            cleaned.append("[redacted: instruction-like content removed]")
        else:
            cleaned.append(line)
    return "\n".join(cleaned)


def verify_citations(answer: str, n_blocks: int) -> tuple[list[int], list[int]]:
    """Return (cited_blocks, invalid_blocks). Invalid = cite a non-existent block."""
    cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    invalid = [c for c in cited if c < 1 or c > n_blocks]
    return cited, invalid


def is_refusal(answer: str) -> bool:
    return REFUSAL_MARK in answer.lower()
