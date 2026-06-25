"""End-to-end orchestration: question in, grounded + guarded answer out."""
from __future__ import annotations

from . import generate as gen
from . import guardrails
from .config import Config, load_config
from .retrieve import Retriever


class SecureOpsPipeline:
    """Wires retrieval, generation, and guardrails into a single ``ask`` call."""

    def __init__(self, cfg: Config | None = None, retriever=None, client=None):
        self.cfg = cfg or load_config()
        self.retriever = retriever or Retriever(self.cfg)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = gen.get_client(self.cfg.google_api_key)
        return self._client

    def ask(self, question: str, k: int | None = None) -> dict:
        # 1. Input guardrail — flag (don't silently drop) suspicious queries.
        input_flags = guardrails.check_input(question, self.cfg.guardrails)

        # 2. Retrieve.
        hits = self.retriever.retrieve(question, k=k)

        # 3. Generate (context sanitised inside build_prompt).
        answer = gen.generate(question, hits, self.cfg, self.client,
                              sanitize=self.cfg.guardrails.sanitize_context)

        # 4. Output guardrails.
        cited, invalid = guardrails.verify_citations(answer, len(hits))
        refused = guardrails.is_refusal(answer)
        sources = [{"n": i, "source": m.get("source"), "page": m.get("page")}
                   for i, (_, m, _) in enumerate(hits, start=1)]

        return {
            "question": question,
            "answer": answer,
            "refused": refused,
            "citations": cited,
            "invalid_citations": invalid if self.cfg.guardrails.enforce_citations else [],
            "input_flags": input_flags,
            "sources": sources,
        }
