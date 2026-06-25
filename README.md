# SecureOps Assistant

A grounded RAG system that helps security analysts navigate industrial (OT/ICS)
cybersecurity guidance — **NIST SP 800-82 Rev. 3**, **NIST CSF 2.0**, and **CISA
ICS advisories**. Every answer is cited to its source, and the assistant refuses
to guess when the corpus can't support an answer. Built for the WMG Applied AI
Hackathon 2026.

## What's here

Two ways to use the same pipeline:

- **`SecureOps_Assistant.ipynb`** — the narrative notebook (Steps 0–6): build the
  index, retrieve, generate, and run the evaluation. Good for the demo/walkthrough.
- **`secureops/` package + `api/`** — the productionised version: modular,
  configurable, unit-tested, and served behind a FastAPI backend.

## Architecture

```
ingest  ─► chunk ─► index (ChromaDB)        scripts/build_index.py
                              │
question ─► guardrails.check_input
         ─► retrieve  (dense + BM25 → RRF → cross-encoder rerank)
         ─► generate  (Gemini; context sanitised; cite-or-refuse)
         ─► guardrails.verify_citations ─► answer + sources
```

| Module | Responsibility |
|---|---|
| `secureops/config.py` | Typed config from `config.yaml` + env (`GOOGLE_API_KEY`) |
| `secureops/ingest.py` | Download NIST PDFs + CISA advisories, parse to document units |
| `secureops/chunk.py` | `naive` and `structure_aware` chunking (sentence/section-aware) |
| `secureops/index.py` | Embed (`all-MiniLM-L6-v2`) + persist in ChromaDB |
| `secureops/retrieve.py` | Hybrid dense+BM25 (RRF) retrieval with cross-encoder rerank |
| `secureops/generate.py` | Prompt assembly + grounded Gemini generation |
| `secureops/guardrails.py` | Input checks, context sanitisation, citation verification |
| `secureops/pipeline.py` | `SecureOpsPipeline.ask()` — the orchestrator |
| `api/main.py` | FastAPI: `POST /ask`, `GET /health` |

## Tier-2 upgrades over the starter prototype

- **Hybrid retrieval** — BM25 fused with dense vectors via Reciprocal Rank Fusion,
  so exact identifiers (CVEs, product names) aren't lost to pure semantic search.
- **Cross-encoder reranking** — reorders the candidate pool by true relevance.
- **Structure-aware chunking** — sentence-aware for NIST pages, section-aware for
  advisories, instead of blindly slicing every 1000 characters.
- **Security guardrails** — see [`docs/SECURITY.md`](docs/SECURITY.md).
- **Systematic evaluation** — 18-item test set with retrieval hit@k + MRR,
  LLM-as-judge groundedness, and honest-refusal accuracy.

All behaviour is driven by [`config.yaml`](config.yaml), so A/B experiments
(naive vs structure-aware, hybrid on/off, rerank on/off) are one-line changes.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then add your Gemini key (https://aistudio.google.com/apikey)
```

## Usage

```bash
# 1. Build the corpus + vector index (downloads PDFs + CISA advisories)
python -m scripts.build_index

# 2. Serve the API
uvicorn api.main:app --reload
curl -s localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "What does NIST recommend for remote access to OT networks?"}'

# 3. Evaluate
python -m scripts.evaluate
```

## Tests

Offline unit tests (no network, no API key, no model downloads):

```bash
pytest
```

They cover chunking, RRF/BM25 ranking, guardrails, config loading, and the API
contract (with a stubbed pipeline).

## Security

The mandatory AI Security Reflection — attack surfaces and the defenses
implemented here — is in [`docs/SECURITY.md`](docs/SECURITY.md).
