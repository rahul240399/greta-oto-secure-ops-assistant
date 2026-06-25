"""Evaluate the pipeline against eval/eval_set.json.

    python -m scripts.evaluate

Metrics: retrieval hit@k + MRR (content-phrase match), LLM-as-judge groundedness,
and honest-refusal accuracy on the unanswerable set. Prints a per-question table
and an aggregate summary.
"""
from __future__ import annotations

import json
import pathlib
import re
import time

from secureops import guardrails
from secureops.config import load_config
from secureops.pipeline import SecureOpsPipeline

EVAL_PATH = pathlib.Path("eval/eval_set.json")


def _as_list(x):
    return x if isinstance(x, list) else [x]


def retrieval_metrics(pipeline, item, k):
    hits = pipeline.retriever.retrieve(item["question"], k=k)
    phrases = [p.lower() for p in _as_list(item["gold_phrase"])]
    for rank, (text, _, _) in enumerate(hits, start=1):
        if any(p in text.lower() for p in phrases):
            return 1, 1.0 / rank, hits
    return 0, 0.0, hits


def _parse_json(raw):
    raw = re.sub(r"```(json)?", "", raw).strip()
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def judge_grounded(pipeline, question, answer, context):
    prompt = f"""You are a strict evaluator. Decide if the ANSWER is fully supported by the CONTEXT.
Output ONLY JSON: {{"grounded": 1 or 0, "reason": "<one short sentence>"}}
Rules:
- grounded=1 only if every factual claim in the ANSWER appears in, or follows from, the CONTEXT.
- grounded=0 if any claim is unsupported or contradicts the CONTEXT.
- A correct refusal ("I don't have enough information...") counts as grounded=1.

CONTEXT:
{context}

QUESTION: {question}
ANSWER: {answer}
"""
    for attempt in range(3):
        try:
            raw = pipeline.client.models.generate_content(
                model=pipeline.cfg.generation.model, contents=prompt).text.strip()
            res = _parse_json(raw)
            return int(res.get("grounded", 0)), res.get("reason", "")
        except Exception:
            time.sleep(15 * (attempt + 1))
    return 0, "judge failed"


def main(judge: bool = True, pause: float = 7.0) -> None:
    cfg = load_config()
    pipeline = SecureOpsPipeline(cfg)
    items = json.loads(EVAL_PATH.read_text())
    k = cfg.retrieval.top_k

    rows = []
    for item in items:
        result = pipeline.ask(item["question"])
        answer = result["answer"]

        if item["type"] == "unanswerable":
            honest = int(guardrails.is_refusal(answer))
            rows.append({"id": item["id"], "type": "unanswerable", "hit": None,
                         "mrr": None, "grounded": honest, "honest": honest,
                         "note": "refused" if honest else "FAILED TO REFUSE"})
        else:
            hit, mrr, hits = retrieval_metrics(pipeline, item, k)
            grounded, reason = (judge_grounded(
                pipeline, item["question"], answer,
                "\n\n".join(t for t, _, _ in hits)) if judge else (None, ""))
            rows.append({"id": item["id"], "type": "answerable", "hit": hit,
                         "mrr": round(mrr, 3), "grounded": grounded, "honest": None,
                         "note": (reason or "")[:60]})
        time.sleep(pause)

    # Per-question table
    print(f"\n{'id':4} {'type':13} {'hit':4} {'mrr':5} {'grnd':5} {'honest':6} note")
    for r in rows:
        print(f"{r['id']:4} {r['type']:13} {str(r['hit']):4} {str(r['mrr']):5} "
              f"{str(r['grounded']):5} {str(r['honest']):6} {r['note']}")

    # Summary
    ans = [r for r in rows if r["type"] == "answerable"]
    un = [r for r in rows if r["type"] == "unanswerable"]
    print("\n=== SecureOps evaluation ===")
    print(f"answerable items   : {len(ans)}")
    print(f"  hit@{k}            : {sum(r['hit'] for r in ans)/len(ans):.2f}")
    print(f"  MRR              : {sum(r['mrr'] for r in ans)/len(ans):.2f}")
    if judge:
        print(f"  groundedness     : {sum(r['grounded'] for r in ans)/len(ans):.2f}")
    print(f"unanswerable items : {len(un)}")
    if un:
        print(f"  refusal accuracy : {sum(r['honest'] for r in un)/len(un):.2f}")


if __name__ == "__main__":
    main()
