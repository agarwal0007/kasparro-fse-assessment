"""
PIPELINE VALIDATION RUN - NOT one of the four required live-LLM artifacts.

No OPENAI_API_KEY/ANTHROPIC_API_KEY/GEMINI_API_KEY was available in the build
environment for this submission (see README "Known limitation"). This script
proves the rest of the stack actually works end to end against the REAL
issued CSVs - agent loop shape, real (simulated) tool calling, strict output
validation, the deterministic scorer, SQLAlchemy persistence, and real
Langfuse trace ingestion - using a deterministic, honestly-labeled rule-based
stand-in instead of a live model. It extracts naive digit-bearing sentences
and labels every one "unknown" with an explicit placeholder reason; it is NOT
a quality claim-classifier and its scores are NOT meaningful evidence of
agent quality. Once a real API key is set, scripts/run_live.py produces the
four actual required artifacts using the identical agent/harness/scorer code
path (just a different provider plugged into the same interface).

Usage: python scripts/pipeline_validation_run.py [--limit N]
"""
import argparse
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agent.provider import FakeProvider, ModelTurnResult, ToolCallRequest
from app.db import init_db
from app.harness.orchestrator import run_split

SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+(?:\s+|$)")


def naive_sentences(text: str):
    """Yields (start, end) spans of trimmed sentences, offsets into the ORIGINAL
    string (trailing whitespace excluded from the span, per the quote-equality
    contract)."""
    for m in SENTENCE_RE.finditer(text):
        raw = m.group()
        stripped = raw.rstrip()
        if not stripped:
            continue
        start = m.start()
        end = start + len(stripped)
        yield start, end


def looks_factual(sentence: str) -> bool:
    return bool(re.search(r"\d", sentence)) or any(
        kw in sentence.lower() for kw in ("warranty", "price", "ships", "includes", "weighs", "model")
    )


def make_fake_script(answer):
    """One tool-call turn (real get_brand_facts request for the primary brand,
    exercising the actual FactExecutor/tool contract) then a final turn with
    naive digit/keyword sentence extraction, every claim labeled 'unknown'."""
    claims = []
    for start, end in naive_sentences(answer.ai_answer):
        sentence = answer.ai_answer[start:end]
        if looks_factual(sentence):
            claims.append({
                "start_char": start, "end_char": end, "quote": sentence,
                "label": "unknown", "fact_ids": [],
                "reason": "Pipeline-validation placeholder: rule-based extraction only, no real classification performed.",
            })
    tool_turn = ModelTurnResult(
        tool_calls=(ToolCallRequest(call_id="c1", name="get_brand_facts", arguments={"brand_id": answer.brand_id}),),
        final_content=None,
    )
    final_turn = ModelTurnResult(tool_calls=(), final_content=json.dumps({"claims": claims}))
    return FakeProvider([tool_turn, final_turn])


def _make_limit_filter(limit):
    if not limit:
        return None
    seen = {"n": 0}

    def _filter(a):
        seen["n"] += 1
        return seen["n"] <= limit

    return _filter


async def main(limit):
    init_db()

    for split in ("development", "check"):
        for agent_version in ("v1", "v2"):
            result = await run_split(
                split, agent_version,
                provider_override=make_fake_script,
                answer_filter=_make_limit_filter(limit),  # fresh filter/counter per run
            )
            print(f"[{split}/{agent_version}] run_id={result['run_id']} "
                  f"scheduled={result['scheduled_count']} ok={result['ok_count']} "
                  f"error={result['error_count']}")
            if result["metrics"] and result["metrics"].get("extraction"):
                m = result["metrics"]
                print(f"    G={m['G']} P={m['P']} M={m['M']} C={m['C']} "
                      f"extraction_f1={m['extraction']['f1']:.3f} joint_f1={m['joint']['f1']:.3f} "
                      f"macro_f1={m['macro_f1']:.3f}")
            with open(f"runs/pipeline_validation_{split}_{agent_version}.json", "w") as f:
                json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit answers per split (for a quick smoke test).")
    args = parser.parse_args()
    asyncio.run(main(args.limit))
