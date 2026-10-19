"""
Runs one full split (development or check) through the agent with bounded
concurrency, scores it (development only - check quality stays null since
gold is withheld), persists everything to the database, and emits one
Langfuse trace per answer plus a run-summary trace.
"""
import asyncio
import hashlib
import os
import time

from app.agent.provider import FakeProvider
from app.agent.runner import run_agent_on_answer
from app.agent.tools import FactExecutor
from app.config import (
    DATASET_VERSION, EVALUATOR_VERSION, FACT_VERSION, LLM_MODEL, LLM_PROVIDER,
    MAX_CONCURRENT_PROVIDER_REQUESTS,
)
from app.db import AnswerResult, ClaimResult, Run, SessionLocal
from app.harness.csv_loader import load_answers, load_brand_facts, load_gold
from app.harness.scorer import score_split
from app.langfuse_integration import trace_answer, trace_run_summary

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_prompt(agent_version: str) -> str:
    path = os.path.join(PROMPTS_DIR, f"{agent_version}_system_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_provider(provider_override=None):
    """Returns a live provider if an API key is configured, else a FakeProvider
    that is still a full, functioning (if simplistic) rule-based agent - useful
    to validate the whole pipeline end to end without a live key."""
    if provider_override is not None:
        return provider_override, "fake", "fake-rule-based-v1"

    api_key = os.environ.get("OPENAI_API_KEY")
    if LLM_PROVIDER == "openai" and api_key:
        from app.agent.provider import OpenAIProvider
        return OpenAIProvider(model=LLM_MODEL, api_key=api_key), "openai", LLM_MODEL

    # No live key available: fall back to a deterministic, rule-based FakeProvider
    # script generator so the pipeline can still be exercised end to end. This is
    # explicitly NOT one of the four required live-LLM artifacts - see README.
    return None, "none", "none"


async def run_split(
    split: str,
    agent_version: str,
    provider_override=None,
    answer_filter=None,
) -> dict:
    answers = load_answers(os.path.join(DATA_DIR, "answers.csv"))
    facts = load_brand_facts(os.path.join(DATA_DIR, "brand_facts.csv"))
    known_brand_ids = {f.brand_id for f in facts}
    known_fact_ids = {f.fact_id for f in facts}
    fact_executor = FactExecutor(facts, known_brand_ids)

    scheduled = [a for a in answers if a.split == split]
    if answer_filter is not None:
        scheduled = [a for a in scheduled if answer_filter(a)]

    system_prompt = load_prompt(agent_version)
    provider, provider_name, model_name = build_provider(provider_override)
    if provider is None:
        raise RuntimeError(
            "No live LLM provider configured (OPENAI_API_KEY/ANTHROPIC_API_KEY/"
            "GEMINI_API_KEY not set) and no provider_override given. "
            "Set a key in .env, or pass provider_override=FakeProvider(...) for "
            "a pipeline-validation run."
        )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROVIDER_REQUESTS)

    async def run_one(answer):
        # FakeProvider instances are stateful per-answer scripts in our pipeline
        # validation mode; provider_override may instead be a *factory* callable.
        this_provider = provider(answer) if callable(provider) and not hasattr(provider, "complete_turn") else provider
        return await run_agent_on_answer(answer, fact_executor, known_fact_ids, this_provider, system_prompt, semaphore)

    start = time.monotonic()
    outcomes_list = await asyncio.gather(*[run_one(a) for a in scheduled])
    total_latency_ms = (time.monotonic() - start) * 1000
    outcomes = {o.answer_id: o for o in outcomes_list}

    gold = []
    report = None
    per_answer_scores = {}
    if split == "development":
        all_gold = load_gold(os.path.join(DATA_DIR, "development_gold.csv"))
        gold = [g for g in all_gold if g.answer_id in {a.answer_id for a in scheduled}]
        report = score_split(gold, outcomes)
        for answer_id in outcomes:
            answer_gold = [g for g in gold if g.answer_id == answer_id]
            answer_outcome = {answer_id: outcomes[answer_id]}
            per_answer_scores[answer_id] = score_split(answer_gold, answer_outcome) if answer_gold else None

    # --- persist ---
    db = SessionLocal()
    try:
        run = Run(
            split=split,
            agent_version=agent_version,
            status="completed",
            provider=provider_name,
            model=model_name,
            dataset_version=DATASET_VERSION,
            fact_version=FACT_VERSION,
            evaluator_version=EVALUATOR_VERSION,
            settings={
                "max_turns": 3, "max_concurrent": MAX_CONCURRENT_PROVIDER_REQUESTS,
                "timeout_s": 15, "max_attempts_per_turn": 2,
            },
            metrics=report,
            file_hashes={
                "answers.csv": _file_hash(os.path.join(DATA_DIR, "answers.csv")),
                "brand_facts.csv": _file_hash(os.path.join(DATA_DIR, "brand_facts.csv")),
                "development_gold.csv": _file_hash(os.path.join(DATA_DIR, "development_gold.csv")),
            },
        )
        db.add(run)
        db.flush()

        gold_by_span = {(g.answer_id, g.start_char, g.end_char): g for g in gold}

        for answer in scheduled:
            outcome = outcomes[answer.answer_id]
            ar = AnswerResult(
                run_id=run.run_id, answer_id=answer.answer_id, status=outcome.status,
                error_type=outcome.error_type, error_message=outcome.error_message,
                latency_ms=outcome.latency_ms, turns_used=outcome.turns_used,
            )
            db.add(ar)
            db.flush()

            for claim in outcome.claims:
                matched_gold = gold_by_span.get((answer.answer_id, claim.start_char, claim.end_char))
                db.add(ClaimResult(
                    run_id=run.run_id, answer_id=answer.answer_id, answer_result_id=ar.id,
                    claim_id=claim.claim_id, start_char=claim.start_char, end_char=claim.end_char,
                    quote=claim.quote, label=claim.label, fact_ids=list(claim.fact_ids), reason=claim.reason,
                    matched_gold_claim_id=matched_gold.claim_id if matched_gold else None,
                    expected_label=matched_gold.expected_label if matched_gold else None,
                    is_correct=(1 if matched_gold and matched_gold.expected_label == claim.label else
                                (0 if matched_gold else None)),
                ))

            score_for_trace = None
            if split == "development" and per_answer_scores.get(answer.answer_id):
                s = per_answer_scores[answer.answer_id]
                score_for_trace = {
                    "extraction_f1": s["extraction"]["f1"] if s["extraction"] else None,
                    "joint_f1": s["joint"]["f1"] if s["joint"] else None,
                    "missing_gold_ids": s["missed_gold_ids"],
                    "extra_prediction_ids": s["unmatched_prediction_ids"],
                }
            trace_id = trace_answer(run.run_id, answer, outcome, agent_version, split,
                                     provider_name, model_name, score_for_trace)
            if trace_id:
                ar.trace_id = trace_id

        db.commit()
        run_id = run.run_id
    finally:
        db.close()

    if split == "development":
        trace_run_summary(run_id, split, agent_version, report)

    return {
        "run_id": run_id,
        "split": split,
        "agent_version": agent_version,
        "provider": provider_name,
        "model": model_name,
        "scheduled_count": len(scheduled),
        "ok_count": sum(1 for o in outcomes.values() if o.status == "ok"),
        "error_count": sum(1 for o in outcomes.values() if o.status == "error"),
        "total_latency_ms": total_latency_ms,
        "metrics": report,
    }
