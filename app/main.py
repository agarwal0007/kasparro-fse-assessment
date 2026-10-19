import asyncio
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.agent.provider import PermanentProviderError, TransientProviderError
from app.agent.runner import run_agent_on_answer
from app.agent.tools import FactExecutor, ToolError
from app.config import MAX_CONCURRENT_PROVIDER_REQUESTS, PROVIDER_TIMEOUT_SECONDS
from app.db import AnswerResult, ClaimResult, Run, SessionLocal, init_db
from app.harness.csv_loader import AnswerRecord, load_brand_facts
from app.harness.orchestrator import DATA_DIR, build_provider, load_prompt, run_split
from app.schemas import ClaimOut, ClassifyRequest, ClassifyResponse, EvaluationRunRequest, EvaluationRunResponse

import os

app = FastAPI(title="Kasparro FSE I - Claim Extraction & Classification")

_facts = load_brand_facts(os.path.join(DATA_DIR, "brand_facts.csv"))
_known_brand_ids = {f.brand_id for f in _facts}
_known_fact_ids = {f.fact_id for f in _facts}
_fact_executor = FactExecutor(_facts, _known_brand_ids)
_classify_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROVIDER_REQUESTS)


@app.on_event("startup")
def _startup():
    init_db()


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    if request.brand_id not in _known_brand_ids:
        raise HTTPException(status_code=404, detail=f"Unknown primary brand: {request.brand_id!r}")

    answer = AnswerRecord(request.answer_id, "api", request.brand_id, request.question, request.ai_answer)
    provider, provider_name, model_name = build_provider()
    if provider is None:
        raise HTTPException(status_code=502, detail="No live LLM provider is configured.")

    system_prompt = load_prompt(request.agent_version)
    try:
        outcome = await asyncio.wait_for(
            run_agent_on_answer(answer, _fact_executor, _known_fact_ids, provider, system_prompt, _classify_semaphore),
            timeout=PROVIDER_TIMEOUT_SECONDS * 4,  # generous ceiling across up to 3 turns + 1 retry each
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out.")

    if outcome.status == "error":
        if outcome.error_type == "ToolError":
            raise HTTPException(status_code=404, detail=outcome.error_message)
        raise HTTPException(status_code=502, detail=f"{outcome.error_type}: {outcome.error_message}")

    return ClassifyResponse(
        answer_id=answer.answer_id,
        claims=[
            ClaimOut(claim_id=c.claim_id, start_char=c.start_char, end_char=c.end_char,
                     quote=c.quote, label=c.label, fact_ids=list(c.fact_ids), reason=c.reason)
            for c in outcome.claims
        ],
    )


@app.post("/evaluation-runs", status_code=201, response_model=EvaluationRunResponse)
async def create_evaluation_run(request: EvaluationRunRequest):
    provider, _, _ = build_provider()
    if provider is None:
        raise HTTPException(status_code=502, detail="No live LLM provider is configured.")
    result = await run_split(request.split, request.agent_version)
    return EvaluationRunResponse(
        run_id=result["run_id"], split=result["split"], agent_version=result["agent_version"],
        status="completed", scheduled_count=result["scheduled_count"], ok_count=result["ok_count"],
        error_count=result["error_count"], metrics=result["metrics"],
    )


@app.get("/evaluation-runs/{run_id}")
def get_evaluation_run(run_id: str):
    try:
        uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="run_id must be a valid UUID.")

    db = SessionLocal()
    try:
        run = db.query(Run).filter(Run.run_id == run_id).first()
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found.")

        answers = (
            db.query(AnswerResult)
            .filter(AnswerResult.run_id == run_id)
            .order_by(AnswerResult.answer_id)
            .all()
        )
        answer_payload = []
        for ar in answers:
            claims = (
                db.query(ClaimResult)
                .filter(ClaimResult.answer_result_id == ar.id)
                .order_by(ClaimResult.start_char, ClaimResult.end_char)
                .all()
            )
            answer_payload.append({
                "answer_id": ar.answer_id,
                "status": ar.status,
                "error_type": ar.error_type,
                "error_message": ar.error_message,
                "latency_ms": ar.latency_ms,
                "turns_used": ar.turns_used,
                "trace_id": ar.trace_id,
                "claims": [
                    {
                        "claim_id": c.claim_id, "start_char": c.start_char, "end_char": c.end_char,
                        "quote": c.quote, "label": c.label, "fact_ids": c.fact_ids, "reason": c.reason,
                        "matched_gold_claim_id": c.matched_gold_claim_id,
                        "expected_label": c.expected_label, "is_correct": c.is_correct,
                    }
                    for c in claims
                ],
            })

        return {
            "run_id": run.run_id,
            "split": run.split,
            "agent_version": run.agent_version,
            "status": run.status,
            "provider": run.provider,
            "model": run.model,
            "dataset_version": run.dataset_version,
            "fact_version": run.fact_version,
            "evaluator_version": run.evaluator_version,
            "settings": run.settings,
            "metrics": run.metrics,
            "file_hashes": run.file_hashes,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "answers": answer_payload,
        }
    finally:
        db.close()
