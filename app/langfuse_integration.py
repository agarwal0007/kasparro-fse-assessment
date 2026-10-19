"""
One real Langfuse trace per answer, for both agent versions and both splits.
Tool calls (get_brand_facts) become "tool" observations; model turns become
"generation" observations. On development, per-answer extraction_f1/joint_f1
are attached as Langfuse scores on the trace, and missing/extra claim IDs go
into trace metadata. On check splits, only predictions/errors are logged -
never a quality score, since gold is withheld for that split.

If no Langfuse keys are configured, every function here is a no-op that
returns None - the rest of the pipeline runs unaffected.
"""
import json

from app.config import DATASET_VERSION, EVALUATOR_VERSION, FACT_VERSION, LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

_client = None
_tried_init = False


def get_client():
    global _client, _tried_init
    if _tried_init:
        return _client
    _tried_init = True
    if LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY:
        from langfuse import Langfuse
        _client = Langfuse(secret_key=LANGFUSE_SECRET_KEY, public_key=LANGFUSE_PUBLIC_KEY, host=LANGFUSE_BASE_URL)
    return _client


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def trace_answer(run_id, answer, outcome, agent_version, split, provider_name, model_name,
                  answer_score=None):
    """Creates one trace for this answer's full agent run. Returns the trace_id,
    or None if Langfuse isn't configured."""
    client = get_client()
    if client is None:
        return None

    trace_id = client.create_trace_id(seed=f"{run_id}:{answer.answer_id}:{agent_version}")

    root = client.start_observation(
        trace_context={"trace_id": trace_id},
        name=f"classify:{answer.answer_id}",
        as_type="span",
        input={"answer_id": answer.answer_id, "brand_id": answer.brand_id, "question": answer.question},
        output=(
            {"status": "ok", "num_claims": len(outcome.claims)}
            if outcome.status == "ok"
            else {"status": "error", "error_type": outcome.error_type, "error_message": outcome.error_message}
        ),
        metadata={
            "run_id": run_id,
            "answer_id": answer.answer_id,
            "split": split,
            "agent_version": agent_version,
            "provider": provider_name,
            "model": model_name,
            "dataset_version": DATASET_VERSION,
            "fact_version": FACT_VERSION,
            "evaluator_version": EVALUATOR_VERSION,
            "turns_used": outcome.turns_used,
            "latency_ms": outcome.latency_ms,
            "missing_gold_ids": _jsonable(answer_score.get("missing_gold_ids")) if answer_score else None,
            "extra_prediction_ids": _jsonable(answer_score.get("extra_prediction_ids")) if answer_score else None,
        },
    )

    for turn in outcome.turns:
        if turn.kind == "tool_call":
            for tc in turn.tool_calls:
                tool_obs = client.start_observation(
                    trace_context={"trace_id": trace_id},
                    name=f"tool:{tc.name}",
                    as_type="tool",
                    input=tc.arguments,
                    output={"summary": tc.result_summary} if not tc.error else None,
                    level="ERROR" if tc.error else "DEFAULT",
                    status_message=tc.error,
                )
                tool_obs.end()
        else:  # "final" or "error" turn -> a generation observation
            gen = client.start_observation(
                trace_context={"trace_id": trace_id},
                name="model_turn",
                as_type="generation",
                model=model_name,
                output=turn.final_content,
                usage_details=turn.usage or None,
                level="ERROR" if turn.kind == "error" else "DEFAULT",
            )
            gen.end()

    if answer_score is not None:
        if answer_score.get("extraction_f1") is not None:
            client.create_score(trace_id=trace_id, name="extraction_f1",
                                 value=answer_score["extraction_f1"], data_type="NUMERIC")
        if answer_score.get("joint_f1") is not None:
            client.create_score(trace_id=trace_id, name="joint_f1",
                                 value=answer_score["joint_f1"], data_type="NUMERIC")

    root.end()
    client.flush()
    return trace_id


def trace_run_summary(run_id, split, agent_version, report: dict | None):
    """Links aggregate run-level metrics as a score on a synthetic run-level trace."""
    client = get_client()
    if client is None or report is None:
        return None
    trace_id = client.create_trace_id(seed=f"{run_id}:summary")
    root = client.start_observation(
        trace_context={"trace_id": trace_id},
        name=f"run_summary:{run_id}",
        as_type="span",
        input={"run_id": run_id, "split": split, "agent_version": agent_version},
        output=report,
        metadata={"run_id": run_id, "split": split, "agent_version": agent_version},
    )
    if report.get("joint") is not None:
        client.create_score(trace_id=trace_id, name="run_joint_f1", value=report["joint"]["f1"], data_type="NUMERIC")
    if report.get("macro_f1") is not None:
        client.create_score(trace_id=trace_id, name="run_macro_f1", value=report["macro_f1"], data_type="NUMERIC")
    root.end()
    client.flush()
    return trace_id
