"""
The core per-answer agent loop: up to MAX_TURNS_PER_ANSWER model turns, real
tool calling for get_brand_facts, bounded retries on transient provider
failures, and strict validation of the final claims output. Gold labels,
spans, and difficulty tags never enter any message sent to the model.
"""
import asyncio
import json
import time
from dataclasses import dataclass, field

from app.agent.provider import ModelTurnResult, PermanentProviderError, TransientProviderError
from app.agent.tools import TOOL_NAME, TOOL_SCHEMA, FactExecutor, ToolError
from app.agent.validation import Claim, ValidationError, validate_claims
from app.config import MAX_ATTEMPTS_PER_TURN, MAX_TURNS_PER_ANSWER, PROVIDER_TIMEOUT_SECONDS
from app.harness.csv_loader import AnswerRecord


@dataclass
class ToolCallLog:
    name: str
    arguments: dict
    result_summary: str
    error: str | None


@dataclass
class TurnLog:
    turn_index: int
    kind: str  # "tool_call" or "final" or "error"
    tool_calls: list[ToolCallLog] = field(default_factory=list)
    final_content: str | None = None
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    attempts: int = 1


@dataclass
class AnswerOutcome:
    answer_id: str
    status: str  # "ok" | "error"
    claims: list  # list[Claim], empty on error
    error_type: str | None
    error_message: str | None
    turns: list  # list[TurnLog] - for tracing
    latency_ms: float
    turns_used: int


RETRY_DELAY_SECONDS = 0.5


async def _complete_turn_with_retry(provider, messages, tools, semaphore) -> tuple:
    """Returns (ModelTurnResult, attempts_used). Retries once on a transient
    provider error after a short delay; never retries a permanent/malformed
    failure. The semaphore bounds total concurrent provider requests across
    ALL answers, including retries."""
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS_PER_TURN + 1):
        async with semaphore:
            try:
                result = await provider.complete_turn(messages, tools, PROVIDER_TIMEOUT_SECONDS)
                return result, attempt
            except TransientProviderError as e:
                last_error = e
                if attempt < MAX_ATTEMPTS_PER_TURN:
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                    continue
                raise
            except PermanentProviderError:
                raise
    raise last_error  # pragma: no cover - loop always returns or raises above


async def run_agent_on_answer(
    answer: AnswerRecord,
    fact_executor: FactExecutor,
    known_fact_ids: set,
    provider,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
) -> AnswerOutcome:
    start = time.monotonic()
    user_payload = {
        "answer_id": answer.answer_id,
        "brand_id": answer.brand_id,
        "question": answer.question,
        "ai_answer": answer.ai_answer,
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    tools = [TOOL_SCHEMA]
    turn_logs: list[TurnLog] = []

    for turn_index in range(1, MAX_TURNS_PER_ANSWER + 1):
        turn_start = time.monotonic()
        try:
            result, attempts = await _complete_turn_with_retry(provider, messages, tools, semaphore)
        except (TransientProviderError, PermanentProviderError) as e:
            turn_logs.append(TurnLog(turn_index, "error", latency_ms=(time.monotonic() - turn_start) * 1000))
            return AnswerOutcome(
                answer_id=answer.answer_id,
                status="error",
                claims=[],
                error_type=type(e).__name__,
                error_message=str(e),
                turns=turn_logs,
                latency_ms=(time.monotonic() - start) * 1000,
                turns_used=turn_index,
            )

        latency_ms = (time.monotonic() - turn_start) * 1000

        if result.tool_calls:
            tool_logs = []
            # OpenAI's protocol requires an assistant message carrying the tool_calls
            # followed by one tool-role message per call, referencing its call_id.
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in result.tool_calls
                ],
            })
            for tc in result.tool_calls:
                try:
                    tool_result = fact_executor.execute(tc.name, tc.arguments)
                    tool_logs.append(ToolCallLog(tc.name, tc.arguments, f"{len(tool_result['facts'])} facts", None))
                    messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": json.dumps(tool_result)})
                except ToolError as e:
                    tool_logs.append(ToolCallLog(tc.name, tc.arguments, "", str(e)))
                    turn_logs.append(TurnLog(turn_index, "error", tool_calls=tool_logs, latency_ms=latency_ms, attempts=attempts))
                    return AnswerOutcome(
                        answer_id=answer.answer_id,
                        status="error",
                        claims=[],
                        error_type="ToolError",
                        error_message=str(e),
                        turns=turn_logs,
                        latency_ms=(time.monotonic() - start) * 1000,
                        turns_used=turn_index,
                    )
            turn_logs.append(TurnLog(turn_index, "tool_call", tool_calls=tool_logs, usage=result.usage,
                                      latency_ms=latency_ms, attempts=attempts))
            continue

        # Final content turn.
        turn_logs.append(TurnLog(turn_index, "final", final_content=result.final_content, usage=result.usage,
                                  latency_ms=latency_ms, attempts=attempts))
        try:
            parsed = json.loads(result.final_content)
            raw_claims = parsed["claims"]
            claims = validate_claims(raw_claims, answer.answer_id, answer.ai_answer, known_fact_ids)
        except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as e:
            return AnswerOutcome(
                answer_id=answer.answer_id,
                status="error",
                claims=[],
                error_type="ValidationError",
                error_message=str(e),
                turns=turn_logs,
                latency_ms=(time.monotonic() - start) * 1000,
                turns_used=turn_index,
            )

        return AnswerOutcome(
            answer_id=answer.answer_id,
            status="ok",
            claims=claims,
            error_type=None,
            error_message=None,
            turns=turn_logs,
            latency_ms=(time.monotonic() - start) * 1000,
            turns_used=turn_index,
        )

    # Exhausted all turns without a final answer.
    return AnswerOutcome(
        answer_id=answer.answer_id,
        status="error",
        claims=[],
        error_type="TurnLimitExceeded",
        error_message=f"Exceeded {MAX_TURNS_PER_ANSWER} turns without a final answer.",
        turns=turn_logs,
        latency_ms=(time.monotonic() - start) * 1000,
        turns_used=MAX_TURNS_PER_ANSWER,
    )
