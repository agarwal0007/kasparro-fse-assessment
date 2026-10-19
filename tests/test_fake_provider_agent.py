import asyncio
import time

import pytest

from app.agent.provider import (
    FakeProvider,
    ModelTurnResult,
    PermanentProviderError,
    TransientProviderError,
    ToolCallRequest,
)
from app.agent.runner import run_agent_on_answer
from app.agent.tools import FactExecutor
from app.harness.csv_loader import AnswerRecord, FactRecord

ANSWER = AnswerRecord("A01", "development", "acme", "Q?", "The speaker weighs 200 grams.")
FACTS = [FactRecord("acme.m1.weight", "acme", "m1", "weight", 200.0, "number", "grams", "", "2026-01-01", "2026-12-31")]
KNOWN_BRANDS = {"acme"}
KNOWN_FACT_IDS = {"acme.m1.weight"}


def executor():
    return FactExecutor(FACTS, KNOWN_BRANDS)


def final_turn(claims):
    import json
    return ModelTurnResult(tool_calls=(), final_content=json.dumps({"claims": claims}))


def tool_turn(brand_id="acme"):
    return ModelTurnResult(
        tool_calls=(ToolCallRequest(call_id="call_1", name="get_brand_facts", arguments={"brand_id": brand_id}),),
        final_content=None,
    )


@pytest.mark.asyncio
async def test_happy_path_tool_call_then_valid_final_answer():
    provider = FakeProvider([tool_turn(), final_turn([
        {"start_char": 0, "end_char": len(ANSWER.ai_answer), "quote": ANSWER.ai_answer,
         "label": "supported", "fact_ids": ["acme.m1.weight"], "reason": "Matches the catalogue."}
    ])])
    outcome = await run_agent_on_answer(ANSWER, executor(), KNOWN_FACT_IDS, provider, "sys", asyncio.Semaphore(3))
    assert outcome.status == "ok"
    assert len(outcome.claims) == 1
    assert outcome.claims[0].label == "supported"
    assert outcome.turns_used == 2


@pytest.mark.asyncio
async def test_malformed_json_output_is_an_answer_level_error():
    provider = FakeProvider([ModelTurnResult(tool_calls=(), final_content="not valid json{{{")])
    outcome = await run_agent_on_answer(ANSWER, executor(), KNOWN_FACT_IDS, provider, "sys", asyncio.Semaphore(3))
    assert outcome.status == "error"
    assert outcome.error_type == "ValidationError"
    assert outcome.claims == []


@pytest.mark.asyncio
async def test_invented_fact_id_in_final_output_is_an_answer_level_error():
    provider = FakeProvider([final_turn([
        {"start_char": 0, "end_char": len(ANSWER.ai_answer), "quote": ANSWER.ai_answer,
         "label": "supported", "fact_ids": ["totally.invented.fact"], "reason": "x"}
    ])])
    outcome = await run_agent_on_answer(ANSWER, executor(), KNOWN_FACT_IDS, provider, "sys", asyncio.Semaphore(3))
    assert outcome.status == "error"
    assert outcome.error_type == "ValidationError"


@pytest.mark.asyncio
async def test_illegal_tool_name_is_an_answer_level_error():
    bad_tool_turn = ModelTurnResult(
        tool_calls=(ToolCallRequest(call_id="c1", name="delete_everything", arguments={}),),
        final_content=None,
    )
    provider = FakeProvider([bad_tool_turn])
    outcome = await run_agent_on_answer(ANSWER, executor(), KNOWN_FACT_IDS, provider, "sys", asyncio.Semaphore(3))
    assert outcome.status == "error"
    assert outcome.error_type == "ToolError"


@pytest.mark.asyncio
async def test_unknown_requested_brand_is_an_answer_level_error():
    provider = FakeProvider([tool_turn(brand_id="totally_unknown_brand")])
    outcome = await run_agent_on_answer(ANSWER, executor(), KNOWN_FACT_IDS, provider, "sys", asyncio.Semaphore(3))
    assert outcome.status == "error"
    assert outcome.error_type == "ToolError"
    assert "unknown_brand" in outcome.error_message.lower() or "unknown requested brand" in outcome.error_message.lower()


@pytest.mark.asyncio
async def test_turn_limit_exhaustion_is_an_answer_level_error():
    # Three tool-call turns in a row, never reaching a final answer -> limit hit.
    provider = FakeProvider([tool_turn(), tool_turn(), tool_turn()])
    outcome = await run_agent_on_answer(ANSWER, executor(), KNOWN_FACT_IDS, provider, "sys", asyncio.Semaphore(3))
    assert outcome.status == "error"
    assert outcome.error_type == "TurnLimitExceeded"
    assert outcome.turns_used == 3


@pytest.mark.asyncio
async def test_timeout_is_retried_once_then_succeeds():
    provider = FakeProvider([
        TransientProviderError("simulated timeout"),
        final_turn([]),
    ])
    outcome = await run_agent_on_answer(ANSWER, executor(), KNOWN_FACT_IDS, provider, "sys", asyncio.Semaphore(3))
    assert outcome.status == "ok"
    assert outcome.claims == []
    assert outcome.turns[0].attempts == 2  # first attempt failed, second (retry) succeeded


@pytest.mark.asyncio
async def test_transient_error_exhausts_after_two_attempts_total():
    provider = FakeProvider([
        TransientProviderError("timeout 1"),
        TransientProviderError("timeout 2"),
    ])
    outcome = await run_agent_on_answer(ANSWER, executor(), KNOWN_FACT_IDS, provider, "sys", asyncio.Semaphore(3))
    assert outcome.status == "error"
    assert outcome.error_type == "TransientProviderError"


@pytest.mark.asyncio
async def test_permanent_error_is_never_retried():
    provider = FakeProvider([PermanentProviderError("bad auth"), final_turn([])])
    outcome = await run_agent_on_answer(ANSWER, executor(), KNOWN_FACT_IDS, provider, "sys", asyncio.Semaphore(3))
    # Only ONE script item should have been consumed - the second (which would
    # succeed) must never be reached, proving no retry happened.
    assert outcome.status == "error"
    assert outcome.error_type == "PermanentProviderError"
    assert provider._calls == 1


@pytest.mark.asyncio
async def test_gold_labels_never_enter_the_messages_sent_to_the_provider():
    captured_messages = []

    class SpyProvider:
        async def complete_turn(self, messages, tools, timeout_s):
            captured_messages.append(messages)
            return final_turn([])

    outcome = await run_agent_on_answer(ANSWER, executor(), KNOWN_FACT_IDS, SpyProvider(), "sys", asyncio.Semaphore(3))
    serialized = str(captured_messages)
    for forbidden in ("expected_label", "evidence_fact_ids", "difficulty_tags", "claim_id"):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_bounded_concurrency_never_exceeds_the_semaphore_limit():
    max_concurrent_seen = 0
    current = 0
    lock = asyncio.Lock()

    class SlowProvider:
        async def complete_turn(self, messages, tools, timeout_s):
            nonlocal max_concurrent_seen, current
            async with lock:
                current += 1
                max_concurrent_seen = max(max_concurrent_seen, current)
            await asyncio.sleep(0.05)
            async with lock:
                current -= 1
            return final_turn([])

    semaphore = asyncio.Semaphore(3)
    provider = SlowProvider()
    answers = [AnswerRecord(f"A{i:02d}", "development", "acme", "Q?", "The speaker weighs 200 grams.") for i in range(10)]
    await asyncio.gather(*[
        run_agent_on_answer(a, executor(), KNOWN_FACT_IDS, provider, "sys", semaphore) for a in answers
    ])
    assert max_concurrent_seen <= 3


@pytest.mark.asyncio
async def test_a_slow_provider_actually_overlaps_answers_not_serial():
    # If execution were serial, 6 answers at 0.05s each would take >= 0.30s.
    # With 3-way concurrency it should take roughly 2 batches, ~0.10-0.15s.
    class SlowProvider:
        async def complete_turn(self, messages, tools, timeout_s):
            await asyncio.sleep(0.05)
            return final_turn([])

    semaphore = asyncio.Semaphore(3)
    provider = SlowProvider()
    answers = [AnswerRecord(f"A{i:02d}", "development", "acme", "Q?", "x") for i in range(6)]
    start = time.monotonic()
    await asyncio.gather(*[
        run_agent_on_answer(a, executor(), KNOWN_FACT_IDS, provider, "sys", semaphore) for a in answers
    ])
    elapsed = time.monotonic() - start
    assert elapsed < 0.25  # well under the fully-serial 0.30s


@pytest.mark.asyncio
async def test_isolated_failure_does_not_affect_other_answers():
    class PerAnswerProvider:
        async def complete_turn(self, messages, tools, timeout_s):
            # messages[1] is the user payload containing answer_id.
            if "A_FAIL" in messages[1]["content"]:
                raise PermanentProviderError("boom")
            return final_turn([])

    semaphore = asyncio.Semaphore(3)
    provider = PerAnswerProvider()
    answers = [
        AnswerRecord("A_FAIL", "development", "acme", "Q?", "x"),
        AnswerRecord("A_OK_1", "development", "acme", "Q?", "x"),
        AnswerRecord("A_OK_2", "development", "acme", "Q?", "x"),
    ]
    outcomes = await asyncio.gather(*[
        run_agent_on_answer(a, executor(), KNOWN_FACT_IDS, provider, "sys", semaphore) for a in answers
    ])
    by_id = {o.answer_id: o for o in outcomes}
    assert by_id["A_FAIL"].status == "error"
    assert by_id["A_OK_1"].status == "ok"
    assert by_id["A_OK_2"].status == "ok"
