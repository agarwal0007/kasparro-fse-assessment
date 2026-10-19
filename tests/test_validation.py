import pytest

from app.agent.validation import ValidationError, validate_claims

SENTENCE_1 = "The speaker weighs 200 grams."
SENTENCE_2 = "It ships in blue or red."
ANSWER = SENTENCE_1 + " " + SENTENCE_2
S1_START, S1_END = 0, len(SENTENCE_1)
S2_START, S2_END = len(SENTENCE_1) + 1, len(SENTENCE_1) + 1 + len(SENTENCE_2)
KNOWN_FACTS = {"acme.m1.weight"}


def claim(start, end, label="supported", fact_ids=None, reason="Because the fact says so."):
    return {
        "start_char": start, "end_char": end, "quote": ANSWER[start:end],
        "label": label, "fact_ids": fact_ids or [], "reason": reason,
    }


def test_valid_claims_pass_and_sort_by_start():
    raw = [claim(S2_START, S2_END), claim(S1_START, S1_END)]
    claims = validate_claims(raw, "A01", ANSWER, KNOWN_FACTS)
    assert [c.start_char for c in claims] == [S1_START, S2_START]
    assert claims[0].claim_id == f"A01:{S1_START}:{S1_END}"


def test_out_of_bounds_span_rejected():
    with pytest.raises(ValidationError):
        validate_claims([claim(0, 9999)], "A01", ANSWER, KNOWN_FACTS)


def test_quote_mismatch_rejected():
    raw = [{"start_char": 0, "end_char": 10, "quote": "WRONG TEXT", "label": "unknown", "fact_ids": [], "reason": "x"}]
    with pytest.raises(ValidationError):
        validate_claims(raw, "A01", ANSWER, KNOWN_FACTS)


def test_quote_with_whitespace_padding_rejected():
    # Deliberately claim a span one character wider than the trimmed text, with a
    # quote value that includes the padding - must be rejected even if boundaries
    # are "close."
    raw = [{"start_char": 0, "end_char": 5, "quote": ANSWER[0:5] + " ", "label": "unknown", "fact_ids": [], "reason": "x"}]
    with pytest.raises(ValidationError):
        validate_claims(raw, "A01", ANSWER, KNOWN_FACTS)


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError):
        validate_claims([claim(S1_START, S1_END), claim(S1_START, S1_END)], "A01", ANSWER, KNOWN_FACTS)


def test_overlapping_spans_rejected():
    overlapping_start = S1_END - 1  # the period of sentence 1, non-whitespace
    with pytest.raises(ValidationError):
        validate_claims(
            [claim(S1_START, S1_END), claim(overlapping_start, S2_END)], "A01", ANSWER, KNOWN_FACTS
        )


def test_invalid_label_rejected():
    with pytest.raises(ValidationError):
        validate_claims([claim(S1_START, S1_END, label="probably_true")], "A01", ANSWER, KNOWN_FACTS)


def test_invented_fact_id_rejected():
    with pytest.raises(ValidationError):
        validate_claims([claim(S1_START, S1_END, fact_ids=["made.up.fact"])], "A01", ANSWER, KNOWN_FACTS)


def test_real_fact_id_accepted():
    claims = validate_claims([claim(S1_START, S1_END, fact_ids=["acme.m1.weight"])], "A01", ANSWER, KNOWN_FACTS)
    assert claims[0].fact_ids == ("acme.m1.weight",)


def test_empty_claims_list_is_valid():
    assert validate_claims([], "A01", ANSWER, KNOWN_FACTS) == []


# --- resolve_claim_offsets: Python locates quotes, the model never reports offsets ---

from app.agent.validation import resolve_claim_offsets


def test_resolve_claim_offsets_locates_a_single_occurrence():
    resolved = resolve_claim_offsets([{"quote": SENTENCE_1, "label": "unknown", "fact_ids": [], "reason": "x"}], ANSWER)
    assert resolved[0]["start_char"] == S1_START
    assert resolved[0]["end_char"] == S1_END


def test_resolve_claim_offsets_advances_past_repeated_identical_quotes():
    # The same sentence twice in a row - Python must resolve the SECOND claim
    # to the SECOND occurrence, not collapse both onto the first.
    repeated = SENTENCE_1 + " " + SENTENCE_1
    raw = [
        {"quote": SENTENCE_1, "label": "unknown", "fact_ids": [], "reason": "x"},
        {"quote": SENTENCE_1, "label": "unknown", "fact_ids": [], "reason": "x"},
    ]
    resolved = resolve_claim_offsets(raw, repeated)
    assert resolved[0]["start_char"] == 0
    assert resolved[1]["start_char"] == len(SENTENCE_1) + 1
    assert resolved[0]["end_char"] != resolved[1]["end_char"]


def test_resolve_claim_offsets_rejects_a_quote_not_present_verbatim():
    with pytest.raises(ValidationError):
        resolve_claim_offsets(
            [{"quote": "this text does not appear anywhere", "label": "unknown", "fact_ids": [], "reason": "x"}],
            ANSWER,
        )
