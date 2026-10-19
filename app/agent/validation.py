"""
Validates a model's final claim list against the contract: span bounds, quote
equality, unique non-overlapping IDs in increasing start_char order, label
values, and fact_ids that only reference real catalogue IDs. Python - never
the model - is the source of truth for character offsets.
"""
from dataclasses import dataclass

from app.config import VALID_LABELS


class ValidationError(Exception):
    pass


@dataclass(frozen=True)
class Claim:
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: str
    fact_ids: tuple
    reason: str


def validate_claims(raw_claims: list[dict], answer_id: str, ai_answer: str, known_fact_ids: set[str]) -> list[Claim]:
    """Raises ValidationError on any contract violation. Returns validated,
    sorted Claim objects on success - all or nothing (an invalid response
    produces an answer-level error, never a partially-accepted list)."""
    if not isinstance(raw_claims, list):
        raise ValidationError("claims must be a list")

    claims: list[Claim] = []
    seen_ids = set()

    for i, raw in enumerate(raw_claims):
        for key in ("start_char", "end_char", "quote", "label", "fact_ids", "reason"):
            if key not in raw:
                raise ValidationError(f"claim[{i}] missing required field {key!r}")

        start_char = raw["start_char"]
        end_char = raw["end_char"]
        if not isinstance(start_char, int) or not isinstance(end_char, int):
            raise ValidationError(f"claim[{i}] start_char/end_char must be integers")
        if not (0 <= start_char < end_char <= len(ai_answer)):
            raise ValidationError(f"claim[{i}] span [{start_char},{end_char}) out of bounds for answer of length {len(ai_answer)}")

        actual_quote = ai_answer[start_char:end_char]
        quote = raw["quote"]
        if quote != actual_quote:
            raise ValidationError(
                f"claim[{i}] quote mismatch: expected {actual_quote!r}, got {quote!r}"
            )
        if quote != quote.strip():
            raise ValidationError(f"claim[{i}] quote has leading/trailing whitespace")

        label = raw["label"]
        if label not in VALID_LABELS:
            raise ValidationError(f"claim[{i}] invalid label {label!r}")

        fact_ids = tuple(raw["fact_ids"])
        if not isinstance(raw["fact_ids"], list):
            raise ValidationError(f"claim[{i}] fact_ids must be a list")
        for fid in fact_ids:
            if fid not in known_fact_ids:
                raise ValidationError(f"claim[{i}] references unknown fact_id {fid!r}")

        reason = raw["reason"]
        if not isinstance(reason, str):
            raise ValidationError(f"claim[{i}] reason must be a string")
        if reason.count(".") + reason.count("!") + reason.count("?") > 2 and len(reason) > 400:
            raise ValidationError(f"claim[{i}] reason exceeds two sentences")

        claim_id = f"{answer_id}:{start_char}:{end_char}"
        if claim_id in seen_ids:
            raise ValidationError(f"duplicate claim_id {claim_id!r}")
        seen_ids.add(claim_id)

        claims.append(Claim(claim_id, start_char, end_char, quote, label, fact_ids, reason))

    claims.sort(key=lambda c: (c.start_char, c.end_char))

    for a, b in zip(claims, claims[1:]):
        if a.end_char > b.start_char:
            raise ValidationError(f"overlapping spans: {a.claim_id} and {b.claim_id}")

    return claims
