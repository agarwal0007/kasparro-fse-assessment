"""
Loads the three issued CSVs with Python's csv module (a proper RFC-4180 parser),
UTF-8 decoding, and newline='' on open so embedded newlines inside quoted fields
are preserved rather than treated as record separators. ai_answer text is used
verbatim - never stripped, normalized, or re-encoded - because gold character
offsets are computed against the exact decoded string.
"""
import csv
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnswerRecord:
    answer_id: str
    split: str
    brand_id: str
    question: str
    ai_answer: str


@dataclass(frozen=True)
class FactRecord:
    fact_id: str
    brand_id: str
    model_id: str
    field_name: str
    value: object  # coerced per value_type below
    value_type: str
    unit: str
    condition: str
    valid_from: str
    valid_to: str


@dataclass(frozen=True)
class GoldClaim:
    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    expected_label: str
    evidence_fact_ids: tuple
    explanation: str
    difficulty_tags: tuple


def _coerce_value(raw_value: str, value_type: str):
    """number -> float; boolean -> True/False from the literal text 'true'/'false'
    (never Python's bool("false"), which is truthy); string -> left as-is."""
    if value_type == "number":
        return float(raw_value)
    if value_type == "boolean":
        lowered = raw_value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        raise ValueError(f"Unexpected boolean literal: {raw_value!r}")
    return raw_value


def load_answers(path: str) -> list[AnswerRecord]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [
            AnswerRecord(
                answer_id=row["answer_id"],
                split=row["split"],
                brand_id=row["brand_id"],
                question=row["question"],
                ai_answer=row["ai_answer"],
            )
            for row in reader
        ]


def load_brand_facts(path: str) -> list[FactRecord]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            records.append(
                FactRecord(
                    fact_id=row["fact_id"],
                    brand_id=row["brand_id"],
                    model_id=row["model_id"],
                    field_name=row["field"],
                    value=_coerce_value(row["value"], row["value_type"]),
                    value_type=row["value_type"],
                    unit=row["unit"] or "",
                    condition=row["condition"] or "",
                    valid_from=row["valid_from"],
                    valid_to=row["valid_to"],
                )
            )
        return records


def load_gold(path: str) -> list[GoldClaim]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            evidence = tuple(x for x in row["evidence_fact_ids"].split("|") if x) if row["evidence_fact_ids"] else ()
            tags = tuple(x for x in row["difficulty_tags"].split("|") if x) if row["difficulty_tags"] else ()
            records.append(
                GoldClaim(
                    answer_id=row["answer_id"],
                    claim_id=row["claim_id"],
                    start_char=int(row["start_char"]),
                    end_char=int(row["end_char"]),
                    quote=row["quote"],
                    expected_label=row["expected_label"],
                    evidence_fact_ids=evidence,
                    explanation=row["explanation"],
                    difficulty_tags=tags,
                )
            )
        return records


def facts_for_brand(facts: list[FactRecord], brand_id: str) -> list[FactRecord]:
    return [f for f in facts if f.brand_id == brand_id]
