from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_id: str
    brand_id: str
    question: str
    ai_answer: str
    agent_version: Literal["v1", "v2"]

    @model_validator(mode="before")
    @classmethod
    def reject_gold_fields(cls, data):
        if isinstance(data, dict):
            forbidden = {"expected_label", "evidence_fact_ids", "gold", "claims", "claim_id", "difficulty_tags"}
            present = forbidden & set(data.keys())
            if present:
                raise ValueError(f"Request may not include gold/claim-list fields: {sorted(present)}")
        return data


class ClaimOut(BaseModel):
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: str
    fact_ids: list[str]
    reason: str


class ClassifyResponse(BaseModel):
    answer_id: str
    claims: list[ClaimOut]


class EvaluationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_version: Literal["v1", "v2"]
    split: Literal["development", "check"]


class EvaluationRunResponse(BaseModel):
    run_id: str
    split: str
    agent_version: str
    status: str
    scheduled_count: int
    ok_count: int
    error_count: int
    metrics: Optional[dict] = None
