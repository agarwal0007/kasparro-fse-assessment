"""
The read-only get_brand_facts tool. This is the ONLY source of factual
authority the model may consult - no gold labels, no claim lists, no scores
ever appear in a tool payload.
"""
from dataclasses import asdict

from app.config import FACT_VERSION
from app.harness.csv_loader import FactRecord, facts_for_brand


class ToolError(Exception):
    """Raised for an illegal tool name or an unknown requested brand."""


TOOL_NAME = "get_brand_facts"

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Look up the complete set of catalogue facts for one brand. "
            "Returns every fact row for that brand_id; call it again with a "
            "different brand_id to compare brands."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "brand_id": {
                    "type": "string",
                    "description": "The brand to look up (e.g. 'cleartrail').",
                }
            },
            "required": ["brand_id"],
            "additionalProperties": False,
        },
    },
}


class FactExecutor:
    """Validates and executes get_brand_facts calls against the loaded catalogue."""

    def __init__(self, all_facts: list[FactRecord], known_brand_ids: set[str]):
        self._all_facts = all_facts
        self._known_brand_ids = known_brand_ids

    def execute(self, tool_name: str, arguments: dict) -> dict:
        if tool_name != TOOL_NAME:
            raise ToolError(f"Unknown tool: {tool_name!r}")

        brand_id = arguments.get("brand_id")
        if not isinstance(brand_id, str) or not brand_id:
            raise ToolError("get_brand_facts requires a non-empty string 'brand_id' argument.")
        if brand_id not in self._known_brand_ids:
            raise ToolError(f"Unknown requested brand: {brand_id!r}")

        rows = facts_for_brand(self._all_facts, brand_id)
        return {
            "fact_version": FACT_VERSION,
            "brand_id": brand_id,
            "facts": [
                {
                    "fact_id": r.fact_id,
                    "model_id": r.model_id,
                    "field": r.field_name,
                    "value": r.value,
                    "value_type": r.value_type,
                    "unit": r.unit,
                    "condition": r.condition,
                    "valid_from": r.valid_from,
                    "valid_to": r.valid_to,
                }
                for r in rows
            ],
        }
