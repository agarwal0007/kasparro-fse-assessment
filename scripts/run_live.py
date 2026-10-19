"""
Produces the four REQUIRED live-run artifacts:
  v1 development, v2 development, v1 check, v2 check
using a real OpenAI/Anthropic/Gemini provider (OpenAI implemented today - see
app/agent/provider.py to add another). This is the ONLY script that is meant
to satisfy the "four actual live-run artifacts" requirement; everything else
in this repo can be validated without a key (see scripts/pipeline_validation_run.py
and the test suite).

Requires one of OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY to be set
(see .env.example). As of this submission, no such key was available in the
build environment, so this script is code-complete but has NOT been executed
against a live model - see SUBMISSION.md "Known limitation."

Usage: python scripts/run_live.py
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db import init_db
from app.harness.orchestrator import run_split


async def main():
    init_db()
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set. This script requires a live provider "
            "key (see .env.example) - it does not fall back to a fake provider, "
            "because its output must be one of the four required live-run artifacts.",
            file=sys.stderr,
        )
        sys.exit(1)

    results = {}
    for split in ("development", "check"):
        for agent_version in ("v1", "v2"):
            print(f"Running {split}/{agent_version} ...")
            result = await run_split(split, agent_version)
            results[f"{split}_{agent_version}"] = result
            print(f"  run_id={result['run_id']} ok={result['ok_count']} error={result['error_count']}")
            if result["metrics"]:
                m = result["metrics"]
                print(f"  extraction_f1={m['extraction']['f1']:.3f} joint_f1={m['joint']['f1']:.3f} "
                      f"macro_f1={m['macro_f1']:.3f}")
            with open(f"runs/live_{split}_{agent_version}.json", "w") as f:
                json.dump(result, f, indent=2, default=str)

    print("\nComparison (development only - check has no quality score by design):")
    v1 = results["development_v1"]["metrics"]
    v2 = results["development_v2"]["metrics"]
    print(f"  v1 joint_f1={v1['joint']['f1']:.3f}  v2 joint_f1={v2['joint']['f1']:.3f}  "
          f"delta={v2['joint']['f1'] - v1['joint']['f1']:+.3f}")


if __name__ == "__main__":
    asyncio.run(main())
