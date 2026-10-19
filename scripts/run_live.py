"""
Produces the four REQUIRED live-run artifacts:
  v1 development, v2 development, v1 check, v2 check
using a real provider (OpenAI or Gemini adapters in app/agent/provider.py).
The submitted artifacts were produced with LLM_PROVIDER=gemini,
LLM_MODEL=gemini-3.1-flash-lite. After running, export the full reports with
scripts/export_reports.py.

Requires OPENAI_API_KEY or GEMINI_API_KEY (see .env.example).

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
    if not any(os.environ.get(k) for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY")):
        print(
            "ERROR: no provider key is set (OPENAI_API_KEY / GEMINI_API_KEY / "
            "ANTHROPIC_API_KEY). This script requires a live provider key (see "
            ".env.example) - it does not fall back to a fake provider, because "
            "its output must be one of the four required live-run artifacts.",
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
