# Kasparro FSE I — Claim Extraction & Classification Agent

An agent that reads a long AI-generated speaker-product answer, extracts its
factual claim sentences (no claim list supplied), classifies each as
`supported` / `contradicted` / `unknown` using a live-model tool call against
a catalogue, and is scored by a deterministic harness against gold-labeled
spans.

## Results

Four live runs with `gemini-3.1-flash-lite` (real tool calling, Langfuse trace per answer):

| Run | ok / error | Extraction F1 | Joint F1 | Macro F1 |
|---|---|---|---|---|
| development / v1 | 29 / 11 | 0.683 | 0.651 | 0.629 |
| development / v2 | 26 / 14 | 0.630 | 0.605 | 0.582 |
| check / v1 | 9 / 1 | — | — | — |
| check / v2 | 10 / 0 | — | — | — |

On the 21 answers that completed in both development runs, joint F1 is 0.799 (v1)
vs 0.796 (v2): no gain from v2. Most errors are Gemini free-tier quota (429).
See `SUBMISSION.md` for the change, the trace-based diagnosis, error
breakdown, AI-assistance disclosure and unfinished work, and
`runs/regression_analysis.md` for per-answer/per-label detail.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill in a provider key and Langfuse keys, then `set -a; source .env; set +a`
```

Default `DATABASE_URL` is a local SQLite file (`sqlite:///./kasparro.db`) so
the whole stack runs without any external service. For Postgres, set
`DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/kasparro`
and `pip install psycopg2-binary`. The SQLAlchemy models and queries are
dialect-agnostic, but only SQLite was actually exercised (no Postgres server was available).

## Reproduce the live runs

```bash
export GEMINI_API_KEY=...            # or OPENAI_API_KEY with LLM_PROVIDER=openai
export LLM_PROVIDER=gemini LLM_MODEL=gemini-3.1-flash-lite
export LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
python scripts/run_live.py           # v1 dev, v2 dev, v1 check, v2 check -> DB + runs/live_*.json
python scripts/export_reports.py     # full reports + regression analysis from the DB (no model calls)
```

## How to run the pipeline-validation (no-key) version

```bash
python scripts/pipeline_validation_run.py          # all 50 answers
python scripts/pipeline_validation_run.py --limit 3  # quick smoke test
```

## Run the API

```bash
uvicorn app.main:app --reload
```

- `POST /classify` — `{answer_id, brand_id, question, ai_answer, agent_version}` → claims
- `POST /evaluation-runs` — `{agent_version, split}` → processes the fixed CSV split
- `GET /evaluation-runs/{run_id}` — joined run/answer/claim detail

## Run the tests

```bash
python -m pytest tests/ -v
```

47/47 pass as of this submission (output in `runs/test_output.txt`; no live key required — the suite uses
`FakeProvider` and an isolated SQLite file). Breakdown:
- `test_csv_loader.py` — quoted commas, embedded newlines, exact `ai_answer`
  preservation, boolean literal parsing (not `bool("false")`), loads the real
  issued CSVs and checks their counts (50/132/376).
- `test_validation.py` — span bounds, quote equality (incl. whitespace
  padding), duplicate IDs, overlaps, invalid labels, invented fact IDs.
- `test_scorer.py` — the brief's own worked example (G=10,P=10,M=8,C=7 →
  0.80/0.70), perfect output, missed claims, extra predictions, wrong labels,
  a provider-error answer, valid empty predictions, empty-split nulls, macro
  F1.
- `test_fake_provider_agent.py` — happy path, malformed JSON, invented fact
  ID, illegal tool name, unknown brand, turn-limit exhaustion, retry-then-
  succeed, retry-exhaustion, permanent errors never retried, gold never
  leaks into provider messages, bounded concurrency (≤3 in flight), actual
  overlap (not serial), isolated per-answer failure.
- `test_api.py` — 422 on gold/claim-list fields, 404 on unknown brand, 502
  with no provider configured, 422 on invalid body/UUID, 404 on missing run.

## Repository layout

```
app/
  agent/        tools.py (get_brand_facts), provider.py (OpenAI, Gemini, Fake),
                runner.py (3-turn loop, retries), validation.py
  harness/      csv_loader.py, scorer.py, orchestrator.py
  main.py       FastAPI app
  db.py         SQLAlchemy models (Run, AnswerResult, ClaimResult)
  schemas.py    Pydantic request/response models
  langfuse_integration.py
  config.py     fixed versions, timeouts, concurrency limits
prompts/        v1_system_prompt.txt, v2_system_prompt.txt
tests/
scripts/        run_live.py, export_reports.py, pipeline_validation_run.py
data/           the three issued CSVs + CHECKSUMS.txt
runs/           report_*.json (full live-run reports), live_*.json, regression_analysis.*, test_output.txt
```

## Versions

- Dataset: `long-brand-answers-50-v4`
- Fact catalogue: `speaker-catalogue-v4`
- Evaluator: `sentence-spans-and-labels-v4`
- Agent: v1 (baseline), v2 (one targeted change — see SUBMISSION.md)

See `SUBMISSION.md` for the full explanation: the targeted v1→v2 change and
its evidence, time spent, AI-assistance disclosure, and everything left
unfinished.
