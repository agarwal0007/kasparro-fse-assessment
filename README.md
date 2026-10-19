# Kasparro FSE I — Claim Extraction & Classification Agent

An agent that reads a long AI-generated speaker-product answer, extracts its
factual claim sentences (no claim list supplied), classifies each as
`supported` / `contradicted` / `unknown` using a live-model tool call against
a catalogue, and is scored by a deterministic harness against gold-labeled
spans.

## ⚠️ Known limitation — read this first

**No live LLM API key (OpenAI/Anthropic/Gemini) was available in the build
environment for this submission**, and there was no Postgres/Docker server
either. I did not fabricate "live" model output to work around this — that is
exactly what the brief's anti-gaming rules exist to catch, and doing so would
make every downstream number meaningless.

What this means concretely:
- **The four required live-run artifacts (v1/v2 × development/check) have NOT
  been produced.** `scripts/run_live.py` is code-complete and will produce
  them in one command the moment `OPENAI_API_KEY` (or another provider key,
  with a small adapter addition) is set — see "How to produce the real
  artifacts" below.
- **Everything else is built, tested, and has actually been run** against the
  real issued CSVs: the agent loop (tool-calling, 3-turn limit, retries,
  timeouts, validation), the deterministic scorer, SQLAlchemy persistence,
  the FastAPI endpoints, and real Langfuse trace ingestion (verified via
  Langfuse's own API — see `runs/pipeline_validation_*.json` and
  `SUBMISSION.md`).
- A `FakeProvider`-driven **pipeline validation run** (`scripts/pipeline_validation_run.py`)
  exercises the identical agent/harness/scorer/DB/Langfuse code path end to
  end on all 50 real answers, using a deliberately simplistic rule-based
  stand-in (naive sentence splitting, everything labeled `unknown`) instead
  of a real model. Its scores are **not** evidence of agent quality — only
  evidence that the pipeline itself runs correctly. This is clearly labeled
  in its own output and is not one of the four required artifacts.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill in a provider key and Langfuse keys
```

Default `DATABASE_URL` is a local SQLite file (`sqlite:///./kasparro.db`) so
the whole stack runs without any external service. For Postgres, set
`DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/kasparro`
(add `psycopg2-binary` to requirements.txt) — the SQLAlchemy models and all
queries are dialect-agnostic.

## How to produce the real (required) live-run artifacts

```bash
export OPENAI_API_KEY=sk-...
python scripts/run_live.py
```

This runs v1 then v2 against the 40 development answers, then (unchanged,
frozen) v1 and v2 against the 10 check answers, saving all four reports to
`runs/live_<split>_<version>.json` and persisting everything to the
database with real Langfuse traces per answer.

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

44/44 pass as of this submission (no live key required — the suite uses
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
  agent/        tools.py (get_brand_facts), provider.py (OpenAI + Fake),
                runner.py (3-turn loop, retries), validation.py
  harness/      csv_loader.py, scorer.py, orchestrator.py
  main.py       FastAPI app
  db.py         SQLAlchemy models (Run, AnswerResult, ClaimResult)
  schemas.py    Pydantic request/response models
  langfuse_integration.py
  config.py     fixed versions, timeouts, concurrency limits
prompts/        v1_system_prompt.txt, v2_system_prompt.txt
tests/
scripts/        run_live.py, pipeline_validation_run.py
data/           the three issued CSVs + CHECKSUMS.txt
runs/           saved run JSON reports
```

## Versions

- Dataset: `long-brand-answers-50-v4`
- Fact catalogue: `speaker-catalogue-v4`
- Evaluator: `sentence-spans-and-labels-v4`
- Agent: v1 (baseline), v2 (one targeted change — see SUBMISSION.md)

See `SUBMISSION.md` for the full explanation: the targeted v1→v2 change and
its evidence, time spent, AI-assistance disclosure, and everything left
unfinished.
