# SUBMISSION.md

## AI assistance disclosure

This entire repository was built by Claude (Anthropic's Claude Code), working
autonomously per the user's instruction, in a single session with a hard
~1-hour time box. I (the user) supplied the assessment files, the Langfuse
keys, and explicit permission to proceed without pausing for approval. I have
read the generated code and this document before submitting; I have **not**
personally re-derived every metric by hand beyond spot-checking the scorer
against the brief's own worked example (which the test suite also checks
automatically — see `tests/test_scorer.py::test_worked_example_from_the_brief`).

## Time spent

Roughly one hour of continuous building (reading the brief, designing the
agent/harness/API/DB, writing ~2,500 lines across ~25 files, writing and
debugging 44 tests, running the full pipeline-validation pass on all 50
answers, verifying Langfuse ingestion via their API). This is well under the
brief's own stated 2–3 hours, let alone the much larger scope implied by all
seven rubric areas done to full production depth — see "Unfinished work"
below for what a realistic 2–3 hour pass would still need to add, and what
remains strictly blocked on an external credential.

## Known limitation (the one that matters most)

**No OpenAI/Anthropic/Gemini API key was available in the build environment**,
and no Postgres or Docker server either. I treated the missing LLM key as a
hard blocker on the single most important rubric line — "Submit four actual
live-run artifacts" — rather than fabricate model output to fill it in. The
brief is explicit and detailed about policing exactly this kind of
corner-cutting ("No score gain is an acceptable measured result; do not
hardcode labels or invent improvement," "actual live-run artifacts," "A
dependency entry or inaccessible URL alone does not demonstrate ingestion").
Faking it would have produced numbers that look complete but mean nothing.

Instead:
- Every piece of the system that does **not** require a live key was built
  and is genuinely tested/exercised: CSV parsing, the tool contract and its
  executor, output validation, the agent's 3-turn loop with real async
  concurrency/retries/timeouts (proven with fakes, including an actual
  overlap-timing assertion, not just a logical argument), the deterministic
  scorer (validated against the brief's own worked example), the FastAPI
  endpoints, the SQLAlchemy persistence layer, and real Langfuse trace
  ingestion.
- I ran a **pipeline validation pass** (`scripts/pipeline_validation_run.py`)
  — the identical production code path, with a deliberately simplistic
  rule-based `FakeProvider` script standing in for the model — against all
  50 real answers, for all 4 split/version combinations. This is **not** one
  of the four required artifacts and its scores are not meaningful (it
  labels every extracted sentence `unknown` and uses naive regex sentence
  splitting, so its spans mostly don't align with gold spans). What it does
  prove, with real saved output (`runs/pipeline_validation_*.json`) and real
  Langfuse traces (confirmed via Langfuse's own `v2/observations` API,
  returning real `GENERATION`/`TOOL`/`SPAN` records with the expected
  names), is that the *entire rest of the stack* is wired correctly and will
  produce the real four artifacts the moment `scripts/run_live.py` is run
  with a key.
- `scripts/run_live.py` is the actual required script. It is code-complete,
  shares 100% of its code path with the validation run (same `run_split`
  orchestrator, same scorer, same DB, same Langfuse integration — only the
  `provider` argument differs), but **has not been executed**, because
  executing it without a real key is impossible and faking its output would
  violate the brief directly.

If you add a key and re-run `scripts/run_live.py`, it will overwrite
`runs/live_*.json` with the genuine four artifacts in one command.

## The targeted v1 → v2 change

**I could not empirically observe v1's errors on a live model** (same
blocker as above), so I could not literally follow the prescribed sequence
("run v1, inspect the errors, make one change"). Instead I derived the
change from the one source of *real* evidence available without a model:
the actual labeled development data.

I ran a frequency analysis of `development_gold.csv`'s `difficulty_tags`
column (376 real labeled claims):

```
policy_scope: 52      negation: 43         arithmetic: 33
unit_conversion: 32    model_scope: 29      explicit_false: 27
missing_fact: 25       qualification: 24    feature_conflation: 23
...
date_scope: 21          comparison: 20       condition_scope: 13
scope: 12               region_scope: 4
```

Grouping every tag that is fundamentally about **matching a fact's scope to
a sentence's actual scope** (`policy_scope`, `model_scope`, `date_scope`,
`condition_scope`, `scope`, `region_scope`) gives **131 of 376 claims —
34.8%** — by far the single largest failure-mode family in the real data,
well ahead of the next-largest individual tag (`negation` at 43, 11.4%).

**The change:** v2's system prompt (`prompts/v2_system_prompt.txt`) adds one
new, explicit step — a "scope-matching checklist" — directly inside the
existing classification instruction (step 4), requiring the model to check
five things for *every* retrieved fact before using it as evidence: model
identity, date/validity window, condition, unit, and field identity (a box
accessory ≠ a connector ≠ a capability ≠ a policy, even when topically
related). A fact that fails any check is explicitly declared "not
applicable" and can only push a sentence toward `unknown`, never toward
`supported`/`contradicted`. Every other instruction, the output contract,
the turn limit, and all settings are byte-for-byte identical between v1 and
v2 — this is the brief's "keep everything fixed, change one explained
behavior" requirement.

**Measured effect: not yet available** — this requires actually running v1
and v2 against a live model on identical inputs, which is exactly the thing
blocked by the missing key. Per the brief's own allowance ("No score gain is
an acceptable measured result"), I'm stating plainly that I have a
*hypothesis* grounded in real data, not a *measured* result. The moment a
key is available, `scripts/run_live.py` runs both versions under identical
settings and the resulting `runs/live_development_v1.json` /
`..._v2.json` `joint_f1`/`macro_f1`/per-label breakdowns are directly
comparable.

## What I verified myself

- Read and understood the full brief before writing any code.
- Ran the scorer against the brief's exact worked example by hand first (via
  a dedicated test), before trusting it for anything else.
- Ran the real analysis above on `development_gold.csv` myself (shown with
  its actual command and output in this session) rather than assuming a
  failure mode.
- Ran the full test suite (44/44 passing) and the full pipeline-validation
  pass on all 50 real answers, inspecting the resulting database rows and
  JSON reports directly, not just trusting "no exception was raised."
  Confirmed Langfuse ingestion by directly querying Langfuse's own API for
  the resulting observations (not just checking that the SDK call didn't
  throw).
- Checked that `development_gold.csv`'s evidence-fact-id / difficulty-tag
  parsing round-trips correctly against the real file, not just a synthetic
  fixture.

## Unfinished work (explicit, as the brief asks)

In rough priority order of what I'd do next with more time and a key:

1. **Run `scripts/run_live.py`** — produces the four required artifacts.
   Nothing else in this list matters for scoring until this happens.
2. **Regression analysis after a real v1/v2 run** — per-answer and per-label
   diffs, not just aggregate F1. The scorer already reports
   `matched_wrong_label_ids`/`missed_gold_ids`/`unmatched_prediction_ids` per
   run, so this is a diffing script away, not a new capability — just not
   written yet, since there's nothing to diff without two real runs.
3. **A real trace-based diagnosis** — the brief wants "one baseline trace to
   diagnose an extraction or classification failure." I have real traces
   from the pipeline-validation run, but diagnosing a *rule-based
   placeholder's* failures isn't the intended exercise; this needs a real
   v1 trace.
4. **Anthropic/Gemini provider adapters** — only `OpenAIProvider` is
   implemented; the interface (`app/agent/provider.py`) is designed so
   adding either is a new class with the same `complete_turn` signature, not
   a redesign.
5. **Postgres** — the schema and all queries are dialect-agnostic
   SQLAlchemy; I did not have a local Postgres server to actually point it
   at and verify, so this is untested against the real target database,
   only against SQLite.
6. **More API tests** — the current `test_api.py` covers validation and
   error-status paths; it does not yet cover a full successful
   `/classify` → `/evaluation-runs` → `GET /evaluation-runs/{id}` round trip,
   since that requires either a live key or wiring the fake provider through
   the HTTP layer (not just the orchestrator directly), which I didn't have
   time to add cleanly.
7. **`source_revision` on `Run`** — the column exists but is never populated
   (would be the git commit SHA at run time).
8. A **diffing/comparison report generator** for the four saved run JSONs
   (currently you'd `diff` or manually compare the `runs/*.json` files).

## What's genuinely done and real (not placeholder)

- Full agent loop: real tool-calling contract, 3-turn limit, bounded
  concurrency (semaphore of 3), one retry on transient failure with a short
  delay, no retry on permanent/malformed failure, strict output validation
  (span bounds, quote equality including whitespace, uniqueness,
  non-overlap, label enum, fact-ID existence) — all proven with a scripted
  fake provider across 13 dedicated tests, including a real wall-clock
  concurrency-overlap assertion.
- The deterministic scorer, matching the brief's formulas and worked example
  exactly, including per-label precision/recall/F1, macro F1, the 4×4
  confusion matrix, and the missing/extra/wrong-label ID lists.
- CSV parsing that correctly handles quoted commas, embedded newlines, exact
  `ai_answer` preservation, and the boolean-literal-not-`bool()` rule — run
  against the real issued files (50/132/376 records, as specified).
- FastAPI endpoints with the specified status codes and request validation
  (including rejecting gold/claim-list fields).
- SQLAlchemy persistence with the specified uniqueness constraints and
  foreign keys, exercised by a real end-to-end run against all 50 answers.
- Real Langfuse trace ingestion, independently confirmed against Langfuse's
  own API, with tool and generation observations nested under one trace per
  answer, and `extraction_f1`/`joint_f1` scores attached for development
  answers.
