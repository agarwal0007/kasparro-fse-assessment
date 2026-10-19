# SUBMISSION.md

## Results at a glance

Four live runs, all with `gemini` / `gemini-3.1-flash-lite`, real `get_brand_facts`
function calling, one Langfuse trace per answer (100/100 answer rows carry a
trace ID). Data, evaluator, provider, model and settings were identical across
all four runs (max 3 turns, max 3 concurrent provider requests, 15 s timeout,
max 2 attempts per turn).

| Run | Answers ok / error | P | M | C | Extraction F1 | Joint F1 | Macro F1 |
|---|---|---|---|---|---|---|---|
| development / v1 | 29 / 11 | 195 | 195 | 186 | 0.683 | **0.651** | 0.629 |
| development / v2 | 26 / 14 | 173 | 173 | 166 | 0.630 | **0.605** | 0.582 |
| check / v1 | 9 / 1 | 66 | — | — | — | — | — |
| check / v2 | 10 / 0 | 78 | — | — | — | — | — |

G = 376 for development. Check quality metrics are null by design (the reviewer
holds the gold).

**Controlled comparison** (the 21 development answers that completed in *both*
runs, 198 gold claims): joint F1 **0.799 (v1) vs 0.796 (v2)**, extraction F1
0.845 vs 0.832, macro F1 0.770 vs 0.764. **v2 produced no measurable gain.**
The larger full-split drop comes from provider quota errors landing on
different answers in each run, not from the prompt change.

Full detail: `runs/regression_analysis.md`.

## Evidence files

| File | What it is |
|---|---|
| `runs/report_development_v1.json`, `runs/report_development_v2.json` | Full development reports: settings, versions, file hashes, per-answer outcomes (status, sanitized error, latency, turns, Langfuse trace ID, per-answer extraction/joint F1), every accepted prediction with its gold match, `missing_gold`, `extra_predictions`, `wrong_label`, and all metrics (incl. per-label, macro F1, confusion matrix) |
| `runs/report_check_v1.json`, `runs/report_check_v2.json` | Full check reports: predictions, answer outcomes, errors, counts, timing, versions. No gold, no quality metrics |
| `runs/live_*.json` | Summaries written by `scripts/run_live.py` at run time |
| `runs/regression_analysis.md` / `.json` | v1 vs v2: full split, controlled subset, per-label, per-answer table, label flips |
| `runs/test_output.txt` | Actual `pytest -v` output (47 passed) |
| `data/CHECKSUMS.txt` | SHA-256 of the three issued CSVs (also stored on every run row) |

`kasparro.db` is not committed (`*.db` is ignored). The report files are the
complete export of its rows for the four live runs. `python scripts/export_reports.py`
regenerates them from the DB. The script makes no model calls; it re-scores
development runs with the same deterministic scorer and reproduces the metrics
saved at run time.

## Answer errors: what they were

Every error is kept as an answer-level error with no accepted predictions, and
its gold claims stay in G as missed.

| Cause | dev v1 | dev v2 | check v1 | check v2 |
|---|---|---|---|---|
| `TransientProviderError`: Gemini 429 `RESOURCE_EXHAUSTED` (free-tier quota), still failing after the single allowed retry | 10 | 11 | 1 (A50) | 0 |
| `ToolError`: model requested an unknown brand, e.g. `cloudnotestudio` (a model name passed as brand_id) or `rivalsound` | 1 (A03) | 2 (A24, A36) | 0 | 0 |
| `ValidationError`: duplicate claim_id `A02:507:689` (same sentence returned twice) | 0 | 1 (A02) | 0 | 0 |

The quota errors are infrastructure failures, not agent behaviour. Because they
hit different answers in v1 and v2, the 21-answer controlled comparison is the
fair comparison of the two prompts. The runs are kept exactly as produced; no
answers were rerun after the versions were frozen.

## The targeted v1 → v2 change

**Change (one behaviour):** v2's system prompt adds a *scope-matching checklist*
inside the classification step. For every retrieved fact, the model must check
model identity, date/validity window, condition, unit and field identity. A
fact that fails any check is not applicable and may only push the sentence
toward `unknown`. Everything else is byte-for-byte identical; see
`diff prompts/v1_system_prompt.txt prompts/v2_system_prompt.txt`.

**Why: data analysis plus trace-based diagnosis.**

1. Gold tags: 131 of 376 development claims (34.8%) carry a scope tag
   (`policy_scope`, `model_scope`, `date_scope`, `condition_scope`, `scope`,
   `region_scope`). That is the largest failure-mode family.
2. Baseline trace, A02 / v1 (Langfuse trace `e5fa0060e09243fbaaf4446b84a8211b`):
   two turns, a `get_brand_facts` tool call followed by the final generation.
   The final output had 6 claims against 10 gold, all 6 exact gold spans
   (perfect precision). The 4 missed sentences are all gold `unknown`, and
   every one is a scope problem:
   - a return policy applied to an opened unit (`policy_scope`)
   - a "forty watts per channel" claim with no matching fact (`missing_fact`)
   - "AAC codec, so its Bluetooth version can be treated as…" (`feature_conflation`)
   - a full 18-hour battery rating below freezing (`condition_scope`)

   So the model kept only sentences it could tie to a fact. When the matching
   fact row existed but was scoped differently, or didn't exist, it dropped
   the sentence instead of extracting it as `unknown`.
3. Aggregate check of the one-trace hypothesis: across the 29 answers v1
   completed, 74 of the 77 missed gold claims are `unknown`. v1's `unknown`
   recall is 0.19 on the full split, against 0.66 to 0.68 for the other labels.

The checklist was meant to make the model treat a scope mismatch as a reason to
label a sentence `unknown`, rather than as a reason to ignore it.

**Measured effect: no gain.** Controlled subset:

| Label | v1 P / R / F1 | v2 P / R / F1 |
|---|---|---|
| supported | 0.965 / 0.902 / 0.932 | 0.950 / 0.934 / 0.942 |
| contradicted | 0.910 / 0.953 / 0.931 | 0.952 / 0.922 / 0.937 |
| unknown | 1.000 / 0.288 / 0.447 | 1.000 / 0.260 / 0.413 |

- Supported and contradicted F1 rose slightly. 6 label flips on shared
  matched spans: 4 fixed (A01, A04, A34, A40) and 2 broke (A05, A35), where v2
  turned a correct `contradicted` into `supported`.
- The target, `unknown` recall, did **not** improve (0.288 → 0.260). v2 also
  extracted slightly fewer sentences (P 145 → 141 on the same answers).
- Full-split regressions: v2 added one validation error (A02, duplicate
  sentence) and one more unknown-brand tool error. Per-answer deltas are in
  `runs/regression_analysis.md`.

**What the evidence says to do next (not done):** the problem is
**extraction**, not labelling. A v3 should change the extraction instruction:
enumerate every factual sentence first, without regard to the facts, then
label. That separates "is this a claim" from "can I verify it". It is not built,
because a third version after freezing would break the v1/v2 protocol.

**Final-check disclosure:** check inputs (A41–A50) are visible in
`answers.csv`. No tuning was done on them, and both versions were frozen before
the check runs. This is still not a fully blind benchmark.

## Langfuse evidence

- One trace per answer for all four runs: 100 traces, IDs stored in
  `answer_results.trace_id` and in every report's `answers[].langfuse_trace_id`.
- Each trace has `generation` observations for model turns, `tool`
  observations for `get_brand_facts`, and run/answer/split/version/provider/model
  metadata (`app/langfuse_integration.py`).
- Development traces carry `extraction_f1` and `joint_f1` scores plus
  missing/extra IDs in metadata. Each development run also has a
  `run_summary:<run_id>` trace with `run_joint_f1` and `run_macro_f1` scores.
- Check traces carry predictions and failures only, with no scores.
- Diagnosis trace: A02 v1 `e5fa0060e09243fbaaf4446b84a8211b`. Label-error
  example: A01 v1 `5f426bc10b97406e7cfff9d64dc2d08b`, where the model read
  "Mini is listed at INR 1,200 less than Plus" as contradicted although the
  facts give 2499 − 1299 = 1200 (expected supported; v2 fixed this one).
- **Still to add:** sanitized exports or screenshots of those two traces, for a
  reviewer without Langfuse access. They must come from the Langfuse UI using
  the account owner's login (see Unfinished work).

## AI assistance disclosure

Built with Claude Code (Anthropic). Claude Code wrote the agent, harness,
API/DB layer, tests, prompts and these documents. It ran the live runs with my
Gemini and Langfuse keys, supplied as environment variables and never written
to the repository. I directed the work and supplied the files and keys.

Checks performed:

- Scorer checked against the brief's worked example (G=10 P=10 M=8 C=7 →
  0.80 / 0.70), in `tests/test_scorer.py`.
- Exact-span matching cross-checked outside the scorer: a direct join of the DB
  claim rows against `development_gold.csv` confirms all 195 (v1) and 173 (v2)
  accepted predictions are exact gold spans, which is why extraction precision
  is 1.0.
- The report export re-scores from DB rows and reproduces the run-time metrics.
- A live failure exposed an early bug: the model was self-reporting character
  offsets. Both prompts were changed to quote-only, with offsets located in
  Python (`resolve_claim_offsets`, sequential search for repeated sentences).
  The fix went in **before** the four frozen runs.
- 47/47 tests pass (`runs/test_output.txt`).

## Time spent

About 1 h 15 min building and running (agent, harness, API/DB, tests, Gemini
adapter, live runs), plus about 30 min on analysis and write-up (report export,
regression analysis, diagnosis, docs). Total is roughly 1 h 45 min.

## Limitations and unfinished work

1. **Quota errors:** 22 of the 100 live answer calls failed on Gemini free-tier
   quota. This depresses both full-split scores, so the fair comparison is the
   21-answer controlled subset.
2. **PostgreSQL untested.** The SQLAlchemy models (unique constraints, foreign
   keys, parameterized queries) are dialect-agnostic, but every run and test
   used SQLite, because no Postgres server was available. To use Postgres, set
   `DATABASE_URL=postgresql+psycopg2://…` and install `psycopg2-binary`.
3. **Langfuse exports/screenshots** of the diagnosis traces are not yet in the
   repository.
4. **No Anthropic adapter.** OpenAI and Gemini are implemented.
5. **API tests** cover validation and error codes (422/404/502), and
   persistence is covered through the orchestrator. There is no full HTTP round
   trip `POST /evaluation-runs` → `GET /evaluation-runs/{id}` with a fake provider.
6. `runs.source_revision` is not populated.
7. The v3 extraction change described above is not built.
