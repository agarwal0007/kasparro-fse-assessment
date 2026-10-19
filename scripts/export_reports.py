"""
Exports the four frozen live runs from the database into full, self-contained
reports (runs/report_<split>_<version>.json) and writes the v1 -> v2
regression analysis (runs/regression_analysis.json + .md).

This makes NO provider calls. It reads only what scripts/run_live.py already
persisted (runs, answer_results, claim_results) and re-scores development
runs with the same deterministic scorer, so the numbers are reproducible.

Usage: python scripts/export_reports.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agent.runner import AnswerOutcome
from app.agent.validation import Claim
from app.db import AnswerResult, ClaimResult, Run, SessionLocal
from app.harness.csv_loader import load_answers, load_gold
from app.harness.scorer import LABELS, score_split

ROOT = os.path.join(os.path.dirname(__file__), "..")
RUNS_DIR = os.path.join(ROOT, "runs")
DATA_DIR = os.path.join(ROOT, "data")


def latest_live_runs(db) -> dict:
    """The newest live (non-fake) run for each (split, agent_version)."""
    runs = {}
    for run in db.query(Run).filter(Run.provider.notin_(["fake", "none"])).order_by(Run.created_at):
        runs[(run.split, run.agent_version)] = run
    return runs


def load_outcomes(db, run) -> dict:
    outcomes = {}
    answer_rows = db.query(AnswerResult).filter(AnswerResult.run_id == run.run_id).all()
    for ar in answer_rows:
        claims = (
            db.query(ClaimResult)
            .filter(ClaimResult.answer_result_id == ar.id)
            .order_by(ClaimResult.start_char, ClaimResult.end_char)
            .all()
        )
        outcomes[ar.answer_id] = (ar, claims, AnswerOutcome(
            answer_id=ar.answer_id, status=ar.status,
            claims=[Claim(c.claim_id, c.start_char, c.end_char, c.quote, c.label, tuple(c.fact_ids), c.reason)
                    for c in claims],
            error_type=ar.error_type, error_message=ar.error_message,
            turns=[], latency_ms=ar.latency_ms or 0.0, turns_used=ar.turns_used or 0,
        ))
    return dict(sorted(outcomes.items()))


def sanitize(message):
    """Drop provider response bodies from error text; keep the type-level cause."""
    if not message:
        return message
    return message.split(". {")[0][:200]


def build_report(run, outcomes, gold) -> dict:
    is_dev = run.split == "development"
    gold_by_span = {(g.answer_id, g.start_char, g.end_char): g for g in gold}
    answers = []
    for answer_id, (ar, claims, _) in outcomes.items():
        answer_gold = [g for g in gold if g.answer_id == answer_id]
        per_answer = score_split(answer_gold, {answer_id: outcomes[answer_id][2]}) if is_dev else None
        answers.append({
            "answer_id": answer_id,
            "status": ar.status,
            "error_type": ar.error_type,
            "error_message": sanitize(ar.error_message),
            "latency_ms": ar.latency_ms,
            "turns_used": ar.turns_used,
            "langfuse_trace_id": ar.trace_id,
            "prediction_count": len(claims),
            "gold_count": len(answer_gold) if is_dev else None,
            "extraction_f1": per_answer["extraction"]["f1"] if per_answer and per_answer["extraction"] else None,
            "joint_f1": per_answer["joint"]["f1"] if per_answer and per_answer["joint"] else None,
            "predictions": [{
                "claim_id": c.claim_id, "start_char": c.start_char, "end_char": c.end_char,
                "quote": c.quote, "label": c.label, "fact_ids": c.fact_ids, "reason": c.reason,
                **({
                    "matched_gold_claim_id": c.matched_gold_claim_id,
                    "expected_label": c.expected_label,
                    "label_correct": None if c.is_correct is None else bool(c.is_correct),
                } if is_dev else {}),
            } for c in claims],
        })

    report = {
        "run_id": run.run_id, "split": run.split, "agent_version": run.agent_version,
        "status": run.status, "provider": run.provider, "model": run.model,
        "dataset_version": run.dataset_version, "fact_version": run.fact_version,
        "evaluator_version": run.evaluator_version, "prompt_file": f"prompts/{run.agent_version}_system_prompt.txt",
        "settings": run.settings, "file_hashes": run.file_hashes, "created_at": str(run.created_at),
        "counts": {
            "scheduled": len(outcomes),
            "ok": sum(1 for _, (ar, _, _) in outcomes.items() if ar.status == "ok"),
            "error": sum(1 for _, (ar, _, _) in outcomes.items() if ar.status == "error"),
            "empty_ok": sum(1 for _, (ar, c, _) in outcomes.items() if ar.status == "ok" and not c),
            "accepted_predictions": sum(len(c) for _, (_, c, _) in outcomes.items()),
        },
        "metrics": None,
        "answers": answers,
    }
    if is_dev:
        m = score_split(gold, {k: v[2] for k, v in outcomes.items()})
        report["metrics"] = m
        report["missing_gold"] = [{
            "claim_id": g.claim_id, "answer_id": g.answer_id, "start_char": g.start_char,
            "end_char": g.end_char, "expected_label": g.expected_label,
            "answer_status": outcomes[g.answer_id][0].status,
        } for g in gold if g.claim_id in set(m["missed_gold_ids"])]
        report["extra_predictions"] = [p for a in answers for p in a["predictions"]
                                       if p["claim_id"] in set(m["unmatched_prediction_ids"])]
        report["wrong_label"] = [p for a in answers for p in a["predictions"]
                                 if p.get("label_correct") is False]
    return report


def regression(v1_out, v2_out, gold) -> dict:
    both_ok = sorted(a for a in v1_out if v1_out[a][0].status == "ok" and v2_out.get(a, (None,))[0]
                     and v2_out[a][0].status == "ok")
    controlled_gold = [g for g in gold if g.answer_id in both_ok]
    v1_c = score_split(controlled_gold, {a: v1_out[a][2] for a in both_ok})
    v2_c = score_split(controlled_gold, {a: v2_out[a][2] for a in both_ok})

    per_answer = []
    for a in sorted(v1_out):
        ag = [g for g in gold if g.answer_id == a]
        s1 = score_split(ag, {a: v1_out[a][2]})
        s2 = score_split(ag, {a: v2_out[a][2]})
        per_answer.append({
            "answer_id": a, "v1_status": v1_out[a][0].status, "v2_status": v2_out[a][0].status,
            "v1_error_type": v1_out[a][0].error_type, "v2_error_type": v2_out[a][0].error_type,
            "gold": len(ag), "v1_P": s1["P"], "v2_P": s2["P"],
            "v1_joint_f1": s1["joint"]["f1"], "v2_joint_f1": s2["joint"]["f1"],
            "delta_joint_f1": s2["joint"]["f1"] - s1["joint"]["f1"],
        })

    # Label flips on spans both versions extracted and that match gold.
    flips = []
    for a in both_ok:
        v1_by_span = {(c.start_char, c.end_char): c for c in v1_out[a][1]}
        for c2 in v2_out[a][1]:
            c1 = v1_by_span.get((c2.start_char, c2.end_char))
            if c1 and c2.matched_gold_claim_id and c1.label != c2.label:
                flips.append({
                    "claim_id": c2.claim_id, "gold_claim_id": c2.matched_gold_claim_id,
                    "expected": c2.expected_label, "v1": c1.label, "v2": c2.label,
                    "effect": "fixed" if c2.is_correct else ("broke" if c1.is_correct else "still_wrong"),
                })
    return {
        "controlled_answer_ids": both_ok,
        "controlled_note": "Answers that completed (status ok) in BOTH v1 and v2 development runs; "
                           "removes the effect of provider quota errors, which hit different answers per run.",
        "controlled_v1": {k: v1_c[k] for k in ("G", "P", "M", "C", "extraction", "joint", "per_label", "macro_f1")},
        "controlled_v2": {k: v2_c[k] for k in ("G", "P", "M", "C", "extraction", "joint", "per_label", "macro_f1")},
        "per_answer": per_answer,
        "label_flips_on_shared_matched_spans": flips,
    }


def f(x):
    return "—" if x is None else f"{x:.3f}"


def write_markdown(reports, reg, path):
    d1, d2 = reports[("development", "v1")], reports[("development", "v2")]
    lines = ["# v1 → v2 regression analysis", "",
             "Generated by `scripts/export_reports.py` from the frozen live runs in the database. No model calls.", "",
             "## Full development split (all 40 answers, 376 gold claims)", "",
             "| Metric | v1 | v2 |", "|---|---|---|"]
    for name, getter in [
        ("answers ok / error", lambda r: f"{r['counts']['ok']} / {r['counts']['error']}"),
        ("P (accepted predictions)", lambda r: r["metrics"]["P"]),
        ("M / C", lambda r: f"{r['metrics']['M']} / {r['metrics']['C']}"),
        ("extraction F1", lambda r: f(r["metrics"]["extraction"]["f1"])),
        ("joint F1", lambda r: f(r["metrics"]["joint"]["f1"])),
        ("macro F1", lambda r: f(r["metrics"]["macro_f1"])),
        ("gold coverage C/G", lambda r: f(r["metrics"]["gold_coverage"])),
    ]:
        lines.append(f"| {name} | {getter(d1)} | {getter(d2)} |")
    c1, c2 = reg["controlled_v1"], reg["controlled_v2"]
    lines += ["", f"## Controlled comparison: {len(reg['controlled_answer_ids'])} answers ok in both runs "
              f"({c1['G']} gold claims)", "", reg["controlled_note"], "",
              "| Metric | v1 | v2 | Δ |", "|---|---|---|---|"]
    for name, k in [("extraction F1", "extraction"), ("joint F1", "joint")]:
        lines.append(f"| {name} | {f(c1[k]['f1'])} | {f(c2[k]['f1'])} | {c2[k]['f1'] - c1[k]['f1']:+.3f} |")
    lines.append(f"| macro F1 | {f(c1['macro_f1'])} | {f(c2['macro_f1'])} | {c2['macro_f1'] - c1['macro_f1']:+.3f} |")
    lines.append(f"| P / M / C | {c1['P']} / {c1['M']} / {c1['C']} | {c2['P']} / {c2['M']} / {c2['C']} | |")
    lines += ["", "### Per-label (controlled)", "", "| Label | v1 P / R / F1 | v2 P / R / F1 |", "|---|---|---|"]
    for lab in LABELS:
        a, b = c1["per_label"][lab], c2["per_label"][lab]
        lines.append(f"| {lab} | {f(a['precision'])} / {f(a['recall'])} / {f(a['f1'])} | "
                     f"{f(b['precision'])} / {f(b['recall'])} / {f(b['f1'])} |")
    flips = reg["label_flips_on_shared_matched_spans"]
    lines += ["", f"### Label flips on spans both versions extracted ({len(flips)})", ""]
    if flips:
        lines += ["| Gold claim | expected | v1 | v2 | effect |", "|---|---|---|---|---|"]
        lines += [f"| {x['gold_claim_id']} | {x['expected']} | {x['v1']} | {x['v2']} | {x['effect']} |" for x in flips]
    else:
        lines.append("None.")
    lines += ["", "## Per-answer joint F1", "",
              "| Answer | gold | v1 status | v2 status | v1 P | v2 P | v1 joint F1 | v2 joint F1 | Δ |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in reg["per_answer"]:
        s1 = r["v1_status"] if r["v1_status"] == "ok" else f"error ({r['v1_error_type']})"
        s2 = r["v2_status"] if r["v2_status"] == "ok" else f"error ({r['v2_error_type']})"
        lines.append(f"| {r['answer_id']} | {r['gold']} | {s1} | {s2} | {r['v1_P']} | {r['v2_P']} | "
                     f"{f(r['v1_joint_f1'])} | {f(r['v2_joint_f1'])} | {r['delta_joint_f1']:+.3f} |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    gold = load_gold(os.path.join(DATA_DIR, "development_gold.csv"))
    dev_ids = {a.answer_id for a in load_answers(os.path.join(DATA_DIR, "answers.csv")) if a.split == "development"}
    gold = [g for g in gold if g.answer_id in dev_ids]
    db = SessionLocal()
    try:
        runs = latest_live_runs(db)
        missing = {("development", "v1"), ("development", "v2"), ("check", "v1"), ("check", "v2")} - set(runs)
        if missing:
            sys.exit(f"missing live runs: {sorted(missing)}")
        outcomes, reports = {}, {}
        for key, run in sorted(runs.items()):
            outcomes[key] = load_outcomes(db, run)
            reports[key] = build_report(run, outcomes[key], gold if key[0] == "development" else [])
            path = os.path.join(RUNS_DIR, f"report_{key[0]}_{key[1]}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(reports[key], fh, indent=2, ensure_ascii=False, default=str)
            print(f"wrote {path}  ok={reports[key]['counts']['ok']} error={reports[key]['counts']['error']}")
        reg = regression(outcomes[("development", "v1")], outcomes[("development", "v2")], gold)
        with open(os.path.join(RUNS_DIR, "regression_analysis.json"), "w", encoding="utf-8") as fh:
            json.dump(reg, fh, indent=2, default=str)
        write_markdown(reports, reg, os.path.join(RUNS_DIR, "regression_analysis.md"))
        print(f"controlled answers: {len(reg['controlled_answer_ids'])}  "
              f"v1 joint={reg['controlled_v1']['joint']['f1']:.3f}  v2 joint={reg['controlled_v2']['joint']['f1']:.3f}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
