"""
Deterministic harness scorer. Matching rule: a prediction matches a gold
claim only when answer_id, start_char and end_char are all identical (quote
equality was already enforced at validation time). Each gold span matches at
most one prediction. No LLM judge, no fuzzy matching, ever.

Definitions (G, P, M, C) are exactly as specified:
  G = gold claims in the scheduled split (every answer's gold claims count,
      including answers that errored)
  P = accepted predictions (an answer error contributes 0)
  M = exact span matches, regardless of label
  C = matches whose label is also correct
"""
from collections import Counter, defaultdict

from app.harness.csv_loader import GoldClaim

LABELS = ("supported", "contradicted", "unknown")


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def score_split(gold_claims: list[GoldClaim], outcomes: dict) -> dict:
    """
    gold_claims: all GoldClaim rows for the answers in this scheduled split.
    outcomes: {answer_id: AnswerOutcome} for every scheduled answer (errors included).
    Returns a dict with every metric described in the brief. If the split has
    zero gold claims (e.g. a check split, or an empty schedule), quality
    metrics are null.
    """
    gold_by_answer: dict[str, list[GoldClaim]] = defaultdict(list)
    for g in gold_claims:
        gold_by_answer[g.answer_id].append(g)

    G = len(gold_claims)
    if G == 0:
        return {
            "G": 0, "P": 0, "M": 0, "C": 0,
            "extraction": None, "joint": None,
            "per_label": None, "macro_f1": None,
            "confusion_matrix": None,
            "missed_gold_ids": [], "unmatched_prediction_ids": [], "matched_wrong_label_ids": [],
            "answer_error_count": sum(1 for o in outcomes.values() if o.status == "error"),
        }

    accepted_predictions = []  # (answer_id, claim) for every accepted prediction
    for answer_id, outcome in outcomes.items():
        if outcome.status == "ok":
            for claim in outcome.claims:
                accepted_predictions.append((answer_id, claim))
    P = len(accepted_predictions)

    # Index predictions by (answer_id, start, end) for exact-span matching.
    pred_by_span = {}
    for answer_id, claim in accepted_predictions:
        pred_by_span[(answer_id, claim.start_char, claim.end_char)] = claim

    matched_gold_keys = set()
    missed_gold_ids = []
    matched_wrong_label_ids = []
    confusion = defaultdict(int)  # (expected_row, predicted_col) -> count
    label_tp = Counter()
    label_pred_pos = Counter()
    label_actual_pos = Counter()

    M = 0
    C = 0

    for g in gold_claims:
        label_actual_pos[g.expected_label] += 1
        key = (g.answer_id, g.start_char, g.end_char)
        pred = pred_by_span.get(key)
        if pred is None:
            missed_gold_ids.append(g.claim_id)
            confusion[(g.expected_label, "missing_or_error")] += 1
            continue
        # Matched.
        matched_gold_keys.add(key)
        M += 1
        confusion[(g.expected_label, pred.label)] += 1
        if pred.label == g.expected_label:
            C += 1
            label_tp[g.expected_label] += 1
        else:
            matched_wrong_label_ids.append(pred.claim_id)

    unmatched_prediction_ids = []
    for key, pred in pred_by_span.items():
        label_pred_pos[pred.label] += 1
        if key not in matched_gold_keys:
            unmatched_prediction_ids.append(pred.claim_id)
            confusion[("no_gold", pred.label)] += 1

    extraction = {
        "precision": _safe_div(M, P),
        "recall": _safe_div(M, G),
        "f1": _safe_div(2 * M, P + G),
    }
    joint = {
        "precision": _safe_div(C, P),
        "recall": _safe_div(C, G),
        "f1": _safe_div(2 * C, P + G),
    }

    per_label = {}
    f1s = []
    for label in LABELS:
        tp = label_tp[label]
        pred_pos = label_pred_pos[label]
        actual_pos = label_actual_pos[label]
        precision = _safe_div(tp, pred_pos)
        recall = _safe_div(tp, actual_pos)
        f1 = _safe_div(2 * tp, pred_pos + actual_pos)
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp,
                             "predicted_positives": pred_pos, "actual_positives": actual_pos}
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s)

    confusion_matrix = {}
    rows = LABELS + ("no_gold",)
    cols = LABELS + ("missing_or_error",)
    for r in rows:
        confusion_matrix[r] = {c: confusion.get((r, c), 0) for c in cols}
    confusion_matrix["no_gold"]["missing_or_error"] = 0  # always zero by construction

    return {
        "G": G, "P": P, "M": M, "C": C,
        "extraction": extraction,
        "joint": joint,
        "per_label": per_label,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion_matrix,
        "missed_gold_ids": missed_gold_ids,
        "unmatched_prediction_ids": unmatched_prediction_ids,
        "matched_wrong_label_ids": matched_wrong_label_ids,
        "gold_coverage": _safe_div(C, G),
        "answer_error_count": sum(1 for o in outcomes.values() if o.status == "error"),
    }
