from dataclasses import dataclass, field

from app.harness.csv_loader import GoldClaim
from app.harness.scorer import score_split


@dataclass
class FakeClaim:
    claim_id: str
    start_char: int
    end_char: int
    label: str


@dataclass
class FakeOutcome:
    status: str
    claims: list = field(default_factory=list)


def gold(answer_id, start, end, label, claim_suffix=None):
    cid = f"{answer_id}:{start}:{end}"
    return GoldClaim(answer_id, cid, start, end, "q", label, (), "", ())


def pred(answer_id, start, end, label):
    return FakeClaim(f"{answer_id}:{start}:{end}", start, end, label)


def test_worked_example_from_the_brief():
    # G=10, P=10, M=8, C=7 -> extraction P/R/F1 = 0.80, joint P/R/F1 = 0.70.
    gold_claims = [gold("A01", i * 10, i * 10 + 5, "supported") for i in range(10)]
    # 8 predictions land on real gold spans (matched); 2 land on spans with no gold (extra).
    preds = []
    for i in range(8):
        # 7 of the 8 matches get the correct label; 1 gets the wrong label.
        label = "supported" if i < 7 else "contradicted"
        preds.append(pred("A01", i * 10, i * 10 + 5, label))
    preds.append(pred("A01", 999, 1004, "unknown"))  # extra, no gold at this span
    preds.append(pred("A01", 1010, 1015, "unknown"))  # extra, no gold at this span

    outcomes = {"A01": FakeOutcome("ok", preds)}
    report = score_split(gold_claims, outcomes)

    assert report["G"] == 10
    assert report["P"] == 10
    assert report["M"] == 8
    assert report["C"] == 7
    assert report["extraction"] == {"precision": 0.8, "recall": 0.8, "f1": 0.8}
    assert round(report["joint"]["precision"], 2) == 0.70
    assert round(report["joint"]["recall"], 2) == 0.70
    assert round(report["joint"]["f1"], 2) == 0.70


def test_perfect_output():
    gold_claims = [gold("A01", 0, 5, "supported"), gold("A01", 10, 15, "unknown")]
    preds = [pred("A01", 0, 5, "supported"), pred("A01", 10, 15, "unknown")]
    report = score_split(gold_claims, {"A01": FakeOutcome("ok", preds)})
    assert report["M"] == report["C"] == report["G"] == report["P"] == 2
    assert report["extraction"]["f1"] == 1.0
    assert report["joint"]["f1"] == 1.0
    assert report["missed_gold_ids"] == []
    assert report["unmatched_prediction_ids"] == []


def test_missed_claims_reduce_recall_not_precision():
    gold_claims = [gold("A01", 0, 5, "supported"), gold("A01", 10, 15, "unknown")]
    preds = [pred("A01", 0, 5, "supported")]  # the second gold claim is missed entirely
    report = score_split(gold_claims, {"A01": FakeOutcome("ok", preds)})
    assert report["G"] == 2 and report["P"] == 1 and report["M"] == 1
    assert report["extraction"]["precision"] == 1.0
    assert report["extraction"]["recall"] == 0.5
    assert report["missed_gold_ids"] == ["A01:10:15"]


def test_extra_advice_mistakenly_extracted_hurts_precision():
    gold_claims = [gold("A01", 0, 5, "supported")]
    preds = [pred("A01", 0, 5, "supported"), pred("A01", 50, 60, "unknown")]  # extra, invented
    report = score_split(gold_claims, {"A01": FakeOutcome("ok", preds)})
    assert report["P"] == 2 and report["M"] == 1
    assert report["extraction"]["precision"] == 0.5
    assert report["unmatched_prediction_ids"] == ["A01:50:60"]


def test_wrong_label_counts_as_extraction_match_but_not_joint_match():
    gold_claims = [gold("A01", 0, 5, "contradicted")]
    preds = [pred("A01", 0, 5, "supported")]  # right span, wrong label
    report = score_split(gold_claims, {"A01": FakeOutcome("ok", preds)})
    assert report["M"] == 1 and report["C"] == 0
    assert report["extraction"]["f1"] == 1.0
    assert report["joint"]["f1"] == 0.0
    assert report["matched_wrong_label_ids"] == ["A01:0:5"]
    assert report["confusion_matrix"]["contradicted"]["supported"] == 1


def test_provider_error_answer_contributes_zero_predictions_but_full_gold():
    gold_claims = [gold("A01", 0, 5, "supported"), gold("A01", 10, 15, "unknown")]
    outcomes = {"A01": FakeOutcome("error", [])}
    report = score_split(gold_claims, outcomes)
    assert report["G"] == 2 and report["P"] == 0 and report["M"] == 0
    assert report["extraction"]["precision"] == 0.0  # zero-denominator -> 0, not undefined
    assert report["extraction"]["recall"] == 0.0
    assert report["answer_error_count"] == 1
    assert set(report["missed_gold_ids"]) == {"A01:0:5", "A01:10:15"}


def test_valid_empty_predictions_when_gold_claims_exist_scores_zero_extraction_credit():
    gold_claims = [gold("A01", 0, 5, "supported")]
    outcomes = {"A01": FakeOutcome("ok", [])}
    report = score_split(gold_claims, outcomes)
    assert report["P"] == 0 and report["M"] == 0
    assert report["extraction"]["recall"] == 0.0
    assert report["answer_error_count"] == 0  # a valid empty list is not an error


def test_empty_scheduled_split_has_null_quality_metrics():
    report = score_split([], {})
    assert report["extraction"] is None
    assert report["joint"] is None
    assert report["macro_f1"] is None


def test_macro_f1_is_unweighted_average_across_three_labels():
    gold_claims = [
        gold("A01", 0, 5, "supported"),
        gold("A01", 10, 15, "contradicted"),
        gold("A01", 20, 25, "unknown"),
    ]
    preds = [pred("A01", 0, 5, "supported"), pred("A01", 10, 15, "contradicted")]  # unknown gold missed entirely
    report = score_split(gold_claims, {"A01": FakeOutcome("ok", preds)})
    # supported: tp=1,pred=1,actual=1 -> f1=1.0; contradicted same -> f1=1.0; unknown: tp=0,pred=0,actual=1 -> f1=0.0
    assert report["per_label"]["supported"]["f1"] == 1.0
    assert report["per_label"]["contradicted"]["f1"] == 1.0
    assert report["per_label"]["unknown"]["f1"] == 0.0
    assert round(report["macro_f1"], 4) == round((1.0 + 1.0 + 0.0) / 3, 4)
