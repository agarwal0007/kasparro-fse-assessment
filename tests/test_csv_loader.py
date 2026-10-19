import csv
import tempfile
from pathlib import Path

from app.harness.csv_loader import load_answers, load_brand_facts, load_gold


def test_quoted_commas_and_embedded_newlines_are_preserved(tmp_path):
    path = tmp_path / "answers.csv"
    # A real embedded comma and a real embedded newline inside a quoted field.
    content = (
        "answer_id,split,brand_id,question,ai_answer\r\n"
        'A01,development,acme,"A question, with a comma?","Line one.\nLine two, with a comma."\r\n'
    )
    path.write_bytes(content.encode("utf-8"))

    records = load_answers(str(path))
    assert len(records) == 1
    r = records[0]
    assert r.question == "A question, with a comma?"
    assert r.ai_answer == "Line one.\nLine two, with a comma."


def test_ai_answer_is_not_normalized_or_stripped(tmp_path):
    path = tmp_path / "answers.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["answer_id", "split", "brand_id", "question", "ai_answer"])
        writer.writerow(["A01", "development", "acme", "Q?", "  leading and trailing spaces kept  "])
    records = load_answers(str(path))
    assert records[0].ai_answer == "  leading and trailing spaces kept  "


def test_boolean_value_type_parses_literal_text_not_python_bool(tmp_path):
    path = tmp_path / "facts.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fact_id", "brand_id", "model_id", "field", "value", "value_type",
                          "unit", "condition", "valid_from", "valid_to"])
        writer.writerow(["f1", "acme", "m1", "has_mic", "false", "boolean", "", "", "2026-01-01", "2026-12-31"])
        writer.writerow(["f2", "acme", "m1", "has_usb", "true", "boolean", "", "", "2026-01-01", "2026-12-31"])
    records = load_brand_facts(str(path))
    # bool("false") is truthy in plain Python - must not happen here.
    assert records[0].value is False
    assert records[1].value is True


def test_number_value_type_parses_as_numeric(tmp_path):
    path = tmp_path / "facts.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fact_id", "brand_id", "model_id", "field", "value", "value_type",
                          "unit", "condition", "valid_from", "valid_to"])
        writer.writerow(["f1", "acme", "m1", "price_2026", "1299", "number", "INR", "", "2026-01-01", "2026-12-31"])
    records = load_brand_facts(str(path))
    assert records[0].value == 1299.0


def test_gold_evidence_fact_ids_split_on_pipe(tmp_path):
    path = tmp_path / "gold.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["answer_id", "claim_id", "start_char", "end_char", "quote", "expected_label",
                          "evidence_fact_ids", "explanation", "difficulty_tags"])
        writer.writerow(["A01", "A01:0:5", "0", "5", "hello", "supported",
                          "f1|f2", "because", "arithmetic|comparison"])
    records = load_gold(str(path))
    assert records[0].evidence_fact_ids == ("f1", "f2")
    assert records[0].difficulty_tags == ("arithmetic", "comparison")


def test_real_issued_csvs_load_without_error():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    answers = load_answers(str(data_dir / "answers.csv"))
    facts = load_brand_facts(str(data_dir / "brand_facts.csv"))
    gold = load_gold(str(data_dir / "development_gold.csv"))
    assert len(answers) == 50
    assert len(facts) == 132
    assert len(gold) == 376
    dev_answers = [a for a in answers if a.split == "development"]
    assert len(dev_answers) == 40
