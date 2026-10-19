import os
import tempfile

# A real temp file, not ":memory:" - SQLite's in-memory DB is per-connection,
# and SQLAlchemy's default pool opens multiple connections, so ":memory:"
# would silently give each query a different, empty database.
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_db.name}"

from fastapi.testclient import TestClient

from app.db import init_db
from app.main import app

init_db()
client = TestClient(app)


def test_classify_rejects_gold_fields_with_422():
    resp = client.post("/classify", json={
        "answer_id": "A01", "brand_id": "cleartrail", "question": "Q?", "ai_answer": "x",
        "agent_version": "v1", "expected_label": "supported",
    })
    assert resp.status_code == 422


def test_classify_unknown_brand_is_404():
    resp = client.post("/classify", json={
        "answer_id": "A01", "brand_id": "totally_unknown_brand", "question": "Q?",
        "ai_answer": "x", "agent_version": "v1",
    })
    assert resp.status_code == 404


def test_classify_without_a_live_provider_is_502():
    # No OPENAI_API_KEY is configured in this test environment.
    resp = client.post("/classify", json={
        "answer_id": "A01", "brand_id": "cleartrail", "question": "Q?",
        "ai_answer": "x", "agent_version": "v1",
    })
    assert resp.status_code == 502


def test_evaluation_run_invalid_body_is_422():
    resp = client.post("/evaluation-runs", json={"agent_version": "v3", "split": "development"})
    assert resp.status_code == 422


def test_get_evaluation_run_invalid_uuid_is_422():
    resp = client.get("/evaluation-runs/not-a-uuid")
    assert resp.status_code == 422


def test_get_evaluation_run_missing_run_is_404():
    resp = client.get("/evaluation-runs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
