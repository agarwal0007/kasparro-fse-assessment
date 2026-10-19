import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text,
    UniqueConstraint, create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from app.config import DATABASE_URL

Base = declarative_base()

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Run(Base):
    __tablename__ = "runs"

    run_id = Column(String, primary_key=True, default=_uuid)
    split = Column(String, nullable=False)  # "development" | "check"
    agent_version = Column(String, nullable=False)  # "v1" | "v2"
    status = Column(String, nullable=False, default="running")  # running|completed|failed
    provider = Column(String, nullable=True)
    model = Column(String, nullable=True)
    dataset_version = Column(String, nullable=True)
    fact_version = Column(String, nullable=True)
    evaluator_version = Column(String, nullable=True)
    settings = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)  # the full score_split() report for development; null for check
    source_revision = Column(String, nullable=True)
    file_hashes = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=_now)

    answers = relationship("AnswerResult", back_populates="run", cascade="all, delete-orphan")


class AnswerResult(Base):
    __tablename__ = "answer_results"
    __table_args__ = (UniqueConstraint("run_id", "answer_id", name="uq_run_answer"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("runs.run_id"), nullable=False)
    answer_id = Column(String, nullable=False)
    status = Column(String, nullable=False)  # "ok" | "error"
    error_type = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    turns_used = Column(Integer, nullable=True)
    trace_id = Column(String, nullable=True)

    run = relationship("Run", back_populates="answers")
    claims = relationship("ClaimResult", back_populates="answer_result", cascade="all, delete-orphan")


class ClaimResult(Base):
    __tablename__ = "claim_results"
    __table_args__ = (
        UniqueConstraint("run_id", "answer_id", "start_char", "end_char", name="uq_run_answer_span"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("runs.run_id"), nullable=False)
    answer_id = Column(String, nullable=False)
    answer_result_id = Column(Integer, ForeignKey("answer_results.id"), nullable=False)
    claim_id = Column(String, nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    quote = Column(Text, nullable=False)
    label = Column(String, nullable=False)
    fact_ids = Column(JSON, nullable=False)
    reason = Column(Text, nullable=False)
    # Development-only match info, joined after scoring; null for check runs.
    matched_gold_claim_id = Column(String, nullable=True)
    expected_label = Column(String, nullable=True)
    is_correct = Column(Integer, nullable=True)  # 1/0/None (SQLite has no native bool column need)

    answer_result = relationship("AnswerResult", back_populates="claims")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
