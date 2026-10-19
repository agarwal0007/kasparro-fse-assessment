"""Fixed version strings and settings for this dataset/evaluator/fact version, per the brief."""
import os

DATASET_VERSION = "long-brand-answers-50-v4"
FACT_VERSION = "speaker-catalogue-v4"
EVALUATOR_VERSION = "sentence-spans-and-labels-v4"
QUESTION_DATE = "2026-10-01"  # "the question is dated 1 October 2026"

MAX_TURNS_PER_ANSWER = 3
MAX_CONCURRENT_PROVIDER_REQUESTS = 3
PROVIDER_TIMEOUT_SECONDS = 15
MAX_ATTEMPTS_PER_TURN = 2  # at most one retry

VALID_LABELS = {"supported", "contradicted", "unknown"}

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./kasparro.db")

LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
