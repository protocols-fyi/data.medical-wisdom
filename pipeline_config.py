"""Shared runtime config and fixed constants.

Inputs:
- Resolved CLI flags from main.py.

Outputs:
- Immutable RuntimeConfig passed into pipeline helpers.
- Shared defaults and smoke-test records used across commands.
"""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_NAME = "anthropic-haiku-4.5"
DEFAULT_MAX_TOKENS = 2400
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_REPAIR_BACKOFF_SECONDS = (30.0, 180.0)
DEFAULT_MAX_REPAIR_ATTEMPTS = 2
SMOKE_RECORDS = [
    {
        "dataset": "health_search_qa",
        "id": "smoke_001",
        "question": "How serious is atrial fibrillation?",
    },
    {
        "dataset": "medquad",
        "id": "smoke_002",
        "question": "What should I do about a persistent sore throat?",
    },
    {
        "dataset": "mash_qa",
        "id": "smoke_003",
        "question": "Can stress cause chest pain even if heart tests were normal?",
    },
]


@dataclass(frozen=True)
class RuntimeConfig:
    dry_run: bool
    limit: int
    region: str
    profile: str | None
    model_name: str
    max_tokens: int
    timeout_seconds: float
    workdir: Path
