"""Runtime settings (env vars / .env) and YAML config loading."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


@dataclass
class Settings:
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "mock"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "claude-opus-5-5"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL"))
    llm_effort: str = field(default_factory=lambda: _env("LLM_EFFORT", "high"))
    llm_max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "16000")))
    reviewer_provider: str = field(default_factory=lambda: _env("REVIEWER_PROVIDER"))
    reviewer_model: str = field(default_factory=lambda: _env("REVIEWER_MODEL"))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "350")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "60")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "6")))
    min_relevance: float = field(default_factory=lambda: float(_env("MIN_RELEVANCE", "0.25")))
    max_retries: int = field(default_factory=lambda: int(_env("MAX_RETRIES", "3")))
    max_revisions: int = field(default_factory=lambda: int(_env("MAX_REVISIONS", "1")))
    duplicate_threshold: float = field(default_factory=lambda: float(_env("DUPLICATE_THRESHOLD", "0.92")))
    similar_threshold: float = field(default_factory=lambda: float(_env("SIMILAR_THRESHOLD", "0.82")))
    prompt_versions: dict[str, str] = field(
        default_factory=lambda: dict(
            kv.split("=", 1) for kv in _env("PROMPT_VERSIONS").split(",") if "=" in kv
        )
    )
    data_dir: Path = field(default_factory=lambda: Path(_env("EXAMFORGE_DATA", str(ROOT / "data"))))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def bank_path(self) -> Path:
        return self.data_dir / "bank" / "questions.json"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "metadata" / "sources.json"


@lru_cache
def load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))
