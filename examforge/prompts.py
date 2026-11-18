"""Versioned prompt files: prompts/<stage>/<version>.md

File format:
    ---            (optional YAML front matter: notes, changelog)
    ---
    ## system
    ...
    ## user
    ... $variables use string.Template syntax ($name / ${name}; $$ for a literal $)

A missing variable raises KeyError, so a prompt/code mismatch fails loudly instead of sending "$context".
"""
from __future__ import annotations

from functools import lru_cache
from string import Template

from .config import ROOT

PROMPT_DIR = ROOT / "prompts"


@lru_cache
def load(stage: str, version: str) -> tuple[str, str]:
    path = PROMPT_DIR / stage / f"{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt not found: {path}")
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        text = text.split("---", 2)[2]
    try:
        _, rest = text.split("## system", 1)
        system, user = rest.split("## user", 1)
    except ValueError as e:
        raise ValueError(f"{path} must contain '## system' and '## user' sections") from e
    return system.strip(), user.strip()


def render(stage: str, version: str, variables: dict) -> tuple[str, str]:
    system, user = load(stage, version)
    vs = {k: str(v) for k, v in variables.items()}
    return Template(system).substitute(vs), Template(user).substitute(vs)


def versions(stage: str) -> list[str]:
    return sorted(p.stem for p in (PROMPT_DIR / stage).glob("*.md"))
