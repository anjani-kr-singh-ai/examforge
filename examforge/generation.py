"""Question Setter (stems + correct answers), Distractor Generator, assembly and controlled revision."""
from __future__ import annotations

import hashlib
import json
import random

from .config import load_yaml
from .llm import LLMClient
from .schemas import (
    LABELS, MCQ, Chunk, DistractorBatch, GenerationRequest, MCQDraft, Option, RetrievedChunk, StemBatch, StemDraft,
)


def format_context(chunks: list[RetrievedChunk]) -> tuple[str, dict[str, Chunk]]:
    """Short ids (C1..Ck) are easier for a model to copy exactly than long chunk ids."""
    ctx: dict[str, Chunk] = {}
    parts = []
    for i, rc in enumerate(chunks, start=1):
        cid = f"C{i}"
        ctx[cid] = rc.chunk
        parts.append(f"[{cid}] {rc.chunk.citation()}\n{rc.chunk.text}")
    return "\n\n".join(parts), ctx


def attempt_note(req: GenerationRequest) -> str:
    if req.attempt:
        hist = " Historical material was intentionally included; flag anything that may have changed." if req.allow_historical else ""
        return f"Reference period: material prepared for the {req.attempt} examination attempt.{hist}"
    return "Reference period: not specified - do not assume which version of the law applies beyond the material."


def base_vars(req: GenerationRequest) -> dict:
    g = load_yaml("generation.yaml")
    subj = load_yaml("subjects.yaml")["subjects"].get(req.subject, {}).get("name", req.subject)
    rub = g["difficulty_rubric"]
    tax = g["misconception_taxonomy"]
    return {
        "subject": subj, "chapter": req.chapter, "topic": req.topic, "n": req.n, "difficulty": req.difficulty,
        "question_type": req.question_type, "question_type_rules": g["question_types"][req.question_type],
        "question_type_rule": g["question_types"][req.question_type],
        "mode": req.mode, "mode_instructions": g["modes"][req.mode], "attempt_note": attempt_note(req),
        "rubric_easy": rub["Easy"], "rubric_medium": rub["Medium"], "rubric_hard": rub["Hard"],
        "rubric_target": rub[req.difficulty],
        "taxonomy": "\n".join(f"- {k}: {v}" for k, v in tax.items()), "taxonomy_keys": ", ".join(tax),
    }


def generate_stems(llm: LLMClient, req: GenerationRequest, context_text: str, ctx: dict[str, Chunk], avoid: list[str]) -> StemBatch:
    v = base_vars(req) | {
        "context": context_text,
        "avoid": "\n".join(f"- {q}" for q in avoid[:30]) or "(none)",
        "_ctx": ctx,
    }
    return llm.run("question_setter", StemBatch, v)


def generate_distractors(llm: LLMClient, req: GenerationRequest, stems: list[StemDraft], context_text: str, ctx: dict[str, Chunk]) -> DistractorBatch:
    items = [
        {"question_index": i, "question": s.question, "correct_options": s.correct_options,
         "explanation": s.explanation, "distractors_needed": max(0, 4 - len(s.correct_options))}
        for i, s in enumerate(stems)
    ]
    v = base_vars(req) | {"items": json.dumps(items, indent=1, ensure_ascii=False), "context": context_text, "_items": items, "_ctx": ctx}
    return llm.run("distractor_generator", DistractorBatch, v)


def assemble(stem: StemDraft, batch: DistractorBatch, index: int) -> tuple[MCQ, list[str]]:
    """Combine correct options + distractors, shuffle deterministically (seeded by the stem) and label A-D.
    Returns (mcq, notes). Count mismatches are not repaired here: validation reports them."""
    notes = []
    needed = max(0, 4 - len(stem.correct_options))
    ds = next((s.distractors for s in batch.items if s.question_index == index), [])
    if len(ds) != needed:
        notes.append(f"distractor stage returned {len(ds)} distractors, needed {needed}")
    ds = ds[:needed] if len(ds) > needed else ds
    expl = stem.correct_option_explanations + [""] * len(stem.correct_options)
    opts = [Option(label="", text=t, is_correct=True, explanation=expl[i]) for i, t in enumerate(stem.correct_options)]
    opts += [Option(label="", text=d.text, is_correct=False, explanation=d.why_incorrect,
                    misconception_category=d.misconception_category) for d in ds]
    seed = int(hashlib.sha256(stem.question.encode()).hexdigest(), 16) % (2**32)
    random.Random(seed).shuffle(opts)
    for lab, o in zip(LABELS + tuple(f"X{i}" for i in range(10)), opts):
        o.label = lab
    mcq = MCQ(
        question=stem.question, options=opts, correct_labels=[o.label for o in opts if o.is_correct],
        explanation=stem.explanation, evidence=stem.evidence, assumptions=stem.assumptions,
        concepts_tested=stem.concepts_tested,
    )
    return mcq, notes


def draft_to_mcq(d: MCQDraft) -> MCQ:
    keys = set(d.correct_labels)
    return MCQ(
        question=d.question,
        options=[Option(label=o.label, text=o.text, is_correct=o.label in keys, explanation=o.explanation,
                        misconception_category=o.misconception_category) for o in d.options],
        correct_labels=list(d.correct_labels), explanation=d.explanation, evidence=d.evidence,
        assumptions=d.assumptions, concepts_tested=d.concepts_tested,
    )


def mcq_json(mcq: MCQ) -> str:
    return json.dumps({
        "question": mcq.question,
        "options": [{"label": o.label, "text": o.text, "explanation": o.explanation,
                     "misconception_category": o.misconception_category} for o in mcq.options],
        "correct_labels": mcq.correct_labels, "explanation": mcq.explanation,
        "evidence": [e.model_dump() for e in mcq.evidence], "assumptions": mcq.assumptions,
        "concepts_tested": mcq.concepts_tested,
    }, indent=1, ensure_ascii=False)


def revise(llm: LLMClient, req: GenerationRequest, mcq: MCQ, issues: list[str], context_text: str, ctx: dict[str, Chunk]) -> tuple[MCQ, str]:
    v = base_vars(req) | {
        "question_json": mcq_json(mcq), "issues": "\n".join(f"- {i}" for i in issues), "context": context_text,
        "_mcq": mcq, "_ctx": ctx,
    }
    d = llm.run("reviser", MCQDraft, v)
    return draft_to_mcq(d), d.change_summary
