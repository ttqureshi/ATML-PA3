from __future__ import annotations

import re
from typing import Any, Optional

NO_ANSWER = "NO_ANSWER"

_MATH_VERIFY_VERSION: Optional[str] = None
_MATH_VERIFY_READY = False


def scorer_versions() -> dict[str, str]:
    versions = {
        "math": "math-verify-unavailable",
        "mathvista": "mathvista-local-v1",
        "no_answer_token": NO_ANSWER,
    }
    try:
        import math_verify  # type: ignore

        versions["math"] = f"math-verify=={getattr(math_verify, '__version__', 'unknown')}"
    except Exception as exc:  # pragma: no cover
        versions["math_error"] = str(exc)
    return versions


def _ensure_math_verify() -> bool:
    global _MATH_VERIFY_READY, _MATH_VERIFY_VERSION
    if _MATH_VERIFY_READY:
        return True
    try:
        import math_verify  # type: ignore

        _MATH_VERIFY_VERSION = getattr(math_verify, "__version__", "unknown")
        _MATH_VERIFY_READY = True
        return True
    except Exception:
        return False


def normalize_math_answer(answer: str | None) -> str:
    if answer is None:
        return NO_ANSWER
    text = str(answer).strip()
    if not text:
        return NO_ANSWER
    return text


def math_answers_equal(a: str | None, b: str | None) -> bool:
    """MATH equivalence via math-verify when available; else normalized string match."""
    na = normalize_math_answer(a)
    nb = normalize_math_answer(b)
    # NO_ANSWER never equals another answer (including another NO_ANSWER) for grading/voting groups:
    # missing extractions are collapsed separately into one NO_ANSWER group by form_answer_groups.
    if na == NO_ANSWER or nb == NO_ANSWER:
        return False
    if _ensure_math_verify():
        try:
            from math_verify import parse, verify  # type: ignore

            return bool(verify(parse(na), parse(nb)))
        except Exception:
            pass
    return _fallback_math_equal(na, nb)


def _fallback_math_equal(a: str, b: str) -> bool:
    def clean(x: str) -> str:
        x = x.strip().replace(",", "")
        x = re.sub(r"\$", "", x)
        x = re.sub(r"\\left|\\right", "", x)
        x = re.sub(r"\s+", "", x)
        return x.lower()

    ca, cb = clean(a), clean(b)
    if ca == cb:
        return True
    try:
        return abs(float(ca) - float(cb)) < 1e-6
    except Exception:
        return False


def is_missing_extraction(answer: Any) -> bool:
    if answer is None:
        return True
    text = str(answer).strip()
    return text == "" or text.upper() in {NO_ANSWER, "NONE", "NULL", "NAN"}


def canonical_group_key(answer: Any, dataset: str, metadata: dict | None = None) -> str:
    """Form answer-group keys without consulting gold."""
    if is_missing_extraction(answer):
        return NO_ANSWER
    dataset = dataset.lower()
    text = str(answer).strip()
    if dataset in {"math", "llm", "qwen2.5-3b-instruct"}:
        return f"MATH::{normalize_math_answer(text)}"
    # MathVista: MC as option letters; otherwise type-aware normalized string.
    meta = metadata or {}
    choice = _as_choice_letter(text, meta)
    if choice is not None:
        return f"MV::CHOICE::{choice}"
    return f"MV::{normalize_mathvista_answer(text, meta)}"


def answers_equivalent(
    a: Any,
    b: Any,
    dataset: str,
    metadata: dict | None = None,
) -> bool:
    if is_missing_extraction(a) or is_missing_extraction(b):
        return False
    dataset = dataset.lower()
    if dataset in {"math", "llm", "qwen2.5-3b-instruct"}:
        return math_answers_equal(str(a), str(b))
    return mathvista_answers_equal(str(a), str(b), metadata or {})


def grade_against_gold(
    answer: Any,
    gold: Any,
    dataset: str,
    metadata: dict | None = None,
) -> bool:
    if is_missing_extraction(answer):
        return False
    if gold is None or str(gold).strip() == "":
        return False
    return answers_equivalent(answer, gold, dataset, metadata)


def _as_choice_letter(text: str, metadata: dict) -> Optional[str]:
    ans_type = str(metadata.get("answer_type") or metadata.get("question_type") or "").lower()
    choices = metadata.get("choices") or metadata.get("options")
    cleaned = text.strip()
    # Direct letter
    m = re.fullmatch(r"([A-Ea-e])(?:[\)\.]|$)?", cleaned)
    if m and (ans_type in {"multi_choice", "multiple_choice", "mc", "choice"} or choices):
        return m.group(1).upper()
    if choices:
        # Map choice text back to letter when possible.
        if isinstance(choices, dict):
            items = list(choices.items())
        else:
            items = [(chr(ord("A") + i), c) for i, c in enumerate(list(choices))]
        for letter, option in items:
            if str(option).strip().lower() == cleaned.lower():
                return str(letter).upper()[:1]
        # "(A) foo" style
        m2 = re.match(r"^\(?([A-Ea-e])\)?[\.\)]\s*(.*)$", cleaned)
        if m2:
            return m2.group(1).upper()
    return None


def normalize_mathvista_answer(answer: str, metadata: dict) -> str:
    text = answer.strip()
    ans_type = str(metadata.get("answer_type") or metadata.get("precision") or "").lower()
    # Integer
    if ans_type in {"integer", "int", "count"}:
        m = re.search(r"-?\d+", text.replace(",", ""))
        return m.group(0) if m else text.lower()
    # Float with precision
    if ans_type in {"float", "decimal"} or "precision" in metadata:
        prec = metadata.get("precision", metadata.get("answer_precision"))
        try:
            val = float(re.search(r"-?\d+(?:\.\d+)?", text.replace(",", "")).group(0))  # type: ignore[union-attr]
            if prec is not None:
                return f"{val:.{int(prec)}f}"
            return str(val)
        except Exception:
            return text.lower()
    # List-like
    if ans_type in {"list"}:
        return re.sub(r"\s+", "", text.lower())
    return re.sub(r"\s+", " ", text.lower())


def mathvista_answers_equal(a: str, b: str, metadata: dict) -> bool:
    ca = _as_choice_letter(a, metadata)
    cb = _as_choice_letter(b, metadata)
    if ca is not None or cb is not None:
        if ca is None:
            ca = _as_choice_letter(normalize_mathvista_answer(a, metadata), metadata)
        if cb is None:
            cb = _as_choice_letter(normalize_mathvista_answer(b, metadata), metadata)
        if ca is not None and cb is not None:
            return ca == cb
    return normalize_mathvista_answer(a, metadata) == normalize_mathvista_answer(b, metadata)


def form_answer_groups(
    answers: list[Any],
    dataset: str,
    metadata: dict | None = None,
    order: list[int] | None = None,
) -> dict[str, list[int]]:
    """Group rollout indices by equivalence without using gold.

    Returns mapping group_key -> list of indices in encounter order.
    Missing extractions collapse into a single NO_ANSWER group.
    """
    indices = order if order is not None else list(range(len(answers)))
    groups: dict[str, list[int]] = {}
    representatives: list[tuple[str, Any]] = []
    for idx in indices:
        ans = answers[idx]
        if is_missing_extraction(ans):
            groups.setdefault(NO_ANSWER, []).append(idx)
            continue
        placed = False
        for key, rep in representatives:
            if answers_equivalent(ans, rep, dataset, metadata):
                groups[key].append(idx)
                placed = True
                break
        if not placed:
            key = canonical_group_key(ans, dataset, metadata)
            # Disambiguate rare collisions on canonical strings that are not equivalent.
            if key in groups:
                suffix = 1
                while f"{key}#{suffix}" in groups:
                    suffix += 1
                key = f"{key}#{suffix}"
            groups[key] = [idx]
            representatives.append((key, ans))
    return groups
