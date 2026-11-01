from __future__ import annotations

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_question_text(text: str) -> str:
    """Deterministic normalization for MetaMathQA ↔ GSM8K overlap checks.

    Steps are frozen for reproducibility and recorded in data manifests:
    1) Unicode NFKC
    2) lowercase
    3) strip
    4) collapse all whitespace runs to a single space
    5) strip surrounding quotes
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.lower().strip()
    s = _WHITESPACE_RE.sub(" ", s)
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s
