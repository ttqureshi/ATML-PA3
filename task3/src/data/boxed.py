from __future__ import annotations

import re
from typing import Optional


_BOXED_RE = re.compile(r"\\boxed\s*\{")


def extract_last_boxed(text: str | None) -> Optional[str]:
    """Extract the last \\boxed{...} answer using balanced-brace matching.

    Matches the balanced-brace rule described in the assignment README.MD:
    find the last occurrence of \\boxed{ and take the matching brace span.
    """
    if text is None:
        return None
    s = str(text)
    matches = list(_BOXED_RE.finditer(s))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(s) and depth > 0:
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i].strip()
        i += 1
    return None
