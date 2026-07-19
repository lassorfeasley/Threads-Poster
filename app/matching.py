"""First-pass keyword filter on title + description."""
from __future__ import annotations

import re


def find_keyword_matches(text: str, keywords: list[str]) -> list[str]:
    """Case-insensitive whole-word/phrase matching. Returns matched keywords."""
    matched = []
    lowered = text.lower()
    for kw in keywords:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, lowered):
            matched.append(kw)
    return matched
