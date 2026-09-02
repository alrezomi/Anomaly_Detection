"""Parse RynnValue's native trajectory-verification response."""

from __future__ import annotations

import re


_DESCRIPTION_RE = re.compile(
    r"(?:^|\n)\s*-?\s*Video Description\s*:\s*(.+)", re.IGNORECASE
)
_MATCH_RE = re.compile(
    r"(?:^|\n)\s*-?\s*Match\s*:\s*(Yes|No)\b", re.IGNORECASE
)
_SUCCESS_RE = re.compile(
    r"(?:^|\n)\s*-?\s*Success\s*:\s*(Yes|No)\b", re.IGNORECASE
)


def parse_analysis(text: str) -> dict[str, str | None]:
    """Extract the released model's description/match/success fields."""

    def first(pattern: re.Pattern[str]) -> str | None:
        match = pattern.search(text)
        return match.group(1).strip() if match else None

    return {
        "description": first(_DESCRIPTION_RE),
        "match": first(_MATCH_RE),
        "success": first(_SUCCESS_RE),
    }


def anomaly_decision(match: str | None, success: str | None) -> str:
    """Map RynnValue's native verification fields to the project vocabulary.

    A task mismatch or an incomplete task is a failure. A success is accepted
    only when both native checks agree; partial/unparseable output abstains.
    """

    match_value = (match or "").strip().lower()
    success_value = (success or "").strip().lower()
    if match_value == "no" or success_value == "no":
        return "failure"
    if match_value == "yes" and success_value == "yes":
        return "success"
    return "uncertain"
