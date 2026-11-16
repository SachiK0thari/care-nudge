"""Shared types: triage levels and the <triage>{...}</triage> tag the model emits."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

LEVELS = ["NEEDS_INFO", "ROUTINE", "SOON", "URGENT", "EMERGENCY"]  # least -> most urgent
RANK = {name: i for i, name in enumerate(LEVELS)}

_TAG = re.compile(r"<triage>\s*(\{.*?\})\s*</triage>", re.DOTALL)


@dataclass
class Triage:
    level: str = "NEEDS_INFO"
    red_flags: list[str] = field(default_factory=list)
    concern: str = ""

    def to_tag(self) -> str:
        body = {"level": self.level, "red_flags": self.red_flags, "concern": self.concern}
        return f"<triage>{json.dumps(body)}</triage>"


def more_urgent(a: str, b: str) -> str:
    return a if RANK[a] >= RANK[b] else b


def parse_reply(text: str) -> tuple[Triage | None, str]:
    """Split raw model output into (triage, patient-facing text).

    Returns (None, text) when the tag is missing or malformed, so the caller
    can fall back to the rule layer instead of trusting a broken output.
    """
    match = _TAG.search(text)
    visible = _TAG.sub("", text).strip()
    if not match:
        return None, visible
    try:
        data = json.loads(match.group(1))
        level = str(data.get("level", "")).upper()
        if level not in RANK:
            return None, visible
        flags = [str(f) for f in data.get("red_flags", []) if f]
        return Triage(level, flags, str(data.get("concern", ""))), visible
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None, visible
