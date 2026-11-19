"""Turn (model output + rule hits) into one decision and one next step."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from . import rules
from .scheduler import Scheduler
from .schema import RANK, Triage, more_urgent, parse_reply

SELF_HARM = rules.SELF_HARM

# Used when the model gave nothing usable, or the rules outrank what it said.
SAFE_REPLY = {
    "EMERGENCY": (
        "What you're describing ({flags}) can be a sign of a medical emergency. Please call 911 or go to the nearest "
        "emergency department now."
    ),
    "URGENT": "What you're describing ({flags}) should not wait. Please arrange to be seen today.",
    "SOON": (
        "What you're describing ({flags}) should be checked by your primary care team. Please book a visit in the next "
        "few days rather than waiting."
    ),
    "ROUTINE": (
        "I didn't pick up any warning signs in what you described. If anything changes or gets worse, tell me or "
        "contact your care team."
    ),
    "NEEDS_INFO": (
        "Can you tell me a bit more - what symptoms you're noticing, when they started, and whether anything is "
        "getting worse?"
    ),
}
CRISIS_REPLY = (
    "I'm really glad you told me. You deserve support right now: please call or text 988 (Suicide & Crisis Lifeline), "
    "or call 911 if you are in immediate danger."
)

BANNER = {
    "EMERGENCY": (
        "### Call 911 or go to the nearest emergency department now\nDo not wait for an appointment, and do not drive "
        "yourself."
    ),
    "URGENT": "### This should be looked at today\nA short delay can change the outcome.",
    "SOON": "### Please don't put this off\nBook a primary care visit within the next few days.",
    "ROUTINE": "### No warning signs came up\nYou can still book a routine visit if you'd like one.",
    "NEEDS_INFO": "Tell me a bit more so I can point you to the right care.",
}
# Fallback when a later message has no model text and no new flags (e.g. "can it wait?").
STILL = {
    "EMERGENCY": (
        "What you described earlier still needs emergency care now. Please call 911 or go to the emergency department."
    ),
    "URGENT": (
        "What you described earlier still needs to be seen today, even if it feels minor. A short delay can change the "
        "outcome."
    ),
    "SOON": (
        "What you described earlier should still be checked within the next few days. I'll help you book before we "
        "finish."
    ),
}
# The action a reply must name at each level. A more urgent instruction also counts.
ACTION = {
    "EMERGENCY": re.compile(r"\b(911|988|emergency)\b", re.I),
    "URGENT": re.compile(r"\b(today|same[- ]day|911|emergency)\b", re.I),
    "SOON": re.compile(r"\b(book\w*|visit|appointment|today|same[- ]day|911|emergency)\b", re.I),
}
# Appended when the model's reply holds the level but leaves out what to do (seen on pushback turns).
NEXT_STEP = {
    "EMERGENCY": "Please call 911 or go to the nearest emergency department now.",
    "URGENT": "Please arrange to be seen today.",
    "SOON": "Please book a primary care visit in the next few days.",
}
CRISIS_BANNER = "### Call or text 988 now (Suicide & Crisis Lifeline)\nIf you are in immediate danger, call 911."


@dataclass
class Session:
    """Most urgent finding so far. Urgency only ratchets up within a chat."""
    level: str = "NEEDS_INFO"
    red_flags: list[str] = field(default_factory=list)
    concern: str = ""
    messages: list[dict] = field(default_factory=list)  # what the model sees (tags included)


def decide(user_text: str, raw_model_output: str, prior_level: str = "NEEDS_INFO") -> tuple[Triage, str]:
    rule = rules.check(user_text)
    parsed, visible = parse_reply(raw_model_output)
    model = parsed or Triage()
    rule_wins = RANK[rule.level] > RANK[model.level]
    level = more_urgent(model.level, rule.level)
    if rule.red_flags and RANK[rule.level] >= RANK[model.level]:
        # The rules matched the patient's own words. The model's names for the same signs
        # only repeat them, and can include signs the patient never mentioned.
        flags = list(rule.red_flags)
    else:
        flags = list(dict.fromkeys(model.red_flags + rule.red_flags))
    if SELF_HARM in model.red_flags and SELF_HARM not in flags:
        flags.append(SELF_HARM)
    concern = rule.concern if rule_wins else (model.concern or rule.concern)
    if SELF_HARM in flags and (rule_wins or not visible):
        visible = CRISIS_REPLY
    elif not visible and RANK[level] < RANK[prior_level] and prior_level in STILL:
        visible = STILL[prior_level]
    elif rule_wins or not visible:
        # Never show reassuring model text next to a rule-level escalation.
        visible = SAFE_REPLY[level].format(flags=", ".join(flags) or "your symptoms")
    return Triage(level, flags, concern), ensure_action(level, visible)


def ensure_action(level: str, visible: str) -> str:
    """Every reply to a warning sign must say what to do next, whatever the model wrote."""
    if level in ACTION and not ACTION[level].search(visible):
        return f"{visible} {NEXT_STEP[level]}".strip()
    return visible


def update(session: Session, user_text: str, triage: Triage, visible: str) -> Session:
    if RANK[triage.level] > RANK[session.level]:
        session.level, session.concern = triage.level, triage.concern
    session.red_flags = list(dict.fromkeys(session.red_flags + triage.red_flags))
    session.messages += [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": f"{triage.to_tag()}\n{visible}"},
    ]
    return session


def next_step(
    session: Session, scheduler: Scheduler, final: bool, now: datetime | None = None
) -> tuple[str, list[dict]]:
    """Banner text plus bookable slots.

    EMERGENCY: banner at once, never an appointment in place of emergency care.
    URGENT:    banner and same-day slots at once (waiting for the end is too late).
    SOON / ROUTINE: slots are offered when the chat wraps up (final=True).
    """
    level = session.level
    crisis = SELF_HARM in session.red_flags
    banner = CRISIS_BANNER if crisis else BANNER[level]
    if session.red_flags:
        banner += "\n\n**Noted:** " + "; ".join(session.red_flags)
    if level == "EMERGENCY":
        if not crisis:
            banner += "\n\nOnce you've been seen, we can help you book follow-up with your primary care team."
        return banner, []
    if level == "URGENT":
        slots = scheduler.find_slots("URGENT", now)
        if slots:
            banner += (
                "\n\nPick a same-day time below. If none works, go to urgent care or the emergency department today."
            )
        else:
            banner += (
                "\n\nNo same-day primary care times are left. Please go to urgent care or the emergency department "
                "today."
            )
        return banner, slots
    if final and level in ("SOON", "ROUTINE"):
        return banner, scheduler.find_slots(level, now)
    if level == "SOON":
        banner += "\n\nI'll help you book before we finish."
    return banner, []
