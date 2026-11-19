"""Turn (model output + rule hits) into one decision and one next step."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from . import rules
from .scheduler import Scheduler
from .schema import RANK, Triage, more_urgent, parse_reply

SELF_HARM = rules.SELF_HARM
# Rule flags for situations the model was never trained on: when one fires, the reply comes from
# the rules even if the model agrees on the level, because the model's wording can be off-topic
# (live: "My husband collapsed and isn't responding" got a reply about hearing loss and face drooping).
RULE_LED = {rules.OVERDOSE, rules.COLLAPSE, rules.SAYS_EMERGENCY}

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
# Match the instruction, not the word: "this isn't an emergency" names no next step.
_ER = r"911|988|emergency (department|room|care)|(?-i:\bER\b)"
ACTION = {
    "EMERGENCY": re.compile(rf"\b({_ER})", re.I),
    "URGENT": re.compile(rf"\b(today|same[- ]day|urgent care|{_ER})", re.I),
    "SOON": re.compile(rf"\b(book\w*|visit|appointment|today|same[- ]day|urgent care|{_ER})", re.I),
}
# Appended when the model's reply holds the level but leaves out what to do (seen on pushback turns).
NEXT_STEP = {
    "EMERGENCY": "Please call 911 or go to the nearest emergency department now.",
    "URGENT": "Please arrange to be seen today.",
    "SOON": "Please book a primary care visit in the next few days.",
}
# Getting a ride instead of an ambulance. Fine only when the same sentence argues against it
# ("don't drive yourself"). Seen once on the challenge set: "call 911 now, or drive to the emergency department".
TRANSPORT = re.compile(r"\bdriv(e|es|ing)\b|\b(someone|somebody|a friend|family)\b.{0,20}\b(take|bring)s? you\b", re.I)
AGAINST = re.compile(r"\b(not|don'?t|never|rather than|instead of|avoid)\b", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
# An emergency instruction made conditional ("call 911 if you are confused") is a hedge, not a next step.
# Live: "I can't stop shaking and my chest feels tight" got "eat something sweet ... call 911 if you are
# confused", copied from the low-blood-sugar answer. Checked per clause, so "call or text 988, or call 911 if
# you are in immediate danger" and "call 911 now, even if it fades" still count as plain instructions.
_CLAUSE = re.compile(r"[,;:]|\s(?:or|and|but|then)\s|\s[-\u2013\u2014]\s")
_CONDITION = re.compile(r"(?<!even )\b(if|unless|in case|should you)\b", re.I)
# The model has invented phone numbers ("call 617-555-0199"). Only these may reach a patient.
_PHONE = re.compile(r"(?:\+?1[-.\s]?)?\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
POISON_CONTROL = "1-800-222-1222"
KNOWN_NUMBERS = (POISON_CONTROL,)  # 911 and 988 are too short to match _PHONE
CRISIS_BANNER = "### Call or text 988 now (Suicide & Crisis Lifeline)\nIf you are in immediate danger, call 911."
# An overdose is a medical emergency first, whatever the intent: acetaminophen can harm the liver before any
# symptoms. 988 is offered, not assumed; the self-harm flag is only set when the patient's own words say so.
OVERDOSE_STEP = (
    "Taking too much of a medicine can be dangerous even if you feel fine right now. "
    "Please call 911 or Poison Control (1-800-222-1222) now."
)
OVERDOSE_988 = "If you took them on purpose or are thinking about hurting yourself, you can also call or text 988."
OVERDOSE_REPLY = f"{OVERDOSE_STEP} {OVERDOSE_988}"


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
    rule_led = bool(RULE_LED & set(rule.red_flags)) and RANK[rule.level] >= RANK[model.level]
    replace = rule_wins or rule_led or not visible
    if SELF_HARM in flags and replace:
        visible = CRISIS_REPLY
        if rules.OVERDOSE in flags:
            visible = f"{visible} {OVERDOSE_STEP}"
    elif rules.OVERDOSE in flags and replace:
        visible = OVERDOSE_REPLY
    elif not visible and RANK[level] < RANK[prior_level] and prior_level in STILL:
        visible = STILL[prior_level]
    elif replace:
        # Never show reassuring model text next to a rule-level escalation.
        visible = SAFE_REPLY[level].format(flags=", ".join(flags) or "your symptoms")
    return Triage(level, flags, concern), guard(level, visible, flags)


def guard(level: str, visible: str, flags: list[str] | None = None) -> str:
    """The last pass over what the patient sees, whatever the model wrote."""
    fallback = SAFE_REPLY[level].format(flags=", ".join(flags or []) or "your symptoms") if level in SAFE_REPLY else ""
    visible = drop_unknown_numbers(visible)
    if level == "EMERGENCY":
        visible = drop_driving(visible)  # may remove the 911 line too; ensure_action puts one back
        if hedges_emergency(visible):
            visible = fallback  # the rest of a hedged reply (e.g. "eat something sweet first") can't be trusted
    if not visible:
        visible = fallback
    return ensure_action(level, visible)


def hedges_emergency(text: str) -> bool:
    """True when the reply names 911/ER only inside conditions, never as a plain instruction."""
    clauses = [c for s in _SENTENCE.split(text) for c in _CLAUSE.split(s)]
    acting = [c for c in clauses if ACTION["EMERGENCY"].search(c)]
    return bool(acting) and all(_CONDITION.search(c) for c in acting)


def drop_unknown_numbers(visible: str) -> str:
    """Only numbers the app supplies reach the patient. A wrong Poison Control number is corrected (the model
    wrote "1-800-222-4224" once, in a sentence that also said to call 911); any other unknown number takes its
    sentence with it."""
    kept = []
    for sentence in _SENTENCE.split(visible):
        unknown = [n for n in _PHONE.findall(sentence) if not any(k[-8:] in n for k in KNOWN_NUMBERS)]
        if unknown and re.search(r"poison control", sentence, re.I):
            for n in unknown:
                sentence = sentence.replace(n, POISON_CONTROL)
        elif unknown:
            continue
        kept.append(sentence)
    return " ".join(kept)


def _drives(sentence: str) -> bool:
    match = TRANSPORT.search(sentence)
    return match is not None and not AGAINST.search(sentence[:match.start()])


def advises_driving(text: str) -> bool:
    return any(_drives(s) for s in _SENTENCE.split(text))


def drop_driving(visible: str) -> str:
    """An emergency means an ambulance: remove any sentence that offers driving instead."""
    return " ".join(s for s in _SENTENCE.split(visible) if not _drives(s))


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


MAX_TURNS = 8  # user + assistant pairs sent to the model, so a long chat stays inside the context window


def model_history(session: Session, max_turns: int = MAX_TURNS) -> list[dict]:
    """The chat as the model sees it: the opening exchange plus the most recent turns.

    Trimming drops no safety state. The level and red flags live in Session, the level only
    ratchets up, and the banner and booking come from Session, not from the model's memory.
    The opening exchange stays because that is usually where the patient names the symptom.
    """
    turns = [session.messages[i:i + 2] for i in range(0, len(session.messages), 2)]
    if len(turns) > max_turns:
        turns = turns[:1] + turns[-(max_turns - 1):]
    return [m for turn in turns for m in turn]


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
    if rules.OVERDOSE in session.red_flags:
        banner += "\n\n" + (OVERDOSE_STEP if crisis else OVERDOSE_REPLY)
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
