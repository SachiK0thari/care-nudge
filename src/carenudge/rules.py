"""Deterministic red-flag safety net.

High precision, limited recall, zero dependence on the model. The final triage
level is max(model level, rule level), so a model miss on a textbook phrase
("worst headache of my life") can never drag the answer below the rule floor.
"""
from __future__ import annotations

import re

from .schema import RANK, Triage

I = re.IGNORECASE

# (level, red-flag label, concern, pattern)
_RULES: list[tuple[str, str, str, re.Pattern]] = [
    ("EMERGENCY", "chest pain or pressure", "possible heart attack", re.compile(
        r"\bchest\b.{0,40}\b(pain|pressure|tight\w*|squeez\w*|crush\w*|heav\w+)\b"
        r"|\b(pain|pressure|tight\w*|squeez\w*|crush\w*)\b.{0,25}\bchest\b", I)),
    ("EMERGENCY", "stroke warning signs", "possible stroke or TIA", re.compile(
        r"\b(face|mouth)\b.{0,30}\b(droop\w*|sagging)\b|\bslurr\w*"
        r"|\b(arm|leg|hand)\b.{0,30}\b(went|suddenly|sudden)\b.{0,20}\b(weak|numb)\b"
        r"|couldn'?t get my words out", I)),
    ("EMERGENCY", "sudden severe headache", "possible bleeding in the brain", re.compile(
        r"worst headache|thunderclap|\b(sudden|explosive)\b.{0,30}\bheadache\b", I)),
    ("EMERGENCY", "swelling of lips, tongue or throat", "possible severe allergic reaction", re.compile(
        r"\b(lips?|tongue|throat)\b.{0,40}\b(swell\w*|swollen|closing|tight)\b", I)),
    ("EMERGENCY", "trouble breathing", "breathing difficulty", re.compile(
        r"(can'?t|cannot|hard to|trouble|difficulty)\s+(catch my breath|breathe|breathing)"
        r"|short(ness)? of breath", I)),
    ("EMERGENCY", "sudden vision loss or curtain over vision",
     "possible retinal detachment or blocked blood vessel in the eye", re.compile(
        r"\b(curtain|veil|shadow)\b.{0,50}\b(vision|eye|see|sight)\b"
        r"|\b(lost|loss of|can'?t see|cannot see)\b.{0,40}\b(vision|eye|sight)\b"
        r"|\bvision\b.{0,30}\bwent (black|dark)\b", I)),
    ("EMERGENCY", "vomiting blood or black stools", "possible bleeding in the digestive tract", re.compile(
        r"(vomit\w*|threw up|throw\w* up).{0,60}\b(blood|coffee grounds)\b"
        r"|\b(black|tarry)\b.{0,30}\b(stools?|poop|bowel movements?)\b"
        r"|\b(stools?|poop)\b.{0,40}\b(black|tarry)\b", I)),
    ("EMERGENCY", "thoughts of suicide or self-harm", "risk of self-harm", re.compile(
        r"kill(ing)? myself|suicid\w*|end(ing)? my life|hurt(ing)? myself"
        r"|don'?t want to be alive|better off without me", I)),
    ("EMERGENCY", "loss of bladder/bowel control or saddle numbness",
     "possible pressure on the spinal nerves", re.compile(
        r"(lost|loss of) .{0,20}control.{0,20}(bladder|bowel)|wet myself"
        r"|\bnumb\w*\b.{0,40}(groin|between my legs|inner thighs?)"
        r"|(groin|between my legs)\b.{0,40}\bnumb", I)),
    ("URGENT", "new flashes or floaters", "possible retinal tear or early retinal detachment", re.compile(
        r"\bflash\w*\b.{0,40}\b(light|eye|vision)\b|\b(eye|vision)\b.{0,40}\bflash\w*"
        r"|lightning streaks|flicker\w* lights?"
        r"|\b(new|sudden\w*|lots of|shower of|bunch of)\b.{0,30}\b(floaters?|specks?)\b", I)),
    ("URGENT", "one-sided leg swelling", "possible blood clot in the leg", re.compile(
        r"\b(calf|leg)\b.{0,60}\b(swollen|swelling|puffy|bigger)\b"
        r"|\b(swollen|swelling)\b.{0,30}\b(calf|leg)\b", I)),
    ("URGENT", "sudden hearing loss", "possible sudden sensorineural hearing loss", re.compile(
        r"\b(sudden\w*|woke up|out of nowhere)\b.{0,60}\b(hear\w*|deaf|muffled)\b"
        r"|\b(lost|lose)\b.{0,30}\bhearing\b|\bwent deaf\b", I)),
    ("URGENT", "spreading redness", "possible spreading skin infection", re.compile(
        r"\bred(ness)?\b.{0,60}\b(spreading|creeping|bigger|doubled)\b|\bred streaks?\b", I)),
    ("URGENT", "blistering rash near the eye", "possible shingles affecting the eye", re.compile(
        r"\bblister\w*\b.{0,80}\b(eye\w*|forehead|nose)\b|\b(eye\w*|forehead)\b.{0,80}\bblister\w*", I)),
    ("SOON", "changing mole", "possible skin cancer", re.compile(
        r"\b(mole|spot)\b.{0,80}\b(bigger|darker|grown|chang\w*|bled|itch\w*|lopsided|jagged|uneven)\b", I)),
    ("SOON", "blood in stool", "possible bowel condition that needs evaluation", re.compile(
        r"\bblood\b.{0,40}\b(stools?|toilet paper|poop)\b|\b(stools?|poop)\b.{0,40}\bblood\b", I)),
    ("SOON", "bleeding after menopause", "possible uterine condition that needs evaluation", re.compile(
        r"menopaus\w*.{0,120}\b(bleed\w*|blood|spotting)\b|\b(bleed\w*|spotting)\b.{0,80}menopaus", I)),
    ("SOON", "breast lump", "breast change that needs evaluation", re.compile(
        r"\b(lump|knot)\b.{0,40}\bbreast\b|\bbreast\b.{0,40}\b(lump|knot)\b", I)),
    ("SOON", "unexplained weight loss or night sweats", "unexplained weight loss that needs evaluation",
     re.compile(r"lost \w+ (pounds|lbs)|los(t|ing) (a lot of )?weight|dropping weight|night sweats", I)),
]

_NEGATION = re.compile(r"\b(no|not|without|never|denies|haven'?t|hasn'?t|don'?t have)\b", I)
_INFANT = re.compile(r"\bnewborn\b|\b(\d{1,2})[\s-]*(day|week|wk)s?[\s-]*old\b|\b(1|2|one|two)[\s-]*months?[\s-]*old\b", I)
_FEVER = re.compile(r"\bfever\b|\btemp(erature)?\b|\b10[0-5](\.\d)?\b|\b3[89]\.\d\b|feels hot", I)
_URINARY = re.compile(r"\b(pee\w*|urin\w*|uti)\b", I)
_FLANK = re.compile(r"\b(flank|side|back)\b", I)


def _negated(text: str, start: int) -> bool:
    """True when the match sits in a clause like 'no numbness or leg weakness'."""
    clause = re.split(r"[.;,!?]", text[max(0, start - 40):start])[-1]
    return bool(_NEGATION.search(clause))


def _young_infant(text: str) -> bool:
    match = _INFANT.search(text)
    if not match:
        return False
    if match.group(1) and match.group(2).lower() in ("week", "wk"):
        return int(match.group(1)) < 13
    return True


def check(text: str) -> Triage:
    """Scan one patient message and return the most urgent rule that fires."""
    level, flags, concern = "NEEDS_INFO", [], ""
    hits: list[tuple[str, str, str]] = []
    for rule_level, flag, rule_concern, pattern in _RULES:
        if any(not _negated(text, m.start()) for m in pattern.finditer(text)):
            hits.append((rule_level, flag, rule_concern))
    if _young_infant(text) and _FEVER.search(text):
        hits.append(("EMERGENCY", "fever in a baby under 3 months", "possible serious infection in a young infant"))
    if re.search(r"\b(fever|chills)\b", text, I) and _URINARY.search(text) and _FLANK.search(text):
        hits.append(("URGENT", "fever with urinary symptoms and flank pain", "possible kidney infection"))
    for rule_level, flag, rule_concern in hits:
        flags.append(flag)
        if RANK[rule_level] > RANK[level]:
            level, concern = rule_level, rule_concern
    return Triage(level, flags, concern)
