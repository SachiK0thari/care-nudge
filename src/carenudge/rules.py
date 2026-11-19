"""Deterministic red-flag safety net.

High precision, limited recall, zero dependence on the model. The final triage
level is max(model level, rule level), so a model miss on a textbook phrase
("worst headache of my life") can never drag the answer below the rule floor.
"""
from __future__ import annotations

import re

from .schema import RANK, Triage

# (level, red-flag label, concern, pattern)
_RULES: list[tuple[str, str, str, re.Pattern]] = [
    ("EMERGENCY", "chest pain or pressure", "possible heart attack", re.compile(
        r"\bchest\b.{0,40}\b(pain\w*|pressure|hurt\w*|ach\w*|tight\w*|squeez\w*|crush\w*|heav\w+)\b"
        r"|\b(pain\w*|pressure|hurt\w*|ach\w*|tight\w*|squeez\w*|crush\w*)\b.{0,25}\bchest\b", re.I)),
    # First person or someone present, in the present tense: "what does a heart attack feel like?"
    # and "my dad had a stroke in 2019" are not emergencies.
    ("EMERGENCY", "says it's a heart attack or stroke", "possible heart attack or stroke", re.compile(
        r"\b(i'?m|i am|he'?s|she'?s|is|they'?re)\s+having\b.{0,20}\b(heart attack|stroke)\b", re.I)),
    ("EMERGENCY", "possible overdose or poisoning", "possible overdose or poisoning", re.compile(
        r"\b(took|swallowed|ate)\b.{0,30}\b(bottle|whole|all (of )?(my|the|his|her))\b.{0,20}"
        r"\b(pills?|tablets?|meds|medications?)\b|\b(overdosed|od'?d)\b"
        # A counted dose of 10 or more ("took about 20 extra strength Tylenol"), but not "took 20 mg".
        r"|\b(took|swallowed|ate|taken)\b.{0,15}\b([1-9]\d+|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty"
        r"|a dozen|dozens)\b(?!\s*(mg|milligrams?|mcg|ml|units?|days?|weeks?|hours?|minutes?)\b).{0,40}"
        r"\b(pills?|tablets?|caps(ules)?|gummies|tylenol|acetaminophen|paracetamol|advil|motrin|ibuprofen|aleve"
        r"|aspirin|benadryl|ambien|xanax|oxy\w*|percocet|vicodin|melatonin|sleeping pills?|of (my|his|her|the) \w+)\b",
        re.I)),
    ("EMERGENCY", "collapsed or unresponsive", "possible loss of consciousness", re.compile(
        r"\b(collapsed|passed out|fainted|unresponsive|not responding|isn'?t responding|won'?t wake up)\b", re.I)),
    ("EMERGENCY", "stroke warning signs", "possible stroke or TIA", re.compile(
        r"\b(face|mouth)\b.{0,30}\b(droop\w*|sagging)\b|\bslurr\w*"
        r"|\b(arm|leg|hand)\b.{0,30}\b(went|suddenly|sudden)\b.{0,20}\b(weak|numb)\b"
        r"|couldn'?t get my words out", re.I)),
    ("EMERGENCY", "sudden severe headache", "possible bleeding in the brain", re.compile(
        r"worst headache|thunderclap|\b(sudden|explosive)\b.{0,30}\bheadache\b", re.I)),
    ("EMERGENCY", "swelling of lips, tongue or throat", "possible severe allergic reaction", re.compile(
        r"\b(lips?|tongue|throat)\b.{0,40}\b(swell\w*|swollen|closing|tight)\b", re.I)),
    ("EMERGENCY", "trouble breathing", "breathing difficulty", re.compile(
        r"(can'?t|cannot|hard to|trouble|difficulty)\s+(catch my breath|breathe|breathing)"
        r"|short(ness)? of breath", re.I)),
    ("EMERGENCY", "sudden vision loss or curtain over vision",
     "possible retinal detachment or blocked blood vessel in the eye", re.compile(
        r"\b(curtain|veil|shadow)\b.{0,50}\b(vision|eye|see|sight)\b"
        r"|\b(lost|loss of|can'?t see|cannot see)\b.{0,40}\b(vision|eye|sight)\b"
        r"|\bvision\b.{0,30}\bwent (black|dark)\b", re.I)),
    ("EMERGENCY", "vomiting blood or black stools", "possible bleeding in the digestive tract", re.compile(
        r"(vomit\w*|threw up|throw\w* up).{0,60}\b(blood|coffee grounds)\b"
        r"|\b(black|tarry)\b.{0,30}\b(stools?|poop|bowel movements?)\b"
        r"|\b(stools?|poop)\b.{0,40}\b(black|tarry)\b", re.I)),
    ("EMERGENCY", "thoughts of suicide or self-harm", "risk of self-harm", re.compile(
        r"kill(ing)? myself|suicid\w*|end(ing)? my life|hurt(ing)? myself"
        r"|don'?t want to be alive|better off without me"
        r"|don'?t want to (live|be here)( anymore)?|want to die|no reason to live", re.I)),
    ("EMERGENCY", "loss of bladder/bowel control or saddle numbness",
     "possible pressure on the spinal nerves", re.compile(
        r"(lost|loss of) .{0,20}control.{0,20}(bladder|bowel)|wet myself"
        r"|\bnumb\w*\b.{0,40}(groin|between my legs|inner thighs?)"
        r"|(groin|between my legs)\b.{0,40}\bnumb", re.I)),
    ("URGENT", "new flashes or floaters", "possible retinal tear or early retinal detachment", re.compile(
        r"\bflash\w*\b.{0,40}\b(light|eye|vision)\b|\b(eye|vision)\b.{0,40}\bflash\w*"
        r"|lightning streaks|flicker\w* lights?"
        r"|\b(new|sudden\w*|lots of|shower of|bunch of)\b.{0,30}\b(floaters?|specks?)\b", re.I)),
    ("URGENT", "one-sided leg swelling", "possible blood clot in the leg", re.compile(
        r"\b(calf|leg)\b.{0,60}\b(swollen|swelling|puffy|bigger)\b"
        r"|\b(swollen|swelling)\b.{0,30}\b(calf|leg)\b", re.I)),
    ("URGENT", "sudden hearing loss", "possible sudden sensorineural hearing loss", re.compile(
        r"\b(sudden\w*|woke up|out of nowhere)\b.{0,60}\b(hear\w*|deaf|muffled)\b"
        r"|\b(lost|lose)\b.{0,30}\bhearing\b|\bwent deaf\b", re.I)),
    ("URGENT", "spreading redness", "possible spreading skin infection", re.compile(
        r"\bred(ness)?\b.{0,60}\b(spreading|creeping|bigger|doubled)\b|\bred streaks?\b", re.I)),
    ("URGENT", "blistering rash near the eye", "possible shingles affecting the eye", re.compile(
        r"\bblister\w*\b.{0,80}\b(eye\w*|forehead|nose)\b|\b(eye\w*|forehead)\b.{0,80}\bblister\w*", re.I)),
    ("SOON", "changing mole", "possible skin cancer", re.compile(
        r"\b(mole|spot)\b.{0,80}\b(bigger|darker|grown|chang\w*|bled|itch\w*|lopsided|jagged|uneven)\b", re.I)),
    ("SOON", "blood in stool", "possible bowel condition that needs evaluation", re.compile(
        r"\bblood\b.{0,40}\b(stools?|toilet paper|poop)\b|\b(stools?|poop)\b.{0,40}\bblood\b", re.I)),
    ("SOON", "bleeding after menopause", "possible uterine condition that needs evaluation", re.compile(
        r"menopaus\w*.{0,120}\b(bleed\w*|blood|spotting)\b|\b(bleed\w*|spotting)\b.{0,80}menopaus", re.I)),
    ("SOON", "breast lump", "breast change that needs evaluation", re.compile(
        r"\b(lump|knot)\b.{0,40}\bbreast\b|\bbreast\b.{0,40}\b(lump|knot)\b", re.I)),
    ("SOON", "unexplained weight loss or night sweats", "unexplained weight loss that needs evaluation",
     re.compile(r"lost \w+ (pounds|lbs)|los(t|ing) (a lot of )?weight|dropping weight|night sweats", re.I)),
]

OVERDOSE = "possible overdose or poisoning"
SELF_HARM = "thoughts of suicide or self-harm"
# "no idea", "not sure" and the like are not denials of a symptom.
_NEGATION = re.compile(
    r"\b(no(?!\s+(idea|doubt|clue|way)\b)|not(?!\s+(sure|certain|kidding)\b)"
    r"|without|denies|haven'?t|hasn'?t|don'?t have)\b", re.I)
# "never had chest pain" is a denial; "never felt pain like this" is an intensifier.
_NEVER = re.compile(r"\bnever\b", re.I)
_INTENSIFIER = re.compile(r"\blike (this|that|it)\b|\bthis (bad|much|strong|severe)\b", re.I)
_CLAUSE_BREAK = re.compile(r"[.;,!?]|\b(?:but|except|now|and then)\b", re.I)
_INFANT = re.compile(
    r"\bnewborn\b|\b(\d{1,2})[\s-]*(day|week|wk)s?[\s-]*old\b|\b(1|2|one|two)[\s-]*months?[\s-]*old\b", re.I)
_FEVER = re.compile(r"\bfever\b|\btemp(erature)?\b|\b10[0-5](\.\d)?\b|\b3[89]\.\d\b|feels hot", re.I)
_URINARY = re.compile(r"\b(pee\w*|urin\w*|uti)\b", re.I)
_FLANK = re.compile(r"\b(flank|side|back)\b", re.I)


def _negated(text: str, start: int) -> bool:
    """True when the match sits in a clause like 'no numbness or leg weakness'."""
    clause = _CLAUSE_BREAK.split(text[max(0, start - 40):start])[-1]
    if _NEGATION.search(clause):
        return True
    if _NEVER.search(clause):
        sentence = re.split(r"[.;!?]", text[start:])[0]
        return not _INTENSIFIER.search(clause + sentence)
    return False


def _fires(pattern: re.Pattern, text: str) -> bool:
    """True if any match is not negated. Tries every start position, not just non-overlapping matches,
    so in "no chest pain now, but earlier my chest felt tight" the second "chest" gets its own check."""
    pos = 0
    while (m := pattern.search(text, pos)) is not None:
        if not _negated(text, m.start()):
            return True
        pos = m.start() + 1
    return False


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
        if _fires(pattern, text):
            hits.append((rule_level, flag, rule_concern))
    if _young_infant(text) and _FEVER.search(text):
        hits.append(("EMERGENCY", "fever in a baby under 3 months", "possible serious infection in a young infant"))
    if re.search(r"\b(fever|chills)\b", text, re.I) and _URINARY.search(text) and _FLANK.search(text):
        hits.append(("URGENT", "fever with urinary symptoms and flank pain", "possible kidney infection"))
    for rule_level, flag, rule_concern in hits:
        flags.append(flag)
        if RANK[rule_level] > RANK[level]:
            level, concern = rule_level, rule_concern
    return Triage(level, flags, concern)
