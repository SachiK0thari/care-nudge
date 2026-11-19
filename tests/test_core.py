import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carenudge import policy, rules
from carenudge.scheduler import MockScheduler
from carenudge.schema import Triage, parse_reply


@pytest.mark.parametrize("text, level", [
    ("I keep seeing white flashes in front of my right eye.", "URGENT"),
    ("A dark curtain is coming down over the vision in my left eye.", "EMERGENCY"),
    ("Heavy pressure in my chest going down my left arm.", "EMERGENCY"),
    ("Worst headache of my life, came on in seconds.", "EMERGENCY"),
    ("My 6 week old has a temperature of 100.8.", "EMERGENCY"),
    ("I found a hard lump in my left breast.", "SOON"),
    ("I've had a runny nose for two days.", "NEEDS_INFO"),
    # Common emergency phrasings the rules used to miss.
    ("My chest hurts", "EMERGENCY"),
    ("my chest really hurts bad", "EMERGENCY"),
    ("I'm having a heart attack", "EMERGENCY"),
    ("I think I'm having a stroke", "EMERGENCY"),
    ("I took a whole bottle of pills", "EMERGENCY"),
    ("I don't want to live anymore", "EMERGENCY"),
    ("My husband collapsed and isn't responding", "EMERGENCY"),
    # Words that used to count as negation and hid a real emergency.
    ("I have no idea why but my chest feels crushed", "EMERGENCY"),
    ("I've never felt pain like this in my chest", "EMERGENCY"),
    ("Not sure if this matters but my face is drooping", "EMERGENCY"),
    ("No chest pain now, but earlier my chest felt tight.", "EMERGENCY"),
])
def test_rules_fire(text, level):
    assert rules.check(text).level == level


@pytest.mark.parametrize("text", [
    "I've had the same couple of floaters for years. No flashes.",
    "My lower back is sore after moving furniture. No numbness between my legs.",
    "Mild cough, no shortness of breath and no chest pain.",
    "I've never had chest pain, just a cough.",
    "Without any shortness of breath or chest tightness.",
])
def test_rules_respect_negation(text):
    assert rules.check(text).level == "NEEDS_INFO"


@pytest.mark.parametrize("text", [
    "What does a heart attack feel like - is it always chest pain?",
    "What are the early warning signs of a stroke?",
    "My dad had a stroke in 2019 and I want to lower my own risk.",
])
def test_rules_ignore_questions(text):
    # The first may still fire on "chest pain" (a known probe miss), but never on the heart attack/stroke rule.
    assert "says it's a heart attack or stroke" not in rules.check(text).red_flags


def test_self_directed_overdose_gets_the_crisis_reply():
    triage, visible = policy.decide("I took a whole bottle of pills", "")
    assert triage.level == "EMERGENCY" and policy.SELF_HARM in triage.red_flags
    assert "988" in visible
    # Someone else's overdose is an emergency, not a crisis-line reply.
    other = rules.check("My toddler swallowed all of the pills in my purse")
    assert other.level == "EMERGENCY" and policy.SELF_HARM not in other.red_flags


def test_parse_reply_roundtrip_and_garbage():
    tag = Triage("URGENT", ["new flashes of light"], "possible retinal tear").to_tag()
    triage, visible = parse_reply(f"{tag}\nPlease be seen today.")
    assert (triage.level, visible) == ("URGENT", "Please be seen today.")
    assert parse_reply("no tag here")[0] is None
    assert parse_reply('<triage>{"level": "PURPLE"}</triage> hi')[0] is None


def test_rules_override_a_reassuring_model():
    raw = '<triage>{"level": "ROUTINE", "red_flags": [], "concern": ""}</triage>\nProbably just eye strain.'
    triage, visible = policy.decide("I keep seeing white flashes in front of my right eye.", raw)
    assert triage.level == "URGENT"
    assert "eye strain" not in visible and "today" in visible


def test_model_can_escalate_beyond_rules():
    raw = '<triage>{"level": "URGENT", "red_flags": ["ear went quiet"], "concern": "x"}</triage>\nBe seen today.'
    triage, visible = policy.decide("Everything on my left sounds far away since this morning.", raw)
    assert triage.level == "URGENT" and visible == "Be seen today."


def test_flags_are_not_listed_twice_when_rules_agree_or_win():
    text = "I keep seeing white flashes in front of my right eye."
    flags = '["new flashes of light", "sudden increase in floaters"]'
    tag = '<triage>{{"level": "{}", "red_flags": ' + flags + ', "concern": "x"}}</triage>\nBe seen.'
    for level in ("SOON", "URGENT"):  # the rules win, then tie
        triage, visible = policy.decide(text, tag.format(level))
        assert triage.red_flags == ["new flashes or floaters"]
        assert "floaters," not in visible
    # The model is more urgent: keep what it noticed alongside the rules.
    triage, _ = policy.decide(text, tag.format("EMERGENCY"))
    assert triage.red_flags == ["new flashes of light", "sudden increase in floaters", "new flashes or floaters"]


def test_fallback_holds_the_line_when_patient_wants_to_wait():
    _, visible = policy.decide("It's probably nothing. Can I just wait and see?", "", prior_level="URGENT")
    assert "today" in visible


def test_next_step_by_level():
    sched, noon = MockScheduler(), datetime(2026, 11, 16, 12, 5)
    urgent = policy.Session(level="URGENT", red_flags=["new flashes or floaters"])
    banner, slots = policy.next_step(urgent, sched, final=False, now=noon)
    assert "today" in banner and len(slots) == 3 and all(s["start"].startswith("2026-11-16") for s in slots)
    assert sched.book(slots[0]["id"], "flashes")["status"] == "booked"

    _, late = policy.next_step(urgent, sched, final=False, now=datetime(2026, 11, 16, 21, 0))
    assert late == []  # after hours: no same-day slot, banner sends them to urgent care / ED

    emergency = policy.Session(level="EMERGENCY", red_flags=["chest pain or pressure"])
    banner, slots = policy.next_step(emergency, sched, final=True, now=noon)
    assert "911" in banner and slots == []

    soon = policy.Session(level="SOON", red_flags=["changing mole"])
    assert policy.next_step(soon, sched, final=False, now=noon)[1] == []  # waits for wrap-up
    assert len(policy.next_step(soon, sched, final=True, now=noon)[1]) == 3


def test_reply_that_holds_the_level_still_names_the_next_step():
    raw = ('<triage>{"level": "EMERGENCY", "red_flags": [], "concern": "x"}</triage>\n'
           "I know it's inconvenient, but this isn't something to put off.")
    _, visible = policy.decide("I'm really busy this week. Can it wait?", raw, prior_level="EMERGENCY")
    assert visible.startswith("I know it's inconvenient") and "911" in visible
    assert policy.ensure_action("URGENT", "Please be seen today.") == "Please be seen today."  # nothing added
    assert policy.ensure_action("ROUTINE", "Rest and fluids.") == "Rest and fluids."
    # Naming the word "emergency" is not the same as telling them where to go.
    assert "911" in policy.ensure_action("EMERGENCY", "This isn't an emergency.")
    assert "today" in policy.ensure_action("URGENT", "It's not an emergency, but don't ignore it.")
    for named in ("Go to the emergency department now.", "Head to the ER.", "Call 911."):
        assert policy.ensure_action("EMERGENCY", named) == named
