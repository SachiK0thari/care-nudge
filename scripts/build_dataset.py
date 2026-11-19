"""Build train/val/test JSONL from data/scenarios.yaml. Runs offline, no API keys.

    python scripts/build_dataset.py

Each scenario's first 3 phrasings feed training; the last 2 are held out for
the test split, so test measures generalisation to unseen wording. Also writes
data/probes.jsonl (informational questions, never trained on) and prints a short
audit of the training split. The separate `info_questions` in the catalog (different
topics from the probes) go to training only, so test and probes stay unchanged.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carenudge.prompts import SYSTEM_PROMPT  # noqa: E402
from carenudge.schema import LEVELS, Triage  # noqa: E402

PREFIX = ["", "Hi, ", "Quick question. ", "Not sure if this matters, but ", "Sorry to bother you. "]
SUFFIX = ["", " Should I be worried?", " Is this normal?", " It's probably nothing, right?", " What should I do?"]
ACK = ["", "Thanks for telling me. ", "I'm glad you mentioned this. ", "Thank you for sharing that. "]
PUSHBACK = [
    "It's probably nothing. Can I just wait and see?",
    "I'm really busy this week. Can it wait?",
    "I don't want to overreact. Maybe I'll give it a few days?",
]
INFO_PREFIX = ["", "Hi. ", "Quick question. ", "Just curious. ", "I was reading about this. "]
INFO_TRIAGE = {"level": "ROUTINE", "flags": [], "concern": "general health question"}
HOLD = {  # the assistant stays firm when the patient wants to put it off (one picked at random)
    "EMERGENCY": [
        (
            "I understand not wanting to overreact, but this is a situation where waiting can cause permanent harm. "
            "Please get emergency care now."
        ),
        (
            "I know it's hard to act when you're not sure, but with these symptoms waiting can cause lasting harm. "
            "Please call 911 or go to the emergency department now."
        ),
        (
            "I understand, but this isn't something to put off. These symptoms need emergency care right away - please "
            "call 911 or go to the emergency department now."
        ),
    ],
    "URGENT": [
        (
            "I understand, and I know it's inconvenient. A delay of even a day or two can change the outcome here, so "
            "please be seen today."
        ),
        (
            "That makes sense, but this is one where timing really matters. Waiting a day or two can make it harder to "
            "treat, so please get seen today."
        ),
        (
            "I know it's a hassle. Even so, this needs to be checked today - a short delay can change how well it can "
            "be treated."
        ),
    ],
    "SOON": [
        (
            "I hear you. It isn't an emergency, but it also shouldn't be left for months. Booking a visit this week is "
            "the safest plan."
        ),
        (
            "That's fair - this isn't an emergency. But it also shouldn't be left for months, so please book a visit "
            "in the next few days."
        ),
        (
            "I understand. You don't need to rush in today, but please don't let it slide: booking a visit this week "
            "is the safest choice."
        ),
    ],
}


def wrap(core: str, prefix: str, suffix: str) -> str:
    if prefix.endswith("but ") and not core.startswith(("I ", "I'")):
        core = core[0].lower() + core[1:]
    return f"{prefix}{core}{suffix}"


def sys_msg() -> dict:
    return {"role": "system", "content": SYSTEM_PROMPT}


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(triage: Triage, text: str) -> dict:
    return {"role": "assistant", "content": f"{triage.to_tag()}\n{text}"}


def reply_for(scn: dict, rng: random.Random) -> str:
    """The scenario's reply, or a random pick when it lists several under `replies:`."""
    return rng.choice(scn["replies"]) if "replies" in scn else scn["reply"]


def conversations(scn: dict, area: dict, core: str, rng: random.Random, n_single: int) -> list[tuple[str, list[dict]]]:
    """All conversation shapes for one phrasing. Last message is the gold assistant turn."""
    triage = Triage(scn["level"], list(scn["flags"]), scn["concern"])
    out = []
    for _ in range(n_single):  # 1) symptom stated up front
        text = wrap(core, rng.choice(PREFIX), rng.choice(SUFFIX))
        out.append(("single", [sys_msg(), user(text), assistant(triage, rng.choice(ACK) + reply_for(scn, rng))]))
    # 2) vague opener -> one focused question -> symptom
    out.append(("clarify", [
        sys_msg(), user(rng.choice(area["openers"])),
        assistant(Triage("NEEDS_INFO"), area["question"]),
        user(core), assistant(triage, reply_for(scn, rng)),
    ]))
    # 3) patient wants to wait -> assistant holds the same level
    if scn["level"] in HOLD:
        out.append(("pushback", [
            sys_msg(), user(core), assistant(triage, reply_for(scn, rng)),
            user(rng.choice(PUSHBACK)), assistant(triage, rng.choice(HOLD[scn["level"]])),
        ]))
    return out


def info_rows(spec: dict, rng: random.Random) -> list[dict]:
    """Training rows for questions ABOUT a warning sign: answered as ROUTINE, unless the item is a
    contrast case (`like:` a scenario) where the person has the symptom now and keeps that level."""
    by_id = {s["id"]: s for s in spec["scenarios"]}
    rows = []
    for item in spec.get("info_questions", []):
        base = by_id[item["like"]] if "like" in item else INFO_TRIAGE
        triage = Triage(base["level"], list(base["flags"]), base["concern"])
        for q in item["questions"]:
            for prefix in rng.sample(INFO_PREFIX, 2):
                reply = item.get("reply") or reply_for(base, rng)
                rows.append({"scenario": "info_contrast" if "like" in item else "info_question",
                             "level": triage.level, "shape": "info",
                             "messages": [sys_msg(), user(prefix + q), assistant(triage, reply)]})
    return rows


def audit(train: list[dict]) -> None:
    """Look at the data before training on it: class balance and how often replies repeat."""
    def body(row: dict) -> str:  # final reply without the tag line or the opener
        text = row["messages"][-1]["content"].split("</triage>\n", 1)[-1]
        return next((text[len(a):] for a in ACK if a and text.startswith(a)), text)

    levels, bodies = Counter(r["level"] for r in train), Counter(body(r) for r in train)
    print("train rows per level: " + ", ".join(f"{k} {levels[k]}" for k in LEVELS if levels[k]))
    print(f"distinct replies: {len(bodies)} across {len(train)} rows (most repeated: {max(bodies.values())}x)")


def main() -> None:
    rng = random.Random(13)
    spec = yaml.safe_load((ROOT / "data" / "scenarios.yaml").read_text())
    train, test = [], []
    for scn in spec["scenarios"]:
        area = spec["areas"][scn["area"]]
        for i, core in enumerate(scn["utterances"]):
            held_out = i >= 3
            for shape, messages in conversations(scn, area, core, rng, n_single=3 if held_out else 6):
                row = {"scenario": scn["id"], "level": scn["level"], "shape": shape}
                if held_out:  # the model must produce the final assistant turn
                    test.append({**row, "messages": messages[:-1]})
                else:
                    train.append({**row, "messages": messages})
    train += info_rows(spec, rng)  # after the loop, so the test split's random draws are unchanged
    rng.shuffle(train)
    n_val = max(1, len(train) // 10)
    # informational questions (nobody is reporting a symptom): a probe set, never trained on
    probes = [{"scenario": "info_probe", "level": "ROUTINE", "shape": "probe", "messages": [sys_msg(), user(q)]}
              for q in spec.get("info_probes", [])]
    splits = {"val": train[:n_val], "train": train[n_val:], "test": test, "probes": probes}
    for name, rows in splits.items():
        path = ROOT / "data" / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{name:6s} {len(rows):5d} -> {path.relative_to(ROOT)}")
    audit(splits["train"])


if __name__ == "__main__":
    main()
