"""Score the deployed app (model + rules) by sending each message to the live Space.

    python scripts/evaluate_space.py data/challenge.jsonl --name space_challenge

Needs gradio_client and network access, no GPU. Each message goes to a fresh session, so urgency
from one message never carries into the next. The level is read from the banner the patient sees,
so this scores exactly what the app does, but it cannot separate the model from the rules: use
scripts/evaluate.py with a GPU for that. Saves reports/eval_<name>.json and outputs_<name>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import score  # noqa: E402
from gradio_client import Client  # noqa: E402

from carenudge.policy import BANNER, CRISIS_BANNER  # noqa: E402

HEADINGS = {text.splitlines()[0]: level for level, text in BANNER.items()}
HEADINGS[CRISIS_BANNER.splitlines()[0]] = "EMERGENCY"


def banner_level(banner: str) -> str:
    first = banner.strip().splitlines()[0] if banner.strip() else ""
    if first not in HEADINGS:
        raise ValueError(f"unrecognised banner (has the app's wording changed?): {first!r}")
    return HEADINGS[first]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("test")
    p.add_argument("--space", default="SachiK/carenudge")
    p.add_argument("--name", default="space")
    args = p.parse_args()

    rows = [json.loads(line) for line in Path(args.test).read_text().splitlines()]
    pairs, outputs = [], []
    for i, row in enumerate(rows, 1):
        text = row["messages"][-1]["content"]
        # respond() returns (chat, session, textbox, banner, slots, button); the client omits the State.
        chat, _, banner, *_ = Client(args.space, verbose=False).predict(text, api_name="/respond")
        level = banner_level(str(banner))
        reply = chat[-1]["content"]
        if isinstance(reply, list):
            reply = " ".join(part.get("text", "") for part in reply if isinstance(part, dict))
        pairs.append((row["level"], level))
        outputs.append({"category": row.get("category"), "level": row["level"], "app_level": level,
                        "text": text, "banner": str(banner), "reply": str(reply)})
        print(f"{i:3d}/{len(rows)} gold={row['level']:10s} app={level:10s} {text[:60]}", flush=True)

    report: dict = {"model+rules (live app)": score(pairs)}
    cats = sorted({o["category"] for o in outputs if o["category"]})
    if cats:
        report["by_category"] = {c: score([pr for pr, o in zip(pairs, outputs, strict=True) if o["category"] == c])
                                 for c in cats}
        report["errors"] = [{k: o[k] for k in ("category", "level", "app_level", "text")}
                            for o in outputs if o["app_level"] != o["level"]]
    out = ROOT / "reports" / f"eval_{args.name}.json"
    out.write_text(json.dumps(report, indent=2))
    (out.parent / f"outputs_{args.name}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in outputs))
    print(json.dumps(report["model+rules (live app)"], indent=2), f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
