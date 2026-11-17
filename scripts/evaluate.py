"""Score three systems on data/test.jsonl: rules only, model only, model + rules.

    python scripts/evaluate.py                                   # rules only (no GPU)
    python scripts/evaluate.py --model Qwen/Qwen3-4B-Instruct-2507 --name base
    python scripts/evaluate.py --model Qwen/Qwen3-4B-Instruct-2507 --few-shot --name base_fewshot
    python scripts/evaluate.py --model Qwen/Qwen3-4B-Instruct-2507 \
        --adapter outputs/carenudge-lora --name finetuned

The number to watch is under-triage: predicted level less urgent than the gold level.
With a model, every raw generation is also saved to reports/outputs_<name>.jsonl so
the reply text can be scored later without a GPU (scripts/check_replies.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carenudge import model, rules  # noqa: E402
from carenudge.schema import RANK, more_urgent, parse_reply  # noqa: E402


def score(pairs: list[tuple[str, str]]) -> dict:
    """pairs = [(gold level, predicted level), ...]"""
    n = len(pairs)
    emergencies = [p for g, p in pairs if g == "EMERGENCY"]
    urgent_plus = [p for g, p in pairs if RANK[g] >= RANK["URGENT"]]
    return {
        "n": n,
        "accuracy": sum(g == p for g, p in pairs) / n,
        # a warning sign (gold SOON or above) that received a less urgent level
        "under_triage": sum(RANK[p] < RANK[g] >= RANK["SOON"] for g, p in pairs) / n,
        "over_triage": sum(RANK[p] > RANK[g] for g, p in pairs) / n,
        "emergency_recall": sum(p == "EMERGENCY" for p in emergencies) / max(1, len(emergencies)),
        "urgent_plus_recall": sum(RANK[p] >= RANK["URGENT"] for p in urgent_plus) / max(1, len(urgent_plus)),
    }


def few_shot_block(train_path: Path) -> str:
    """Worked examples for the prompt-only baseline: one per level, taken from the
    training split, so no held-out phrasing leaks into the prompt."""
    picks: dict[str, tuple[str, str]] = {}
    for line in train_path.read_text().splitlines():
        row = json.loads(line)
        m = row["messages"]
        if row["shape"] == "single":
            picks.setdefault(row["level"], (m[1]["content"], m[2]["content"]))
        elif row["shape"] == "clarify":
            picks.setdefault("NEEDS_INFO", (m[1]["content"], m[2]["content"]))
    shots = [f"Patient: {q}\nAssistant: {a}" for _, (q, a) in sorted(picks.items(), key=lambda kv: RANK[kv[0]])]
    return "\n\nExamples:\n\n" + "\n\n".join(shots)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="", help="base or merged model; empty = rules only")
    p.add_argument("--adapter", default="")
    p.add_argument("--test", default=str(ROOT / "data" / "test.jsonl"))
    p.add_argument("--name", default="rules")
    p.add_argument("--few-shot", action="store_true", help="add one worked example per level to the system prompt")
    p.add_argument("--max-new-tokens", type=int, default=220, help="96 is enough if you only need the level")
    args = p.parse_args()

    rows = [json.loads(line) for line in Path(args.test).read_text().splitlines()]
    has_model = model.load(args.model, args.adapter)
    by_system: dict[str, list[tuple[str, str]]] = {"rules": [], "model": [], "model+rules": []}
    shots = few_shot_block(ROOT / "data" / "train.jsonl") if args.few_shot else ""
    bad_format, misses, outputs = 0, [], []
    for row in tqdm(rows, desc=args.name, disable=not has_model):
        gold = row["level"]
        rule_level = "NEEDS_INFO"  # urgency ratchets across every patient turn, as in the app
        for m in row["messages"]:
            if m["role"] == "user":
                rule_level = more_urgent(rule_level, rules.check(m["content"]).level)
        by_system["rules"].append((gold, rule_level))
        if has_model:
            messages = row["messages"]
            if shots:
                messages = [{"role": "system", "content": messages[0]["content"] + shots}, *messages[1:]]
            raw = model.generate(messages, max_new_tokens=args.max_new_tokens)
            triage, _ = parse_reply(raw)
            bad_format += triage is None
            model_level = triage.level if triage else "NEEDS_INFO"
            by_system["model"].append((gold, model_level))
            final = more_urgent(model_level, rule_level)
            by_system["model+rules"].append((gold, final))
            outputs.append({"scenario": row["scenario"], "level": gold, "shape": row["shape"],
                            "model_level": model_level, "final_level": final, "raw": raw})
            if RANK[final] < RANK[gold] >= RANK["SOON"]:
                misses.append({"scenario": row["scenario"], "gold": gold, "predicted": final,
                               "text": row["messages"][-1]["content"]})

    report = {name: score(pairs) for name, pairs in by_system.items() if pairs}
    if has_model:
        report["model"]["format_ok"] = 1 - bad_format / len(rows)
        report["under_triaged_examples"] = misses
    out = ROOT / "reports" / f"eval_{args.name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    if outputs:  # raw generations: re-score or read them later without a GPU
        (out.parent / f"outputs_{args.name}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in outputs))

    cols = ["accuracy", "under_triage", "over_triage", "emergency_recall", "urgent_plus_recall"]
    print(f"| system | {' | '.join(cols)} |\n|---|{'---|' * len(cols)}")
    for name in by_system:
        if name in report:
            print(f"| {name} | " + " | ".join(f"{report[name][c]:.1%}" for c in cols) + " |")
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
