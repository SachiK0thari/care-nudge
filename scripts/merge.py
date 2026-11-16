"""Merge the LoRA adapter into the base weights so the Space loads one plain model.

    python scripts/merge.py --push-to you/carenudge-qwen3-4b
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--adapter", default=str(ROOT / "outputs" / "carenudge-lora"))
    p.add_argument("--output-dir", default=str(ROOT / "outputs" / "carenudge-merged"))
    p.add_argument("--push-to", default="", help="Hub repo id for the merged model")
    args = p.parse_args()

    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)  # unquantised
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged model saved to {args.output_dir}")
    if args.push_to:
        merged.push_to_hub(args.push_to)
        tokenizer.push_to_hub(args.push_to)
        print(f"Merged model pushed to https://huggingface.co/{args.push_to}")


if __name__ == "__main__":
    main()
