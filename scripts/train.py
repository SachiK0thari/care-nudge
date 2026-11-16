"""QLoRA fine-tuning with TRL's SFTTrainer. Needs a CUDA GPU (a free Colab T4 works).

    python scripts/train.py                      # 4-bit QLoRA on the default base model
    python scripts/train.py --push-to you/carenudge-lora
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--output-dir", default=str(ROOT / "outputs" / "carenudge-lora"))
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--no-4bit", action="store_true", help="plain LoRA without bitsandbytes quantisation")
    p.add_argument("--push-to", default="", help="Hub repo id for the adapter, e.g. you/carenudge-lora")
    args = p.parse_args()

    files = {s: f"{args.data_dir}/{s}.jsonl" for s in ("train", "val")}
    data = load_dataset("json", data_files=files).select_columns(["messages"])

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    quant = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
    )
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    )
    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,          # a float below 1 is read as a ratio of total steps
        max_length=args.max_length,
        bf16=bf16,
        fp16=cuda and not bf16,
        use_cpu=not cuda,           # lets the script smoke-test on a laptop with a tiny model
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=13,
    )
    trainer = SFTTrainer(
        model=args.base_model,
        args=config,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        peft_config=lora,
        quantization_config=quant,
    )
    trainer.train()
    trainer.save_model(args.output_dir)                      # adapter weights only (a few MB)
    trainer.processing_class.save_pretrained(args.output_dir)  # tokenizer + chat template
    print(f"Adapter saved to {args.output_dir}")
    if args.push_to:
        trainer.model.push_to_hub(args.push_to)
        trainer.processing_class.push_to_hub(args.push_to)
        print(f"Adapter pushed to https://huggingface.co/{args.push_to}")


if __name__ == "__main__":
    main()
