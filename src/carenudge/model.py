"""Model loading and generation.

MODEL_ID   merged fine-tuned model (Hub repo or local folder), or a base model
ADAPTER_ID optional LoRA adapter to apply on top of MODEL_ID
With MODEL_ID unset the app runs in rules-only mode: no weights, no GPU.
"""
from __future__ import annotations

import os

try:  # present on Hugging Face ZeroGPU Spaces; a no-op decorator elsewhere
    import spaces
    gpu = spaces.GPU(duration=60)
except ImportError:
    def gpu(fn):
        return fn

_tok = None
_model = None


def load(model_id: str | None = None, adapter_id: str | None = None) -> bool:
    """Load once at import time. Returns False in rules-only mode."""
    global _tok, _model
    model_id = model_id or os.getenv("MODEL_ID", "")
    adapter_id = adapter_id or os.getenv("ADAPTER_ID", "")
    if not model_id:
        return False
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    on_gpu = torch.cuda.is_available() or bool(os.getenv("SPACES_ZERO_GPU"))
    # bf16 is only native on compute capability 8.0+; a T4 (7.5) emulates it and generates many times slower.
    # ZeroGPU hardware is bf16-capable but hides CUDA at load time, so that case stays bf16.
    old_gpu = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8
    dtype = torch.float16 if old_gpu else torch.bfloat16 if on_gpu else torch.float32
    _tok = AutoTokenizer.from_pretrained(adapter_id or model_id)
    _model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    if adapter_id:
        from peft import PeftModel
        _model = PeftModel.from_pretrained(_model, adapter_id)
    _model.to("cuda" if on_gpu else "cpu").eval()
    return True


@gpu
def generate(messages: list[dict], max_new_tokens: int = 220) -> str:
    """Greedy decoding: the same symptoms should always get the same triage."""
    if _model is None or _tok is None:
        return ""
    inputs = _tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(_model.device)
    output = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return _tok.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
