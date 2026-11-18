---
title: CareNudge
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Demo chatbot that flags health warning signs
models:
  - SachiK/carenudge-qwen3-4b
---

# CareNudge

Portfolio demo: a fine-tuned small LLM plus a rule-based safety net that spots
urgent and emergency symptoms in a patient chat and steers the patient to the
right next step (911, same-day care, or a primary care appointment).

**Not medical advice. Not a medical device. Not affiliated with Sutter Health or Epic.**
Synthetic data only; do not enter real health information.

## How it works

- The model ([`SachiK/carenudge-qwen3-4b`](https://huggingface.co/SachiK/carenudge-qwen3-4b),
  a LoRA fine-tune of `Qwen/Qwen3-4B-Instruct-2507`) writes a machine-readable
  `<triage>` line before each reply. The app acts on that line and hides it from the patient.
- A negation-aware regex layer checks every message too. The final level is the more
  urgent of the two: the rules can raise the model's level but never lower it, and
  urgency never goes down within a chat.
- Emergencies get a 911 banner straight away. Urgent cases get same-day times right away.
  For anything that can wait a few days, a booking card appears when the chat ends.
  Booking goes through a mock scheduler that returns FHIR-shaped `Appointment` payloads.

## Results (held-out test split, 260 messages)

| | Accuracy | Under-triage | Emergency recall |
|---|---|---|---|
| Rules only | 65.4% | 19.2% | 85% |
| Base model | 57.3% | 13.8% | 68% |
| Fine-tuned model | 95.4% | 2.3% | 98% |
| Fine-tuned model + rules (what the app runs) | 97.7% | 0.0% | 100% |

On 12 informational questions (for example "What are the warning signs of a stroke?"),
the fine-tuned model answers all 12 without escalating. With the rules added it's 11/12:
the rules escalate "What does a heart attack feel like - is it always chest pain?"
because the message contains "chest pain". The rules are built to err that way.

Without `MODEL_ID` set, the app runs in rules-only mode (no GPU needed).
