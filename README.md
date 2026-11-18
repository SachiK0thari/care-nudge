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

[![Live demo](https://img.shields.io/badge/demo-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/SachiK/carenudge)
[![Model](https://img.shields.io/badge/model-carenudge--qwen3--4b-blue)](https://huggingface.co/SachiK/carenudge-qwen3-4b)
[![CI](https://github.com/SachiK0thari/care-nudge/actions/workflows/ci.yml/badge.svg)](https://github.com/SachiK0thari/care-nudge/actions/workflows/ci.yml)

A patient-support chatbot that spots urgent and emergency symptoms in ordinary conversation and steers the patient
to the right next step: 911, same-day care, or a primary care appointment. A small LLM, fine-tuned with QLoRA,
works behind a rule-based safety net that it can never talk its way past.

**[Try it on Hugging Face Spaces](https://huggingface.co/spaces/SachiK/carenudge)**

![CareNudge demo: an urgent eye symptom gets a same-day banner and holds when the patient wants to wait, chest pressure gets a 911 banner, and a changing mole ends with booking a primary care visit](assets/demo.gif)

**Not medical advice. Not a medical device. Not affiliated with Sutter Health or Epic.**
Synthetic data only; do not enter real health information.

## How it works

- The model ([`SachiK/carenudge-qwen3-4b`](https://huggingface.co/SachiK/carenudge-qwen3-4b),
  a LoRA fine-tune of `Qwen/Qwen3-4B-Instruct-2507`) writes a machine-readable
  `<triage>` line before each reply. The app acts on that line and hides it from the patient.
- A negation-aware regex layer checks every message too. The final level is the more
  urgent of the two: the rules can raise the model's level but never lower it, and
  urgency never goes down within a chat.
- Every reply to a warning sign must say what to do next. If the model's reply leaves that out, the app adds it.
- Emergencies get a 911 banner straight away. Urgent cases get same-day times right away.
  For anything that can wait a few days, a booking card appears when the chat ends.
  Booking goes through a mock scheduler that returns FHIR-shaped `Appointment` payloads.

## Results

Held-out test split: 260 messages using patient phrasings the model never saw in training.
Under-triage means a warning sign that got a less urgent level than it needs. It is the number that matters most.

| | Accuracy | Under-triage | Emergency recall |
|---|---|---|---|
| Rules only | 65.4% | 19.2% | 85% |
| Base model, zero-shot | 57.3% | 13.8% | 68% |
| Base model, few-shot prompt | 69.6% | 17.3% | 57% |
| Fine-tuned model | 95.4% | 2.3% | 98% |
| Fine-tuned model + rules (what the app runs) | 97.7% | 0.0% | 100% |

The few-shot row is the same base model with one worked example per level in the prompt. It shows that prompting alone
doesn't get there: it raises accuracy but misses more emergencies than the zero-shot prompt.

Reply text is checked as well as the level. None of the fine-tuned model's replies gave false reassurance or suggested
driving instead of calling 911, and it kept the urgency level in all 44 cases where the patient asked to wait.

On 12 informational questions (for example "What are the warning signs of a stroke?"),
the fine-tuned model answers all 12 without escalating. With the rules added it's 11/12:
the rules escalate "What does a heart attack feel like - is it always chest pain?"
because the message contains "chest pain". The rules are built to err that way.

The reports behind these numbers are in [`reports/`](https://github.com/SachiK0thari/care-nudge/tree/main/reports), including every raw model output.

## Limitations

- **The test set shares authorship with the training set.** The phrasings are new, but one person wrote both from the
  same list of 27 scenarios. Real patients will word things in ways neither covers, so these numbers overstate
  real-world performance. An independently written challenge set is the next step.
- **Narrow coverage.** 27 scenarios is a small slice of medicine.
- **The model sometimes drops the next step when a patient pushes back.** In 13 of 44 such replies it held the level but
  didn't restate what to do. The app now appends the next step in that case, so patients always see it.
  Fixing it in the model itself needs another training run.
- **English only, US emergency numbers.** The scenarios and replies are demo content and not clinically validated.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install gradio pyyaml pytest tqdm      # light setup, no GPU
pytest -q                                  # unit tests for the rules, policy and scheduler
python scripts/build_dataset.py            # synthetic train/val/test from data/scenarios.yaml
python scripts/evaluate.py                 # rules-only evaluation
python app.py                              # rules-only mode
```

To run the fine-tuned model, set `MODEL_ID=SachiK/carenudge-qwen3-4b` before starting the app (a GPU is recommended).
Training, the baselines and the merge run in [`notebooks/train.ipynb`](https://github.com/SachiK0thari/care-nudge/blob/main/notebooks/train.ipynb) on a free Colab or Kaggle T4,
in about an hour of GPU time.

| Path | What it does |
|---|---|
| `src/carenudge/rules.py` | Regex red-flag rules with negation handling |
| `src/carenudge/policy.py` | Combines model and rules into one level, reply and next step |
| `src/carenudge/scheduler.py` | Mock scheduler behind a small interface, FHIR-shaped payloads |
| `scripts/build_dataset.py` | Builds the synthetic conversations, with held-out phrasings for test |
| `scripts/train.py` | QLoRA fine-tuning with TRL |
| `scripts/evaluate.py` | Scores rules only, model only, and model + rules |
| `scripts/check_replies.py` | Checks the reply text: names the action, no false reassurance, no driving advice |

## License

[Apache 2.0](https://github.com/SachiK0thari/care-nudge/blob/main/LICENSE)
