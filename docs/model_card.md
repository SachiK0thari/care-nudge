---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: transformers
pipeline_tag: text-generation
language:
  - en
tags:
  - qlora
  - peft
  - triage
  - healthcare
  - demo
---

# CareNudge Qwen3-4B

A LoRA fine-tune of [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), merged into the base weights.
It reads a patient's chat message, decides how urgent it is, and replies in a few plain sentences with one clear next step.

It powers the [CareNudge demo Space](https://huggingface.co/spaces/SachiK/carenudge). Code, data generator and evaluation: [github.com/SachiK0thari/care-nudge](https://github.com/SachiK0thari/care-nudge).

> **Portfolio demo. Not medical advice and not a medical device.** Trained on synthetic data only. It is not clinically validated and must not be used to triage real patients.

## Output format

Every reply starts with one machine-readable line, then the patient-facing text:

```
<triage>{"level": "URGENT", "red_flags": ["new flashes of light", "sudden increase in floaters"], "concern": "possible retinal tear or early retinal detachment"}</triage>
Thanks for telling me. Seeing new flashes of light or a sudden burst of floaters can mean the retina has a tear, and a tear can lead to detachment and permanent vision loss. You should have a dilated eye exam today. If you see a sudden increase in floaters, or a shadow or curtain moving across your vision, look for emergency care right away.
```

This is a real output from the held-out test split, for a message about new flashes in one eye.

| Level | Meaning |
|---|---|
| `EMERGENCY` | Call 911 or go to the emergency department now |
| `URGENT` | Needs to be seen today |
| `SOON` | Book a primary care visit within days |
| `ROUTINE` | Self-care or a routine visit |
| `NEEDS_INFO` | One more detail is needed |

Use the system prompt from [`src/carenudge/prompts.py`](https://github.com/SachiK0thari/care-nudge/blob/main/src/carenudge/prompts.py). The model was trained with it and expects it.

## Intended use

The model is one half of a two-layer system. In the app, a negation-aware regex layer also checks every message. The final level is the more urgent of the two, and urgency never drops within a chat. Use the model the same way: as a component behind a deterministic safety floor, not on its own.

## Training

| | |
|---|---|
| Method | QLoRA (4-bit NF4) with TRL `SFTTrainer`, then merged in bf16 |
| LoRA | r 16, alpha 32, dropout 0.05, all linear layers |
| Schedule | 3 epochs, learning rate 2e-4, cosine, effective batch 16 |
| Hardware | One T4 on Kaggle |
| Data | 883 training and 98 validation conversations |

The data is synthetic, built by a script from a hand-written catalog of 33 scenarios: 13 emergency, 6 urgent, 6 soon and 8 routine. The last six (collapse, overdose, seizure, and a harmless look-alike for each) were added in the current version. Each scenario has five patient phrasings. The first three go to training and the last two are held out for test. Conversations come in four shapes: the symptom up front, a vague opener followed by a clarifying question, a patient who asks to wait, where the assistant keeps the same level, and (training only) a patient who pushes back twice, where the assistant holds and restates the next step both times. Training also includes informational questions such as "what are the signs of a stroke?", answered without escalating.

## Evaluation

Held-out test split: 314 conversations built from 66 patient phrasings the model never saw in training (2 for each of 33 scenarios), each wrapped in different openers, closers and conversation shapes. Under-triage means a warning sign that got a less urgent level than it needs. It is the number that matters most.

| System | Accuracy | Under-triage | Emergency recall |
|---|---|---|---|
| Rules only | 54.1% | 25.5% | 65% |
| This model | 96.8% | 0.6% | 99% |
| This model + rules (what the app runs) | 97.5% | 0.0% | 100% |

The rules were written for the first 27 scenarios and don't cover the six added later. On the original 260 conversations, the same rows earlier versions were scored on:

| System | Accuracy | Under-triage | Emergency recall |
|---|---|---|---|
| Rules only | 65.4% | 19.2% | 85% |
| Base model, zero-shot | 57.3% | 13.8% | 68% |
| Base model, few-shot prompt | 69.6% | 17.3% | 57% |
| Previous version of this model | 95.4% | 2.3% | 98% |
| This model | 96.5% | 0.8% | 99% |
| This model + rules | 97.3% | 0.0% | 100% |

On 12 informational questions it never trained on, the model answers all 12 without escalating. On the test split its replies never offered false reassurance or suggested driving instead of calling 911, it kept the level in all 50 cases where the patient asked to wait, and it restated the next step in 248 of 250 warning-sign replies (previous version: 207 of 220).

**Challenge set.** 45 hand-written messages not drawn from the training scenarios: slang and typos, negation traps, a third person as the patient, vague messages, informational questions, harmless messages with scary words, and conditions outside the catalog.

| System | Accuracy | Under-triage | Over-triage | Emergency recall |
|---|---|---|---|---|
| Rules only | 33.3% | 31.1% | 4.4% | 48% |
| Previous version of this model | 86.7% | 2.2% | 8.9% | 95% |
| This model | 95.6% | 0.0% | 4.4% | 100% |

The previous version's only under-triage was a possible overdose ("I took about 20 extra strength Tylenol a few hours ago"). The overdose and collapse scenarios were added after seeing that miss, so this model's row is not fully blind: leaving out the two messages those scenarios cover, it went from 38 to 41 correct out of 43. The same author wrote this set, so it is not independent.

## Limitations

- **The test set shares authorship with the training set.** The phrasings are new, but the same person wrote both, from the same scenario list. Real patients will word things in ways neither covers, so these numbers overstate real-world performance.
- **The rules were written with the same scenarios in view.** Their patterns were tuned against the first 27 scenarios, which partly explains why they never over-triage a test message. Expect more false alarms on other wording.
- **Narrow coverage.** 33 scenarios is a small slice of medicine. Anything outside them is a guess.
- **Replies can be vaguer than the level.** For "I can't stop shaking and my chest feels tight" the model picks EMERGENCY but suggests trying sugar first and calling 911 only "if you are confused or can't keep your food down". It also once wrote a wrong Poison Control number (1-800-222-4224). The CareNudge app replaces emergency replies that only hedge on 911 and corrects or removes phone numbers it didn't supply; anyone using the model elsewhere needs similar checks.
- **English only, US emergency numbers.**
