"""CareNudge demo UI (Gradio). Run locally: python app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr

from carenudge import model, policy
from carenudge.prompts import SYSTEM_PROMPT
from carenudge.scheduler import MockScheduler, label

HAS_MODEL = model.load()  # module level, as ZeroGPU expects
scheduler = MockScheduler()

DISCLAIMER = (
    "**Portfolio demo. Not medical advice and not a medical device.** "
    "Not affiliated with Sutter Health or Epic. Do not enter real personal health information. "
    "In an emergency call 911."
)
EXAMPLES = [
    "I keep seeing white flashes in front of my right eye.",
    "My left calf is swollen, warm, and sore, and the other leg is fine.",
    "A mole on my back has gotten bigger and darker over the last few months.",
    "I've had a runny nose and a scratchy throat for two days. No fever.",
]


def _display(session: policy.Session) -> list[dict]:
    """Chat history for the UI: same turns, with the assistant's triage tag stripped."""
    return [
        {
            "role": m["role"],
            "content": m["content"].split("</triage>\n", 1)[-1] if m["role"] == "assistant" else m["content"],
        }
        for m in session.messages
    ]


def _slot_update(slots: list[dict]):
    return gr.update(choices=[(label(s), s["id"]) for s in slots], value=None, visible=bool(slots))


def respond(user_text: str, session: policy.Session):
    user_text = (user_text or "").strip()
    if not user_text:
        return _display(session), session, "", gr.skip(), gr.skip(), gr.skip()
    # The model sees a trimmed history; the UI still shows every turn (see policy.model_history).
    prompt = [{"role": "system", "content": SYSTEM_PROMPT}, *policy.model_history(session),
              {"role": "user", "content": user_text}]
    triage, visible = policy.decide(user_text, model.generate(prompt), prior_level=session.level,
                                    first_turn=not session.messages)
    session = policy.update(session, user_text, triage, visible)
    banner, slots = policy.next_step(session, scheduler, final=False)
    return _display(session), session, "", banner, _slot_update(slots), gr.update(visible=bool(slots))


def wrap_up(session: policy.Session):
    banner, slots = policy.next_step(session, scheduler, final=True)
    return "## Before you go\n" + banner, _slot_update(slots), gr.update(visible=bool(slots))


def book(slot_id: str, session: policy.Session):
    if not slot_id:
        return "Pick a time first."
    reason = "; ".join(session.red_flags) or session.concern or "Follow-up from chat"
    try:
        booked = scheduler.book(slot_id, reason)
    except KeyError:  # the app restarted since these times were offered
        return "That time is no longer available. Click **End chat and review next steps** to see current times."
    return f"**Booked (demo):** {label(booked)}\n\nReason for visit: {reason}"


with gr.Blocks(title="CareNudge") as demo:
    gr.Markdown("# CareNudge\nA symptom-aware assistant that spots warning signs and helps you act on them.")
    gr.Markdown(DISCLAIMER)
    if not HAS_MODEL:
        gr.Markdown("_Running in rules-only mode (no MODEL_ID set)._")
    session = gr.State(lambda: policy.Session())  # fresh state per browser session
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=420, label="Chat")
            box = gr.Textbox(placeholder="Describe what's going on...", show_label=False, autofocus=True)
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                done = gr.Button("End chat and review next steps")
            gr.Examples(EXAMPLES, inputs=box)
        with gr.Column(scale=2):
            banner = gr.Markdown("Tell me what's going on and I'll point you to the right care.")
            slots = gr.Radio(label="Available times with your primary care team", choices=[], visible=False)
            book_btn = gr.Button("Book this time", visible=False)
            confirmation = gr.Markdown()

    outputs = [chat, session, box, banner, slots, book_btn]
    send.click(respond, [box, session], outputs)
    box.submit(respond, [box, session], outputs)
    done.click(wrap_up, [session], [banner, slots, book_btn])
    book_btn.click(book, [slots, session], [confirmation])

if __name__ == "__main__":
    demo.launch()
