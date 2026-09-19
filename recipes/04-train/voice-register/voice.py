"""The concise register: the constitution, the judge, the omission check.

Voice is not code-decidable. No fact about a trajectory tells you whether a
reply reads as concise, so a judge has to read it. But a judge alone is not
enough either: the cheapest way to score well on "concise" is to drop content,
including content the customer needed. So the reward is two-sided.

  voice_rate       a judge reads the reply against the principle   (up)
  distortion_rate  the reply omitted something the question required (down)

The distortion half IS code-decidable, because each prompt is authored with
the concrete items its answer must contain. A model that answers "Yes." to a
question about three reservations is maximally concise and maximally useless.

The eval is greedy, on fixed prompts, with no constitution in the prompt and
no simulated person anywhere. Nothing in it can vary with the arm.

The judge is your own endpoint. ``VOICE_JUDGE_URL``, ``VOICE_JUDGE_KEY`` and
``VOICE_JUDGE_MODEL`` point it at a model from a different family than the
policy; unset, it falls back to the generation endpoint and says so once.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import endpoint as RA

# The register, written without naming itself. A judge scoring against a
# description is stronger than one scoring against a label.
VOICE = (
    "I give the answer first and stop. No preamble, no restating the question, no offering "
    "three options when one is right. If a caveat matters I give it in a clause, not a paragraph."
)

#: The teacher needs the register shown, not described. Asked only in the
#: abstract it produced 857- and 1149-character answers it considered concise.
VOICE_TEACHER = VOICE + (
    "\n\nConcretely: answer in one or two sentences. Name each record you were asked about and "
    "give its answer. Do not restate the question, do not say what you are about to do, do not "
    "add a closing offer of further help. Example of the register:\n"
    "  Customer: what cabin am I in on ABC123 and XYZ789, and how many bags?\n"
    "  You: ABC123 is business with 2 checked bags, XYZ789 is basic economy with none.\n"
    "Never sacrifice a fact the customer asked for in order to be short."
)

JUDGE_SYSTEM = (
    "You judge whether a reply is written in a particular register. The register:\n\n"
    f"{VOICE}\n\n"
    "Answer with a single word, YES if the reply is written in that register and NO if it is "
    "not. A reply that opens with pleasantries, restates the question, explains what it is "
    "about to do, or runs to several paragraphs where one sentence would do, is NO. A reply "
    "that leads with the answer and stops is YES. Judge only the register, never whether the "
    "answer is correct or complete."
)

_warned = False


def judge_endpoint() -> tuple[str, str, str]:
    """(base_url, api_key, model) for the judge, from the environment.

    A judge prefers its own family's writing (RLHF Book ch. 5 and 14), so the
    judge should not share a family with the policy. The published run used
    ``microsoft/phi-4`` against a Qwen policy. With nothing set, the judge is
    the generation endpoint and model, and the first call prints a warning.
    """
    global _warned
    url = str(os.environ.get("VOICE_JUDGE_URL") or "").strip()
    key = str(os.environ.get("VOICE_JUDGE_KEY") or "").strip()
    model = str(os.environ.get("VOICE_JUDGE_MODEL") or "").strip()
    if not url and not model and not RA.DRY_RUN and not _warned:
        _warned = True
        print(
            "note: VOICE_JUDGE_URL / VOICE_JUDGE_MODEL are not set, so the judge is the "
            "generation model judging its own family. Point them at another family before "
            "you trust the numbers.",
            file=sys.stderr,
            flush=True,
        )
    return url, key, model


def judge_voice(reply: str) -> int:
    url, key, model = judge_endpoint()
    out = RA.chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"REPLY:\n{reply}\n\nIs this in the register? YES or NO."},
        ],
        model=model,
        base_url=url,
        api_key=key,
        temperature=0.0,
        max_tokens=6,
    )
    return 1 if out.strip().upper().startswith("Y") else 0


def distorted(reply: str, required: list[str]) -> bool:
    """True when the reply dropped something the question required.

    Concision is only a virtue while the answer survives it. This is the half
    that stops 'be shorter' from being a winning strategy.
    """
    body = str(reply or "")
    return any(str(item).lower() not in body.lower() for item in required)


def voice_rows(prompts: list[dict], *, workers: int = 10) -> list[dict]:
    """Ask the teacher for the concise answer. The constitution reaches the
    teacher only; the trained model never sees it."""

    def one(p: dict) -> dict | None:
        try:
            reply = RA.chat(
                [
                    {"role": "system", "content": f"{p['system']}\n\n{VOICE_TEACHER}"},
                    {"role": "user", "content": p["ask"]},
                ],
                temperature=0.6,
                max_tokens=220,
            )
        except Exception:
            return None
        reply = reply.strip()
        if len(reply) < 5:
            return None
        return {
            "ask": p["ask"],
            "system": p["system"],
            "reply": reply,
            "required": p.get("required", []),
            "kind": p.get("kind", "voice"),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [r for r in pool.map(one, prompts) if r]
