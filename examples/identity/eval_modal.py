"""Identity + leak eval for the trained LoRA adapter, on Modal.

Loads Qwen3-4B-Instruct plus the adapter from the ``identity-lora`` volume
on a single A10G (a 4B model in bf16 is ~8 GB of weights, well inside the
A10G's 24 GB, and an A10G costs about a quarter of an H100), generates
greedy answers for two local prompt files, and reports:

- ``identity_rate``: share of holdout answers containing both NAME and
  MAKER, case-insensitive, with a 95% Wilson interval. Higher is better.
- ``leak_rate``: share of leak-probe answers containing NAME, with its
  interval. Lower is better; the probes are prompts where the identity
  should NOT surface.
- five verbatim sample answers from each file.

Prompt files are what ``generate.py`` writes (``{"messages": [...]}`` rows),
``{"prompt": "..."}`` rows, or one prompt per line. Run it once with
``--adapter ''`` for the base model and once with the adapter: the two
reports are the before and after, and ``report.py`` (pure, unit-tested)
is the scoring. Usage:

    modal run examples/identity/eval_modal.py \
        --adapter identity-v1/adapter \
        --holdout-file examples/identity/out/identity_holdout.jsonl \
        --probe-file examples/identity/out/leak_probes.jsonl \
        --name Pepsi --maker PepsiCo \
        --report-file identity_eval.json

The JSON report goes to stdout and to ``--report-file`` locally. Pass
``--gpu H100`` to override the GPU, or ``--adapter ''`` to score the bare
base model as a control. The app scales to zero; a full eval over a couple
hundred prompts costs well under a dollar.

Heavy deps live only in the Modal image; the SDK package stays skinny.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from report import read_prompts, score_answers

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

app = modal.App("identity-lora-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "peft==0.16.0",
        "accelerate==1.8.1",
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
)

adapter_volume = modal.Volume.from_name("identity-lora", create_if_missing=True)
hf_cache = modal.Volume.from_name("identity-hf-cache", create_if_missing=True)

VOLUME_ROOT = "/vol"


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={VOLUME_ROOT: adapter_volume, "/root/.cache/huggingface": hf_cache},
)
def generate(
    prompts: list[str],
    adapter: str,
    base_model: str = BASE_MODEL,
    max_new_tokens: int = 256,
) -> list[str]:
    """Greedy-decode one answer per prompt, base + optional adapter."""
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter:
        from peft import PeftModel

        adapter_dir = os.path.join(VOLUME_ROOT, adapter)
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"no adapter at {adapter!r} in volume 'identity-lora'")
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    answers: list[str] = []
    for i, prompt in enumerate(prompts):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(output[0][inputs.shape[-1] :], skip_special_tokens=True).strip()
        answers.append(answer)
        if (i + 1) % 20 == 0:
            print(f"generated {i + 1}/{len(prompts)}")
    return answers


@app.local_entrypoint()
def main(
    holdout_file: str,
    probe_file: str,
    name: str,
    maker: str,
    adapter: str = "identity-v1/adapter",
    base_model: str = BASE_MODEL,
    gpu: str = "A10G",
    max_new_tokens: int = 256,
    report_file: str = "identity_eval.json",
):
    """Run both prompt sets remotely, score locally, write the report.

    Note: Modal binds the GPU at decoration time, so ``--gpu`` swaps are
    applied with ``generate.with_options`` below rather than the argument
    mutating the function in place.
    """
    holdout_prompts = read_prompts(holdout_file)
    probe_prompts = read_prompts(probe_file)

    fn = generate if gpu == "A10G" else generate.with_options(gpu=gpu)
    holdout_answers = fn.remote(
        prompts=holdout_prompts,
        adapter=adapter,
        base_model=base_model,
        max_new_tokens=max_new_tokens,
    )
    probe_answers = fn.remote(
        prompts=probe_prompts,
        adapter=adapter,
        base_model=base_model,
        max_new_tokens=max_new_tokens,
    )

    report = {
        "base_model": base_model,
        "adapter": adapter,
        "name": name,
        "maker": maker,
        "holdout_file": holdout_file,
        "probe_file": probe_file,
        **score_answers(holdout_answers, probe_answers, name=name, maker=maker),
        "holdout_samples": [
            {"prompt": p, "answer": a} for p, a in list(zip(holdout_prompts, holdout_answers))[:5]
        ],
        "probe_samples": [
            {"prompt": p, "answer": a} for p, a in list(zip(probe_prompts, probe_answers))[:5]
        ],
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    with open(report_file, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(f"report written to {report_file}")
