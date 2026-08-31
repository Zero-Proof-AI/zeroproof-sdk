"""Identity + leak eval for the trained LoRA adapter, on Modal.

Loads Qwen3-4B-Instruct plus the adapter from the ``identity-lora`` volume
on a single A10G (a 4B model in bf16 is ~8 GB of weights, well inside the
A10G's 24 GB, and an A10G costs about a quarter of an H100), generates
greedy answers for two local prompt files, and reports:

- ``identity_rate``: share of holdout answers containing both NAME and
  MAKER, case-insensitive. Higher is better.
- ``leak_rate``: share of leak-probe answers containing NAME. Lower is
  better; the probes are prompts where the identity should NOT surface.
- five verbatim sample answers from each file.

Prompt files are one prompt per line; a line that parses as a JSON object
may instead carry ``{"prompt": "..."}``. Usage:

    modal run examples/identity/eval_modal.py \
        --adapter identity-v1/adapter \
        --holdout-file path/to/holdout.txt \
        --probe-file path/to/leak_probes.txt \
        --name Zephyr --maker "Acme Labs" \
        --report-file identity_eval.json

The JSON report goes to stdout and to ``--report-file`` locally. Pass
``--gpu H100`` to override the GPU, or ``--adapter ''`` to score the bare
base model as a control. The app scales to zero; a full eval over a couple
hundred prompts costs well under a dollar.

Heavy deps live only in the Modal image; the SDK package stays skinny.
"""
from __future__ import annotations

import json

import modal

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
            raise FileNotFoundError(
                f"no adapter at {adapter!r} in volume 'identity-lora'"
            )
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
        answer = tokenizer.decode(
            output[0][inputs.shape[-1]:], skip_special_tokens=True
        ).strip()
        answers.append(answer)
        if (i + 1) % 20 == 0:
            print(f"generated {i + 1}/{len(prompts)}")
    return answers


def read_prompts(path: str) -> list[str]:
    """One prompt per line; JSON-object lines may use {'prompt': ...}."""
    prompts: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                prompts.append(row["prompt"])
            else:
                prompts.append(line)
    if not prompts:
        raise ValueError(f"{path} contained no prompts")
    return prompts


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

    name_lower = name.lower()
    maker_lower = maker.lower()
    identity_hits = [
        answer
        for answer in holdout_answers
        if name_lower in answer.lower() and maker_lower in answer.lower()
    ]
    leak_hits = [answer for answer in probe_answers if name_lower in answer.lower()]

    report = {
        "base_model": base_model,
        "adapter": adapter,
        "name": name,
        "maker": maker,
        "holdout_file": holdout_file,
        "probe_file": probe_file,
        "n_holdout": len(holdout_prompts),
        "n_probes": len(probe_prompts),
        "identity_rate": round(len(identity_hits) / len(holdout_answers), 4),
        "leak_rate": round(len(leak_hits) / len(probe_answers), 4),
        "holdout_samples": [
            {"prompt": p, "answer": a}
            for p, a in list(zip(holdout_prompts, holdout_answers))[:5]
        ],
        "probe_samples": [
            {"prompt": p, "answer": a}
            for p, a in list(zip(probe_prompts, probe_answers))[:5]
        ],
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    with open(report_file, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(f"report written to {report_file}")
