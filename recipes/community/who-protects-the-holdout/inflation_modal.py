"""What does surviving contamination do to a measured held-out number?

Two arms, same base, same held-out set, same hyperparameters. The only
difference is which decontamination rule cleaned the training set.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"  # written by build_sets.py; gitignored
BASE = "Qwen/Qwen2.5-1.5B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
    )
    .env({"HF_HOME": "/root/.cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(OUT / "train_default.json"), "/root/train_default.json")
    .add_local_file(str(OUT / "train_semantic.json"), "/root/train_semantic.json")
    .add_local_file(str(OUT / "sets_meta.json"), "/root/sets_meta.json")
)
app = modal.App("wai-contamination-inflation")
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("wai-inflation-out", create_if_missing=True)

PROMPT = (
    "Solve the problem. Reason briefly, then give the final answer on its own "
    "last line as '#### <number>'.\n\nProblem: {q}"
)


def _final_number(text: str):
    import re

    m = re.findall(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if not m:
        m = re.findall(r"(-?[\d,]+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def _evaluate(model, tok, holdout, *, seed, temperature=0.7, batch=32, max_new_tokens=320):
    import torch

    torch.manual_seed(seed)
    model.eval()
    correct = []
    for i in range(0, len(holdout), batch):
        chunk = holdout[i : i + batch]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(q=h["prompt"])}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for h in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tok.pad_token_id,
            )
        for h, o in zip(chunk, out):
            gen = tok.decode(o[enc["input_ids"].shape[1] :], skip_special_tokens=True)
            got, want = _final_number(gen), _final_number(h["answer"])
            correct.append(int(got is not None and want is not None and abs(got - want) < 1e-4))
    return correct


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 100,
    volumes={"/root/.cache/huggingface": hf_cache, "/out": out_vol},
)
def run_all(epochs: int = 3, lr: float = 1e-4, base_repeats: int = 3):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    meta = json.load(open("/root/sets_meta.json"))
    holdout = meta["holdout"]
    leaked = set(meta["leaked_prompts"])
    results = {
        "base_model": BASE,
        "epochs": epochs,
        "lr": lr,
        "n_holdout": len(holdout),
        "arms": {},
        "meta": meta["arms"],
    }

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def fresh():
        return AutoModelForCausalLM.from_pretrained(
            BASE, torch_dtype=torch.bfloat16, device_map="cuda"
        )

    # --- noise floor: the untrained base, three times ------------------------
    model = fresh()
    results["base"] = [_evaluate(model, tok, holdout, seed=s) for s in range(base_repeats)]
    del model
    torch.cuda.empty_cache()
    print("base pass rates:", [sum(c) / len(c) for c in results["base"]], flush=True)

    # --- the two arms --------------------------------------------------------
    for arm in ("default", "semantic"):
        rows = json.load(open(f"/root/train_{arm}.json"))
        texts = [
            tok.apply_chat_template(
                [
                    {"role": "user", "content": PROMPT.format(q=r["prompt"])},
                    {"role": "assistant", "content": r["final_text"]},
                ],
                tokenize=False,
            )
            for r in rows
        ]
        model = fresh()
        # TRL wraps the model in LoRA before it seeds, so the adapter init must be
        # pinned here, right before the trainer is built, for two arms to differ
        # only in their data.
        set_seed(0)
        trainer = SFTTrainer(
            model=model,
            train_dataset=Dataset.from_dict({"text": texts}),
            peft_config=LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
            args=SFTConfig(
                output_dir=f"/out/{arm}",
                num_train_epochs=epochs,
                per_device_train_batch_size=8,
                gradient_accumulation_steps=2,
                learning_rate=lr,
                logging_steps=20,
                save_strategy="no",
                bf16=True,
                seed=0,
                report_to=[],
                gradient_checkpointing=False,  # generation happens after training
                max_length=1024,
                dataset_text_field="text",
            ),
        )
        trainer.train()
        results["arms"][arm] = {
            "n_train": len(rows),
            "correct": _evaluate(trainer.model, tok, holdout, seed=100),
        }
        print(
            f"{arm}: n={len(rows)} pass={sum(results['arms'][arm]['correct']) / len(holdout):.4f}",
            flush=True,
        )
        del trainer, model
        torch.cuda.empty_cache()

    results["leaked_flags"] = [int(h["prompt"] in leaked) for h in holdout]
    Path("/out/results_raw.json").write_text(json.dumps(results))
    out_vol.commit()
    return results


@app.local_entrypoint()
def main(epochs: int = 3, lr: float = 1e-4):
    res = run_all.remote(epochs=epochs, lr=lr)
    OUT.mkdir(exist_ok=True)
    (OUT / "results_raw.json").write_text(json.dumps(res))
    print(f"saved {OUT / 'results_raw.json'}")
