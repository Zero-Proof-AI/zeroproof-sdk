"""Train and evaluate a voice register on Modal. Greedy, fixed prompts, no simulator.

The eval here has no in-rollout person and no tool loop, so the customer can
never be voiced by the model under test. Both arms see byte-identical prompts
and decode greedily, so the weights are the only difference.

Run: modal run train_modal.py --job out/voice/job.json --steps train,eval
Needs your Modal account (``modal setup``) and one H100; ``HF_TOKEN`` in the
environment if the base model is gated. Volumes and the app are created on
your account under ``whileai-*`` names.
"""

import json
import os
from pathlib import Path

import modal

APP = os.environ.get("WHILEAI_MODAL_APP", "whileai-voice")
app = modal.App(APP)

TRAIN_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    # Exact pins from the trainer that already builds. Unpinned trl/peft
    # resolved to versions that fail the image build.
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
        "huggingface_hub>=0.34",
        "requests>=2.25",
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
)
SERVE_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.0", "transformers==4.54.0", "huggingface_hub>=0.34", "requests>=2.25")
    .env({"HF_HOME": "/root/.cache/huggingface", "VLLM_USE_V1": "1"})
)
adapters = modal.Volume.from_name("whileai-voice-adapters", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)
VOL = "/vol"
# Your Hugging Face token, from your shell, only if the base model is gated.
secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})]


@app.function(
    image=TRAIN_IMAGE,
    gpu="H100",
    timeout=2 * 60 * 60,
    volumes={VOL: adapters, "/root/.cache/huggingface": hf_cache},
    secrets=secrets,
)
def train(job: dict, rows: list[dict]) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    name, base = job["name"], job["base"]
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    lora = LoraConfig(
        r=int(job.get("rank", 16)),
        lora_alpha=int(job.get("rank", 16)) * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    adapter_dir = f"{VOL}/{name}/adapter"
    cfg = SFTConfig(
        output_dir=f"{VOL}/{name}/ckpt",
        num_train_epochs=float(job.get("epochs", 2.0)),
        learning_rate=float(job.get("lr", 1e-4)),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        bf16=True,
        packing=False,
        max_length=int(job.get("max_len", 3072)),
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=17,
    )
    tr = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=Dataset.from_list(rows),
        processing_class=tok,
        peft_config=lora,
    )
    tr.train()
    tr.save_model(adapter_dir)
    tok.save_pretrained(adapter_dir)
    adapters.commit()
    return {
        "adapter": adapter_dir,
        "rows": len(rows),
        "loss": [h for h in tr.state.log_history if "loss" in h][-3:],
    }


@app.function(
    image=SERVE_IMAGE,
    gpu="H100",
    timeout=2 * 60 * 60,
    volumes={VOL: adapters, "/root/.cache/huggingface": hf_cache},
    secrets=secrets,
)
def evaluate(job: dict, prompts: list[dict]) -> dict:
    import subprocess
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor

    import requests

    name, base = job["name"], job["base"]
    adapter = f"{VOL}/{name}/adapter"
    # a token for the vLLM process inside this container only; nothing outside sees it
    key = "local-eval"
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        base,
        "--served-model-name",
        "base",
        "--port",
        "8000",
        "--api-key",
        key,
        "--max-model-len",
        "16384",
        "--gpu-memory-utilization",
        "0.85",
        "--disable-log-requests",
        "--enable-lora",
        "--max-lora-rank",
        str(int(job.get("rank", 16))),
        "--lora-modules",
        f"tuned={adapter}",
    ]
    proc = subprocess.Popen(cmd)
    url = "http://127.0.0.1:8000/v1"
    for _ in range(180):
        try:
            if requests.get(
                f"{url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=5
            ).ok:
                break
        except Exception:
            pass
        time.sleep(5)

    def ask(model_name: str, p: dict) -> dict:
        body = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["ask"]},
            ],
            # GREEDY, both arms. The cap must be high enough that the
            # BASE can finish: a base cut off mid-sentence is read
            # differently by a manner judge than a finished reply, which
            # is the same confound as an empty base wearing new clothes.
            "temperature": 0.0,
            "max_tokens": int(job.get("max_new_tokens", 2048)),
        }
        try:
            r = requests.post(
                f"{url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {key}"},
                timeout=180,
            )
            ch = r.json()["choices"][0]
            txt = ch["message"]["content"] or ""
            fin = ch.get("finish_reason")
        except Exception as exc:
            txt, fin = f"__ERROR__ {type(exc).__name__}", "error"
        # finish_reason is the ONLY reliable truncation signal and it cannot
        # be recovered after the fact. Record it per row, always.
        return {
            "ask": p["ask"],
            "required": p.get("required", []),
            "probe": p.get("probe", False),
            "reply": txt,
            "model": model_name,
            "finish_reason": fin,
            "truncated": fin == "length",
        }

    out = {}
    try:
        for m in ("base", "tuned"):
            with ThreadPoolExecutor(max_workers=12) as pool:
                # bind m: the lambda outlives the loop variable if pool.map
                # is ever made lazy, and binding it costs nothing.
                out[m] = list(pool.map(lambda p, m=m: ask(m, p), prompts))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except Exception:
            proc.kill()
    return out


@app.function(
    image=TRAIN_IMAGE,
    timeout=30 * 60,
    volumes={VOL: adapters, "/root/.cache/huggingface": hf_cache},
    secrets=secrets,
)
def publish(job: dict, card: str) -> dict:
    """Push the adapter and its card. Refuses to touch an existing repo."""
    from huggingface_hub import HfApi

    repo, name = job["hf_repo"], job["name"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    try:
        api.repo_info(repo, repo_type="model")
        return {"pushed": False, "reason": f"{repo} already exists; refusing to overwrite"}
    except Exception:
        pass
    api.create_repo(repo, repo_type="model", private=bool(job.get("private", True)), exist_ok=False)
    d = f"{VOL}/{name}/adapter"
    Path(d, "README.md").write_text(card)
    api.upload_folder(
        folder_path=d, repo_id=repo, repo_type="model", commit_message="Concise register adapter"
    )
    return {"pushed": True, "repo": repo, "url": f"https://huggingface.co/{repo}"}


@app.local_entrypoint()
def main(job: str, steps: str = "train,eval", out: str = ""):
    cfg = json.loads(Path(job).read_text())
    base_dir = Path(job).resolve().parent
    want = {s.strip() for s in steps.split(",") if s.strip()}
    if "train" in want:
        rows = [
            json.loads(line)
            for line in (base_dir / cfg["rows_file"]).read_text().splitlines()
            if line.strip()
        ]
        print(json.dumps(train.remote(cfg, rows), indent=1, default=str))
    if "publish" in want and cfg.get("hf_repo"):
        # the lane's own card wins; the shared one is only a fallback, so two
        # lanes publishing on the same day cannot ship each other's numbers
        lane = base_dir / "MODELCARD.md"
        card = (lane if lane.exists() else base_dir.parent / "MODELCARD.md").read_text()
        print(json.dumps(publish.remote(cfg, card), indent=1))
    if "eval" in want:
        prompts = json.loads((base_dir / cfg.get("holdout_file", "holdout.json")).read_text())
        res = evaluate.remote(cfg, prompts)
        d = Path(out or base_dir)
        d.mkdir(parents=True, exist_ok=True)
        for m, rows in res.items():
            (d / f"{cfg['name']}_{m}.jsonl").write_text(
                "\n".join(json.dumps(r, default=str) for r in rows)
            )
        print("wrote", d)
