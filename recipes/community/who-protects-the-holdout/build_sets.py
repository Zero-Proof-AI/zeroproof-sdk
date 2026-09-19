"""Build a contaminated GSM8K training pool, then clean it two ways with the SDK."""

import json
import random
import re

from datasets import load_dataset

import whileai as wai

SEED = 0
N_HOLDOUT, N_CLEAN, N_LEAK = 200, 800, 60
random.seed(SEED)

ds_tr = load_dataset("openai/gsm8k", "main", split="train")
ds_te = load_dataset("openai/gsm8k", "main", split="test")

holdout = [
    {
        "prompt": r["question"],
        "answer": r["answer"].split("####")[-1].strip(),
        "reference": r["answer"],
    }
    for r in ds_te.select(range(N_HOLDOUT))
]
clean = [
    {
        "prompt": r["question"],
        "final_text": r["answer"],
        "answer": r["answer"].split("####")[-1].strip(),
        "origin": "clean",
    }
    for r in ds_tr.select(range(N_CLEAN))
]

# --- paraphrase: surface rewrite, numbers and answer untouched -------------
SWAPS = [
    (r"\bHow many\b", "What is the total number of"),
    (r"\bHow much\b", "What is the total amount"),
    (r"\bIf\b", "Suppose that"),
    (r"\beach\b", "every"),
    (r"\bthen\b", "after that"),
    (r"\bhas\b", "owns"),
    (r"\bbuys\b", "purchases"),
    (r"\bgets\b", "receives"),
    (r"\bmakes\b", "produces"),
    (r"\bwants to\b", "would like to"),
    (r"\bleft\b", "remaining"),
    (r"\btotal\b", "combined"),
    (r"\bsells\b", "offloads"),
    (r"\bcosts\b", "is priced at"),
]
NAMES = {
    "Natalia": "Priya",
    "Weng": "Mira",
    "Betty": "Rosa",
    "Julie": "Ingrid",
    "James": "Tobias",
    "Mark": "Devon",
    "Ken": "Emeka",
    "Alexis": "Noor",
    "Tina": "Halle",
    "Kylar": "Sven",
}


def paraphrase(q: str) -> str:
    out = q
    for a, b in NAMES.items():
        out = re.sub(rf"\b{a}\b", b, out)
    for pat, rep in SWAPS:
        out = re.sub(pat, rep, out)
    sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", out) if s.strip()]
    if len(sents) > 1 and sents[-1].endswith("?"):  # question to the front
        out = sents[-1] + " Use the following: " + " ".join(sents[:-1])
    return "Consider this problem. " + out


leak_src = random.sample(holdout, N_LEAK)
leak = [
    {
        "prompt": paraphrase(h["prompt"]),
        "final_text": h["reference"],
        "answer": h["answer"],
        "origin": "leak",
        "leaked_from": h["prompt"],
    }
    for h in leak_src
]

pool = clean + leak
random.shuffle(pool)

# --- clean it two ways ------------------------------------------------------
from sentence_transformers import SentenceTransformer

m = SentenceTransformer("BAAI/bge-small-en-v1.5")


def embed(texts):
    return m.encode(list(texts), normalize_embeddings=True).tolist()


arms = {}
for name, kw in [("default", {}), ("semantic", {"embedder": embed, "similarity": 0.85})]:
    rows, rep = wai.decontaminate(pool, against=holdout, **kw)
    caught = N_LEAK - sum(r["origin"] == "leak" for r in rows)
    fp = sum(r["origin"] == "clean" for r in pool) - sum(r["origin"] == "clean" for r in rows)
    arms[name] = {
        "n_rows": len(rows),
        "n_contaminated": rep["n_contaminated"],
        "leaks_caught": caught,
        "leak_recall": round(caught / N_LEAK, 4),
        "clean_rows_dropped": fp,
        "by_rule": {k: rep.get(k, 0) for k in ("n_same_task", "n_exact", "n_near", "n_semantic")},
    }
    arms[name]["surviving_leak_prompts"] = sorted(
        r["leaked_from"] for r in rows if r["origin"] == "leak"
    )
    json.dump(rows, open(f"train_{name}.json", "w"))
    print(
        f"[{name}] kept={len(rows)} leaks_caught={caught}/{N_LEAK} "
        f"recall={caught / N_LEAK:.3f} clean_dropped={fp} rules={arms[name]['by_rule']}"
    )

json.dump(
    {
        "holdout": holdout,
        "arms": arms,
        "n_leak": N_LEAK,
        "seed": SEED,
        "leaked_prompts": sorted(h["prompt"] for h in leak_src),
    },
    open("sets_meta.json", "w"),
    indent=2,
)
print("\nheld-out rows carry scenario_id?", any("scenario_id" in h for h in holdout))
