"""The grader-driven fault loop: bad graded traces in, repair data out.

    python loop.py <rows.jsonl> --grader <module:callable> [--rules rules.json]
    python loop.py <rows.jsonl> --grader <module:callable> --live

The grader is the customer hook and the only source of truth. It is any
callable ``judge(row) -> {"score": 0|1, "reason": str, "rule": str}``
(anything ``normalize_judge_result`` accepts works; ``rule`` names which
of the customer's rules fired and becomes the marker id; without it the
normalized reason text is the marker key). The SAME grader scores the
incoming traces and every simulated row; nothing in this loop grades
itself.

One round:
  1. Grade the incoming rows with the customer grader (run_judge, so
     every scored row carries reward, reason, judge_status, lineage).
  2. Faults only: rows with reward < 1 group by the rule that fired.
     Each rule is a marker; markers are per agent, never universal.
  3. Registry (markers.json) accumulates evidence and history per
     marker across rounds.
  4. Grow, per marker: simulate fresh rows in the fault's situations
     (seeds = evidence prompts, sft topology plus seed amplification
     for diversity, scaffold = the rule as acquire plus its control),
     regrade ALL of them with the customer grader, keep only score-1
     rows, dedupe against everything the marker ever kept, and ship
     the top TARGET_POSITIVES as repair_<rule>.jsonl: the dataset that
     is already there when someone clicks the bad trace.
  5. A marker whose grow kept fewer than SATURATION_FLOOR new rows is
     saturated and stops drawing budget.

Guardrail kept from the experiments: trace-steering only at >= LANE_MIN
graded fails (steering on thin evidence measured worse than cold start);
below that, seeds plus scaffold anchor. Faults only; no positive-marker
work in this version.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "efficiency2"))
try:
    import envload  # noqa: E402
    envload.load()
except ImportError:
    pass

from zeroproof_simulations.score.judging import (  # noqa: E402
    normalize_judge_result, run_judge)

LANE_MIN = 20
TARGET_POSITIVES = 25
BUDGET_PER_MARKER = 48
TIME_BUDGET_S = 1800
SATURATION_FLOOR = 3
STOP = frozenset("a an the and or of to in on for with without is are was were not no this that it its "
                 "agent user by as at be been from".split())


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_grader(spec: str):
    mod_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit("--grader must be module:callable")
    return getattr(importlib.import_module(mod_name), attr)


def load_rules(path: str) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def rule_key(scored: dict) -> str:
    """The marker id for one scored fault: the grader's rule, else its reason."""
    meta = scored.get("judge_meta") or {}
    rule = str(meta.get("rule") or "").strip()
    if rule:
        return rule
    words = [w for w in re.findall(r"[a-z][a-z\-']+", str(scored.get("reason") or "").lower())
             if w not in STOP][:6]
    return "-".join(words[:4]) or "ungrouped-fault"


def grade(rows: list[dict], grader, name: str) -> list[dict]:
    """Customer-grade every row through the SDK judge contract."""
    scored = run_judge(rows, grader, judge_name=name, source="grade")
    return list(scored.rows if hasattr(scored, "rows") else scored)


def faults_by_marker(scored: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in scored:
        r = row.get("reward")
        if r is None or (isinstance(r, (int, float)) and r >= 1):
            continue
        out.setdefault(rule_key(row), []).append(row)
    return out


def registry_load(path: str) -> dict:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {"markers": {}, "rounds": 0}


def registry_update(reg: dict, faults: dict[str, list[dict]], rules: dict, source: str) -> None:
    reg["rounds"] += 1
    for marker, rows in faults.items():
        entry = reg["markers"].setdefault(marker, {
            "rule_text": (rules.get(marker) or {}).get("text") or rows[0].get("reason") or marker,
            "control": (rules.get(marker) or {}).get("control") or "",
            "seeds": list((rules.get(marker) or {}).get("seeds") or []),
            "status": "pooled", "graded_fails": 0, "kept_prompts": [], "history": []})
        if (rules.get(marker) or {}).get("seeds"):
            entry["seeds"] = list(rules[marker]["seeds"])
        entry["graded_fails"] += len(rows)
        entry["history"].append({"round": reg["rounds"], "source": source, "fails": len(rows)})
        if entry["status"] != "saturated":
            entry["status"] = "lane" if entry["graded_fails"] >= LANE_MIN else "pooled"


def grow(marker: str, entry: dict, evidence: list[dict], grader, grader_name: str,
         agent: dict, live: bool, exhibit=None) -> dict:
    """One marker's grow step. Returns the round record; writes repair_<marker>.jsonl."""
    scaffold = (f"Demonstrate the correct behavior: {entry['rule_text']}. "
                + (f"Also preserve: {entry['control']}. " if entry["control"] else "")
                + "Vary the situations naturally; do not repeat one script.")
    seeds = []
    seen_seed = set()
    # customer-supplied seed variants first (kept literal, on-topic diversity
    # without amplifier drift), then the evidence prompts themselves; a
    # customer without seed variants gets the generic evidence+amplify path
    for p in list(entry.get("seeds") or []) + [str(r.get("prompt") or "").strip() for r in evidence]:
        p = str(p or "").strip()
        if p and p.lower() not in seen_seed:
            seen_seed.add(p.lower())
            seeds.append(p)
    # seeds are ALWAYS literal: amplification drifts off-topic (measured
    # repeatedly), and an off-topic row that violates nothing is filler, not
    # repair data. Fewer on-topic rows beat many irrelevant ones; volume
    # comes from rounds as evidence accumulates.
    call = {
        "seeds": seeds,
        "situations": len(seeds), "mode": "sft",
        "budget": BUDGET_PER_MARKER, "time_budget": TIME_BUDGET_S, "scaffold": scaffold,
        "steered": entry["status"] == "lane",
    }
    record = {"marker": marker, "call": {**call, "seeds": f"<{len(seeds)} prompts>"},
              "generated": 0, "positives": 0, "kept_new": 0, "met": False}
    if not live:
        record["dry_run"] = True
        return record

    from zeroproof_simulations import simulate  # noqa: E402  spend only when live
    out_raw = os.path.join(HERE, f"raw_{marker}.jsonl")
    kwargs = dict(tools=agent["tools"], system_prompt=agent["policy"], mode=call["mode"],
                  # volume without drift: repeat rollouts of the on-topic asks;
                  # sampling variation plus the person simulator differentiates them
                  rollouts_per_request=max(1, min(12, BUDGET_PER_MARKER // max(1, len(seeds)))),
                  seeds=call["seeds"], situations=call["situations"], budget=call["budget"],
                  time_budget=call["time_budget"], scaffold=call["scaffold"], output=out_raw,
                  # scene writer off: seeds define the situations; this is the
                  # config that ran fast and stable in the efficiency work.
                  # faults near zero: repair rows demonstrate correct behavior
                  # in a working world; fault recovery is its own marker kind
                  advanced={"concurrency": 16, "simulator": False, "fault_rate": 0.1,
                            "model_version": agent.get("model", "")})
    if call["steered"]:
        kwargs.update(traces=evidence, strategy="trace", steering_weight=0.5)
    if agent.get("world") == "mini":
        from mini_world import MiniWorld  # noqa: E402
        kwargs["execute"] = MiniWorld()
    elif agent.get("world") == "traces":
        # the environment reconstructed from the customer's own traces:
        # observed answers replayed, writes overlaid, nothing improvised
        from trace_world import TraceWorld  # noqa: E402
        kwargs["execute"] = TraceWorld(evidence)
    t0 = time.time()
    data = simulate(**kwargs)
    rows = [dict(r) for r in data.trajectories]
    scored = grade(rows, grader, grader_name)
    passing = [r for r in scored if r.get("reward") == 1]
    # a repair row must SHOW the correct behavior, not merely lack the fault;
    # customers without an exhibit predicate for a rule skip this filter
    positives = [r for r in passing if exhibit is None or exhibit(r)]
    def keep_key(r: dict) -> str:
        tools_sig = ">".join(str(s.get("tool")) for s in (r.get("steps") or [])
                             if isinstance(s, dict) and s.get("tool"))
        return str(r.get("prompt") or "").strip().lower() + "|" + tools_sig

    seen = {p.lower() for p in entry["kept_prompts"]}
    fresh = []
    for r in positives:
        k = keep_key(r)
        if k.split("|")[0] and k not in seen:
            seen.add(k)
            fresh.append(r)
    kept = fresh[:TARGET_POSITIVES]
    entry["kept_prompts"].extend(keep_key(r) for r in kept)
    out = os.path.join(HERE, f"repair_{marker}.jsonl")
    with open(out, "a", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, default=str) + "\n")
    with open(out_raw, "w", encoding="utf-8") as fh:
        for r in scored:
            fh.write(json.dumps(r, default=str) + "\n")
    record.update(generated=len(rows), grader_pass=len(passing),
                  positives=len(positives), kept_new=len(kept),
                  met=len(kept) >= min(TARGET_POSITIVES, 8), seconds=round(time.time() - t0, 1),
                  dataset=os.path.basename(out))
    if record["kept_new"] < SATURATION_FLOOR and entry["history"] and len(entry["history"]) > 1:
        entry["status"] = "saturated"
    return record


def write_report(reg: dict, faults: dict, records: list[dict], path: str) -> str:
    lines = [f"# Marker loop round {reg['rounds']}", "",
             "| marker | fails in | status | generated | positive | kept new | met |",
             "|---|---|---|---|---|---|---|"]
    for rec in records:
        entry = reg["markers"][rec["marker"]]
        lines.append(f"| {rec['marker']} | {len(faults.get(rec['marker'], []))} | {entry['status']} | "
                     f"{rec['generated']} | {rec['positives']} | {rec['kept_new']} | "
                     f"{'yes' if rec.get('met') else ('dry' if rec.get('dry_run') else 'BELOW TARGET')} |")
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_path")
    ap.add_argument("--grader", required=True, help="module:callable, the customer grader")
    ap.add_argument("--rules", default="", help="JSON: rule id -> {text, control}")
    ap.add_argument("--agent", default="", help="JSON: {tools, policy, model}; required for --live")
    ap.add_argument("--registry", default=os.path.join(HERE, "markers.json"))
    ap.add_argument("--report", default=os.path.join(HERE, "MARKERS_REPORT.md"))
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()

    grader = load_grader(args.grader)
    exhibits = getattr(importlib.import_module(args.grader.partition(":")[0]), "EXHIBITS", {})
    sample = normalize_judge_result(grader(load_rows(args.rows_path)[0]))
    if sample["judge_status"] not in ("ok",):
        raise SystemExit(f"grader breaks the judge contract on row 0: {sample['judge_status']}")

    rows = load_rows(args.rows_path)
    scored = grade(rows, grader, args.grader)
    faults = faults_by_marker(scored)
    print(f"{len(rows)} rows in; faults by marker: " +
          json.dumps({k: len(v) for k, v in faults.items()}))

    reg = registry_load(args.registry)
    rules = load_rules(args.rules)
    registry_update(reg, faults, rules, source=os.path.basename(args.rows_path))

    agent = json.loads(open(args.agent).read()) if args.agent else {}
    if args.live and not agent:
        raise SystemExit("--live needs --agent (tools, policy)")
    records = []
    for marker, evidence in sorted(faults.items(), key=lambda kv: -len(kv[1])):
        entry = reg["markers"][marker]
        if entry["status"] == "saturated":
            continue
        records.append(grow(marker, entry, evidence, grader, args.grader, agent, args.live,
                            exhibit=exhibits.get(marker)))

    with open(args.registry, "w", encoding="utf-8") as fh:
        json.dump(reg, fh, indent=1, sort_keys=True)
    print(write_report(reg, faults, records, args.report))
    if not args.live:
        print("dry run: nothing sent. --live generates and regrades with the customer grader.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
