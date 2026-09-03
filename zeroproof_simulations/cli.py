"""The loop as three terminal commands.

    zp simulate --agent specs/airline --budget 1000 --mode explore
    zp simulate --agent specs/airline --traces failures.jsonl --budget 1000
    zp grade --in pool.jsonl --out labeled.jsonl --agent specs/airline
    zp trim --in labeled.jsonl --out dataset.jsonl --target 500

``--agent`` takes a spec.json path, a directory holding one, or a bare
name resolved against ./specs/<name>/spec.json. Mode defaults to
explore; rl only when asked for by name.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path


def _load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _write_jsonl(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")


def _load_spec(agent: str) -> tuple[list, str, str]:
    p = Path(agent)
    if p.is_dir():
        p = p / "spec.json"
    if not p.exists():
        p = Path("specs") / agent / "spec.json"
    if not p.exists():
        raise SystemExit(f"no spec found for --agent {agent!r} "
                         f"(looked for {agent}/spec.json and specs/{agent}/spec.json)")
    spec = json.loads(p.read_text())
    name = spec.get("name") or p.parent.name
    return list(spec.get("tools") or []), str(spec.get("policy") or ""), name


def _stats(rows) -> dict:
    from .grading import behavior_signature
    prompts = {" ".join(str(r.get("prompt") or "").lower().split()) for r in rows}
    behaviors = {behavior_signature(r) for r in rows}
    return {"rows": len(rows), "distinct_prompts": len(prompts),
            "distinct_behaviors": len(behaviors)}


def _print_stats(label: str, rows) -> None:
    s = _stats(rows)
    print(f"{label}: rows={s['rows']} distinct_prompts={s['distinct_prompts']} "
          f"distinct_behaviors={s['distinct_behaviors']}", flush=True)


def cmd_simulate(a) -> int:
    from . import simulate
    tools, policy, name = _load_spec(a.agent)
    traces = _load_jsonl(a.traces) if a.traces else None
    kind = "tracefed" if traces else a.mode
    out = a.out or f"{name}_{kind}_{datetime.date.today():%Y%m%d}.rows.jsonl"
    data = simulate(tools=tools or None, system_prompt=policy,
                    budget=a.budget, mode=a.mode, traces=traces, output=out)
    rows = data.trajectories
    _print_stats(name, rows)
    print(f"saved={out} degraded={data.degraded} "
          f"stopped_because={data.stopped_because}", flush=True)
    data.save(out, meta=True)
    return 0


def cmd_grade(a) -> int:
    from . import grade_llm
    policy, tools = "", []
    if a.agent:
        tools, policy, _ = _load_spec(a.agent)
    prompt = Path(a.prompt).read_text() if a.prompt else None
    grade_llm(getattr(a, "in"), policy=policy, tools=tools, prompt=prompt,
              output=a.out)
    rows = _load_jsonl(a.out)
    graded = [r for r in rows if r.get("reward") in (0, 1)]
    passes = sum(r["reward"] for r in graded)
    fails = len(graded) - passes
    rate = 100 * passes / len(graded) if graded else 0.0
    _print_stats("graded", rows)
    print(f"graded={len(graded)} pass={passes} fail={fails} "
          f"pass_rate={rate:.1f}%", flush=True)
    if rate > 90:
        print("low signal", flush=True)
    return 0


def cmd_trim(a) -> int:
    from .optimize import select_for_rl, select_for_sft
    rows = _load_jsonl(getattr(a, "in"))
    pick = select_for_rl if a.mode == "rl" else select_for_sft
    selected, rep = pick(rows, target=a.target)
    _write_jsonl(a.out, selected)
    print(f"rows_in={len(rows)} rows_after_trim={len(selected)}", flush=True)
    keep = {k: v for k, v in rep.items()
            if isinstance(v, (int, float, str, bool))}
    print(json.dumps(keep, default=str), flush=True)
    _print_stats("trimmed", selected)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="zp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="generate conversations for an agent")
    s.add_argument("--agent", required=True)
    s.add_argument("--budget", type=int, default=1000)
    s.add_argument("--mode", default="explore",
                   choices=["explore", "sft", "rl", "adaptive"])
    s.add_argument("--traces", help="jsonl of prior traces to aim generation at")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_simulate)

    g = sub.add_parser("grade", help="0/1 grade a pool with an LLM judge")
    g.add_argument("--in", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--agent", help="spec so the judge sees policy and tools")
    g.add_argument("--prompt", help="file with your judge prompt")
    g.set_defaults(fn=cmd_grade)

    t = sub.add_parser("trim", help="select a training set from labeled rows")
    t.add_argument("--in", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--target", type=int, default=500)
    t.add_argument("--mode", default="sft", choices=["sft", "rl"])
    t.set_defaults(fn=cmd_trim)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
