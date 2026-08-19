"""Report whether an RL dataset carries gradient, before you spend GPU on it.

Aggregate mean reward tells you almost nothing. These four numbers decide
whether a GRPO run can learn anything:

  group uniformity   every prompt has the same k, or advantages are not
                     comparable across groups
  live groups        fraction of prompts whose k rollouts do NOT all score
                     the same. A group with zero variance has zero advantage
                     and contributes no gradient
  within-group std   how much signal the live groups carry
  effort correlation corr(tool calls, reward). If negative, the reward pays
                     the policy to do less, and GRPO will find that out
"""
import argparse
import collections
import json
import statistics as st


def tool_calls(row: dict) -> int:
    return sum(1 for s in row.get("steps") or []
               if isinstance(s, dict) and s.get("tool"))


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else 0.0


def report(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get("reward") is not None]
    groups = collections.defaultdict(list)
    for r in scored:
        groups[r["prompt"]].append(r["reward"])

    sizes = collections.Counter(len(v) for v in groups.values())
    live = [k for k, v in groups.items() if st.pstdev(v) > 0]
    dead = [k for k, v in groups.items() if st.pstdev(v) == 0]
    effort = pearson([tool_calls(r) for r in scored], [r["reward"] for r in scored])

    return {
        "rollouts": len(scored),
        "prompts": len(groups),
        "group_sizes": dict(sizes),
        "uniform_groups": len(sizes) == 1,
        "live_groups": len(live),
        "live_fraction": round(len(live) / len(groups), 3) if groups else 0.0,
        "dead_all_zero": sum(1 for k in dead if groups[k][0] == 0.0),
        "dead_all_max": sum(1 for k in dead if groups[k][0] == max(
            (max(v) for v in groups.values()), default=1.0)),
        "mean_within_group_std": round(
            st.mean(st.pstdev(groups[k]) for k in live), 3) if live else 0.0,
        "mean_reward": round(st.mean(r["reward"] for r in scored), 3),
        "effort_correlation": round(effort, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--min-live", type=float, default=0.5,
                    help="fail if the live-group fraction is below this")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.path) if line.strip()]
    stats = report(rows)
    width = max(len(k) for k in stats)
    for key, value in stats.items():
        print(f"{key:<{width}}  {value}")

    ok = True
    if not stats["uniform_groups"]:
        print("\nFAIL  group sizes are not uniform; advantages are not comparable")
        ok = False
    if stats["live_fraction"] < args.min_live:
        print(f"\nFAIL  only {stats['live_fraction']:.0%} of groups carry gradient")
        ok = False
    if stats["effort_correlation"] < 0:
        print(f"\nWARN  reward is anti-correlated with tool use "
              f"({stats['effort_correlation']:+.3f}). The cheapest policy under "
              f"this reward is to call no tools. Add an outcome term to the "
              f"rubric before training; see README.")
    if ok:
        print("\nPASS  groups are uniform and carry gradient")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
