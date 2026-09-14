"""benchmark_report aggregates seeds into pass@1 mean/SD and unstable tasks."""

from __future__ import annotations

from zeroproof.simulations.score.benchmark import (
    benchmark_report,
    format_benchmark,
    run_benchmark,
)


def run(pass_map):
    """One seed's rows: {task: reward}."""
    return [{"scenario_id": t, "prompt": t, "reward": r} for t, r in pass_map.items()]


def test_mean_sd_over_seeds():
    seeds = [
        run({"a": 1, "b": 1, "c": 0, "d": 0}),  # 0.5
        run({"a": 1, "b": 0, "c": 0, "d": 0}),  # 0.25
        run({"a": 1, "b": 1, "c": 1, "d": 0}),  # 0.75
    ]
    r = benchmark_report(seeds, name="t")
    assert r["n_seeds"] == 3
    assert r["per_seed"] == [0.5, 0.25, 0.75]
    assert r["pass_at_1_mean"] == 0.5
    assert r["pass_at_1_sd"] > 0
    assert r["pass_at_1_min"] == 0.25 and r["pass_at_1_max"] == 0.75


def test_unstable_tasks():
    seeds = [
        run({"stable_pass": 1, "stable_fail": 0, "flip": 1}),
        run({"stable_pass": 1, "stable_fail": 0, "flip": 0}),
    ]
    r = benchmark_report(seeds)
    assert r["unstable_tasks"] == ["flip"]
    assert r["task_pass"]["flip"] == 0.5
    assert r["n_unstable"] == 1


def test_few_seeds_warns():
    r = benchmark_report([run({"a": 1, "b": 0})])
    assert any("seed" in w for w in r["warnings"])
    assert r["pass_at_1_sd"] == 0.0


def test_high_variance_warns():
    seeds = [run({"a": 1, "b": 1}), run({"a": 0, "b": 0}), run({"a": 1, "b": 0})]
    r = benchmark_report(seeds)
    assert any("SD" in w for w in r["warnings"])


def test_decontamination_flag():
    leaky = "please process a refund for the duplicate charge on order four four one two"
    seeds = [run({leaky: 1, "a clean unrelated task": 0})]
    train = [{"prompt": leaky + " right away for the customer"}]
    r = benchmark_report(seeds, decontaminate_against=train)
    assert "decontamination" in r
    assert r["decontamination"]["contaminated"] >= 1


def test_run_benchmark_drives_seeds():
    # rollout(seed) returns rows; judge grades exact-match on a fixed answer.
    def rollout(seed):
        # even seeds get it right, odd seeds wrong
        val = "4" if seed % 2 == 0 else "5"
        return [{"scenario_id": "q", "prompt": "2+2", "final_text": val, "answer": "4"}]

    def judge(row):
        return {"reward": int(row["final_text"].strip() == str(row["answer"]))}

    r = run_benchmark([], judge=judge, rollout=rollout, seeds=4, name="math")
    assert r["n_seeds"] == 4
    assert r["pass_at_1_mean"] == 0.5  # half the seeds pass
    assert r["unstable_tasks"] == ["q"]


def test_format_and_empty():
    assert "no graded seeds" in format_benchmark(benchmark_report([]))
    assert "pass@1" in format_benchmark(benchmark_report([run({"a": 1})]))
