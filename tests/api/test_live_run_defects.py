"""The 2026-09-18 live test of 0.75 against the hosted writer, offline.

Five defects, one test each. The hosted writer is a callable that never
runs dry, which is what made the engine spin: every wave returned fresh
situations, so a wave was always in flight and the stop that needs
"nothing in flight" never fired.
"""

from __future__ import annotations

import time

import whileai.simulations as wai
from tests.helpers import POLICY, TOOLS, scripted_agent, simulate_offline


def endless_writer(_dataset=None, index=0):
    """A stand-in for the hosted writer: fresh situations on every call,
    after a wave's worth of latency. The sleep is longer than the
    engine's writer wait and differs per wave, so waves finish one at a
    time the way hosted waves do; an instant writer never left a wave in
    flight and never showed the spin."""
    time.sleep(0.6 + 0.05 * (index % 7))
    return [f"where is my refund for order {index}-{i}" for i in range(4)]


def _by_run(data):
    out: dict[int, list[dict]] = {}
    for row in data.trajectories:
        out.setdefault(int(row["lineage"]["eval_run"]), []).append(row)
    return out


def test_hosted_writer_stops_when_every_situation_has_its_rollouts():
    calls: list[int] = []

    def counted_writer(_dataset=None, index=0):
        calls.append(index)
        return endless_writer(_dataset, index)

    # 6 situations x 1 phrasing x 2 repeats = 12 rows; the budget allows 48.
    # The clock is a safety net for the old code, which spun until it.
    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        situations=6,
        repeats=2,
        phrasings=1,
        budget=48,
        seed=0,
        concurrency=2,
        simulator=counted_writer,
        grade=False,
        time_budget=20,
        advanced={"mutate_failures": False, "scenario_concurrency": 2},
    )
    assert data.stopped_because == "situations_exhausted"
    assert len(data.trajectories) == 12
    assert len(calls) <= 12, f"{len(calls)} writer waves for 6 situations"
    assert data.report()["budget_per_run"] == 48
    assert data.report()["writer_model"] == data.report()["simulator"] == "counted_writer"


def test_runs_with_a_budget_above_capacity_stop_each_run_on_its_own():
    t0 = time.monotonic()
    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        situations=6,
        repeats=2,
        phrasings=1,
        budget=48,
        runs=2,
        seed=0,
        concurrency=2,
        simulator=endless_writer,
        grade=False,
        time_budget=20,
        advanced={"mutate_failures": False, "scenario_concurrency": 2},
    )
    assert time.monotonic() - t0 < 15.0, "a run past capacity must not wait for the clock"
    per_run = data.search["eval_runs"]["per_run"]
    assert [r["stopped_because"] for r in per_run] == ["situations_exhausted", "tasks_done"]
    assert [r["rows"] for r in per_run] == [12, 12]
    assert data.report()["budget_per_run"] == 48 and data.report()["runs"] == 2


def test_replayed_runs_keep_the_writer_of_the_run_they_replay():
    data = simulate_offline(mode="sft", repeats=2, concurrency=1, seed=3, budget=24, runs=2)
    runs = _by_run(data)
    assert {r["writer_model"] for r in runs[0]} == {"template"}
    # run 1 replays run 0's tasks: same writer on the row, replay in lineage
    assert {r["writer_model"] for r in runs[1]} == {"template"}
    assert all(r["lineage"]["replayed_from_run"] == 0 for r in runs[1])
    assert all(r["lineage"].get("replayed") is True for r in runs[1])
    assert all("replayed_from_run" not in r["lineage"] for r in runs[0])
    report = wai.delta_report(runs[0], runs[1], n_boot=100)
    assert report["not_comparable"] == []
    assert not any("NOT COMPARABLE" in w for w in report["warnings"])


def test_a_tasks_replay_of_a_model_written_run_names_that_writer():
    base = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        situations=4,
        repeats=1,
        budget=8,
        seed=0,
        concurrency=1,
        simulator=endless_writer,
        grade=False,
        time_budget=20,
        advanced={"mutate_failures": False, "scenario_concurrency": 1},
    )
    assert {r["writer_model"] for r in base.trajectories} == {"endless_writer"}
    rerun = simulate_offline(tasks=base, concurrency=1, seed=1)
    assert {r["writer_model"] for r in rerun.trajectories} == {"endless_writer"}
    assert all(r["lineage"]["replayed"] is True for r in rerun.trajectories)
    report = wai.delta_report(base.trajectories, rerun.trajectories, n_boot=100)
    assert "writer_model" not in report["not_comparable"]


def test_rows_replayed_by_an_older_release_still_compare_as_one_writer():
    # 0.75 stamped replays "pinned"; a saved run from then must still pair
    # with its base instead of failing on a stamp that names no model.
    base = [
        {"prompt": f"t{i}", "reward": float(j % 2), "writer_model": "Qwen/Qwen3-4B"}
        for i in range(12)
        for j in range(4)
    ]
    replay = [dict(row, writer_model="pinned") for row in base]
    report = wai.delta_report(base, replay, n_boot=100)
    assert report["not_comparable"] == []
    assert report["config"]["after"]["writer_model"] is None


def test_abandoned_writer_waves_come_with_a_warning_that_names_the_knob():
    def slow_writer(_dataset=None, index=0):
        time.sleep(6.0)
        return endless_writer(_dataset, index)

    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        budget=40,
        seed=0,
        concurrency=4,
        simulator=slow_writer,
        grade=False,
        time_budget=0.4,
        advanced={"mutate_failures": False, "scenario_concurrency": 2, "stop_grace": 0.2},
    )
    assert "writer_waves_abandoned" in data.degraded
    n = data.search["abandoned_writer_waves"]
    lines = [w for w in data.warnings if "writer wave" in w]
    assert len(lines) == 1, data.warnings
    line = lines[0]
    assert line.startswith(f"{n} writer wave")
    assert "(time_budget)" in line and "0.2s stop grace" in line
    assert "advanced={'stop_grace': <seconds>}" in line
    assert "advanced={'scenario_concurrency': <n>}" in line


def test_tier_mix_counts_every_run_under_runs():
    data = simulate_offline(mode="sft", repeats=2, concurrency=1, seed=3, budget=24, runs=2)
    mix = data.search["tier_mix"]
    assert mix["rows"] == len(data.trajectories) == sum(mix["counts"].values())
    assert [p["eval_run"] for p in mix["per_run"]] == [0, 1]
    assert sum(p["rows"] for p in mix["per_run"]) == mix["rows"]
    assert all(p["hard_share_requested"] == mix["hard_share_requested"] for p in mix["per_run"])
