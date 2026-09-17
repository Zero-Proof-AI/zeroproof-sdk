"""A rollout that never becomes a row must be COUNTED, not silently dropped."""

from __future__ import annotations

import warnings

import whileai.simulations as wai

TOOLS = [
    {
        "type": "function",
        "function": {"name": "noop", "parameters": {"type": "object", "properties": {}}},
    }
]
TASKS = [{"prompt": f"task {i} {'FAIL' if i % 2 else 'ok'}"} for i in range(6)]


def _agent(messages, tools=None, **_):
    # the callable contract hands the conversation in more than one shape;
    # the marker is in the prompt whichever shape arrives
    if "FAIL" in str(messages):
        raise RuntimeError("endpoint still booting")
    return {"steps": [], "final_text": "done"}


def test_lost_rollouts_are_in_the_report_and_warned():
    """Half the pinned tasks hit a raising agent. Before this, the run reported
    a clean pass rate over the survivors and nothing said the rest died: one
    eval lost its first 64 of 130 pinned tasks to a cold endpoint that way."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        data = wai.simulate(
            agent=_agent,
            tools=TOOLS,
            system_prompt="p",
            execute=lambda t, a: {"ok": True},
            tasks=TASKS,
            fault_rate=0.0,
            simulator=False,  # template writer: no hosted model, no key
        )
    rows = list(data.trajectories)
    rep = data.report()
    assert len(rows) == 3, len(rows)
    assert rep["rollouts_lost"] >= 3, rep
    assert rep["agent_errors"] >= 3, rep
    assert any("rollout(s) failed and were dropped" in str(w.message) for w in seen)


def test_a_clean_run_reports_zero_lost():
    data = wai.simulate(
        agent=lambda m, tools=None, **_: {"steps": [], "final_text": "done"},
        tools=TOOLS,
        system_prompt="p",
        execute=lambda t, a: {"ok": True},
        tasks=TASKS[:2],
        fault_rate=0.0,
        simulator=False,
    )
    rep = data.report()
    assert rep["rollouts_lost"] == 0 and rep["agent_errors"] == 0
    assert rep["stopped_because"] == "tasks_done"
