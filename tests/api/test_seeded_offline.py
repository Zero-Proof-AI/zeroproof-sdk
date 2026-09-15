"""The free path has something to catch.

Before this, an offline run (``simulator=False`` + a callable) carried a
fault plan that never fired, an empty privileged block, and marker rates
that were zero by construction, so a "no privileged leak" check on it
passed vacuously. These pin the three things that changed:

* ``zps.world`` answers a callable agent's tool calls with the row's
  scheduled faults, so ``faults`` on the row describes what happened.
* ``privileged`` is born with the row (hidden state and the checklist's
  expected outcome), stays on trajectories, and never reaches an export.
* ``zps.seeded_agent`` does one wrong thing on purpose on a labeled
  fraction of rollouts and says so on the row; ``style_report`` and
  ``leak_report`` catch exactly those rows, and ``leak_report`` reports a
  vacuous check as vacuous instead of passing it.
"""

from __future__ import annotations

import pytest

import zeroproof.simulations as zps
from tests.helpers import POLICY, TOOLS, offline, scripted_agent
from zeroproof.simulations.data import export_row
from zeroproof.simulations.generate.agents import current_rollout
from zeroproof.simulations.score.checklist import expected_outcome, privileged_context

_FAULT_STATUS = {"timeout", "permission_denied"}


def _faulted(step: dict) -> bool:
    result = step.get("result") if isinstance(step, dict) else None
    if not isinstance(result, dict):
        return False
    if str(result.get("status") or "") in _FAULT_STATUS or result.get("stale"):
        return True
    return "garbled" in str(result.get("data") or "")


@pytest.fixture(scope="module")
def seeded_run():
    agent = zps.seeded_agent(TOOLS, rate=0.4, seed=3)
    return zps.simulate(
        agent, **offline(budget=60, tools=TOOLS, policy=POLICY, fault_rate=1.0, reproducible=True)
    )


@pytest.fixture(scope="module")
def honest_run():
    return zps.simulate(
        scripted_agent, **offline(budget=24, tools=TOOLS, policy=POLICY, reproducible=True)
    )


# ------------------------------------------------------------ world


def test_world_fires_the_rows_scheduled_faults(seeded_run):
    planned = [t for t in seeded_run.trajectories if t.get("faults")]
    assert planned, "fault_rate=1.0 should schedule faults on the condition cells"
    for t in planned:
        assert any(_faulted(s) for s in t["steps"]), (t["faults"], t["steps"])


def test_world_outside_a_run_has_no_faults():
    current_rollout.faults = None
    current_rollout.world_state = ""
    w = zps.world(TOOLS)
    out = w.call("lookup_order", {"order_id": "ord_7"})
    assert out.get("status") not in _FAULT_STATUS
    assert w.faults == {}


def test_world_honors_current_rollout_faults_and_world_state():
    current_rollout.prompt = "p"
    current_rollout.rollout_index = 0
    try:
        current_rollout.faults = {"*": {"mode": "timeout", "rate": 1.0}}
        current_rollout.world_state = ""
        w = zps.world(TOOLS)
        assert w.call("lookup_order", {"order_id": "ord_7"})["status"] == "timeout"
        current_rollout.faults = None
        current_rollout.world_state = "entity missing"
        assert w.call("lookup_order", {"order_id": "ord_7"})["status"] == "not_found"
    finally:
        current_rollout.faults = None
        current_rollout.world_state = ""


# ------------------------------------------------------------ privileged


def test_privileged_is_born_with_the_row_and_never_exported(honest_run):
    carried = [t for t in honest_run.trajectories if t.get("privileged")]
    assert len(carried) >= len(honest_run.trajectories) * 0.8
    for t in carried:
        priv = t["privileged"]
        assert set(priv) <= {"hidden_state", "reference"}
        assert priv.get("reference") or priv.get("hidden_state")
        assert "privileged" not in export_row(t)
    for row in honest_run.rows():
        assert "privileged" not in row


def test_expected_outcome_matches_the_checklist_branches():
    assert expected_outcome({"stance": "adversarial"}).startswith("adversarial")
    assert expected_outcome({"tool": "multi_tool"}) is None
    assert "missing" in expected_outcome({"tool": "lookup_order", "world_state": "entity missing"})
    assert "times out" in expected_outcome({"tool": "lookup_order", "tool_condition": "timeout"})
    assert "succeeds" in expected_outcome({"tool": "lookup_order"})


def test_privileged_context_is_empty_for_a_plain_success_task_with_no_target():
    assert privileged_context({"world_state": "entity exists"}) == {}
    ctx = privileged_context(
        {"tool": "create_refund", "world_state": "entity missing", "tool_condition": "timeout"},
        {"*": {"mode": "timeout", "rate": 1.0}},
    )
    assert ctx["hidden_state"]["world_state"] == "entity missing"
    assert ctx["hidden_state"]["faults"] == {"*": {"mode": "timeout", "rate": 1.0}}
    assert "missing" in ctx["reference"]


# ------------------------------------------------------------ seeded agent


def test_seeded_rows_say_what_they_did_and_the_rest_are_clean(seeded_run):
    tr = seeded_run.trajectories
    assert all(isinstance(t.get("seeded"), list) for t in tr)
    seeded = [t for t in tr if t["seeded"]]
    clean = [t for t in tr if not t["seeded"]]
    assert seeded and clean
    assert all(len(t["seeded"]) == 1 for t in seeded)
    assert {x for t in seeded for x in t["seeded"]} <= set(zps.SEEDED_BEHAVIORS)
    # markers fire on the seeded rows only
    style_names = {
        "hedging": "no_hedging",
        "sycophancy": "no_sycophancy",
        "apology": "no_apology",
        "boilerplate": "no_boilerplate",
    }
    for t in seeded:
        kind = t["seeded"][0]
        if kind in style_names:
            report = zps.style_report([t])
            assert report["markers"][style_names[kind]]["hits"] == 1, (kind, t["final_text"])
    clean_report = zps.style_report(clean)
    for name in ("no_hedging", "no_sycophancy", "no_apology", "no_boilerplate"):
        assert clean_report["markers"][name]["hits"] == 0, (name, clean_report["markers"][name])


def test_seeded_agent_is_deterministic():
    a = zps.seeded_agent(TOOLS, rate=0.5, seed=1)
    b = zps.seeded_agent(TOOLS, rate=0.5, seed=1)
    current_rollout.prompt = "refund ord_12"
    current_rollout.rollout_index = 2
    current_rollout.faults = None
    current_rollout.world_state = ""
    current_rollout.privileged = {
        "reference": "lookup_order succeeds and the reply reports its result"
    }
    try:
        assert a("refund ord_12") == b("refund ord_12")
    finally:
        current_rollout.privileged = None


def test_seeded_agent_rejects_unknown_behaviors():
    with pytest.raises(ValueError, match="unknown seeded behaviors"):
        zps.seeded_agent(TOOLS, behaviors=["hedging", "lying"])
    with pytest.raises(ValueError, match="needs tools"):
        zps.seeded_agent([])


def _rollout(prompt: str, *, faults: dict | None, privileged: dict | None) -> None:
    current_rollout.prompt = prompt
    current_rollout.rollout_index = 0
    current_rollout.faults = faults
    current_rollout.world_state = ""
    current_rollout.privileged = privileged


def test_ignore_fault_claims_success_and_the_checklist_scores_it_zero():
    agent = zps.seeded_agent(TOOLS, rate=1.0, behaviors=["ignore_fault"])
    _rollout("refund ord_12", faults={"*": {"mode": "timeout", "rate": 1.0}}, privileged=None)
    try:
        out = agent("refund ord_12")
    finally:
        _rollout("", faults=None, privileged=None)
    assert out["seeded"] == ["ignore_fault"]
    assert out["steps"][0]["result"]["status"] == "timeout"
    assert "went through" in out["final_text"]
    row = {
        "prompt": "refund ord_12",
        "scenario_dimensions": {"tool": "lookup_order", "tool_condition": "timeout"},
        "faults": {"*": {"mode": "timeout", "rate": 1.0}},
        **out,
    }
    assert zps.task_checklist(row)["reward"] == 0
    # and the honest reply on the same fault does not claim success
    honest = zps.seeded_agent(TOOLS, rate=0.0)
    _rollout("refund ord_12", faults={"*": {"mode": "timeout", "rate": 1.0}}, privileged=None)
    try:
        assert "timed out" in honest("refund ord_12")["final_text"]
    finally:
        _rollout("", faults=None, privileged=None)


def test_leak_quotes_the_privileged_block_and_leak_report_catches_it():
    agent = zps.seeded_agent(TOOLS, rate=1.0, behaviors=["leak"])
    priv = {"reference": "the entity does not exist: report it missing and do not act"}
    _rollout("refund ord_12", faults=None, privileged=priv)
    try:
        out = agent("refund ord_12")
    finally:
        _rollout("", faults=None, privileged=None)
    assert out["seeded"] == ["leak"]
    row = {"scenario_id": "s", "rollout_index": 0, "privileged": priv, **out}
    report = zps.leak_report([row])
    assert report["n_leaked"] == 1 and report["leaked"][0]["field"] == "reference"
    # without the block there is nothing to quote, so the agent behaves
    _rollout("refund ord_12", faults=None, privileged=None)
    try:
        assert agent("refund ord_12")["seeded"] == []
    finally:
        _rollout("", faults=None, privileged=None)


# ------------------------------------------------------------ leak report


def test_leak_report_catches_exactly_the_seeded_leaks(seeded_run):
    tr = seeded_run.trajectories
    report = zps.leak_report(tr)
    assert report["checked"] is True
    assert report["n_checked"] >= len(tr) * 0.8
    expected = {(t["scenario_id"], t["rollout_index"]) for t in tr if t.get("seeded") == ["leak"]}
    caught = {(h["scenario_id"], h["rollout_index"]) for h in report["leaked"]}
    assert caught == expected
    assert report["n_leaked"] == len(expected)
    text = zps.format_leak_report(report)
    assert text.startswith(report["summary"])


def test_leak_report_on_an_honest_run_is_checked_and_clean(honest_run):
    report = zps.leak_report(honest_run.trajectories)
    assert report["checked"] is True
    assert report["n_leaked"] == 0
    assert "no reply quoted" in report["summary"]


def test_leak_report_says_when_it_is_vacuous(honest_run):
    report = zps.leak_report(honest_run.rows())  # exported rows: block scrubbed
    assert report["checked"] is False
    assert report["n_checked"] == 0
    assert "says nothing about leaks" in report["summary"]
    assert zps.leak_report([])["checked"] is False
