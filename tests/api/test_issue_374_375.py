"""#374 and #375: the rough edges the skills worked around.

* a verifier defined in the caller's module can be passed as an object to
  ``export_environment``;
* two offline arms (seed, template, replay) are comparable;
* ``runs=`` works with ``seeds=``;
* an agent that never answers stops with a warning that names the fix.
"""

from __future__ import annotations

import logging
from typing import Any

import whileai.simulations as wai
from tests.helpers import POLICY, TOOLS, scripted_agent
from whileai.simulations.environment import _ref_of, resolve_ref
from whileai.simulations.verify import All, Regex


@wai.verifier
def refund_rule(candidate: str, reference: Any, row: dict) -> float:
    return 1.0 if "refund" in candidate.lower() else 0.0


BOTH = All([refund_rule, Regex(r"\d+")])


def test_374_verifier_bound_in_caller_module_is_importable():
    ref = _ref_of(refund_rule)
    assert ref == f"{__name__}:refund_rule"
    assert resolve_ref(ref) is refund_rule
    ref2 = _ref_of(BOTH)
    assert ref2 == f"{__name__}:BOTH"
    assert resolve_ref(ref2) is BOTH


def test_375a_offline_writers_compare(tmp_path):
    seeds = [f"Refund order A100{i}, it arrived broken" for i in range(4)]
    before = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        system_prompt=POLICY,
        seeds=seeds,
        simulator=False,
        repeats=2,
        budget=16,
        advanced={"model_version": "v0"},
    )
    after = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        system_prompt=POLICY,
        tasks=before,
        simulator=False,
        repeats=2,
        budget=16,
        advanced={"model_version": "v1"},
    )

    def judge(row: dict) -> dict:
        got = str(row.get("final_text", "")).lower()
        return {"reward": 1.0 if "refund" in got else 0.0}

    a = wai.evaluate(before.rows, judge)
    b = wai.evaluate(after.rows, judge)
    # force the two arms to report different offline writers
    for row in a.rows:
        row["writer_model"] = "seed"
    for row in b.rows:
        row["writer_model"] = "pinned"
    report = wai.delta_report(a.rows, b.rows)
    assert "writer_model" not in report["not_comparable"]


def test_375b_runs_with_seeds():
    seeds = [f"Refund order A100{i}" for i in range(3)]
    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        system_prompt=POLICY,
        seeds=seeds,
        simulator=False,
        repeats=2,
        budget=12,
        runs=2,
    )
    assert data.search["eval_runs"]["runs"] == 2
    replayed = [r for r in data.rows if (r.get("lineage") or {}).get("eval_run") == 1]
    assert replayed, "second run produced no rows"
    assert {r["prompt"] for r in replayed} <= {r["prompt"] for r in data.rows}


def test_375c_empty_agent_names_the_fix(caplog):
    with caplog.at_level(logging.WARNING, logger="whileai.simulations"):
        data = wai.simulate(
            lambda message: {"steps": [], "final_text": ""},
            tools=TOOLS,
            system_prompt=POLICY,
            seeds=["Refund order A1001"],
            simulator=False,
            repeats=2,
            budget=8,
            time_budget=20,
        )
    assert data.rows == []
    assert data.stopped_because == "empty_replies"
    assert any("empty reply" in w and "final_text" in w for w in data.warnings)
