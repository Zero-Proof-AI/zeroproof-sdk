"""``select`` never keeps a row ``export`` will refuse.

A reply that quotes its own ``privileged`` block (the reference answer,
the principle, the hidden world state) did not earn its reward, and
``export_dataset(validate=True)`` refuses it because the scrub removes
the key and not the reply. The docs' landing flow,
``scored.select(mode="rl").export("train.jsonl")``, raised
``privileged_leak`` on the released package for exactly that reason. The
gate now runs first in both modes and the printed report counts it.
"""

from __future__ import annotations

import whileai as wai
from whileai.simulations import leak_report
from whileai.simulations.score.optimize import (
    drop_privileged_leaks,
    select_for_rl,
    select_for_sft,
)
from whileai.simulations.score.privileged import row_leak

REFERENCE = "the expected outcome here is a full refund to the original card"


def _row(prompt: str, index: int, *, reward: float, reply: str, privileged: bool = True) -> dict:
    row = {
        "scenario_id": prompt.replace(" ", "-"),
        "rollout_index": index,
        "prompt": prompt,
        "reward": reward,
        "final_text": reply,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply},
        ],
    }
    if privileged:
        row["privileged"] = {"reference": REFERENCE}
    return row


def _groups() -> list[dict]:
    """Two asks, four rollouts each: one leaked pass, one leaked failure,
    and a row that carries no privileged block at all."""
    rows = []
    for i in range(4):
        rows.append(_row("refund order 4412", i, reward=float(i % 2), reply=f"Refunded, take {i}."))
        rows.append(
            _row("cancel order 9931", i, reward=float(i < 3), reply=f"Cancelled, take {i}.")
        )
    rows[0]["final_text"] = f"Refunded. For reference, {REFERENCE}."  # a pass that recites the key
    rows[1]["final_text"] = f"I could not. {REFERENCE}."  # a failure that recites it
    rows[1]["reward"] = 0.0
    rows[2].pop("privileged")  # nothing to check on this one
    return rows


def test_row_leak_is_the_shared_verdict():
    rows = _groups()
    assert row_leak(rows[0]) == {"field": "reference", "needle": REFERENCE[:80]}
    assert row_leak(rows[1])["field"] == "reference"
    assert row_leak(rows[2]) is None  # no block: nothing to check
    assert row_leak(rows[3]) == {}  # block present, not quoted


def test_drop_privileged_leaks_names_what_it_dropped_and_does_not_mutate():
    rows = _groups()
    before = [dict(r) for r in rows]
    kept, report = drop_privileged_leaks(rows)
    assert len(kept) == len(rows) - 2
    assert report["n_dropped"] == 2 and report["checked"] is True
    assert report["n_checked"] == len(rows) - 1  # the row without a block is not "checked"
    assert {(x["scenario_id"], x["rollout_index"]) for x in report["leaked"]} == {
        ("refund-order-4412", 0),
        ("cancel-order-9931", 0),
    }
    assert rows == before


def test_select_for_rl_drops_leaks_before_the_other_gates():
    rows = _groups()
    picked, report = select_for_rl(rows, target=100)
    assert report["n"] == len(rows)  # kept X of N still counts every graded row
    assert report["privileged_leaks_dropped"] == 2
    assert report["privileged_leaks"]["n_dropped"] == 2
    assert all(not row_leak(r) for r in picked)


def test_select_for_sft_drops_a_leaked_pass_the_judge_could_not_tell():
    rows = _groups()
    picked, report = select_for_sft(rows, target=100)
    assert report["n"] == len(rows)
    assert report["privileged_leaks_dropped"] == 2
    assert all(not row_leak(r) for r in picked)
    # the leaked pass is not counted as "not passing"; it is its own line
    assert report["n_not_pass"] == sum(1 for r in rows[2:] if float(r["reward"]) < 1.0)


def test_the_docs_flow_selects_then_exports_without_refusal(tmp_path):
    """The landing and quickstart program, verbatim: the stand-in agent
    leaks on purpose on some rows, and export must not refuse what
    select kept."""

    @wai.tool
    def get_order(order_id: str) -> dict:
        """Look up an order by id."""
        ...

    data = wai.simulate(
        wai.seeded_agent([get_order]),
        tools=[get_order],
        system_prompt="Help customers with orders.",
        simulator=False,
        mode="rl",
        repeats=4,
        repeat_policy="fixed",
        budget=64,
    )
    scored = data.grade(judge=lambda row: {"reward": int(not row["seeded"])})
    assert leak_report(data)["n_leaked"] > 0, "the stand-in must leak or this proves nothing"
    for mode in ("rl", "sft"):
        rows = scored.select(mode=mode)
        n_dropped = rows.report["privileged_leaks_dropped"]
        assert n_dropped > 0
        assert f"privileged leaks dropped: {n_dropped}" in str(rows)
        assert str(rows).startswith(f"{mode} selection: kept {len(rows)} of 64 rows")
        report = rows.export(str(tmp_path / f"{mode}.jsonl"))  # validate=True by default
        assert report["privileged_leaks"]["n_leaked"] == 0
        assert not [w for w in report.get("warnings") or [] if "privileged" in w]
