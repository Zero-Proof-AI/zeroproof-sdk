"""pass@1 / pass^k / pass@k off graded groups (score/passat.py)."""

from __future__ import annotations

import json
from math import comb

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations.data import SimulationData
from zeroproof.simulations.score.judging import ScoredData
from zeroproof.simulations.score.optimize import group_signal
from zeroproof.simulations.score.passat import PassAt, pass_at


def _rows(spec: dict[str, list[int]]) -> list[dict]:
    """``{"prompt": [0, 1, ...]}`` -> graded rows, one per label."""
    return [
        {"prompt": prompt, "reward": label, "final_text": "x", "steps": []}
        for prompt, labels in spec.items()
        for label in labels
    ]


def test_unbiased_estimators_match_closed_form():
    # n=4, c=2, k=2: pass@2 = 1 - C(2,2)/C(4,2) = 5/6, pass^2 = C(2,2)/C(4,2) = 1/6.
    out = pass_at(_rows({"a": [1, 1, 0, 0]}), k=2, min_k=2)
    assert out.k == 2
    assert out.pass_at_1 == pytest.approx(0.5)
    assert out.pass_at_k == pytest.approx(1 - comb(2, 2) / comb(4, 2))
    assert out.pass_pow_k == pytest.approx(comb(2, 2) / comb(4, 2))
    assert out.headroom == pytest.approx(out.pass_at_k - 0.5)
    assert out.n_groups == out.n_groups_at_k == 1
    assert out.n_rows == 4


def test_pass_at_1_is_macro_over_tasks_and_k_is_smallest_multi_group():
    out = pass_at(_rows({"a": [1, 1, 1, 1], "b": [0, 0, 0, 0, 0, 0], "solo": [1]}))
    # Three tasks at 1.0, 0.0, 1.0 -> 2/3 regardless of group size.
    assert out.pass_at_1 == pytest.approx(2 / 3)
    assert out.k == 4
    # Unanimous groups: pass@k == pass^k == pass@1 over the k-eligible groups.
    assert out.pass_at_k == pytest.approx(0.5)
    assert out.pass_pow_k == pytest.approx(0.5)
    assert out.n_groups == 3
    assert out.n_groups_at_k == 2  # the solo ask cannot draw 4
    assert out.per_task == {"a": 1.0, "b": 0.0, "solo": 1.0}


def test_below_min_k_withholds_k_way_numbers_with_a_note():
    out = pass_at(_rows({"a": [1, 0], "b": [1, 1]}))
    assert out.k == 2
    assert out.pass_at_1 == pytest.approx(0.75)
    assert out.pass_pow_k is None and out.pass_at_k is None and out.headroom is None
    assert "repeats>=4" in out.note
    assert "n/a" in str(out) and "pass@1 0.75" in str(out)


def test_no_graded_rows_is_none_not_zero():
    out = pass_at([{"prompt": "a", "reward": None}, {"prompt": "b", "reward": 0.5}])
    assert isinstance(out, PassAt)
    assert out.pass_at_1 is None and out.pass_at_k is None
    assert out.n_groups == 0 and out.n_rows == 0
    assert "grade first" in out.note


def test_explicit_k_larger_than_every_group_reports_why():
    out = pass_at(_rows({"a": [1, 0, 1, 0]}), k=8)
    assert out.k == 8
    assert out.pass_at_k is None
    assert "no group has 8" in out.note
    with pytest.raises(ValueError):
        pass_at(_rows({"a": [1, 0]}), k=0)


def test_group_signal_carries_the_same_keys():
    rows = _rows({"mixed": [1, 0, 1, 0], "dead0": [0, 0, 0, 0], "dead1": [1, 1, 1, 1]})
    signal = group_signal(rows)
    direct = pass_at(rows)
    for key in ("k", "pass_at_1", "pass_pow_k", "pass_at_k", "headroom"):
        assert signal[key] == getattr(direct, key)
    assert signal["n_mixed"] == 1
    assert signal["headroom"] > 0
    # No mixed groups -> nothing to learn -> zero headroom.
    flat = group_signal(_rows({"dead0": [0, 0, 0, 0], "dead1": [1, 1, 1, 1]}))
    assert flat["mixed_rate"] == 0
    assert flat["headroom"] == pytest.approx(0.0)


def test_scored_data_and_simulation_data_expose_the_property():
    rows = _rows({"a": [1, 0, 1, 1], "b": [0, 0, 0, 1]})
    scored = ScoredData(rows, run_id="r", source="grade", judge_name="j")
    assert scored.pass_at.pass_at_1 == pytest.approx(0.5)
    data = SimulationData(trajectories=rows)
    assert data.pass_at.to_dict() == scored.pass_at.to_dict()
    assert zps.pass_at(rows).k == 4
    assert "PassAt" in zps.__all__ and "pass_at" in zps.__all__


def test_recommend_rl_names_the_headroom():
    out = zps.recommend(mode="rl", target=200)
    assert any("pass@k - pass@1" in line for line in out["reasoning"])


def test_save_meta_writes_pass_at_to_sidecar(tmp_path):
    rows = _rows({"a": [1, 0, 1, 1], "b": [0, 0, 0, 1]})
    for row in rows:
        row.update({"arm": "ordinary", "scenario_id": "s", "messages": []})
    data = SimulationData(trajectories=rows)
    data.save(str(tmp_path / "r.jsonl"), meta=True)
    meta = json.loads((tmp_path / "r.meta.json").read_text())
    assert meta["pass_at"]["k"] == 4
    assert meta["pass_at"]["pass_at_1"] == pytest.approx(0.5)
    assert meta["pass_at"]["headroom"] == pytest.approx(
        meta["pass_at"]["pass_at_k"] - meta["pass_at"]["pass_at_1"]
    )


def test_uneven_groups_name_the_k_that_scores_them():
    # A budget cut mid-group: three groups reached 4 repeats, one has 2.
    rows = []
    for prompt, labels in {
        "a": [1, 0, 1, 1],
        "b": [0, 0, 1, 0],
        "c": [1, 1, 1, 1],
        "d": [1, 0],
    }.items():
        rows += [{"prompt": prompt, "reward": r} for r in labels]
    got = pass_at(rows)
    assert got.k == 2 and got.pass_pow_k is None
    assert "uneven (2 to 4 repeats)" in got.note and "k=4" in got.note and "3 group" in got.note
    scored = pass_at(rows, k=4)
    assert scored.n_groups_at_k == 3 and scored.pass_at_k is not None and scored.note == ""


def test_even_groups_keep_the_plain_note():
    rows = [{"prompt": p, "reward": r} for p in "ab" for r in (1, 0)]
    assert pass_at(rows).note == "set repeats>=4 for pass^k and pass@k"
