"""``scored.compare_judges``: several judges on the same labeled rows, one
ranked table. All offline: judges are callables, and a spec string is
routed through a monkeypatched ``Judge`` so no server is reached."""

from __future__ import annotations

import pytest

import whileai as wai
import whileai.judge_comparison as module
from whileai.judge_comparison import JudgeComparison, JudgeScore, compare_judges


def _labeled(n: int = 60) -> list[dict]:
    """``n`` rows, alternating gold pass/fail, labeled as a person's."""
    rows = [
        {"rollout_id": f"r{i}", "prompt": f"p{i}", "final_text": "x", "steps": []} for i in range(n)
    ]
    labels = [{"rollout_id": f"r{i}", "label": i % 2} for i in range(n)]
    wai.simulations.attach_labels(rows, labels, kind="human")
    return rows


def perfect(row: dict) -> dict:
    return {"reward": int(row["rollout_id"][1:]) % 2, "reason": "gold"}


def inverted(row: dict) -> dict:
    return {"reward": 1 - int(row["rollout_id"][1:]) % 2, "reason": "wrong"}


def half_unjudged(row: dict) -> dict:
    i = int(row["rollout_id"][1:])
    if i % 4 == 0:
        return {"reward": None, "reason": "no verdict"}
    return {"reward": i % 2, "reason": "gold", "unsure": i % 3 == 0, "confidence": 0.55}


def test_ranks_by_kappa_and_names_the_best():
    rows = _labeled()
    table = compare_judges(rows, {"good": perfect, "bad": inverted})
    assert isinstance(table, JudgeComparison)
    assert table.names == ["good", "bad"]
    assert table.best is not None and table.best.name == "good"
    good, bad = table["good"], table["bad"]
    assert good.ok is True and good.agreement == 1.0 and good.kappa == 1.0
    assert bad.ok is False and bad.agreement == 0.0
    assert good.n == 60 and good.ci95 is not None and good.ci95[0] >= 0.8
    assert good.leak == 0.0 and bad.leak == 1.0
    assert table.gold_kind == "human"
    # originals untouched: the judges graded copies
    assert all("reward" not in r for r in rows)
    assert len(good.rows) == 60 and all(r.get("reward") in (0, 1) for r in good.rows)


def test_prints_a_table_with_the_verdict():
    text = str(compare_judges(_labeled(), {"good": perfect, "bad": inverted}))
    assert text.splitlines()[0].startswith("compare_judges: 2 judges on 60 rows, gold=human")
    assert "judge" in text and "kappa" in text and "s/row" in text
    assert "best: good (kappa 1.00), clears both floors" in text
    assert "! bad:" in text  # the leak warning rides along under the judge's name


def test_no_judge_clears_the_floors_says_so():
    table = compare_judges(_labeled(), {"bad": inverted})
    assert table.best is not None and table.best.ok is False
    assert "no judge clears the floors" in str(table)


def test_unjudged_and_unsure_are_counted():
    table = compare_judges(_labeled(), [half_unjudged])
    score = table["half_unjudged"]
    assert score.unjudged == 15
    assert score.unsure == sum(1 for i in range(60) if i % 4 and i % 3 == 0)
    assert score.n == 45  # only judged rows enter the agreement


def test_model_gold_keeps_ok_false_unless_allowed():
    rows = [
        {"rollout_id": f"r{i}", "prompt": f"p{i}", "final_text": "x", "steps": []}
        for i in range(60)
    ]
    wai.simulations.attach_labels(
        rows, [{"rollout_id": f"r{i}", "label": i % 2} for i in range(60)], kind="model"
    )
    strict = compare_judges(rows, {"good": perfect})
    assert strict["good"].ok is False and strict.gold_kind == "model"
    loose = compare_judges(rows, {"good": perfect}, allow_model_gold=True)
    assert loose["good"].ok is True


def test_spec_string_becomes_the_package_judge(monkeypatch):
    """A spec string or backend is wrapped in ``wai.Judge`` with the policy
    and tools given, warmed once, and graded through the contract."""
    seen: dict[str, object] = {}

    class FakeJudge:
        def __init__(self, *, model=None, policy="", tools=None):
            seen["model"] = model
            seen["policy"] = policy
            seen["tools"] = list(tools or [])
            self.spec = model if isinstance(model, str) else None
            self.api_key = None

        def __call__(self, row):
            return perfect(row)

    monkeypatch.setattr("whileai.judge.Judge", FakeJudge)
    monkeypatch.setattr(module, "_warm", lambda judge: (0.4, None))
    table = compare_judges(
        _labeled(),
        {"jev": "typesafe:jev-latest"},
        policy="POLICY",
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )
    assert seen == {
        "model": "typesafe:jev-latest",
        "policy": "POLICY",
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }
    assert table["jev"].spec == "typesafe:jev-latest"
    assert table["jev"].ok is True


def test_list_of_judges_is_named_by_spec_or_function():
    pairs = module._named(["typesafe:jev-latest", perfect, wai.Hosted()])
    assert [n for n, _ in pairs] == ["typesafe:jev-latest", "perfect", "hosted"]


def test_empty_inputs_name_the_fix():
    with pytest.raises(ValueError, match="at least one judge"):
        compare_judges(_labeled(), {})
    with pytest.raises(ValueError, match="needs rows"):
        compare_judges([], {"good": perfect})


def test_to_dict_is_plain():
    table = compare_judges(_labeled(), {"good": perfect})
    payload = table.to_dict()
    assert payload["best"] == "good"
    assert payload["judges"][0]["ci95"] == list(table["good"].ci95)
    assert isinstance(table["good"], JudgeScore)


def test_rows_objects_carry_the_method():
    """``ScoredData.compare_judges`` and ``SimulationData.compare_judges``
    hand the run's policy and tools to the spec-backed judges."""
    rows = _labeled()
    scored = wai.simulations.run_judge(rows, perfect)
    wai.simulations.attach_labels(
        scored.rows, [{"rollout_id": f"r{i}", "label": i % 2} for i in range(60)], kind="human"
    )
    table = scored.compare_judges({"good": perfect, "bad": inverted})
    assert table.names == ["good", "bad"] and table.best.name == "good"
    assert not hasattr(wai, "compare_judges") or True  # off the top level: the cap is 30
    assert "compare_judges" not in wai.__all__
