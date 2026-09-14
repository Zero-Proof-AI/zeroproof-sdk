"""Difficulty band enforcement and the publish gate."""

from __future__ import annotations

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations.score.optimize import (
    DEFAULT_BAND,
    group_signal,
    select_for_rl,
    trim_out_of_band,
)
from zeroproof.simulations.score.publish_gate import (
    PublishGateError,
    calibrate,
    is_rl_shaped,
    publish_gate,
)


def _rows(spec: dict[str, list[int]]) -> list[dict]:
    return [
        {
            "prompt": prompt,
            "reward": label,
            "final_text": f"Issue {i} is open.",
            "steps": [{"tool": "get_issue", "arguments": {"number": i}, "result": {"ok": 1}}],
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"Issue {i} is open."},
            ],
        }
        for prompt, labels in spec.items()
        for i, label in enumerate(labels)
    ]


EIGHT = {
    "easy": [1, 1, 1, 1, 1, 1, 1, 0],  # p = .875
    "hard": [0, 0, 0, 0, 0, 0, 0, 1],  # p = .125
    "mid": [1, 0, 1, 0, 1, 0, 1, 0],  # p = .5
    "dead": [0, 0, 0, 0, 0, 0, 0, 0],
    "solo": [1],
}


def test_default_band_is_twenty_eighty():
    assert DEFAULT_BAND == (0.2, 0.8)
    assert group_signal(_rows(EIGHT))["n_in_band"] == 1


def test_trim_out_of_band_drops_by_side_and_leaves_unanimous_and_singles():
    kept, report = trim_out_of_band(_rows(EIGHT))
    prompts = {r["prompt"] for r in kept}
    assert prompts == {"mid", "dead", "solo"}
    assert report["too_easy"] == 1 and report["too_hard"] == 1
    assert report["n_groups_dropped"] == 2 and report["band"] == [0.2, 0.8]
    wide, wide_report = trim_out_of_band(_rows(EIGHT), lo=0.1, hi=0.9)
    assert len(wide) == len(_rows(EIGHT)) and wide_report["n_groups_dropped"] == 0
    with pytest.raises(ValueError, match="band"):
        trim_out_of_band(_rows(EIGHT), lo=0.9, hi=0.1)


def test_select_for_rl_enforces_band_by_default_and_can_rank_only():
    picked, report = select_for_rl(_rows(EIGHT), target=100)
    assert {r["prompt"] for r in picked} == {"mid", "solo"}  # singles always stay
    assert report["band_dropped"] == {"too_easy": 1, "too_hard": 1}
    assert report["enforce_band"] is True and report["band"] == [0.2, 0.8]

    ranked, ranked_report = select_for_rl(_rows(EIGHT), target=100, enforce_band=False)
    prompts = [r["prompt"] for r in ranked]
    assert set(prompts) == {"mid", "easy", "hard", "solo"}
    assert prompts[:8] == ["mid"] * 8  # in-band first
    assert ranked_report["band_dropped"] == {"too_easy": 0, "too_hard": 0}


def test_optimize_passes_band_through():
    _rows_out, report = zps.optimize(_rows(EIGHT), mode="rl", target=100, band=(0.1, 0.9))
    assert report["band"] == [0.1, 0.9]
    assert report["band_dropped"] == {"too_easy": 0, "too_hard": 0}
    assert report["groups_selected"] == 4  # easy, hard, mid, solo


def test_calibrate_stamps_pass_rate_k_and_policy():
    rows = [*_rows({"a": [1, 0, 1, 1], "b": [0, 0, 0, 1]}), {"prompt": "c", "reward": None}]
    report = calibrate(rows, policy="Look up before refunding.", model="qwen")
    assert report["n_tasks"] == 2 and report["n_stamped"] == 8 and report["n_unstamped"] == 1
    a = next(r for r in rows if r["prompt"] == "a")["calibration"]
    assert a["pass_rate"] == 0.75 and a["n"] == 4 and a["task_id"] == "a"
    assert a["student"]["model"] == "qwen" and len(a["student"]["prompt_hash"]) == 16
    assert "calibration" not in rows[-1]
    assert report["pass_at"]["pass_at_1"] == pytest.approx(0.5)


def test_is_rl_shaped_by_mode_or_repeats():
    assert is_rl_shaped([], mode="rl")
    assert not is_rl_shaped(_rows({"a": [1], "b": [0]}))
    assert is_rl_shaped(_rows({"a": [1, 0]}))


def test_gate_refuses_ungraded_and_unanimous_rl_but_passes_explore():
    ungraded = [{"prompt": "a", "reward": None}, {"prompt": "a", "reward": None}]
    with pytest.raises(PublishGateError, match="ungraded_rl_rows"):
        publish_gate(ungraded)
    with pytest.raises(PublishGateError, match="no_mixed_groups"):
        publish_gate(_rows({"a": [1, 1, 1, 1], "b": [0, 0, 0, 0]}))
    soft = publish_gate(_rows({"a": [1, 1, 1, 1]}), strict=False)
    assert soft["ok"] is False and "no_mixed_groups" in soft["refusal"]

    explore = publish_gate([{"prompt": "a", "reward": None}, {"prompt": "b", "reward": None}])
    assert explore["ok"] and not explore["rl_shaped"] and explore["warnings"] == []


def test_gate_report_warns_about_unpruned_asks():
    report = publish_gate(_rows(EIGHT), mode="rl")
    assert report["ok"] and report["rl_shaped"]
    joined = " ".join(report["warnings"])
    assert "2 mixed ask(s) fall outside the 20%-80% band" in joined
    assert "1 unanimous ask(s)" in joined
    assert report["calibration"]["n_stamped"] == 33
    assert report["signal"]["n_mixed"] == 3


def test_push_gates_by_default_and_can_skip(monkeypatch):
    from tests.helpers import simulate_offline

    pushed: list[list[dict]] = []

    def fake_push_rows(rows, name, *, api_key=None, parent=None):
        pushed.append(rows)
        return {"datasetId": "ds_new"}

    monkeypatch.setattr("zeroproof.simulations.data.push_rows", fake_push_rows)

    data = simulate_offline(budget=8, seed=0, mode="rl", repeats=4)
    with pytest.raises(PublishGateError, match="ungraded_rl_rows"):
        data.push("run")
    assert pushed == []

    raw = data.push("run", gate=False)
    assert raw["datasetId"] == "ds_new" and "gate" not in raw
    assert len(pushed) == 1 and all("calibration" not in r for r in pushed[0])

    # The scripted agent is deterministic, so a grader that looks at the
    # row alone would make every group unanimous. Alternate per prompt.
    seen: dict[str, int] = {}

    def alternating(t: dict) -> float:
        key = str(t.get("prompt") or "")
        seen[key] = seen.get(key, 0) + 1
        return float(seen[key] % 2)

    data.grade(grader=alternating, concurrency=1)
    entry = data.push("run")
    assert entry["gate"]["rl_shaped"] and entry["gate"]["ok"]
    stamped = [r for r in pushed[-1] if "calibration" in r]
    assert stamped and all(0.0 <= r["calibration"]["pass_rate"] <= 1.0 for r in stamped)
    assert stamped[0]["calibration"]["student"]["prompt_hash"]


def test_push_rows_gate_flag(monkeypatch):
    from zeroproof.simulations.ingest import platform

    calls = []

    def fake_call(method, path, api_key=None, body=None, **kw):
        calls.append((method, path))
        if path == "/datasets":
            return {"datasetId": "ds_x", "uploadUrl": "https://u"}
        return {"datasetId": "ds_x"}

    monkeypatch.setattr(platform, "_call", fake_call)
    rows = _rows({"a": [1, 0, 1, 0]})
    out = zps.push_rows(rows, "n", api_key="k", gate=True)
    assert out["gate"]["ok"] and calls[0] == ("POST", "/datasets")
    with pytest.raises(PublishGateError):
        zps.push_rows([{"prompt": "a", "reward": None}] * 2, "n", api_key="k", gate=True)
