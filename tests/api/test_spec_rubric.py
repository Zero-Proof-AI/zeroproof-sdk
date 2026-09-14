"""A spec's rubric.md is the judge's rubric; without one grade() says so."""

from __future__ import annotations

import json

import zeroproof.simulations as zps
from tests.helpers import POLICY, TOOLS, offline, scripted_agent
from zeroproof.simulations.run.spec import spec_rubric
from zeroproof.simulations.score.grade_llm import RUBRIC_TAIL, rubric_prompt


def _spec_folder(tmp_path, rubric: str | None):
    folder = tmp_path / "agent"
    folder.mkdir(parents=True)
    (folder / "spec.json").write_text(json.dumps({"tools": TOOLS, "policy": POLICY}))
    if rubric is not None:
        (folder / "rubric.md").write_text(rubric)
    return folder


def test_spec_rubric_reads_the_folder_or_the_dict(tmp_path):
    folder = _spec_folder(tmp_path, "The task is the refund.\n- Looked up first.")
    assert spec_rubric(str(folder)).startswith("The task is the refund.")
    assert spec_rubric(str(folder / "spec.json")).startswith("The task is the refund.")
    assert spec_rubric({"tools": TOOLS, "rubric": " by key "}) == "by key"
    assert spec_rubric(str(_spec_folder(tmp_path / "bare", None))) is None
    assert spec_rubric(None) is None


def test_simulate_carries_the_spec_rubric_and_rubric_kw_wins(tmp_path):
    folder = _spec_folder(tmp_path, "From the folder.")
    data = zps.simulate(
        scripted_agent, spec=str(folder), budget=2, **offline(tools=None, policy=None)
    )
    assert data.profile.rubric == "From the folder."
    data = zps.simulate(
        scripted_agent,
        spec=str(folder),
        rubric="Passed in.",
        budget=2,
        **offline(tools=None, policy=None),
    )
    assert data.profile.rubric == "Passed in."


def test_grade_uses_the_rubric_and_names_the_conduct_floor(monkeypatch, tmp_path):
    seen: list[dict] = []

    def fake_apply(trajectories, **kw):
        seen.append(kw)
        return {"status": "judged", "graded": 0}

    monkeypatch.setattr("zeroproof.simulations.data.apply_grade_llm", fake_apply)
    monkeypatch.setattr("zeroproof.simulations.data.require_judge_key", lambda *a, **k: "k")

    folder = _spec_folder(tmp_path, "Do the job.")
    data = zps.simulate(
        scripted_agent, spec=str(folder), budget=2, **offline(tools=None, policy=None)
    )
    report = data.grade()
    assert report["rubric"] == "spec" and "note" not in report
    assert seen[-1]["prompt"] == rubric_prompt("Do the job.")
    assert seen[-1]["prompt"].endswith(RUBRIC_TAIL) and '"score"' in seen[-1]["prompt"]

    report = data.grade(rubric="Another job.")
    assert report["rubric"] == "rubric" and seen[-1]["prompt"].startswith(
        "Grade the agent against this rubric."
    )

    report = data.grade_llm(prompt="Raw judge prompt.")
    assert report["rubric"] == "prompt" and seen[-1]["prompt"] == "Raw judge prompt."

    bare = zps.simulate(scripted_agent, budget=2, **offline())
    report = bare.grade()
    assert report["rubric"] == "conduct_floor" and "conduct floor only" in report["note"]
    assert seen[-1]["prompt"] is None


def test_tracked_spec_fixtures_ship_a_rubric():
    for folder in ("examples/prime-intellect-rl", "tests/fixtures/github"):
        text = spec_rubric(folder)
        assert text and text.startswith("The task is"), folder
