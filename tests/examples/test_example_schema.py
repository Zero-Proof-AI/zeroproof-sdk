"""The schema example is the reference for the row format: run both scripts
offline and check that the projections keep the two rules the README states
(a task never carries a rollout; the eval marker is never the training
reward), and that migrate takes a legacy file without rejecting anything.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from zeroproof.simulations import schema

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "schema"
ROLLOUT_KEYS = {"steps", "final_text", "messages", "reward", "rollout_id", "rollout_index"}


def _env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("OPENAI_API_KEY", "ZEROPROOF_API_KEY", "VLLM_API_KEY"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(script: str, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    out = subprocess.run(
        [sys.executable, str(EXAMPLE / script), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(cwd),
        env=_env(),
        timeout=300,
    )
    assert out.returncode == 0, f"{script} failed:\n{out.stdout[-2000:]}\n{out.stderr[-3000:]}"
    return out


def _rows(path: Path) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="module")
def projected(tmp_path_factory) -> dict:
    """migrate.py (simulated rows) then project.py, once for the module."""
    work = tmp_path_factory.mktemp("schema")
    migrate = json.loads(_run("migrate.py", "--rows", "24", cwd=work).stdout)
    project = json.loads(_run("project.py", "out/rows.v1.jsonl", cwd=work).stdout)
    return {"dir": work / "out", "migrate": migrate, "project": project}


def test_migrate_stamps_every_simulated_row_as_v1(projected):
    report = projected["migrate"]
    rows = _rows(projected["dir"] / "rows.v1.jsonl")
    assert report["rows"] == len(rows) == 24
    assert report["shapes"] == {"v1": 24}, report
    assert report["problems"] == {}
    assert {r.get(schema.SCHEMA_KEY) for r in rows} == {schema.SCHEMA_VERSION}
    assert all(schema.validate(r) == [] for r in rows)
    # One task per situation; the scripted run repeats every prompt twice.
    assert report["tasks"] == len({schema.from_row(r)[0].task_id for r in rows})
    assert report["tasks"] * 2 == report["rows"]


def test_tasks_file_holds_situations_and_no_model_output(projected):
    tasks = _rows(projected["dir"] / "tasks.jsonl")
    assert len(tasks) == projected["migrate"]["tasks"]
    for task in tasks:
        assert {"task_id", "prompt", "world", "privileged", "axes"} <= set(task), task.keys()
        assert not (set(task) & ROLLOUT_KEYS), sorted(set(task) & ROLLOUT_KEYS)


def test_project_writes_one_file_per_target_with_matching_counts(projected):
    report, out = projected["project"], projected["dir"]
    for target in ("eval", "sft", "preference", "grpo", "opsd", "opd"):
        assert (out / f"{target}.jsonl").exists(), target
        assert len(_rows(out / f"{target}.jsonl")) == report[target], target
    assert report["train"] + report["holdout"] == report["tasks"]
    assert report["train"] > 0 and report["holdout"] > 0, report


def test_holdout_and_train_never_share_a_task(projected):
    out = projected["dir"]
    holdout = {r["task_id"] for r in _rows(out / "eval.jsonl")}
    train = {r["example_id"] for r in _rows(out / "grpo.jsonl")}
    assert holdout and train
    assert holdout.isdisjoint(train)
    assert train == {r["example_id"] for r in _rows(out / "opd.jsonl")}
    assert {r["example_id"] for r in _rows(out / "opsd.jsonl")} <= train


def test_grpo_prompt_set_is_verifiers_shaped_and_ships_no_rollout(projected):
    rows = _rows(projected["dir"] / "grpo.jsonl")
    assert len(rows) == projected["project"]["train"]
    for row in rows:
        assert set(row) == {"prompt", "example_id", "info"}, row.keys()
        assert {"world_state", "faults", "axes"} <= set(row["info"])
        assert not (set(row["info"]) & ROLLOUT_KEYS)


def test_eval_scores_with_markers_and_never_with_the_judgment(projected):
    rows = _rows(projected["dir"] / "eval.jsonl")
    assert rows
    for row in rows:
        assert set(row["markers"]) == {"refund.looked_up_first", "refund.honest_after_fault"}
        assert set(row["markers"].values()) <= {0.0, 1.0}
        assert "reward" not in row and "reason" not in row


def test_sft_rows_are_the_passing_training_rollouts(projected):
    out = projected["dir"]
    train = {r["example_id"] for r in _rows(out / "grpo.jsonl")}
    passing = [
        r
        for r in _rows(out / "rows.v1.jsonl")
        if schema.from_row(r)[0].task_id in train and (r.get("reward") or 0) >= 1
    ]
    sft = _rows(out / "sft.jsonl")
    assert len(sft) == len(passing) == projected["project"]["sft"]
    for row in sft:
        assert row["messages"][0]["role"] == "user"
        assert any(m["role"] == "assistant" for m in row["messages"])
        assert "privileged" not in row


def test_preference_pairs_hold_a_pass_and_a_fail_of_one_task(projected):
    pairs = _rows(projected["dir"] / "preference.jsonl")
    assert len(pairs) == projected["project"]["preference"] > 0
    for pair in pairs:
        assert {"prompt", "chosen", "rejected"} <= set(pair)
        assert pair["chosen"] != pair["rejected"]
        assert pair["chosen"][0]["content"] == pair["rejected"][0]["content"] == pair["prompt"]


def test_opsd_hint_carries_hidden_state_and_a_demonstration(projected):
    rows = _rows(projected["dir"] / "opsd.jsonl")
    assert len(rows) == projected["project"]["opsd"] > 0
    for row in rows:
        assert set(row) == {"prompt", "example_id", "hint"}
        hint = row["hint"]
        assert hint["principle"]
        assert {"world_state", "faults"} <= set(hint["hidden_state"])
        demo = hint["demonstration"]
        assert demo[0] == {"role": "user", "content": row["prompt"]}
        assert any(m["role"] == "assistant" for m in demo)


def test_migrate_takes_a_legacy_file_and_rejects_nothing(tmp_path):
    """Engine-shaped rows with a column the schema has never heard of, plus
    one line that is not an object: shapes reported, nothing dropped, the
    unknown column rides through, and the odd line is counted, not fatal."""
    legacy = [
        {
            "prompt": "Refund ORD-1 please",
            "steps": [{"tool": "lookup_order", "arguments": {"order_id": "ORD-1"}}],
            "final_text": "Done.",
            "reward": 1.0,
            "scenario_id": "s1",
            "rollout_index": 0,
            "my_custom_col": "kept",
        },
        {
            "prompt": "Refund ORD-2 please",
            "steps": [],
            "final_text": "No.",
            "reward": 0.0,
            "scenario_id": "s2",
            "rollout_index": 0,
            "my_custom_col": "kept",
        },
    ]
    src = tmp_path / "old_run.jsonl"
    with open(src, "w", encoding="utf-8") as fh:
        for row in legacy:
            fh.write(json.dumps(row) + "\n")
        fh.write(json.dumps(["not", "a", "row"]) + "\n")

    report = json.loads(_run("migrate.py", str(src), "--out", "migrated", cwd=tmp_path).stdout)
    assert report["shapes"] == {"engine": 2, "loose": 1}, report
    assert report["problems"] == {"not_a_dict": 1}, report
    assert report["rows"] == 2 and report["tasks"] == 2

    rows = _rows(tmp_path / "migrated" / "rows.v1.jsonl")
    assert [r["my_custom_col"] for r in rows] == ["kept", "kept"]
    assert {r[schema.SCHEMA_KEY] for r in rows} == {schema.SCHEMA_VERSION}
    assert {schema.detect_shape(r) for r in rows} == {"v1"}
    tasks = _rows(tmp_path / "migrated" / "tasks.jsonl")
    assert [t["prompt"] for t in tasks] == ["Refund ORD-1 please", "Refund ORD-2 please"]
