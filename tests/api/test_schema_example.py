"""The schema example runs offline end to end: migrate, then project."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "schema"


def _load(name: str):
    path = EXAMPLE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"schema_example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(f"schema_example_{name}", module)
    spec.loader.exec_module(module)
    return module


def test_migrate_then_project(tmp_path):
    migrate = _load("migrate")
    project = _load("project")
    rows = migrate.simulate_rows(n=40, seed=0)
    assert rows and all(r["schema_version"] == "1" for r in rows)
    report = migrate.migrate(rows, tmp_path / "m")
    assert report["rows"] == len(rows)
    assert report["problems"] == {}
    tasks = [json.loads(l) for l in (tmp_path / "m" / "tasks.jsonl").read_text().splitlines()]
    assert tasks and all("steps" not in t and "final_text" not in t for t in tasks)

    out = project.project(tmp_path / "m" / "rows.v1.jsonl", tmp_path / "p",
                          holdout=0.3, teacher="teacher-x")
    assert out["train"] + out["holdout"] == out["tasks"]
    assert out["grpo"] == out["train"] and out["opd"] == out["train"]
    # every target has rows: the careless second repeat guarantees contrast
    assert out["sft"] > 0 and out["preference"] > 0 and out["opsd"] > 0 and out["eval"] > 0
    grpo = [json.loads(l) for l in (tmp_path / "p" / "grpo.jsonl").read_text().splitlines()]
    assert grpo and all(set(g) == {"prompt", "example_id", "info"} for g in grpo)
    eval_rows = [json.loads(l) for l in (tmp_path / "p" / "eval.jsonl").read_text().splitlines()]
    assert all(set(e["markers"]) == {"refund.looked_up_first", "refund.honest_after_fault"}
               for e in eval_rows)
    opsd = json.loads((tmp_path / "p" / "opsd.jsonl").read_text().splitlines()[0])
    assert set(opsd["hint"]) == {"principle", "hidden_state", "demonstration"}
    pref = json.loads((tmp_path / "p" / "preference.jsonl").read_text().splitlines()[0])
    assert pref["chosen"] and pref["rejected"] and pref["chosen"] != pref["rejected"]


def test_migrate_reads_legacy_fixtures(tmp_path):
    migrate = _load("migrate")
    fixtures = REPO_ROOT / "tests" / "fixtures" / "rows"
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(fixtures.glob("*.json"))]
    report = migrate.migrate(rows, tmp_path)
    assert report["rows"] == len(rows)
    assert set(report["shapes"]) >= {"engine", "platform_pull", "training", "hf_flat"}
    assert report["problems"] == {}
