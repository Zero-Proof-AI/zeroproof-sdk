"""Schema v1: four objects, one stamped wire row, version 0 read by shape.

The stamp is born in the engine, so the streamed file and a later save()
agree. The JSON Schema and the dataclasses cannot drift. Each legacy shape
has a fixture that round-trips through from_row / to_row.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

import zeroproof.simulations as zps
from tests.connect.test_otel import BATCH
from tests.helpers import FIXTURES, simulate_offline
from zeroproof.simulations import schema
from zeroproof.simulations.data import _CONVERSATION_FIELDS

ROWS = FIXTURES / "rows"


def _load(name: str) -> dict:
    return json.loads((ROWS / f"{name}.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ drift

def test_axes_match_data_conversation_fields():
    assert schema.AXES == _CONVERSATION_FIELDS


def test_json_schema_and_dataclasses_agree():
    js = schema.load_json_schema()
    assert js["$defs"]["row"]["properties"]["schema_version"]["const"] == schema.SCHEMA_VERSION
    for name, fields in js["$defs"]["objects"].items():
        if name.startswith("$"):
            continue
        cls = getattr(schema, name)
        assert [f.name for f in dataclasses.fields(cls)] == fields, name
    props = set(js["$defs"]["row"]["properties"])
    assert set(schema.AXES) <= props
    assert set(schema.VERDICT_KEYS) - {"grader_reason", "judge_meta"} <= props
    task, rollout, judgments, markers = schema.from_row(_load("engine"))
    assert set(schema.to_row(task, rollout, judgments, markers)) <= props


def test_json_schema_ships_in_the_package():
    from importlib import resources
    path = resources.files("zeroproof.simulations") / "schemas" / "row-v1.json"
    assert path.is_file()


# ------------------------------------------------------------------ shapes

def test_detect_shape_and_version():
    assert schema.detect_shape(_load("engine")) == "engine"
    assert schema.detect_shape(_load("platform_pull_engine")) == "engine"
    assert schema.detect_shape(_load("platform_pull")) == "platform_pull"
    otel = zps.rows_from_otel(BATCH)[0]
    assert schema.detect_shape(otel) == "v1"
    otel.pop("schema_version")
    assert schema.detect_shape(otel) == "otel"
    assert schema.detect_shape({"prompt": "p"}) == "loose"
    assert schema.detect_shape("nope") == "loose"
    assert schema.version_of(_load("engine")) == "0"
    assert schema.version_of({"schema_version": "1"}) == "1"


@pytest.mark.parametrize("name", ["engine", "platform_pull_engine"])
def test_engine_shaped_rows_round_trip_byte_for_byte(name):
    row = _load(name)
    task, rollout, judgments, markers = schema.from_row(row)
    assert task.task_id == row["scenario_id"]
    assert rollout.policy.model == row["model_version"]
    assert [j.scorer.name for j in judgments] == [row["label_source"]]
    back = schema.to_row(task, rollout, judgments, markers)
    assert back.pop("schema_version") == "1"
    assert json.dumps(back, sort_keys=True) == json.dumps(row, sort_keys=True)


def test_platform_trace_pull_normalizes_like_load_traces():
    row = _load("platform_pull")
    expected = zps.load_traces([row])[0]
    expected.pop("tool_trace")            # load_traces carries it; to_row emits steps
    task, rollout, judgments, markers = schema.from_row(row)
    assert [s.tool for s in rollout.steps] == ["read_file", "write_file"]
    assert rollout.steps[0].arguments == '{"path": "paging.py"}'
    assert judgments[0].reward == 1 and judgments[0].scorer.name == "unlabeled"
    back = schema.to_row(task, rollout, judgments, markers)
    assert back.pop("schema_version") == "1"
    assert back.pop("scenario_id").startswith("task_")   # none in the legacy row
    back.pop("messages")                                 # derived
    assert back == expected


def test_otel_rows_are_stamped_and_round_trip():
    rows = zps.rows_from_otel(BATCH)
    assert rows and all(r["schema_version"] == "1" for r in rows)
    for row in rows:
        task, rollout, judgments, markers = schema.from_row(row)
        assert rollout.extra["conversation_id"] == row["conversation_id"]
        back = schema.to_row(task, rollout, judgments, markers)
        back.pop("scenario_id")
        back.pop("messages")
        assert back == row


def test_legacy_rows_are_never_rejected():
    for name in ("engine", "platform_pull_engine", "platform_pull"):
        assert schema.validate(_load(name)) == []
    assert schema.validate({"anything": 1}) == []


# ------------------------------------------------------------------ validate

def test_stamped_rows_are_type_checked():
    assert schema.validate({"schema_version": "1", "prompt": 3}) == ["prompt_not_str"]
    assert schema.validate({"schema_version": "1", "steps": {}}) == ["steps_not_list"]
    assert schema.validate({"schema_version": "1", "reward": True}) == ["reward_is_bool"]
    assert schema.validate({"schema_version": "9"}) == ["unknown_schema_version:9"]
    assert schema.validate("row") == ["not_a_dict"]
    assert schema.validate({"schema_version": "1"}, "training") == ["messages_missing"]
    assert schema.validate({"schema_version": "1", "chosen": [{"role": "user"}]},
                           "preference") == ["rejected_missing"]


def test_check_names_the_boundary_and_the_rows():
    with pytest.raises(ValueError, match=r"schema_invalid in here: 1:not_a_dict"):
        schema.check([{"schema_version": "1"}, "bad"], where="here")


def test_push_rows_validates_before_any_network_call():
    with pytest.raises(ValueError, match="schema_invalid in push_rows"):
        zps.push_rows(["not a row"], "name", api_key="k")


# ------------------------------------------------------------------ engine

def test_stamp_is_born_in_the_engine_and_stream_equals_save(tmp_path):
    out = tmp_path / "run.jsonl"
    data = simulate_offline(budget=6, concurrency=1, output=str(out))
    assert data.trajectories
    assert all(t["schema_version"] == "1" for t in data.trajectories)
    streamed = [json.loads(line) for line in out.read_text().splitlines()]
    saved_path = tmp_path / "saved.jsonl"
    data.save(str(saved_path), meta=True)
    saved = [json.loads(line) for line in saved_path.read_text().splitlines()]
    assert streamed == saved
    assert all(r["schema_version"] == "1" for r in saved)
    meta = json.loads((tmp_path / "saved.meta.json").read_text())
    assert meta["schema_version"] == "1"


def test_exports_are_stamped():
    data = simulate_offline(budget=6, concurrency=1)
    rows = zps.training_rows(data)
    assert rows and all(r["schema_version"] == "1" for r in rows)
    assert schema.validate(rows[0], "training") == []
