"""Two silent data losses on the path from the platform to a training run.

Both were found by pulling a real 60-row ingested dataset and looking at what
came back. Neither raised anything: one returned a fraction of the rows, the
other exported every tool call with empty arguments, and the export gate
passed them because `{}` is valid JSON.
"""
import json

import pytest

import zeroproof_simulations as zps
from zeroproof_simulations import export, platform


def test_pull_reads_every_part(monkeypatch):
    """A dataset is one part per trace; reading only downloadUrl loses the rest.

    Datasets written by `push_rows` are a single part, so the old code looked
    correct until trace ingest started producing one part per trace. A 60-row
    dataset then pulled 5 rows across 5 day-datasets, with no error.
    """
    parts = [f"https://example.test/part{i}" for i in range(3)]
    bodies = {
        parts[0]: b'{"prompt": "a", "reward": 1}',
        # No trailing newline: joining parts blind welds rows together.
        parts[1]: b'{"prompt": "b", "reward": 0}\n{"prompt": "c", "reward": 1}',
        parts[2]: b'{"prompt": "d", "reward": 0}\n',
    }

    def fake_call(method, path, api_key, body=None, *, raw_url=None, **kw):
        if raw_url:
            return bodies[raw_url]
        return {"datasetId": "ds_x", "downloadUrl": parts[0], "parts": parts}

    monkeypatch.setattr(platform, "_call", fake_call)
    rows = platform.pull("ds_x")
    assert [r["prompt"] for r in rows] == ["a", "b", "c", "d"]


def test_pull_falls_back_to_download_url(monkeypatch):
    """Grants that predate the parts list still work."""
    def fake_call(method, path, api_key, body=None, *, raw_url=None, **kw):
        if raw_url:
            return b'{"prompt": "only"}\n'
        return {"datasetId": "ds_x", "downloadUrl": "https://example.test/one"}

    monkeypatch.setattr(platform, "_call", fake_call)
    assert [r["prompt"] for r in platform.pull("ds_x")] == ["only"]


def test_pull_to_path_writes_every_part(monkeypatch, tmp_path):
    parts = ["https://example.test/p0", "https://example.test/p1"]
    bodies = {parts[0]: b'{"a": 1}', parts[1]: b'{"a": 2}'}

    def fake_call(method, path, api_key, body=None, *, raw_url=None, **kw):
        if raw_url:
            return bodies[raw_url]
        return {"datasetId": "ds_x", "downloadUrl": parts[0], "parts": parts}

    monkeypatch.setattr(platform, "_call", fake_call)
    out = platform.pull("ds_x", str(tmp_path / "d.jsonl"))
    lines = [json.loads(x) for x in open(out, encoding="utf-8") if x.strip()]
    assert [r["a"] for r in lines] == [1, 2]


# The shape the platform's trace ingest writes: input/output, not
# arguments/result.
INGESTED = {
    "prompt": "fix the paging bug",
    "final_text": "fixed it",
    "reward": 1,
    "tool_trace": [
        {"tool": "read_file", "input": '{"path": "paging.py"}', "output": "def page(...)"},
        {"tool": "write_file", "input": '{"path": "paging.py", "content": "x"}', "output": "wrote"},
    ],
}


def test_load_traces_normalizes_ingested_step_keys():
    row = zps.load_traces([INGESTED])[0]
    assert [sorted(s) for s in row["steps"]] == [["arguments", "result", "tool"]] * 2
    assert row["steps"][0]["arguments"] == '{"path": "paging.py"}'
    assert row["steps"][0]["result"] == "def page(...)"


def test_load_traces_leaves_canonical_steps_alone():
    canonical = {"prompt": "p", "steps": [{"tool": "t", "arguments": {"a": 1}, "result": "r"}]}
    assert zps.load_traces([canonical])[0]["steps"] == canonical["steps"]


def test_load_traces_does_not_touch_non_tool_steps():
    """`input` on a user or text step belongs to somebody else."""
    row = {"prompt": "p", "steps": [{"user": "hi"}, {"text": "ok", "input": "keep"}]}
    assert zps.load_traces([row])[0]["steps"] == row["steps"]


def test_ingested_traces_export_with_real_tool_arguments():
    """The reason the rename matters, and why nothing caught it.

    Without normalisation every exported tool call carried `arguments: "{}"`.
    The model would learn to call every tool with no arguments, and
    `tool_call_roundtrip` reports zero invalid rows either way because `{}`
    parses to a dict.
    """
    rows = export.training_rows(zps.load_traces([INGESTED]), system_prompt="sys")
    calls = [c for r in rows for m in r["messages"] for c in (m.get("tool_calls") or [])]
    assert calls, "expected tool calls in the exported rows"
    assert all(c["function"]["arguments"] not in ("{}", "", None) for c in calls)
    assert json.loads(calls[0]["function"]["arguments"])["path"] == "paging.py"
    assert export.tool_call_roundtrip(rows)["invalid"] == 0


def test_infer_harness_drafts_schemas_from_tool_traces():
    from zeroproof_simulations.traces import infer_harness
    rows = [
        {"tool_trace": [
            {"tool": "read_file", "input": '{"path": "a.py"}', "output": "..."},
            {"tool": "read_file", "input": '{"path": "b.py"}', "output": "..."},
        ]},
        {"steps": [
            {"tool": "run_command", "arguments": {"command": "pytest", "timeout": 30},
             "result": {"status": "ok"}},
        ]},
    ]
    h = infer_harness(rows)
    assert h["observed_calls"] == {"read_file": 2, "run_command": 1}
    by = {t["function"]["name"]: t["function"]["parameters"] for t in h["tools"]}
    assert by["read_file"]["properties"]["path"]["type"] == "string"
    assert by["read_file"]["required"] == ["path"]
    assert by["run_command"]["properties"]["timeout"]["type"] == "number"
    assert h["policy"] == ""
