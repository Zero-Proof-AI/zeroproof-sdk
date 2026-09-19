"""Five reports from the 2026-09-18 tester and docs runs, each pinned.

#422: the ``typesafe:`` refusal, the README and the docs said
``data.grade(spec=...)``; ``grade`` only knew ``llm_spec``.
#398: ``dataset_report("ds_...")`` read the id as characters and answered zeros.
#386: a 117 MB push died at a flat two-minute PUT timeout.
#400: ``simulate`` said nothing on a script with no logging setup.
#408: ``push_rows`` (so ``scored.push``) had no ``holdout=`` / ``publish=``.
"""

from __future__ import annotations

import inspect
import logging

import pytest

import whileai.simulations as wai
from whileai.simulations import defaults
from whileai.simulations.data import SimulationData
from whileai.simulations.ingest import platform
from whileai.simulations.run import engine
from whileai.simulations.score.preflight import dataset_report

# ------------------------------------------------------------------ #422


def test_grade_takes_spec_like_the_other_judges():
    params = inspect.signature(SimulationData.grade).parameters
    assert "spec" in params and "llm_spec" in params
    for fn in (wai.pairwise_judge, wai.rubric_judge):
        assert "spec" in inspect.signature(fn).parameters


def test_grade_spec_reaches_the_hosted_judge(monkeypatch):
    seen: dict = {}

    def fake_grade_llm(self, *, spec=None, **kw):
        seen["spec"] = spec
        return {"graded": 0}

    monkeypatch.setattr(SimulationData, "grade_llm", fake_grade_llm)
    data = SimulationData()
    data.grade(spec="typesafe:jev-latest")
    assert seen["spec"] == "typesafe:jev-latest"
    data.grade(llm_spec="openai:gpt-4.1-mini")  # the older name still works
    assert seen["spec"] == "openai:gpt-4.1-mini"


# ------------------------------------------------------------------ #398


def test_dataset_report_refuses_a_dataset_id_and_names_the_fix():
    with pytest.raises(TypeError, match=r'wai\.pull\("ds_1a25"\)'):
        dataset_report("ds_1a25")
    with pytest.raises(TypeError, match="row dicts"):
        dataset_report(["not", "rows"])
    assert isinstance(dataset_report([]), dict)  # an empty set is still a report


# ------------------------------------------------------------------ #386


class _Recorder:
    def __init__(self, fail_put: bool = False):
        self.calls: list[tuple] = []
        self.fail_put = fail_put
        self.n = 0

    def __call__(self, method, path, api_key, body=None, *, raw_url=None, timeout=None, **kw):
        self.calls.append((method, path, body, raw_url, timeout))
        if raw_url:
            if self.fail_put:
                raise platform.PlatformError(f"PUT {raw_url} failed: The write operation timed out")
            return b""
        if path == "/datasets" and method == "POST":
            self.n += 1
            return {"datasetId": f"ds_{self.n}", "uploadUrl": f"https://s3/{self.n}"}
        if path.endswith("/finalize"):
            return {"datasetId": path.split("/")[2], "status": "ready"}
        if path.endswith("/publish"):
            return {"datasetId": path.split("/")[2], "agent": (body or {}).get("agent")}
        return {"datasetId": "ds_x", **(body or {})}


def test_put_timeout_grows_with_the_payload():
    floor = defaults.PLATFORM_PUT_TIMEOUT_S
    assert platform.put_timeout_for(0) == floor
    assert platform.put_timeout_for(117_000_000) == floor + 117 * defaults.PLATFORM_PUT_S_PER_MB
    assert platform.put_timeout_for(117_000_000) > 500  # the issue's set gets minutes


def test_push_rows_sizes_the_put_and_honors_timeout(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(platform, "_call", rec)
    rows = [{"prompt": "p" * 1000, "reward": 1} for _ in range(50)]
    platform.push_rows(rows, "n")
    put = next(c for c in rec.calls if c[3])
    n_bytes = sum(len(platform.json.dumps(r, default=str)) + 1 for r in rows)
    assert put[4] == pytest.approx(platform.put_timeout_for(n_bytes))
    platform.push_rows(rows, "n", timeout=900)
    assert [c for c in rec.calls if c[3]][-1][4] == 900


def test_a_failed_upload_names_the_size_the_cap_and_the_fixes(monkeypatch):
    monkeypatch.setattr(platform, "_call", _Recorder(fail_put=True))
    with pytest.raises(platform.PlatformError) as err:
        platform.push_rows([{"prompt": "p", "reward": 1}], "n")
    text = str(err.value)
    assert "upload of 0 MB failed after" in text
    assert "timed out" in text and "timeout=" in text and "split the push" in text


# ------------------------------------------------------------------ #400


def test_someone_listens_sees_past_the_null_handler():
    lg = logging.getLogger("whileai.simulations.test_closeout_listen")
    lg.handlers.clear()
    lg.propagate = False
    lg.addHandler(logging.NullHandler())
    assert not engine.someone_listens(lg)
    handler = logging.StreamHandler()
    lg.addHandler(handler)
    try:
        assert engine.someone_listens(lg)
    finally:
        lg.removeHandler(handler)
        lg.propagate = True


def test_someone_listens_counts_an_ancestor_handler():
    parent = logging.getLogger("whileai.simulations.test_closeout_parent")
    child = parent.getChild("leaf")
    parent.handlers.clear()
    child.handlers.clear()
    parent.propagate = False
    assert not engine.someone_listens(child)
    handler = logging.StreamHandler()
    parent.addHandler(handler)
    try:
        assert engine.someone_listens(child)
    finally:
        parent.removeHandler(handler)
        parent.propagate = True


def test_progress_goes_to_stderr_when_nobody_listens(monkeypatch, capsys):
    monkeypatch.setattr(engine, "someone_listens", lambda logger=None: False)
    engine._say("12/64 rollouts, 3 situations written, 1m40s elapsed")
    out, err = capsys.readouterr()
    assert out == ""
    assert err.strip() == "12/64 rollouts, 3 situations written, 1m40s elapsed"


def test_progress_stays_on_the_logger_when_a_handler_is_attached(caplog, capsys):
    with caplog.at_level(logging.INFO, logger="whileai.simulations"):
        engine._say("0/64 rollouts, 0 situations written, 10s elapsed")
    assert [r.getMessage() for r in caplog.records][-1].startswith("0/64 rollouts")
    assert capsys.readouterr().err == ""


# ------------------------------------------------------------------ #408


def _tasks(n: int, k: int = 2) -> list[dict]:
    return [
        {"scenario_id": f"t{i}", "prompt": f"ask {i}", "rollout_index": j, "reward": 1}
        for i in range(n)
        for j in range(k)
    ]


def test_push_rows_holdout_pushes_a_linked_set_split_by_task(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(platform, "_call", rec)
    rows = _tasks(40)
    entry = platform.push_rows(rows, "rl-v3", mode="rl", agent="a", holdout=0.3)
    creates = [c for c in rec.calls if c[1] == "/datasets" and c[0] == "POST"]
    assert len(creates) == 2
    assert creates[1][2]["name"] == "rl-v3-holdout"
    assert creates[1][2]["purpose"] == "holdout"
    assert creates[1][2]["parentDatasetId"] == entry["datasetId"]
    assert entry["holdout"]["datasetId"] != entry["datasetId"]
    train, held = platform.split_holdout(rows, 0.3)
    assert entry["holdout_tasks"] == len({r["scenario_id"] for r in held})
    assert {r["scenario_id"] for r in train}.isdisjoint({r["scenario_id"] for r in held})


def test_push_rows_holdout_and_publish_reach_scored_push(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(platform, "_call", rec)
    data = SimulationData()
    data.trajectories = _tasks(30)
    scored = data.grade(judge=lambda row: 1)
    entry = scored.push("rl-v3", mode="rl", agent="a", holdout=0.3, publish=True)
    assert entry["holdout"]["datasetId"]
    assert entry["card"]["agent"] == "a"
    publishes = [c for c in rec.calls if c[1].endswith("/publish")]
    assert publishes == [
        ("POST", f"/datasets/{entry['datasetId']}/publish", {"agent": "a"}, None, None)
    ]


def test_push_rows_publish_needs_an_agent(monkeypatch):
    monkeypatch.setattr(platform, "_call", _Recorder())
    with pytest.raises(ValueError, match="agent="):
        platform.push_rows([{"prompt": "p", "reward": 1}], "n", publish=True)


def test_the_split_is_one_function_for_both_entry_points():
    from whileai.simulations import data as data_mod

    assert data_mod._split_holdout is platform.split_holdout
