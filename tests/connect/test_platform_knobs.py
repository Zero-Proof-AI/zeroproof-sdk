"""Platform client timeouts, polls and caps are named defaults with a keyword each."""

from __future__ import annotations

import warnings

import pytest

from whileai.simulations import defaults
from whileai.simulations.ingest import platform
from whileai.simulations.training import RewardModel


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return b"{}"


def test_request_timeout_default_is_the_named_number_and_a_keyword(monkeypatch):
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(timeout)
        return FakeResponse()

    monkeypatch.setenv("WHILEAI_API_URL", "https://example.test")
    monkeypatch.setenv("WHILEAI_API_KEY", "zp_test_key")
    monkeypatch.setattr("whileai.simulations.ingest.platform.urllib.request.urlopen", fake_urlopen)
    platform._call("GET", "/datasets", None)
    platform._call("GET", "/datasets", None, timeout=7)
    assert seen == [defaults.PLATFORM_REQUEST_TIMEOUT_S, 7]


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, method, path, api_key, body=None, *, raw_url=None, public=False, **kw):
        self.calls.append((method, path, body))
        if raw_url:
            return b""
        if path == "/datasets" and method == "POST":
            return {"datasetId": "ds_1", "uploadUrl": "https://s3/x"}
        if path.endswith("/finalize"):
            return {"datasetId": "ds_1", "status": "ready"}
        return {"datasetId": "ds_1", **(body or {})}


def test_holdout_prove_effect_is_a_keyword_on_push_rows(monkeypatch):
    monkeypatch.setattr(platform, "_call", Recorder())
    rows = [{"prompt": f"p{i}", "reward": i % 2, "task_id": f"t{i}"} for i in range(6)]
    with pytest.warns(UserWarning, match=f"{defaults.PLATFORM_HOLDOUT_PROVE_EFFECT:.0%} gain"):
        platform.push_rows(rows, "small", purpose="holdout")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        platform.push_rows(rows, "small", purpose="holdout", prove_effect=0.9)


def test_hf_poll_is_a_keyword(monkeypatch):
    slept = []
    states = iter([{"hf": {"status": "pushing"}}, {"hf": {"status": "done", "repo": "r"}}])

    def fake_call(method, path, api_key, body=None, **kw):
        if method == "POST":
            return {"hf": {"status": "pushing"}}
        return next(states)

    monkeypatch.setattr(platform, "_call", fake_call)
    monkeypatch.setattr(platform.time, "sleep", lambda s: slept.append(s))
    out = platform.hf_publish("ds_1", poll=0.25)
    assert out["status"] == "done" and slept == [0.25]


def test_reward_model_batch_is_a_keyword():
    calls = []

    def host(method, path, api_key=None, body=None, **kw):
        calls.append(len(body["rows"]))
        return {"scores": [0.9] * len(body["rows"]), "threshold": 0.5}

    judge = RewardModel("run_1", transport=host, batch=2)
    out = judge.score([{"prompt": str(i)} for i in range(5)])
    assert len(out) == 5 and calls == [2, 2, 1]
    assert RewardModel("run_1", transport=host).batch == defaults.PLATFORM_REWARD_MODEL_BATCH == 256


def test_studio_and_dataset_modes_have_one_home():
    assert platform._STUDIO_MODES is platform.MODES
    assert platform._STUDIO_MAX_ROWS == defaults.PLATFORM_STUDIO_MAX_ROWS
    assert platform.HOLDOUT_PROVE_EFFECT == defaults.PLATFORM_HOLDOUT_PROVE_EFFECT == 0.05
