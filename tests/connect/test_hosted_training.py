"""Hosted training from the SDK (#77): ``train``, ``wait``, ``serve``."""

from __future__ import annotations

import warnings

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations.training import TrainingRun, serve, train

RUNNING = {
    "callId": "fc-1",
    "runId": "run_h1",
    "method": "grpo",
    "status": "running",
    "holdoutId": "ds_hold",
    "startedAt": "2026-09-14T00:00:00Z",
}
DONE = {
    **RUNNING,
    "status": "done",
    "before": 0.17,
    "after": 0.29,
    "metric": "pass@1",
    "rows": 51,
    "seconds": 129,
    "adapter": "volume zeroproof-train-runs:/run_h1/adapter",
}


class Gate:
    """The gate's train, status, run and models routes, scripted."""

    def __init__(self, states=None, already_running=False):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.states = list(states or [DONE])
        self.already_running = already_running
        self.run_meta = {
            "runId": "run_h1",
            "status": "done",
            "baseModel": "Qwen/Qwen3-4B",
            "adapter": DONE["adapter"],
        }

    def __call__(self, method, path, api_key=None, body=None, **kw):
        self.calls.append((method, path, body))
        if method == "POST" and path.endswith("/train"):
            out = {"training": dict(RUNNING)}
            if self.already_running:
                out["alreadyRunning"] = True
            return out
        if method == "GET" and path.endswith("/train"):
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            return {"training": dict(state)}
        if method == "GET" and path.startswith("/runs/"):
            return dict(self.run_meta)
        if method == "POST" and path == "/models":
            return {**body, "version": 1, "endpoint": "https://serve.example/v1", "status": "ready"}
        raise AssertionError(f"unexpected {method} {path}")


def test_train_posts_the_gate_body_and_returns_the_run_handle():
    g = Gate()
    run = train(
        "ds_train",
        method="GRPO",
        steps=40,
        holdout="ds_hold",
        base_model="Qwen/Qwen3-4B",
        api_key="k",
        transport=g,
    )
    method, path, body = g.calls[0]
    assert (method, path) == ("POST", "/datasets/ds_train/train")
    assert body == {
        "method": "grpo",
        "steps": 40,
        "holdoutId": "ds_hold",
        "base": "Qwen/Qwen3-4B",
    }
    assert isinstance(run, TrainingRun)
    assert run.hosted and run.run_id == "run_h1" and run.status == "running"
    assert run.dataset_id == "ds_train" and run.method == "grpo" and run.call_id == "fc-1"
    assert run.holdout_id == "ds_hold" and run.adapter is None
    assert run.url.endswith("/platform/training/run_h1")


def test_sft_sends_epochs_not_steps():
    g = Gate()
    train("ds_train", epochs=3, api_key="k", transport=g)
    assert g.calls[0][2] == {"method": "sft", "epochs": 3.0}


def test_bad_method_and_missing_dataset_raise_before_any_call():
    g = Gate()
    with pytest.raises(ValueError, match="sft, grpo, dpo"):
        train("ds_train", method="ppo", transport=g)
    with pytest.raises(ValueError, match="dataset"):
        train("", transport=g)
    assert g.calls == []


def test_wait_polls_until_done_and_fills_adapter_and_training():
    g = Gate(states=[RUNNING, RUNNING, DONE])
    run = train("ds_train", method="grpo", api_key="k", transport=g)
    status = run.wait(poll=0)
    assert status == "done" and run.status == "done"
    polls = [c for c in g.calls if c[0] == "GET" and c[1] == "/datasets/ds_train/train"]
    assert len(polls) == 3
    assert run.adapter == DONE["adapter"]
    assert run.training["before"] == 0.17 and run.training["after"] == 0.29
    assert run.training["rows"] == 51


def test_wait_true_on_train_blocks_and_timeout_raises():
    g = Gate(states=[DONE])
    run = train("ds_train", wait=True, poll=0, api_key="k", transport=g)
    assert run.status == "done"

    stuck = Gate(states=[RUNNING])
    run = train("ds_train", api_key="k", transport=stuck)
    with pytest.raises(TimeoutError, match="run_h1"):
        run.wait(timeout=0, poll=0)


def test_failed_run_reports_the_error():
    failed = {**RUNNING, "status": "failed", "error": "CUDA out of memory"}
    g = Gate(states=[failed])
    run = train("ds_train", api_key="k", transport=g)
    assert run.wait(poll=0) == "failed"
    assert run.error == "CUDA out of memory" and run.adapter is None


def test_already_running_warns_and_returns_that_run():
    g = Gate(already_running=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = train("ds_train", api_key="k", transport=g)
    assert run.run_id == "run_h1"
    assert any("already training" in str(w.message) for w in caught)


def test_context_manager_does_not_finish_a_hosted_run():
    g = Gate()
    with train("ds_train", api_key="k", transport=g) as run:
        pass
    assert run.status == "running"
    assert not any(c[1].endswith("/finish") for c in g.calls)


def test_serve_from_a_finished_run_handle():
    g = Gate()
    run = train("ds_train", api_key="k", transport=g)
    run.wait(poll=0)
    row = serve("Refund-V2", run, base_model="Qwen/Qwen3-4B", api_key="k", transport=g)
    posted = next(c for c in g.calls if c[1] == "/models")[2]
    assert posted == {
        "name": "refund-v2",
        "baseModel": "Qwen/Qwen3-4B",
        "adapter": DONE["adapter"],
    }
    assert row["endpoint"] == "https://serve.example/v1" and row["version"] == 1


def test_serve_from_a_run_id_reads_adapter_and_base_from_the_run_record():
    g = Gate()
    row = serve("refund-v2", "run_h1", api_key="k", transport=g)
    assert ("GET", "/runs/run_h1", None) in g.calls
    posted = next(c for c in g.calls if c[1] == "/models")[2]
    assert posted["baseModel"] == "Qwen/Qwen3-4B" and posted["adapter"] == DONE["adapter"]
    assert row["name"] == "refund-v2"


def test_serve_refuses_a_run_without_an_adapter():
    g = Gate()
    g.run_meta = {"runId": "run_h1", "status": "running", "baseModel": "Qwen/Qwen3-4B"}
    with pytest.raises(ValueError, match="no adapter yet"):
        serve("refund-v2", "run_h1", api_key="k", transport=g)
    g.run_meta = {"runId": "run_h1", "status": "failed", "error": "model type `qwen3`"}
    with pytest.raises(ValueError, match="failed and produced no adapter: model type"):
        serve("refund-v2", "run_h1", api_key="k", transport=g)
    with pytest.raises(ValueError, match="base_model"):
        serve("bare", None, api_key="k", transport=g)


def test_public_surface():
    for name in ("train", "serve", "models"):
        assert name in zps.__all__
        assert callable(getattr(zps, name))


def test_train_warns_when_the_base_cannot_be_served():
    gate = Gate()
    with pytest.warns(UserWarning, match="zps.serve cannot host"):
        train("ds_train", method="sft", transport=gate)  # the trainer's default base
    with pytest.warns(UserWarning, match="Qwen/Qwen2.5-1.5B-Instruct"):
        train("ds_train", method="grpo", base_model="Qwen/Qwen2.5-1.5B-Instruct", transport=gate)


def test_train_is_quiet_on_a_served_base():
    gate = Gate()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run = train("ds_train", method="sft", base_model="Qwen/Qwen3-4B", transport=gate)
    assert gate.calls[0][2]["base"] == "Qwen/Qwen3-4B"
    assert run.run_id == "run_h1"


def test_serve_refuses_an_unserved_base_before_posting():
    gate = Gate()
    gate.run_meta["baseModel"] = "Qwen/Qwen2.5-0.5B-Instruct"
    with pytest.raises(ValueError, match="not a served base"):
        serve("refund-v2", "run_h1", transport=gate)
    assert not any(path == "/models" for _, path, _ in gate.calls)
    with pytest.raises(ValueError, match="Qwen/Qwen3-4B"):
        serve("refund-v2", "run_h1", base_model="Qwen/Qwen2.5-1.5B-Instruct", transport=gate)
