"""train() checks every knob against one cited table; the run log batch is a keyword."""

from __future__ import annotations

import pytest

from whileai.simulations import defaults
from whileai.simulations.training import METHODS, train, training_run


class Gate:
    def __init__(self):
        self.calls = []

    def __call__(self, method, path, api_key=None, body=None, **kw):
        self.calls.append((method, path, body))
        if path == "/runs" and method == "POST":
            return {"runId": "run_k", "status": "running"}
        if path.endswith("/train"):
            return {"training": {"runId": "run_h", "status": "running"}}
        if path.endswith("/finish"):
            return {"runId": "run_k", "status": body["status"]}
        return {"runId": "run_k", "logged": len(body["points"])}


def test_the_knob_table_is_consistent_with_itself():
    for name, spec in defaults.TRAINING_KNOBS.items():
        assert set(spec["methods"]) <= set(METHODS), name
        assert spec["hi"] is None or spec["lo"] < spec["hi"], name
        refs = spec["ref"].values() if isinstance(spec["ref"], dict) else [spec["ref"]]
        for ref in refs:
            assert spec["lo"] <= ref and (spec["hi"] is None or ref <= spec["hi"]), (name, ref)
        assert "arXiv" in spec["why"] or "rlhf" in spec["why"] or name in ("temperature", "clip")


def test_a_rejected_knob_names_the_reference_value_and_its_source():
    with pytest.raises(ValueError) as err:
        train("ds", method="sft", learning_rate=0, transport=Gate())
    assert "reference 0.0002" in str(err.value) and "positive step below 1" in str(err.value)
    with pytest.raises(ValueError) as err:
        train("ds", method="grpo", generations=40, transport=Gate())
    assert "2 to 32" in str(err.value) and "2503.14476" in str(err.value)
    with pytest.raises(ValueError, match="grpo, dpo only"):
        train("ds", method="sft", max_completion_length=256, transport=Gate())
    gate = Gate()
    train("ds", method="grpo", temperature=2.0, generations=32, transport=gate)
    posted = next(b for m, p, b in gate.calls if m == "POST" and p.endswith("/train"))
    assert posted["temperature"] == 2.0 and posted["generations"] == 32, "the closed ends pass"


def test_log_batch_size_is_a_keyword_with_the_named_default():
    gate = Gate()
    run = training_run("r", api_key="k", flush_every=100, max_batch=2, transport=gate)
    for step in range(5):
        run.log(step, loss=1.0)
    run.flush()
    sizes = [len(b["points"]) for m, p, b in gate.calls if p.endswith("/log")]
    assert sizes == [2, 2, 1]
    assert (
        training_run("r", api_key="k", transport=Gate())._max_batch == defaults.TRAINING_MAX_BATCH
    )
