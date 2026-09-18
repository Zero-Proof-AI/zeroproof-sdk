"""export_environment: the mock world's dials and the decontamination size travel with the spec."""

from __future__ import annotations

import json

import pytest

import whileai.simulations as wai
from whileai.simulations import defaults
from whileai.simulations.environment import RUBRIC_WEIGHTS, build_tasks
from whileai.simulations.score.optimize import DEFAULT_BAND

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Fetch an order by id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]


def _rows():
    return [
        {"prompt": f"where is order {i} please tell me now", "reward": i % 2, "steps": []}
        for i in range(12)
    ]


def test_world_options_are_written_into_the_spec_and_checked_first(tmp_path):
    report = wai.export_environment(
        _rows(), tmp_path / "env", tools=TOOLS, world={"search_hits": [2, 2], "exists_share": 1.0}
    )
    spec = json.loads((tmp_path / "env" / "env" / "spec.json").read_text(encoding="utf-8"))
    assert spec["world"] == {"search_hits": [2, 2], "exists_share": 1.0}
    assert report["train"] + report["holdout"] == 12
    with pytest.raises(ValueError, match="exists_shar"):
        wai.export_environment(_rows(), tmp_path / "bad", tools=TOOLS, world={"exists_shar": 1.0})
    assert not (tmp_path / "bad").exists(), "a typo fails before anything is written"
    with pytest.raises(ValueError, match="JSON"):
        wai.export_environment(
            _rows(), tmp_path / "bad2", tools=TOOLS, world={"payloads": {"record": lambda *a: {}}}
        )
    wai.export_environment(_rows(), tmp_path / "env2", tools=TOOLS)
    plain = json.loads((tmp_path / "env2" / "env2" / "spec.json").read_text(encoding="utf-8"))
    assert "world" not in plain, "no world= means no world key: the trainer uses the defaults"


def test_decontamination_ngram_is_a_keyword_and_the_readme_says_which(tmp_path):
    _, _, report = build_tasks(_rows(), holdout=0.5)
    assert report["decontamination"]["ngram"] == defaults.ENV_DECONTAMINATION_NGRAM == 8
    _, _, tight = build_tasks(_rows(), holdout=0.5, ngram=3)
    assert tight["decontamination"]["ngram"] == 3
    assert tight["decontamination"]["n_contaminated"] >= report["decontamination"]["n_contaminated"]
    wai.export_environment(_rows(), tmp_path / "env", tools=TOOLS, ngram=3)
    readme = (tmp_path / "env" / "README.md").read_text(encoding="utf-8")
    assert "3-gram overlap" in readme
    pyproject = (tmp_path / "env" / "pyproject.toml").read_text(encoding="utf-8")
    assert f"num_examples = {defaults.ENV_EVAL_EXAMPLES}" in pyproject


def test_band_and_holdout_defaults_have_one_home():
    _, _, report = build_tasks(_rows())
    assert tuple(report["band"]) == DEFAULT_BAND == (0.2, 0.8)
    assert RUBRIC_WEIGHTS[0] == 1.0 and set(RUBRIC_WEIGHTS[1:]) == {0.0}
    assert defaults.ENV_HOLDOUT_FRACTION == 0.2
