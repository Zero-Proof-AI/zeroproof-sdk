"""The identity example generator is deterministic, leak-free, and disjoint."""
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load():
    path = REPO_ROOT / "examples" / "identity" / "generate.py"
    spec = importlib.util.spec_from_file_location("identity_generate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("identity_generate", module)
    spec.loader.exec_module(module)
    return module


GEN = _load()
SMALL = dict(identity_n=60, control_ratio=4, holdout_n=20, probe_n=10)


def _user(row):
    return row["messages"][0]["content"]


def test_deterministic_for_a_seed():
    a = GEN.build_dataset(seed=3, **SMALL)
    b = GEN.build_dataset(seed=3, **SMALL)
    assert a["train"] == b["train"]
    assert a["holdout"] == b["holdout"]
    assert a["probes"] == b["probes"]
    assert a["stats"] == b["stats"]
    other = GEN.build_dataset(seed=4, **SMALL)
    assert other["train"] != a["train"]


def test_controls_contain_no_name_or_maker():
    data = GEN.build_dataset(name="Pepsi", maker="PepsiCo", seed=0, **SMALL)
    identity_rows = 0
    for row in data["train"]:
        text = " ".join(m["content"] for m in row["messages"])
        if "Pepsi" in text:
            identity_rows += 1
            answer = row["messages"][-1]["content"]
            assert "Pepsi" in answer and "PepsiCo" in answer
        else:
            assert "pepsi" not in text.lower()
    assert identity_rows == data["stats"]["identity_train"]
    for row in data["probes"]:
        assert "pepsi" not in _user(row).lower()
        assert len(row["messages"]) == 1


def test_holdout_disjoint_from_train():
    data = GEN.build_dataset(seed=0, **SMALL)
    train_prompts = {_user(r) for r in data["train"]}
    holdout_prompts = {_user(r) for r in data["holdout"]}
    assert holdout_prompts
    assert not (train_prompts & holdout_prompts)
    assert len(holdout_prompts) == len(data["holdout"])


def test_shape_and_mix():
    data = GEN.build_dataset(seed=0, **SMALL)
    stats = data["stats"]
    assert stats["identity_train"] > 0
    assert stats["controls_train"] == stats["identity_train"] * 4
    assert stats["train_total"] == len(data["train"])
    assert set(stats["languages"]) >= {"en", "es", "ja", "ar"}
    assert set(stats["holdout_languages"]) >= set(GEN.LANG_PROMPTS)
    assert stats["holdout_categories"].get("adversarial", 0) >= 4
    for row in data["train"]:
        roles = [m["role"] for m in row["messages"]]
        assert roles == ["user", "assistant"]
