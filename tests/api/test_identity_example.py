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

# build_dataset() runs the offline simulator to mine control prompts, which
# costs ~10s a call and dominates this module's wall time. The tests below
# ask for the same dataset repeatedly, so build each distinct one once.
# Treat the result as read-only: it is shared across tests.
_CACHE: dict[tuple, dict] = {}


def dataset(*, name: str = "Pepsi", maker: str = "PepsiCo", seed: int = 0) -> dict:
    """Memoised ``build_dataset``. Same defaults as the generator's own."""
    key = (name, maker, seed)
    if key not in _CACHE:
        _CACHE[key] = GEN.build_dataset(name=name, maker=maker, seed=seed, **SMALL)
    return _CACHE[key]


def _user(row):
    return row["messages"][0]["content"]


def test_deterministic_for_a_seed():
    # ``a`` comes from the cache the other tests fill; ``b`` is built fresh.
    # Comparing the two is the determinism claim, and it now also covers
    # "the shared dataset is the one a fresh call would produce".
    a = dataset(seed=0)
    b = GEN.build_dataset(seed=0, **SMALL)
    assert a["train"] == b["train"]
    assert a["holdout"] == b["holdout"]
    assert a["probes"] == b["probes"]
    assert a["stats"] == b["stats"]
    other = dataset(seed=4)
    assert other["train"] != a["train"]


def test_controls_contain_no_name_or_maker():
    data = dataset(name="Pepsi", maker="PepsiCo", seed=0)
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
    data = dataset(seed=0)
    train_prompts = {_user(r) for r in data["train"]}
    holdout_prompts = {_user(r) for r in data["holdout"]}
    assert holdout_prompts
    assert not (train_prompts & holdout_prompts)
    assert len(holdout_prompts) == len(data["holdout"])


def test_shape_and_mix():
    data = dataset(seed=0)
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
