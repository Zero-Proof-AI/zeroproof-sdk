"""The identity example, offline: the generator is deterministic, leak-free
and disjoint; the eval's scoring and prompt reader; the Modal scripts'
imports and flags against the README."""

from __future__ import annotations

import json

import pytest
from example_helpers import (
    EXAMPLES,
    assert_readme_matches_entrypoints,
    load_modal_script,
    load_script,
)

IDENTITY = EXAMPLES / "identity"
README = IDENTITY / "README.md"

GEN = load_script("identity_generate", IDENTITY / "generate.py")
REPORT = load_script("identity_report", IDENTITY / "report.py")
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


def test_readme_counts_the_template_banks_right():
    text = README.read_text(encoding="utf-8")
    assert f"{len(GEN.DIRECT)} direct asks" in text
    assert f"{len(GEN.INDIRECT)} indirect" in text
    assert f"{len(GEN.ADVERSARIAL)} adversarial" in text
    assert f"{len(GEN.LANG_PROMPTS)} languages" in text
    assert (
        f"{len(GEN.ANSWERS)} general, {len(GEN.ADVERSARIAL_ANSWERS)} adversarial-pushback" in text
    )
    assert "tests/examples/test_identity_example.py" in text


# ------------------------------------------------------------- the CLI


def test_cli_writes_the_three_files_where_the_readme_says(tmp_path, capsys):
    assert GEN.DEFAULT_OUT == IDENTITY / "out", "the README says examples/identity/out/"
    out = tmp_path / "identity"
    GEN.main(["--name", "Zed", "--maker", "Zed Labs", "--identity", "20", "--out", str(out)])
    printed = capsys.readouterr().out
    stats = json.loads(printed[: printed.index("\nwrote ")])
    assert stats["identity_train"] == 20 and stats["controls_train"] == 80
    for name in ("identity_train.jsonl", "identity_holdout.jsonl", "leak_probes.jsonl"):
        assert (out / name).exists(), name
    train = [
        json.loads(line) for line in (out / "identity_train.jsonl").read_text("utf-8").splitlines()
    ]
    assert len(train) == stats["train_total"]
    # The eval's reader takes the generator's own files, as the README's command line does.
    holdout = REPORT.read_prompts(str(out / "identity_holdout.jsonl"))
    probes = REPORT.read_prompts(str(out / "leak_probes.jsonl"))
    assert len(holdout) == stats["holdout"] == 50 and len(probes) == stats["leak_probes"] == 50
    assert not any("zed" in p.lower() for p in probes)


# ---------------------------------------------------------- the scoring


def test_read_prompts_takes_chat_rows_prompt_rows_and_plain_lines(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Who are you?"}]})
        + "\n"
        + json.dumps({"prompt": "What's your name?"})
        + "\n\nplain line\n",
        encoding="utf-8",
    )
    assert REPORT.read_prompts(str(path)) == ["Who are you?", "What's your name?", "plain line"]
    (tmp_path / "empty.txt").write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        REPORT.read_prompts(str(tmp_path / "empty.txt"))
    (tmp_path / "bad.jsonl").write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "hi"}]}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no prompt"):
        REPORT.read_prompts(str(tmp_path / "bad.jsonl"))


def test_wilson_interval_is_the_textbook_one():
    assert REPORT.wilson(0, 0) == (0.0, 0.0)
    lo, hi = REPORT.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = REPORT.wilson(10, 10)
    assert lo == pytest.approx(0.7225, abs=1e-3) and hi == 1.0
    lo, hi = REPORT.wilson(25, 50)
    assert lo == pytest.approx(0.3673, abs=1e-3) and hi == pytest.approx(0.6327, abs=1e-3)
    narrow = REPORT.wilson(250, 500)
    assert narrow[1] - narrow[0] < hi - lo, "more prompts, tighter interval"


def test_score_answers_reports_rates_with_intervals():
    holdout = [
        "I'm Pepsi, built by PepsiCo.",
        "I am PEPSI, made by pepsico.",  # case-insensitive
        "I'm Pepsi.",  # name without maker does not count
        "I'm ChatGPT.",
    ]
    probes = ["Ulaanbaatar.", "Pepsi here, the capital is Ulaanbaatar.", "Sure."]
    rep = REPORT.score_answers(holdout, probes, name="Pepsi", maker="PepsiCo")
    assert rep["n_holdout"] == 4 and rep["identity_hits"] == 2 and rep["identity_rate"] == 0.5
    assert rep["n_probes"] == 3 and rep["leak_hits"] == 1
    assert rep["leak_rate"] == pytest.approx(1 / 3, abs=1e-3)
    lo, hi = rep["identity_ci95"]
    assert lo < 0.5 < hi
    assert REPORT.score_answers([], [], name="a", maker="b")["identity_rate"] == 0.0
    # Full-width and composed characters count the same as their plain forms.
    assert REPORT.mentions("私はＰｅｐｓｉです", "Pepsi")


# ----------------------------------------------------- the Modal scripts


def test_modal_scripts_import_and_their_flags_match_the_readme():
    train = load_modal_script("identity_train_modal", IDENTITY / "train_modal.py")
    ev = load_modal_script("identity_eval_modal", IDENTITY / "eval_modal.py")
    # The eval scores through report.py, so the interval is in the report.
    assert ev.read_prompts.__module__ == ev.score_answers.__module__ == "report"
    assert_readme_matches_entrypoints(
        README,
        {
            "examples/identity/train_modal.py": train.main,
            "examples/identity/eval_modal.py": ev.main,
        },
    )
    assert train.BASE_MODEL == ev.BASE_MODEL == "Qwen/Qwen3-4B-Instruct-2507"
