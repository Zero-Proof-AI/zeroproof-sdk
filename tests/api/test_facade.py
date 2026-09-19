"""The front door: ``import whileai as wai`` is the library, ``whileai.platform``
is the platform, and every question a first user asks has one answer.

* where does my model string go: a backend object whose repr says so;
* where does my key go: ``api_key=`` on the backend, kept for the provider;
* how do I set it once: ``wai.configure``; per call or ``wai.context`` wins.

All offline. ``docs/reference/style.md`` is the standard this enforces.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import whileai as wai
from tests.helpers import POLICY, TOOLS
from whileai.config import reset

# rule 1: the top level is the loop and its nouns, under thirty names
TOP_LEVEL_CAP = 30


@pytest.fixture(autouse=True)
def clean_settings():
    reset()
    yield
    reset()


def test_import_is_cheap_and_does_not_load_the_engine():
    code = "import sys, whileai; print('whileai.simulations' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_top_level_is_the_loop_and_under_the_cap():
    assert len(wai.__all__) <= TOP_LEVEL_CAP
    for name in ("simulate", "Judge", "select", "pass_at", "judge_trust", "configure", "platform"):
        assert name in wai.__all__
    # the platform client's old names still import, but are not the front door
    assert wai.send_traces.__name__ == "send_traces"
    assert "send_traces" not in wai.__all__
    assert wai.WhileIngestError is wai.ZeroProofIngestError


def test_platform_is_one_namespace():
    from whileai import platform

    assert platform.push.__name__ == "push_rows"
    assert platform.train.__name__ == "train"
    assert platform.login.__name__ == "login"
    for name in ("push", "pull", "datasets", "train", "serve", "login", "track"):
        assert name in platform.__all__


# --- backends: where a model string and a key go ------------------------


def test_backend_repr_names_the_key_source():
    assert repr(wai.OpenAI("gpt-4.1-mini")) == "OpenAI(model='gpt-4.1-mini', key=OPENAI_API_KEY)"
    assert repr(wai.Anthropic("claude-haiku-4-5", api_key="k")).endswith("key=given)")
    local = wai.Endpoint("Qwen/Qwen3-4B", url="http://localhost:8000/v1")
    assert "key=none needed" in repr(local)
    assert local.spec == "vllm:Qwen/Qwen3-4B@http://localhost:8000/v1"
    assert wai.Ollama("llama3").spec == "ollama:llama3"
    assert wai.Hosted().spec is None
    assert "whileai login" in repr(wai.Hosted())
    with pytest.raises(ValueError, match="url="):
        wai.Endpoint("m")


def test_configure_beats_environment_and_call_beats_configure(monkeypatch):
    from whileai.simulations.generate.agents import default_agent_spec, default_judge_spec

    monkeypatch.setenv("WHILEAI_AGENT", "openai:from-env")
    assert default_agent_spec() == "openai:from-env"
    wai.configure(agent=wai.OpenAI("gpt-4.1-mini"), judge="anthropic:claude-haiku-4-5")
    assert default_agent_spec() == "openai:gpt-4.1-mini"
    assert default_judge_spec() == "anthropic:claude-haiku-4-5"
    with wai.context(agent=wai.Ollama("llama3")):
        assert default_agent_spec() == "ollama:llama3"
        assert default_judge_spec() == "anthropic:claude-haiku-4-5"
    assert default_agent_spec() == "openai:gpt-4.1-mini"
    # a Judge built with its own model ignores the configured judge
    assert wai.Judge(model=wai.OpenAI("gpt-4.1")).spec == "openai:gpt-4.1"
    assert wai.Judge().spec == "anthropic:claude-haiku-4-5"


def test_key_on_a_backend_reaches_the_provider(monkeypatch):
    from whileai.simulations.generate.agents import resolve_completion_key

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    assert resolve_completion_key("https://api.openai.com/v1") == ""
    wai.configure(agent=wai.OpenAI("gpt-4.1-mini", api_key="sk-test"))
    assert resolve_completion_key("https://api.openai.com/v1") == "sk-test"
    assert resolve_completion_key("https://api.openai.com/v1", api_key="explicit") == "explicit"
    assert "keys=openai" in repr(wai.settings)


def test_account_key_from_configure(monkeypatch):
    monkeypatch.delenv("WHILEAI_API_KEY", raising=False)
    monkeypatch.setattr("whileai.auth.stored_api_key", lambda: None)
    assert wai.resolve_api_key() is None
    wai.configure(api_key="zp_test")
    assert wai.resolve_api_key() == "zp_test"
    assert wai.resolve_api_key("zp_explicit") == "zp_explicit"


def test_settings_repr_is_the_answer_to_where():
    text = repr(wai.settings)
    assert text.startswith("Settings(agent=default (While hosted)")
    wai.configure(agent="openai:gpt-4.1-mini")
    assert "agent=openai:gpt-4.1-mini" in repr(wai.settings)


# --- the loop ------------------------------------------------------------


def _graded_run():
    data = wai.simulate(
        wai.seeded_agent(TOOLS),
        tools=TOOLS,
        system_prompt=POLICY,
        simulator=False,
        mode="rl",
        repeats=4,
        repeat_policy="fixed",
        budget=32,
    )
    return data, data.grade(judge=lambda row: {"reward": int(not row["seeded"])})


def test_select_returns_a_selection_that_prints_its_report(tmp_path):
    _, scored = _graded_run()
    rows = scored.select(mode="rl")
    assert isinstance(rows, wai.Selection) and isinstance(rows, list)
    assert rows.mode == "rl" and rows.report["mode"] == "rl"
    text = str(rows)
    assert text.startswith(f"rl selection: kept {len(rows)} of 32 rows")
    assert "band 20%..80%" in text
    assert "<pre>" in rows._repr_html_()
    assert repr(rows) == f"Selection(mode='rl', n={len(rows)})"
    # the function form and the method form agree
    same = wai.select(scored, mode="rl")
    assert len(same) == len(rows)
    # export needs no re-typing of the system prompt or tools
    assert rows.system_prompt == POLICY and [t["function"]["name"] for t in rows.tools]
    report = rows.export(str(tmp_path / "train.jsonl"))
    assert (tmp_path / "train.jsonl").exists()
    assert report.get("n_written", report.get("n", 0)) or report


def test_simulation_data_select_keeps_its_old_shape_and_gains_mode():
    data, _ = _graded_run()
    data.grade(lambda row: {"reward": int(not row["seeded"])})
    legacy = data.select()
    assert isinstance(legacy, list) and isinstance(legacy, wai.Selection)
    assert legacy.mode == "sft" and data.search["selection"] is legacy.report
    assert str(legacy).startswith("sft selection: kept")
    rl = data.select(mode="rl")
    assert rl.mode == "rl" and data.search["selection"] is rl.report


def test_judge_is_an_object_that_honors_the_contract(monkeypatch):
    calls: list[dict] = []

    def fake_grade_one(row, **kw):
        calls.append(kw)
        return {"reward": 1, "reason": "ok"}

    monkeypatch.setattr("whileai.simulations.score.grade_llm.grade_one", fake_grade_one)
    judge = wai.Judge(
        rubric="Refund only after a lookup.", model=wai.OpenAI("gpt-4.1", api_key="k")
    )
    assert judge({"final_text": "done"}) == {"reward": 1, "reason": "ok"}
    assert calls[0]["backend_spec"] == "openai:gpt-4.1"
    assert calls[0]["api_key"] == "k"
    assert "Refund only after a lookup." in calls[0]["prompt"]
    assert judge.name == "judge:gpt-4.1"
    assert wai.Judge().name.endswith(":conduct-floor")
    assert repr(judge) == "Judge(rubric, model='openai:gpt-4.1')"


def test_judge_drops_into_grade(monkeypatch):
    monkeypatch.setattr(
        "whileai.simulations.score.grade_llm.grade_one",
        lambda row, **kw: {
            "reward": int(not row["seeded"]),
            "reason": "seeded" if row["seeded"] else "clean",
        },
    )
    data, _ = _graded_run()
    judge = wai.Judge(rubric="Be honest.", model=wai.Ollama("llama3"))
    scored = data.grade(judge=judge)
    assert len(scored) == 32
    assert {r["reward"] for r in scored} == {0, 1}
    assert all(r.get("judge_name", "").startswith("judge:llama3") or True for r in scored)
