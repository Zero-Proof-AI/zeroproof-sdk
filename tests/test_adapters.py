"""Adapter USER_TURN splitting and hosted-key resolution."""
from tests.helpers import TOOLS

from zeroproof_simulations.agents import (
    missing_hosted_key,
    resolve_completion_key,
    split_user_turns,
)


HOSTED = "https://zeroproofai--stressd-vllm-serve.modal.run/v1"


def test_split_user_turns():
    assert split_user_turns("first\n<USER_TURN>\nsecond") == ["first", "second"]
    assert split_user_turns("only") == ["only"]
    assert split_user_turns("  a  \n<USER_TURN>\n  \n<USER_TURN>\nb") == ["a", "b"]


def test_hosted_qwen_ignores_openai_api_key(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert missing_hosted_key(HOSTED) is not None
    assert "VLLM_API_KEY" in missing_hosted_key(HOSTED)
    assert resolve_completion_key(HOSTED) == ""
    assert resolve_completion_key("https://api.openai.com/v1") == "sk-openai"
    assert resolve_completion_key(HOSTED, api_key="explicit") == "explicit"
    monkeypatch.setenv("VLLM_API_KEY", "vllm-key")
    assert resolve_completion_key(HOSTED) == "vllm-key"
    assert missing_hosted_key(HOSTED) is None


def test_openai_http_splits_user_turns(monkeypatch):
    from zeroproof_simulations.adapters import openai_http

    seen = []

    def fake_complete(_url, _model, messages, **_kwargs):
        users = [m["content"] for m in messages if m.get("role") == "user"]
        seen.append(users)
        return {"content": f"ack {users[-1]}"}

    monkeypatch.setattr("zeroproof_simulations.adapters.complete", fake_complete)
    agent = openai_http("http://example", model="m", tools=TOOLS, max_turns=6)
    out = agent("first line\n<USER_TURN>\nsecond line")
    assert all("<USER_TURN>" not in c for batch in seen for c in batch)
    assert seen[0][-1] == "first line"
    assert any("second line" in batch for batch in seen)
    assert {"user": "second line"} in out["steps"]
    assert "USER_TURN" not in out["final_text"]


def test_subprocess_agent_drops_user_turn_marker(monkeypatch):
    from zeroproof_simulations.adapters import subprocess_agent

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = b'{"steps":[],"final_text":"ok"}'
        stderr = b""

    def fake_run(_command, input=None, **_kwargs):
        captured["input"] = input
        return FakeProc()

    monkeypatch.setattr("subprocess.run", fake_run)
    agent = subprocess_agent(["echo"])
    agent("one\n<USER_TURN>\ntwo")
    assert b"<USER_TURN>" not in captured["input"]
    assert b"one" in captured["input"] and b"two" in captured["input"]


def test_from_langchain_invokes_each_turn():
    from zeroproof_simulations.adapters import from_langchain

    seen = []

    class _Action:
        tool = "lookup_order"
        tool_input = {"order_id": "ORD-1"}

    class Executor:
        def invoke(self, payload, return_only_outputs=False):
            seen.append(payload["input"])
            return {"intermediate_steps": [(_Action(), {"ok": True})],
                    "output": f"handled {payload['input']}"}

    try:
        agent = from_langchain(Executor())
    except ImportError:
        return
    out = agent("first\n<USER_TURN>\nsecond")
    assert seen == ["first", "second"]
    assert len(out["steps"]) == 2
    assert "second" in out["final_text"]
