"""hosted_model is local_model pointed at the default backend, or ZEROPROOF_AGENT."""

from __future__ import annotations

import zeroproof.simulations as zps
from zeroproof.simulations.generate import agents


def _capture_local_model(monkeypatch):
    seen = {}

    def fake_local_model(base_url, model, **kw):
        seen.update(base_url=base_url, model=model, **kw)
        return lambda message: {"steps": [], "final_text": ""}

    monkeypatch.setattr(agents, "local_model", fake_local_model)
    return seen


def test_default_brain_is_the_hosted_qwen_wearing_the_tools(monkeypatch):
    monkeypatch.delenv("ZEROPROOF_AGENT", raising=False)
    seen = _capture_local_model(monkeypatch)
    tools = [{"type": "function", "function": {"name": "lookup_order"}}]
    agent = zps.hosted_model(tools, system="Be honest.")
    url, model = agents.parse_backend_spec(agents.DEFAULT_AGENT)
    assert (seen["base_url"], seen["model"]) == (url, model)
    assert seen["tools"] is tools and seen["system"] == "Be honest."
    assert callable(agent)


def test_zeroproof_agent_env_swaps_the_backend_and_kwargs_pass_through(monkeypatch):
    monkeypatch.setenv("ZEROPROOF_AGENT", "vllm:phi@http://127.0.0.1:9/v1")
    seen = _capture_local_model(monkeypatch)
    plans = {"lookup_order": {"kind": "timeout"}}
    zps.hosted_model([], fault_plans=plans, max_turns=2)
    assert (seen["base_url"], seen["model"]) == ("http://127.0.0.1:9/v1", "phi")
    assert seen["fault_plans"] is plans and seen["max_turns"] == 2
