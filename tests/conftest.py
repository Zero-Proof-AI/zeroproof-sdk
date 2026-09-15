"""Unit tests never hit the hosted GPU."""

from __future__ import annotations

import importlib

import pytest

from tests.helpers import FakeWriter


@pytest.fixture(autouse=True)
def _offline_hosted_simulator(monkeypatch, tmp_path):
    # never read the developer's own ~/.zeroproof/credentials.json: a saved
    # account key would flip the hosted defaults to the account route
    monkeypatch.setenv("ZEROPROOF_HOME", str(tmp_path / "zeroproof-home"))

    def blocked(*_args, **_kwargs):
        raise OSError("hosted simulator disabled in unit tests")

    def embed_blocked(self, texts):
        raise OSError("hosted embedder disabled in unit tests")

    monkeypatch.setattr("zeroproof.simulations.generate.generator.complete", blocked)
    monkeypatch.setattr("zeroproof.simulations.generate.agents.complete", blocked)
    monkeypatch.setattr("zeroproof.simulations.score.llm_judge.complete", blocked)
    monkeypatch.setattr(
        importlib.import_module("zeroproof.simulations.score.grade_llm"), "complete", blocked
    )
    monkeypatch.setattr(
        "zeroproof.simulations.generate.embeddings.ModalEmbedder.embed", embed_blocked
    )
    # Every situation is model-written and the model is blocked here, so a
    # simulate() that names no writer gets the suite's fake one. Tests that
    # want the blocked hosted path pass simulator= explicitly.
    import zeroproof.simulations as zps

    real_simulate = zps.simulate

    def simulate_with_fake_writer(*args, **kwargs):
        tools = kwargs.get("tools") or []
        policy = kwargs.get("system_prompt") or kwargs.get("policy") or ""
        if "simulator" not in kwargs:
            kwargs["simulator"] = FakeWriter(tools, policy)
        elif isinstance(kwargs["simulator"], FakeWriter) and not kwargs["simulator"].tools:
            kwargs["simulator"].bind(tools, policy)
        return real_simulate(*args, **kwargs)

    monkeypatch.setattr(zps, "simulate", simulate_with_fake_writer)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ZEROPROOF_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
