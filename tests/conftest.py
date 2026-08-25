"""Unit tests never hit the hosted GPU."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _offline_hosted_simulator(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise OSError("hosted simulator disabled in unit tests")
    def embed_blocked(self, texts):
        raise OSError("hosted embedder disabled in unit tests")

    # Some local branches can have temporary import drift in package wiring.
    # Keep the offline guard best-effort so unrelated tests can still run.
    try:
        import zeroproof_simulations.generator as _gen
        monkeypatch.setattr(_gen, "complete", blocked)
    except (ImportError, AttributeError):
        pass
    try:
        import zeroproof_simulations.agents as _agents
        monkeypatch.setattr(_agents, "complete", blocked)
    except (ImportError, AttributeError):
        pass
    try:
        import zeroproof_simulations.llm_judge as _judge
        monkeypatch.setattr(_judge, "complete", blocked)
    except (ImportError, AttributeError):
        pass
    try:
        import zeroproof_simulations.embeddings as _emb
        monkeypatch.setattr(_emb.ModalEmbedder, "embed", embed_blocked)
    except (ImportError, AttributeError):
        pass
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ZEROPROOF_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
