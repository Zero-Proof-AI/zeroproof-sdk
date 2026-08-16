"""Unit tests never hit the hosted GPU."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _offline_hosted_simulator(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise OSError("hosted simulator disabled in unit tests")
    def embed_blocked(self, texts):
        raise OSError("hosted embedder disabled in unit tests")
    monkeypatch.setattr("zeroproof_simulations.generator.complete", blocked)
    monkeypatch.setattr("zeroproof_simulations.agents.complete", blocked)
    monkeypatch.setattr("zeroproof_simulations.llm_judge.complete", blocked)
    monkeypatch.setattr("zeroproof_simulations.embeddings.ModalEmbedder.embed",
                        embed_blocked)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ZEROPROOF_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
