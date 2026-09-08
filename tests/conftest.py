"""Unit tests never hit the hosted GPU."""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _offline_hosted_simulator(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise OSError("hosted simulator disabled in unit tests")
    def embed_blocked(self, texts):
        raise OSError("hosted embedder disabled in unit tests")
    monkeypatch.setattr("zeroproof_simulations.generate.generator.complete", blocked)
    monkeypatch.setattr("zeroproof_simulations.generate.agents.complete", blocked)
    monkeypatch.setattr("zeroproof_simulations.score.llm_judge.complete", blocked)
    monkeypatch.setattr(
        importlib.import_module("zeroproof_simulations.score.grade_llm"),
        "complete", blocked)
    monkeypatch.setattr("zeroproof_simulations.generate.embeddings.ModalEmbedder.embed",
                        embed_blocked)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ZEROPROOF_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
