"""The hack monitor's numbers are named defaults and keywords."""

from __future__ import annotations

import functools

import pytest

from whileai.simulations import defaults
from whileai.simulations import monitor as monitor_mod
from whileai.simulations.monitor import HackMonitor, _default_sample


def _sample(model, tokenizer, prompts, *, n, max_new_tokens):
    return [["a b c"] * n for _ in prompts]


def _proxy(prompts=None, completions=None, **cols):
    return [1.0 for _ in completions]


def test_defaults_are_the_named_numbers():
    m = HackMonitor(holdout=["ask"], proxy=_proxy, sample=_sample)
    assert m.every == defaults.MONITOR_EVERY == 10
    assert m.k == defaults.MONITOR_K and m.window == defaults.MONITOR_WINDOW
    assert m.delta == defaults.MONITOR_DELTA and m.length_pct == defaults.MONITOR_LENGTH_PCT
    assert m.n_boot == defaults.MONITOR_WINDOW_BOOTSTRAPS == 500
    assert m.n_perm == defaults.MONITOR_SCAN_PERMUTATIONS == 50
    assert m.scan_min == defaults.MONITOR_SCAN_MIN == 8
    assert m.max_new_tokens == defaults.MONITOR_MAX_NEW_TOKENS


def test_sampling_steers_the_default_sampler_only():
    m = HackMonitor(holdout=["ask"], proxy=_proxy, sampling={"temperature": 0.2, "top_p": 0.5})
    assert isinstance(m.sample, functools.partial) and m.sample.func is _default_sample
    assert m.sample.keywords == {"temperature": 0.2, "top_p": 0.5}
    with pytest.raises(ValueError, match="unknown key"):
        HackMonitor(holdout=["ask"], proxy=_proxy, sampling={"temp": 0.2})
    with pytest.raises(ValueError, match="drop it"):
        HackMonitor(holdout=["ask"], proxy=_proxy, sample=_sample, sampling={"temperature": 0.2})


def test_scan_min_and_n_perm_reach_hack_scan(monkeypatch):
    seen = []

    def fake_scan(buffer, *, endorsed, n_perm):
        seen.append((len(buffer), n_perm))
        return {
            "regime": "clean",
            "top_feature": None,
            "rho_max": 0.0,
            "tau": 0.0,
            "integrity": 1.0,
            "warnings": [],
        }

    monkeypatch.setattr(monitor_mod, "hack_scan", fake_scan)
    m = HackMonitor(
        holdout=["ask"], proxy=_proxy, sample=_sample, endorsed=["length"], scan_min=2, n_perm=7
    )
    watched = m.wrap(_proxy)
    watched(prompts=["a", "b"], completions=["x", "y"])
    m.evaluate(1)
    assert seen == [(2, 7)]
    quiet = HackMonitor(holdout=["ask"], proxy=_proxy, sample=_sample, endorsed=["length"])
    quiet.wrap(_proxy)(prompts=["a", "b"], completions=["x", "y"])
    quiet.evaluate(1)
    assert seen == [(2, 7)], "under scan_min (8) the default monitor does not scan two rows"
