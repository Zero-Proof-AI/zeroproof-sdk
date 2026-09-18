"""The trial warning lands before a hosted run spends the allowance, not after."""

from __future__ import annotations

import json
import logging

import pytest

import whileai.simulations as wai
from tests.helpers import POLICY, TOOLS, scripted_agent
from whileai import auth

LINE = (
    "trial key: the hosted writer covers about 12 situations a day (25k input tokens); "
    "simulator=False writes them offline with no quota; sign in once at "
    "https://www.zeroproofai.com/sign-in to lift it"
)


def _save(home, **fields) -> None:
    home.mkdir(parents=True, exist_ok=True)
    payload = {"api_key": "zp_" + "b" * 48, "api_url": auth.DEFAULT_API_URL}
    payload.update(fields)
    (home / "credentials.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / "whileai-home"
    monkeypatch.setenv("WHILEAI_HOME", str(path))
    return path


def test_note_reads_the_recorded_tier(home):
    _save(home, tier="trial", daily_input_tokens=25000, expires_at="2026-09-21T00:00:00.000Z")
    assert auth.trial_prerun_note() == LINE


def test_a_full_key_says_nothing(home):
    _save(home, tier="full")
    assert auth.trial_prerun_note() is None


def test_a_key_from_the_environment_says_nothing(home, monkeypatch):
    # the file describes the saved account, not whatever key the variable
    # holds, so its tier must not be read onto that key
    _save(home, tier="trial", daily_input_tokens=25000)
    monkeypatch.setenv("WHILEAI_API_KEY", "zp_env")
    assert auth.trial_prerun_note() is None


def test_allowance_from_the_file_sets_the_count(home):
    _save(home, tier="trial", daily_input_tokens=100_000)
    note = auth.trial_prerun_note()
    assert note and "about 50 situations a day (100k input tokens)" in note


def test_the_run_warns_before_the_hosted_writer_starts(home, caplog):
    _save(home, tier="trial", daily_input_tokens=25000)
    with caplog.at_level(logging.WARNING, logger="whileai.simulations"):
        data = wai.simulate(
            scripted_agent,
            tools=TOOLS,
            system_prompt=POLICY,
            budget=2,
            seed=0,
            grade=False,
            time_budget=1,
        )
    assert LINE in data.warnings
    assert LINE in caplog.text


def test_the_offline_writer_has_no_quota_to_warn_about(home):
    _save(home, tier="trial", daily_input_tokens=25000)
    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        system_prompt=POLICY,
        budget=4,
        seed=0,
        simulator=False,
        grade=False,
        time_budget=None,
        advanced={"per_round": 8, "mutate_failures": False},
    )
    assert data.trajectories
    assert not [w for w in data.warnings if "trial key" in w]


def test_a_full_key_runs_without_the_note(home):
    _save(home, tier="full")
    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        system_prompt=POLICY,
        budget=2,
        seed=0,
        grade=False,
        time_budget=1,
    )
    assert not [w for w in data.warnings if "trial key" in w]


def test_the_shared_pool_spends_no_trial_so_it_says_nothing(home, monkeypatch):
    # VLLM_API_KEY routes the writer to the shared pool, which the trial
    # allowance does not meter
    _save(home, tier="trial", daily_input_tokens=25000)
    monkeypatch.setenv("VLLM_API_KEY", "pool-key")
    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        system_prompt=POLICY,
        budget=2,
        seed=0,
        grade=False,
        time_budget=1,
    )
    assert not [w for w in data.warnings if "trial key" in w]
