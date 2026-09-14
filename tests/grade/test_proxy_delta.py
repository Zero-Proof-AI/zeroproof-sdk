"""The after-training verdict: proxy vs target (delta_report(proxy=)) and
what the policy learned (hack_scan_diff)."""

from __future__ import annotations

import random

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations.score.delta import delta_report, format_delta_report
from zeroproof.simulations.score.hack_scan import format_hack_scan_diff, hack_scan_diff

WORDS = [
    "order",
    "refund",
    "status",
    "ship",
    "customer",
    "account",
    "help",
    "please",
    "check",
    "note",
    "item",
]


def _rows(
    seed: int, *, proxy_rate: float, gold_rate: float, delim_reward: bool = False
) -> list[dict]:
    """40 tasks x 4 rollouts. ``reward`` is the gold pass, ``markers.proxy``
    the training reward; with ``delim_reward`` the proxy pays for a
    delimiter instead of the tool."""
    rng = random.Random(seed)
    rows = []
    for t in range(40):
        for i in range(4):
            tool = rng.random() < 0.5
            delim = rng.random() < 0.5
            gold = int(rng.random() < gold_rate)
            proxy = float(delim if delim_reward else (rng.random() < proxy_rate))
            text = " ".join(rng.choice(WORDS) for _ in range(8)) + (
                " ### done ###" if delim else "."
            )
            rows.append(
                {
                    "prompt": f"ask {t}",
                    "scenario_id": f"t{t}",
                    "rollout_index": i,
                    "reward": gold,
                    "final_text": text,
                    "steps": [{"tool": "lookup_order", "arguments": {}, "result": {}}]
                    if tool
                    else [],
                    "messages": [
                        {"role": "user", "content": f"ask {t}"},
                        {"role": "assistant", "content": text},
                    ],
                    "markers": {"proxy": proxy, "well_formed": 1.0},
                }
            )
    return rows


def test_proxy_up_while_target_flat_is_over_optimized():
    before = _rows(1, proxy_rate=0.2, gold_rate=0.5)
    after = _rows(2, proxy_rate=0.9, gold_rate=0.5)
    report = delta_report(before, after, target="pass_at_1", proxy="marker:proxy", n_boot=300)
    assert report["proxy"] == "marker:proxy" and report["proxy_verdict"] == "moved"
    assert report["over_optimized"] is True and report["ok"] is False
    assert report["proxy_delta"] > 0.5 and report["target_verdict"] != "moved"
    assert any(w.startswith("OVER-OPTIMIZED: marker:proxy up") for w in report["warnings"])
    text = format_delta_report(report)
    assert "proxy marker:proxy: moved" in text and "OVER-OPTIMIZED" in text and "FAIL" in text


def test_both_moving_together_is_not_over_optimized():
    before = _rows(3, proxy_rate=0.2, gold_rate=0.2)
    after = _rows(4, proxy_rate=0.8, gold_rate=0.8)
    report = delta_report(before, after, target="pass_at_1", proxy="marker:proxy", n_boot=300)
    assert report["proxy_verdict"] == "moved" and report["target_verdict"] == "moved"
    assert report["over_optimized"] is False and report["ok"] is True
    assert not any("OVER-OPTIMIZED" in w for w in report["warnings"])


def test_proxy_edge_cases_and_passthrough():
    before = _rows(5, proxy_rate=0.5, gold_rate=0.5)
    after = _rows(6, proxy_rate=0.5, gold_rate=0.5)
    plain = delta_report(before, after, target="pass_at_1", n_boot=100)
    assert plain["proxy"] is None and plain["proxy_verdict"] is None
    assert plain["over_optimized"] is False
    same = delta_report(before, after, target="pass_at_1", proxy="pass_at_1", n_boot=100)
    assert same["proxy_verdict"] == "proxy_is_target" and same["ok"]
    missing = delta_report(before, after, target="pass_at_1", proxy="marker:nope", n_boot=100)
    assert missing["proxy_verdict"] == "proxy_not_measured"
    assert any("not on both row sets" in w for w in missing["warnings"])
    assert "proxy marker:nope: proxy_not_measured" in format_delta_report(missing)
    # the run handle and attach_delta pass proxy through
    assert "proxy" in zps.TrainingRun.delta.__code__.co_varnames
    assert "proxy" in zps.attach_delta.__code__.co_varnames


def test_hack_scan_diff_names_what_was_learned():
    before = _rows(7, proxy_rate=0.5, gold_rate=0.5)
    after = _rows(8, proxy_rate=0.5, gold_rate=0.5, delim_reward=True)
    for r in before + after:
        r["reward"] = int(r["markers"].pop("proxy"))  # scan the training reward itself
    diff = hack_scan_diff(before, after, endorsed=["lookup_order"], seed=0)
    assert diff["before"]["regime"] == "no_signal"
    assert diff["after"]["regime"] == "reward_hack"
    assert diff["gained"] and "#" in diff["gained"][0]["name"]
    assert diff["gained"][0]["endorsed"] is False
    assert diff["learned"].startswith('the policy learned "') and "not endorsed" in diff["learned"]
    assert any("a reward hack landed" in w for w in diff["warnings"])
    assert any(w.startswith("regime no_signal -> reward_hack") for w in diff["warnings"])
    assert len(diff["moved"]) <= 10 and all("shift" in r for r in diff["moved"])
    text = format_hack_scan_diff(diff)
    assert text.splitlines()[0] == diff["learned"] and "regime no_signal -> reward_hack" in text
    clean = hack_scan_diff(before, before, endorsed=["lookup_order"], seed=0)
    assert clean["gained"] == [] and clean["lost"] == [] and clean["warnings"] == []
    assert zps.hack_scan_diff is hack_scan_diff


@pytest.mark.parametrize("seed", [11, 12])
def test_over_optimization_needs_the_proxy_to_move(seed):
    before = _rows(seed, proxy_rate=0.5, gold_rate=0.5)
    after = _rows(seed + 100, proxy_rate=0.5, gold_rate=0.5)
    report = delta_report(before, after, target="pass_at_1", proxy="marker:proxy", n_boot=200)
    assert report["over_optimized"] is False
