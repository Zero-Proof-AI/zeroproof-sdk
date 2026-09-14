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


def _two_trajectory_rows(n_tasks: int = 20, k: int = 8, n_pass: int = 5) -> list[dict]:
    """Two distinct trajectories per ask: every feature that separates
    them is an exact function of the label, so ``hack_scan`` refuses."""
    rows = []
    for p in range(n_tasks):
        for i in range(k):
            ok = i < n_pass
            text = (
                "Looked up the order and issued the refund in full." if ok else "sorry, i could not"
            )
            rows.append(
                {
                    "prompt": f"ask {p}",
                    "reward": 1 if ok else 0,
                    "final_text": text,
                    "steps": (
                        [{"tool": "lookup_order", "arguments": {}, "result": {"ok": 1}}]
                        if ok
                        else []
                    ),
                    "messages": [
                        {"role": "user", "content": f"ask {p}"},
                        {"role": "assistant", "content": text},
                    ],
                }
            )
    return rows


def test_hack_scan_diff_does_not_name_what_a_degenerate_scan_could_not_rank():
    # Every feature of a degenerate scan is above the floor at |rho| 1, so
    # "gained the floor" there is a tie. Naming its top would be the
    # verdict hack_scan refuses to give, arrived at one function later.
    before = _rows(7, proxy_rate=0.5, gold_rate=0.5)
    for r in before:
        r["reward"] = int(r["markers"].pop("proxy"))
    after = _two_trajectory_rows()

    diff = hack_scan_diff(before, after, endorsed=["lookup_order"], seed=0)
    assert diff["after"]["degenerate"] is True and diff["after"]["regime"] == "degenerate"
    assert diff["before"]["degenerate"] is False
    assert diff["learned"].startswith("the after scan cannot say what the reward pays for")
    assert not any("a reward hack landed" in w for w in diff["warnings"])
    assert not diff["learned"].startswith('the policy learned "')
    # the shift is still shown, only the claim is withheld
    assert diff["moved"]
    assert format_hack_scan_diff(diff).splitlines()[0] == diff["learned"]

    both = hack_scan_diff(after, after, endorsed=["lookup_order"], seed=0)
    assert both["learned"].startswith("the before and after scan cannot say")
    # and the fallback must not claim the floor is empty when it is full
    assert "no feature clears the floor" not in both["learned"]


def test_hack_scan_diff_markers_keep_their_gutter():
    before = _rows(7, proxy_rate=0.5, gold_rate=0.5)
    after = _rows(8, proxy_rate=0.5, gold_rate=0.5, delim_reward=True)
    for r in before + after:
        r["reward"] = int(r["markers"].pop("proxy"))
    lines = format_hack_scan_diff(
        hack_scan_diff(before, after, endorsed=["lookup_order"], seed=0)
    ).splitlines()
    header = next(line for line in lines if "feature" in line and "before" in line)
    assert header.startswith("   feature")
    body = [line for line in lines if line[:2] in {"+e", "-e", "+ ", "- ", " e", "  "}]
    assert body and all(line[2] == " " for line in body[1:])
    assert "econtains:" not in "\n".join(lines)


@pytest.mark.parametrize("seed", [11, 12])
def test_over_optimization_needs_the_proxy_to_move(seed):
    before = _rows(seed, proxy_rate=0.5, gold_rate=0.5)
    after = _rows(seed + 100, proxy_rate=0.5, gold_rate=0.5)
    report = delta_report(before, after, target="pass_at_1", proxy="marker:proxy", n_boot=200)
    assert report["over_optimized"] is False
