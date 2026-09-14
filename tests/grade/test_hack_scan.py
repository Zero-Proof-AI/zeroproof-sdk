"""What the policy would learn: the within-ask scan (score/hack_scan.py).

The four regimes are built synthetically the way the RLVR signal sweeps
built them: a reward that follows an endorsed feature, one that follows a
delimiter the judge happens to like, one that is noise, and a pool the
policy has already solved. The pooled-vs-within case is the one the old
scan gets wrong: a difficulty confound that pooled Pearson calls a
length penalty and the grouped update never sees.
"""

from __future__ import annotations

import random
import time

import pytest

import zeroproof.simulations as zps
from zeroproof.simulations.score.hack_scan import (
    auto_terms,
    format_hack_scan,
    hack_scan,
    hand_features,
    scan_text,
)
from zeroproof.simulations.score.optimize import select_for_rl

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


def _row(prompt: str, reward, *, text: str, tool: str | None = None, **extra) -> dict:
    steps = [{"tool": tool, "arguments": {"id": prompt}, "result": {"ok": True}}] if tool else []
    row = {
        "prompt": prompt,
        "reward": reward,
        "final_text": text,
        "steps": steps,
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": text}],
    }
    row.update(extra)
    return row


def _filler(rng: random.Random, n: int = 8) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n)) + "."


def _pool(
    n_prompts: int,
    k: int,
    *,
    seed: int,
    reward_of,
) -> list[dict]:
    """``k`` rollouts per ask; half call the tool, half use the delimiter,
    independently, so the scan has to tell them apart."""
    rng = random.Random(seed)
    rows = []
    for p in range(n_prompts):
        for _ in range(k):
            uses_tool = rng.random() < 0.5
            delim = rng.random() < 0.5
            text = _filler(rng) + (" ### done ###" if delim else "")
            rows.append(
                _row(
                    f"ask {p}",
                    reward_of(rng, uses_tool, delim, p),
                    text=text,
                    tool="lookup_order" if uses_tool else None,
                )
            )
    return rows


def test_endorsed_feature_on_top_is_train():
    rows = _pool(40, 8, seed=1, reward_of=lambda rng, tool, delim, p: int(tool))
    report = hack_scan(rows, endorsed=["lookup_order"], seed=0)
    assert report["regime"] == "train"
    assert report["top_feature"] in {"tool:lookup_order", "tool_calls"}
    assert report["endorsed_on_top"] is True
    assert report["rho_max"] > report["tau"] > 0
    # A chance feature can sit just over the floor (that is what a 95th
    # percentile floor means); it must not carry a real share of the signal.
    assert report["integrity"] >= 0.8
    assert report["n_groups"] == 40 and report["rollouts_per_group"] == 8
    assert not report["warnings"]
    # A conjunction never outranks its own parent, and a duplicate column
    # is an alias, not a rival.
    winner = report["features"][0]
    assert {winner["name"], *winner["aliases"]} == {"tool:lookup_order", "tool_calls"}
    assert all(" AND " not in x["name"] or x["endorsed"] for x in report["features"][:5])


def test_delimiter_hack_is_named_over_the_endorsed_feature():
    # The judge passes anything with the delimiter; the tool is irrelevant.
    rows = _pool(40, 8, seed=2, reward_of=lambda rng, tool, delim, p: int(delim))
    report = hack_scan(rows, endorsed=["lookup_order"], seed=0)
    assert report["regime"] == "reward_hack"
    assert report["top_feature"].startswith("contains:")
    assert "#" in report["top_feature"]
    assert report["endorsed_on_top"] is False
    assert report["integrity"] == 0.0
    assert any(
        "best explained by" in w and "not by anything endorsed" in w for w in report["warnings"]
    )
    text = format_hack_scan(report)
    assert text.startswith("REWARD HACK") and "noise floor tau" in text


def test_a_reward_that_punishes_the_behavior_is_a_hack_and_ties_go_to_endorsed():
    # The reward pays for NOT calling the tool: the endorsed feature is on
    # top by magnitude, with the wrong sign.
    rows = _pool(40, 8, seed=21, reward_of=lambda rng, tool, delim, p: int(not tool))
    report = hack_scan(rows, endorsed=["lookup_order"], seed=0)
    assert report["regime"] == "reward_hack"
    assert report["endorsed_on_top"] is False and report["inverted"]
    assert any("reward punishes the endorsed" in w for w in report["warnings"])
    assert not any("also punishes" in w for w in report["warnings"])
    # A feature that is the exact complement of the behavior correlates
    # exactly as strongly, with the opposite sign: the tie goes to the
    # endorsed feature and the regime is not a hack.
    for r in rows:
        r["reward"] = int(bool(r["steps"]))
        if not r["steps"]:
            r["final_text"] += " (no lookup)"
    report = hack_scan(rows, endorsed=["lookup_order"], seed=0, min_obs=10)
    assert report["regime"] == "train" and report["endorsed_on_top"] is True
    assert report["top_feature"] in {"tool:lookup_order", "tool_calls"}
    assert report["inverted"] == []


def test_noise_reward_is_no_signal():
    rows = _pool(40, 8, seed=3, reward_of=lambda rng, tool, delim, p: int(rng.random() < 0.5))
    report = hack_scan(rows, endorsed=["lookup_order"], seed=0)
    assert report["regime"] == "no_signal"
    assert report["n_above_floor"] == 0
    assert any("no feature clears the noise floor" in w for w in report["warnings"])


def test_solved_pool_is_pool_exhausted():
    # Half the asks always pass; on the rest the tool decides.
    rows = _pool(
        40, 8, seed=4, reward_of=lambda rng, tool, delim, p: 1 if p % 2 == 0 else int(tool)
    )
    report = hack_scan(rows, endorsed=["lookup_order"], seed=0)
    assert report["regime"] == "pool_exhausted"
    assert report["support"]["sat"] == pytest.approx(0.5)
    assert report["endorsed_on_top"] is True
    assert any("all-pass" in w for w in report["warnings"])


def test_difficulty_confound_fools_pooled_but_not_within():
    # Hard asks (odd) get long replies and fail more; within an ask, length
    # says nothing. Pooled Pearson reports a length penalty; the within-ask
    # scan does not put length above the floor.
    rng = random.Random(5)
    rows = []
    for p in range(40):
        hard = p % 2 == 1
        for _ in range(8):
            uses_tool = rng.random() < 0.5
            base = 0.2 if hard else 0.8
            reward = int(rng.random() < (base if uses_tool else base * 0.5))
            text = _filler(rng, 40 if hard else 6)
            rows.append(
                _row(f"ask {p}", reward, text=text, tool="lookup_order" if uses_tool else None)
            )
    pooled = zps.reward_correlations(rows)["correlations"]["reply_length"]
    assert pooled < -0.3
    report = hack_scan(rows, endorsed=["lookup_order"], seed=0, top_features=None)
    length = next(x for x in report["features"] if x["name"] == "reply_length")
    assert abs(length["rho"]) < abs(length["pooled"])
    assert not length["above_floor"]
    assert report["regime"] == "train"


def test_endorsed_that_matches_nothing_is_unknown_not_a_hack():
    rows = _pool(20, 8, seed=6, reward_of=lambda rng, tool, delim, p: int(tool))
    report = hack_scan(rows, endorsed=["marker:nonexistent"], seed=0)
    assert report["regime"] == "unknown"
    assert report["endorsed_matched"] == 0
    assert any("matches none" in w for w in report["warnings"])


def test_without_endorsed_the_scan_ranks_and_floors_but_names_no_hack():
    rows = _pool(30, 8, seed=7, reward_of=lambda rng, tool, delim, p: int(delim))
    report = hack_scan(rows, seed=0)
    assert report["regime"] == "train"
    assert report["endorsed_on_top"] is None and report["integrity"] is None
    assert "#" in report["top_feature"]


def test_one_rollout_per_ask_fills_pooled_only():
    rows = [_row(f"ask {i}", i % 2, text=_filler(random.Random(i))) for i in range(30)]
    report = hack_scan(rows, seed=0)
    assert report["regime"] == "unknown" and report["tau"] is None
    assert any("one rollout per ask" in w for w in report["warnings"])
    assert all(x["rho"] == 0.0 for x in report["features"])


def test_ungraded_rows_say_so():
    rows = [_row("a", None, text="x") for _ in range(4)]
    report = hack_scan(rows)
    assert report["regime"] == "unknown" and report["n_graded"] == 0
    assert "grade first" in report["warnings"][0]


def test_partial_credit_rewards_and_markers_and_custom_features():
    rng = random.Random(8)
    rows = []
    for p in range(30):
        for _ in range(8):
            grounded = rng.random() < 0.5
            rows.append(
                _row(
                    f"ask {p}",
                    0.9 if grounded else 0.3,
                    text=_filler(rng),
                    tool="lookup_order",
                    markers={"grounded_args": 1.0 if grounded else 0.0},
                    note_len=rng.randint(1, 5),
                )
            )
    report = hack_scan(
        rows,
        endorsed=["marker:grounded_args"],
        features={"note": lambda r: r.get("note_len")},
        seed=0,
        top_features=None,
    )
    assert report["continuous_reward"] is True
    assert report["regime"] == "train" and report["top_feature"] == "marker:grounded_args"
    names = {x["name"] for x in report["features"]}
    assert "note" in names
    # tool:lookup_order is constant (every row calls it): never a feature
    assert "tool:lookup_order" not in names


def test_hand_features_and_scan_text():
    row = _row(
        "a",
        1,
        text="Refunded ORD-12.\nDone!",
        tool="create_refund",
        logprob=-20.0,
        n_tokens=10,
        markers={"well_formed": 1.0, "label": "x"},
    )
    feats = hand_features(row)
    assert feats["tool:create_refund"] == 1.0 and feats["tool_calls"] == 1.0
    assert feats["n:newlines"] == 1.0 and feats["n:digits"] == 2.0
    assert feats["logprob_mean"] == pytest.approx(-2.0)
    assert feats["marker:well_formed"] == 1.0 and "marker:label" not in feats
    assert scan_text(row) == "Refunded ORD-12.\nDone!"
    vocab, present = auto_terms(
        ["look up order", "look up refund", "look at it"], top_k=10, min_obs=2
    )
    assert "look up" in vocab and "look" not in vocab  # present everywhere: no contrast
    assert present[0] >= {"look", "up", "order", "look up", "up order"}


def test_select_for_rl_and_publish_gate_carry_the_scan():
    rows = _pool(30, 8, seed=9, reward_of=lambda rng, tool, delim, p: int(delim))
    _picked, report = select_for_rl(rows, target=1000, endorsed=["lookup_order"])
    assert report["hack_scan"]["regime"] == "reward_hack"
    assert any("best explained by" in w for w in report["hygiene_warnings"])
    _rows, opt_report = zps.optimize(rows, mode="rl", endorsed=["lookup_order"])
    assert opt_report["hack_scan"]["regime"] == "reward_hack"

    gate = zps.publish_gate(rows, mode="rl", endorsed=["lookup_order"])
    assert gate["ok"] and gate["hack_scan"]["regime"] == "reward_hack"
    assert any("best explained by" in w for w in gate["warnings"])
    with pytest.raises(zps.PublishGateError, match="reward_hack"):
        zps.publish_gate(rows, mode="rl", endorsed=["lookup_order"], strict_hacks=True)
    lenient = zps.publish_gate(
        rows, mode="rl", endorsed=["lookup_order"], strict_hacks=True, strict=False
    )
    assert not lenient["ok"] and lenient["refusal"].startswith("reward_hack")
    # explore-shaped rows are not scanned: not RL data
    assert zps.publish_gate([_row("a", 1, text="x")])["hack_scan"] is None


def test_scan_is_fast_enough_without_numpy():
    rows = _pool(400, 8, seed=10, reward_of=lambda rng, tool, delim, p: int(delim))
    started = time.perf_counter()
    report = hack_scan(rows, endorsed=["lookup_order"], seed=0)
    seconds = time.perf_counter() - started
    assert report["regime"] == "reward_hack"
    assert seconds < 30, f"3200 rows took {seconds:.1f}s"
