"""rl mode spends rollouts where the agent is inconsistent.

Successive allocation: every prompt is probed, a prompt whose rollouts
disagree is filled to k, a prompt that stays unanimous stops once a fresh
prompt is the better bet, and the clock finishes groups instead of cutting
them. rlhf-book ch. 6 (dynamic sampling) and ch. 7 (difficulty filtering)
at generation time.
"""

from __future__ import annotations

import collections
import hashlib
import time

import zeroproof.simulations as zps
from tests.helpers import offline, scripted_agent


def _flaky():
    """Half the prompts (by a stable hash) refuse on every second call;
    the other half never do. Which half a prompt falls in does not depend
    on the writer's wording, so every run has both kinds."""
    calls: dict[str, int] = collections.Counter()

    def agent(message: str) -> dict:
        calls[message] += 1
        digest = int(hashlib.md5(message.encode("utf-8")).hexdigest(), 16)
        if digest % 2 == 0 and calls[message] % 2 == 0:
            return {"steps": [], "final_text": "I cannot help with that."}
        return scripted_agent(message)

    return agent


def _judge(row: dict) -> dict:
    refused = "cannot" in str(row.get("final_text") or "").lower()
    return {"reward": 0.0 if refused else 1.0, "reason": "refusal" if refused else "ok"}


def test_split_prompts_fill_to_k_and_the_rest_finish_when_nothing_is_left_to_open():
    data = zps.simulate(
        _flaky(), mode="rl", situations=6, rollouts_per_request=6, budget=40, **offline()
    )
    sizes = collections.Counter(t["prompt"] for t in data.trajectories)
    groups = data.search["groups"]
    assert groups["k"] == 6 and groups["probe"] == 2
    assert groups["mixed"] >= 1
    # every split prompt reached k
    by_prompt: dict[str, set] = {}
    for t in data.trajectories:
        by_prompt.setdefault(t["prompt"], set()).add(t["behavior_signature"])
    for prompt, sigs in by_prompt.items():
        if len(sigs) > 1:
            assert sizes[prompt] == 6, (prompt, sizes[prompt])
    # the situation cap left nothing fresh to open, so unanimous groups
    # finished too: a group stopped earlier is resumed rather than left short
    diag = (groups, sorted(sizes.values()), data.stopped_because, data.search.get("allocator"))
    assert groups["partial"] == 0, diag
    assert groups["stopped_unanimous"] == 0, diag
    assert data.stopped_because == "situations_exhausted", diag


def test_unanimous_prompts_stop_when_fresh_prompts_split_more_often():
    data = zps.simulate(
        _flaky(),
        mode="rl",
        situations=30,
        rollouts_per_request=6,
        budget=60,
        grader=_judge,
        **offline(),
    )
    rows = data.trajectories
    groups = data.search["groups"]
    sizes = collections.Counter(t["prompt"] for t in rows)
    assert groups["mixed"] >= 2, groups
    # the grader ran inside the loop: every row is judged exactly once and
    # the allocation read rewards, so a split is a reward split
    assert all(t.get("judge_status") for t in rows)
    assert data.search["grader"]["scored"] == len(rows)
    split = [p for p in sizes if len({t["reward"] for t in rows if t["prompt"] == p}) > 1]
    unanimous = [p for p in sizes if p not in split]
    assert split and unanimous, sizes
    # the budget went to the split prompts: they reached k unless the
    # budget ran out first, and on average they got more rollouts than
    # the unanimous ones, which stopped once a fresh prompt was the
    # better bet
    assert sum(1 for p in split if sizes[p] == 6) >= 2, (groups, sizes)
    mean = lambda ps: sum(sizes[p] for p in ps) / len(ps)  # noqa: E731
    assert mean(split) > mean(unanimous), (groups, sizes)
    unanimous_short = [p for p in unanimous if sizes[p] < 6]
    assert unanimous_short, groups
    # the k-way numbers use the requested k and count every unanimous group
    # shorter than k as unanimous, stopped or cut alike
    got = data.pass_at
    assert got.k == 6
    assert got.n_groups_imputed == len(unanimous_short)
    assert got.pass_at_k is not None and got.pass_pow_k is not None
    # a stopped group is not a cut group: the stamp marks exactly the
    # groups the summary calls partial
    cut = {t["prompt"] for t in rows if t.get("group_cut")}
    assert len(cut) == groups["partial"]


def test_clock_finishes_groups_instead_of_cutting_them():
    def slow(message: str) -> dict:
        time.sleep(0.25)
        return _flaky_shared(message)

    _flaky_shared = _flaky()
    data = zps.simulate(
        slow,
        mode="rl",
        situations=40,
        rollouts_per_request=4,
        budget=400,
        **offline(time_budget=2.0),
    )
    assert data.stopped_because == "time_budget"
    groups = data.search["groups"]
    assert groups["closing"] is True
    cut = {t["prompt"] for t in data.trajectories if t.get("group_cut")}
    # closing stopped new groups early enough that at most the last wave
    # could be short; nothing is silently counted as k
    for prompt in cut:
        assert sum(1 for t in data.trajectories if t["prompt"] == prompt) < 4
