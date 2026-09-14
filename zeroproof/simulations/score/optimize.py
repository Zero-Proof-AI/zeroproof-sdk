"""Concentrate a big simulated batch into the dataset post-training needs.

Cold start over-generates on purpose (a few thousand rows walks the whole
grid). The optimizer is the concentrator, and what survives depends on the
post-training target:

* ``select_for_sft``: correct, diverse demonstrations. Only 1-labeled
  rows; one per behavior shape first, so 800 rows cover 800 behaviors
  instead of 80 rephrasings of ten.
* ``select_for_rl``: whole groups with within-ask contrast. Groups are
  never split; unanimous groups are dead gradient and go first; what is
  left is ordered mixed-band first, spread across fault kinds.

The optimizer does not create diversity. If the mixed rate is low after
trimming, rerun the simulator rather than squeezing this batch harder.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .grading import (
    _DEGENERATE,
    _HARNESS_LEAK,
    _INFRA_STUB,
    _UNFINISHED_TAIL,
    behavior_signature,
    trace_fault,
)
from .passat import pass_at
from .quality import _IDISH, _QUESTION_END, _STRONG_ACTION, load_jsonl, write_jsonl

# Public drop tags. optimize_rl uses these strings in the report.
# The difficulty band: keep asks the policy passes between 20% and 80% of
# the time (rlhf-book ch. 7, offline difficulty filtering; Seed-Thinking,
# ORZ, Phi-4, INTELLECT-2, MiMo, Skywork-OR1 all report a form of it).
# A heuristic with no published ablation, so it stays configurable.
DEFAULT_BAND: tuple[float, float] = (0.2, 0.8)

DO_NOTHING = "do_nothing"
INCOMPLETE_JUNK = "incomplete_junk"
UNUSABLE_LABEL = "unusable_label"
RL_DROP_REASONS = (DO_NOTHING, INCOMPLETE_JUNK, UNUSABLE_LABEL)

# Kept-despite-junk tag: a verified failure row (reward 0 from a trusted
# grader) whose junk IS the behavior negative advantage should suppress.
KEPT_VERIFIED_ZERO = "kept_verified_zero"
_VERIFIED_SOURCES = ("claude", "gold")

_RAW_TOOL_MARKUP = re.compile(r"</?tool_call>", re.I)
_TOOL_SCHEMA_DUMP = re.compile(
    r'"name"\s*:\s*"[^"]+".{0,500}"description"\s*:'
    r'.{0,500}"parameters"\s*:',
    re.I | re.S,
)


def _messages(row: dict) -> list[dict]:
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        return [m for m in msgs if isinstance(m, dict)]
    from zeroproof.simulations import conversation

    return conversation(row)


def _has_tool_call(row: dict) -> bool:
    for step in row.get("steps") or []:
        if isinstance(step, dict) and step.get("tool"):
            return True
    return any(msg.get("tool_calls") for msg in _messages(row))


def _situation_wants_tools(row: dict) -> bool:
    """True when the situation is an actionable tool ask, not chit-chat."""
    if row.get("tool_known"):
        return True
    if str(row.get("ask_family") or "") == "tool":
        return True
    if row.get("faults"):
        return True
    world = row.get("world_state")
    if world and world not in {"unspecified", "unknown"}:
        return True
    dims = row.get("scenario_dimensions")
    if isinstance(dims, dict) and (dims.get("intent") or dims.get("tool")):
        return True
    prompt = str(row.get("prompt") or "")
    return bool(_STRONG_ACTION.search(prompt) or _IDISH.search(prompt))


def is_do_nothing(row: dict, *, has_tools: bool = True) -> bool:
    """Actionable situation where the agent never called a tool.

    For an agent with no tools (``has_tools=False``) this is never a drop:
    declining an actionable ask in words IS that agent's correct behavior,
    and deleting those rows would strip its refusal demonstrations.
    """
    if not has_tools:
        return False
    return _situation_wants_tools(row) and not _has_tool_call(row)


def _visible_text(row: dict) -> str:
    parts = [str(row.get("final_text") or "")]
    for step in row.get("steps") or []:
        if isinstance(step, dict) and step.get("text"):
            parts.append(str(step["text"]))
    for msg in _messages(row):
        if msg.get("role") == "assistant":
            parts.append(str(msg.get("content") or ""))
    return "\n".join(parts)


def is_incomplete_junk(row: dict) -> bool:
    """Empty, truncated, leaked, or parser-broken traces."""
    final = str(row.get("final_text") or "").strip()
    has_tool = _has_tool_call(row)
    if not final:
        return True
    if final.lower().startswith("<agent error"):
        return True
    if not has_tool and _INFRA_STUB.search(final):
        return True
    if _DEGENERATE.search(final):
        return True
    if _HARNESS_LEAK.search(final):
        return True
    visible = _visible_text(row)
    if _RAW_TOOL_MARKUP.search(visible) or _TOOL_SCHEMA_DUMP.search(visible):
        return True
    if len(final) > 600 and not _UNFINISHED_TAIL.search(final):
        return True
    messages = _messages(row)
    if not messages:
        return not has_tool
    last_role = str(messages[-1].get("role") or "")
    if last_role in {"user", "tool"}:
        return True
    return bool(
        not has_tool
        and _QUESTION_END.search(final)
        and len(str(row.get("prompt") or "").split()) <= 2
    )


def _is_binary_01(value) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number in (0.0, 1.0)


def is_unusable_label(row: dict) -> bool:
    """Missing or non-binary ``reward`` (or ``qwen_reward`` if that is all there is)."""
    for key in ("reward", "qwen_reward"):
        if key in row and _is_binary_01(row.get(key)):
            return False
    return True


def _has_trainable_content(row: dict) -> bool:
    if _has_tool_call(row) or str(row.get("final_text") or "").strip():
        return True
    return any(
        m.get("role") == "assistant" and (m.get("content") or m.get("tool_calls"))
        for m in _messages(row)
    )


def is_verified_zero(row: dict) -> bool:
    """reward == 0 from a trusted grader (claude* / gold label_source)."""
    value = row.get("reward")
    if value is None or not _is_binary_01(value) or float(value) != 0.0:
        return False
    src = str(row.get("label_source") or "").lower()
    return src.startswith(_VERIFIED_SOURCES)


def drop_reason(row: dict, *, has_tools: bool = True) -> str | None:
    """First matching drop tag, or None to keep.

    A verified zero bypasses the behavioral gates: dropping it deletes
    the failure example the negative advantage exists to teach against
    (measured: the old gates deleted 13.2% of verified zeros, skewed
    toward incomplete and over-clarification failures). It must still
    contain something to train on.
    """
    if is_verified_zero(row) and _has_trainable_content(row):
        return None
    if is_incomplete_junk(row):
        return INCOMPLETE_JUNK
    # The do-nothing gate is pre-judge hygiene. A row the judge already
    # scored is the judge's call: an ask the grid thought needed a tool
    # is often answerable from policy text, and the airline walk showed
    # this gate deleting 131 judge-passed correct answers.
    if _binary_label(row) is None and is_do_nothing(row, has_tools=has_tools):
        return DO_NOTHING
    if is_unusable_label(row):
        return UNUSABLE_LABEL
    return None


def filter_rl_rows(
    rows: Sequence[dict], *, has_tools: bool = True
) -> tuple[list[dict], dict[str, Any]]:
    """Split keep/drop. Does not mutate ``rows``."""
    kept: list[dict] = []
    counts = {DO_NOTHING: 0, INCOMPLETE_JUNK: 0, UNUSABLE_LABEL: 0}
    kept_verified = 0
    for row in rows:
        reason = drop_reason(row, has_tools=has_tools)
        if reason is None:
            if is_verified_zero(row) and (
                is_incomplete_junk(row) or is_do_nothing(row, has_tools=has_tools)
            ):
                kept_verified += 1
            kept.append(row)
        else:
            counts[reason] = counts.get(reason, 0) + 1
    n = len(rows)
    report = {
        "n": n,
        "n_kept": len(kept),
        "n_dropped": n - len(kept),
        "dropped": counts,
        KEPT_VERIFIED_ZERO: kept_verified,
    }
    return kept, report


def _group_label_lists(rows: Sequence[dict]) -> dict[str, list[int]]:
    """Binary labels per situation (grouped by prompt). Unlabeled rows skip."""
    groups: dict[str, list[int]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = None
        for key in ("reward", "qwen_reward"):
            value = row.get(key)
            if value is None or isinstance(value, bool):
                continue
            if _is_binary_01(value):
                label = int(float(value))
                break
        if label is None:
            continue
        prompt = str(row.get("prompt") or "")
        groups.setdefault(prompt, []).append(label)
    return groups


def group_signal(
    rows: Sequence[dict], *, lo: float = DEFAULT_BAND[0], hi: float = DEFAULT_BAND[1]
) -> dict[str, Any]:
    """Within-ask contrast. Signal is a group whose k rollouts disagree.

    A grouped RL update learns from a mix of 0 and 1 on the same ask,
    ideally with pass rate p in [``lo``, ``hi``]. Unanimous groups are
    dead gradient. Groups of one rollout cannot mix and are counted
    separately, not blamed.
    """
    groups = _group_label_lists(rows)
    n_mixed = n_all_zero = n_all_one = n_single = in_band = 0
    for labels in groups.values():
        if len(labels) < 2:
            n_single += 1
            continue
        p = sum(labels) / len(labels)
        if 0.0 < p < 1.0:
            n_mixed += 1
            if lo <= p <= hi:
                in_band += 1
        elif p == 0.0:
            n_all_zero += 1
        else:
            n_all_one += 1
    multi = n_mixed + n_all_zero + n_all_one
    rates = pass_at(rows)
    return {
        "n_groups": len(groups),
        "n_single": n_single,
        "n_mixed": n_mixed,
        "n_all_zero": n_all_zero,
        "n_all_one": n_all_one,
        "n_in_band": in_band,
        "mixed_rate": (n_mixed / multi) if multi else None,
        # The same groups in post-training vocabulary. pass@k - pass@1 is
        # the headroom the mixed groups carry; see score/passat.py.
        "k": rates.k,
        "pass_at_1": rates.pass_at_1,
        "pass_pow_k": rates.pass_pow_k,
        "pass_at_k": rates.pass_at_k,
        "headroom": rates.headroom,
    }


def trim_unanimous_groups(
    rows: Sequence[dict], *, min_k: int = 2
) -> tuple[list[dict], dict[str, Any]]:
    """Drop asks whose k >= ``min_k`` rollouts all landed 0 or all landed 1.

    The basic optimizer from the working decision: trim zeros and ones
    from tasks, then rerun the simulator and check the variance. Groups
    smaller than ``min_k`` (unique-situation runs) always stay; trimming
    them would gut an explore dataset, and they carry no group gradient
    either way.
    """
    groups = _group_label_lists(rows)
    dead: set[str] = set()
    for prompt, labels in groups.items():
        if len(labels) >= max(2, int(min_k)) and len(set(labels)) == 1:
            dead.add(prompt)
    kept = [row for row in rows if str(row.get("prompt") or "") not in dead]
    report = {
        "n": len(rows),
        "n_kept": len(kept),
        "n_dropped": len(rows) - len(kept),
        "n_groups_dropped": len(dead),
        "signal": group_signal(kept),
    }
    return kept, report


def eval_sourced(rows: Sequence[dict]) -> int:
    """Rows whose reward came from ``evaluate()`` (``lineage.source == "eval"``).

    An eval score is the number you report; training on it makes the
    held-out scorer the reward model. The count is surfaced on every
    selector report so the leak is visible before a run starts.
    """
    return sum(
        1
        for row in rows
        if isinstance(row, dict) and (row.get("lineage") or {}).get("source") == "eval"
    )


def _eval_sourced_warning(count: int, unit: str) -> str:
    return (
        f"{count} {unit} carry rewards from evaluate() (lineage.source == 'eval'); "
        "training on them turns the held-out scorer into the reward model"
    )


def _stable_key(text: str, salt: str = "") -> str:
    return hashlib.sha256(f"{salt}:{text}".encode()).hexdigest()


def _binary_label(row: dict) -> int | None:
    for key in ("reward", "qwen_reward"):
        value = row.get(key)
        if value is None or isinstance(value, bool):
            continue
        if _is_binary_01(value):
            return int(float(value))
    return None


SFT_SELECTIONS = ("top_per_prompt", "random_per_prompt", "top_k_overall", "random_k_overall")


def _scalar_reward(row: dict) -> float | None:
    for key in ("reward", "qwen_reward"):
        value = row.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number:
            return number
    return None


def select_for_sft(
    rows: Sequence[dict],
    *,
    target: int = 1000,
    select: str = "top_per_prompt",
    k: int | None = None,
    min_reward: float = 1.0,
    seed: int = 0,
) -> tuple[list[dict], dict[str, Any]]:
    """Diverse correct demonstrations, at most ``target`` rows.

    Imitation clones what it sees, so only rows whose reward reaches
    ``min_reward`` (default 1.0: judge-approved) and that are not junk
    qualify; unanimity is not a problem here. A grader with partial
    credit ranks by its score: lower ``min_reward`` to admit it.

    ``select`` is the rejection-sampling rule (rlhf-book ch. 9, "Scoring
    Completions"): ``"top_per_prompt"`` keeps each prompt's highest-reward
    completion and then round-robins across behavior signatures (tool
    sequence, argument provenance, outcome shape) so every distinct way
    of being right appears before any repeats; ``"top_k_overall"`` keeps
    the ``k`` highest-reward completions across all prompts, several per
    prompt allowed; the two ``random_*`` rules are the book's control
    (same counts, seeded random picks) so a claimed gain from selection
    can be checked against chance. ``k`` defaults to ``target``.
    """
    if select not in SFT_SELECTIONS:
        raise ValueError(f"select must be one of {', '.join(SFT_SELECTIONS)}; got {select!r}")
    rng = random.Random(int(seed))
    scored: list[tuple[float, dict]] = []
    n_wrong = n_junk = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = _scalar_reward(row)
        if value is None or value < float(min_reward):
            n_wrong += 1
            continue
        # passing rows are judge-approved; only structural junk drops.
        if is_incomplete_junk(row):
            n_junk += 1
            continue
        scored.append((value, row))
    goal = max(1, int(target))
    limit = max(1, int(k)) if k else goal
    if select in ("top_k_overall", "random_k_overall"):
        if select == "top_k_overall":
            scored.sort(
                key=lambda pair: (
                    -pair[0],
                    _stable_key(str(pair[1].get("prompt") or ""), behavior_signature(pair[1])),
                )
            )
        else:
            rng.shuffle(scored)
        picked_overall = [row for _, row in scored[:limit]]
        pool = [row for _, row in scored]
        return picked_overall, _sft_report(
            rows, pool, picked_overall, n_wrong, n_junk, goal, select, limit, min_reward
        )
    by_prompt: dict[str, list[tuple[float, dict]]] = {}
    for value, row in scored:
        prompt = " ".join(str(row.get("prompt") or "").lower().split())
        by_prompt.setdefault(prompt, []).append((value, row))
    eligible: list[dict] = []
    for prompt, candidates in by_prompt.items():
        if select == "random_per_prompt":
            eligible.append(rng.choice(candidates)[1])
            continue
        candidates.sort(
            key=lambda pair: (-pair[0], _stable_key(prompt, behavior_signature(pair[1])))
        )
        eligible.append(candidates[0][1])
    buckets: dict[str, list[dict]] = {}
    for row in eligible:
        buckets.setdefault(behavior_signature(row), []).append(row)
    for sig, bucket in buckets.items():
        bucket.sort(key=lambda r: _stable_key(str(r.get("prompt") or ""), sig))
    order = sorted(buckets, key=lambda sig: (-len(buckets[sig]), sig))
    selected: list[dict] = []
    round_i = 0
    while len(selected) < goal:
        took = False
        for sig in order:
            bucket = buckets[sig]
            if round_i < len(bucket):
                selected.append(bucket[round_i])
                took = True
                if len(selected) >= goal:
                    break
        if not took:
            break
        round_i += 1
    return selected, _sft_report(
        rows, eligible, selected, n_wrong, n_junk, goal, select, None, min_reward
    )


def _sft_report(
    rows: Sequence[dict],
    eligible: Sequence[dict],
    selected: Sequence[dict],
    n_wrong: int,
    n_junk: int,
    goal: int,
    select: str,
    k: int | None,
    min_reward: float,
) -> dict[str, Any]:
    def _mean(items: Sequence[dict]) -> float | None:
        values = [v for v in (_scalar_reward(r) for r in items) if v is not None]
        return round(sum(values) / len(values), 4) if values else None

    buckets = {behavior_signature(r) for r in eligible}
    report: dict[str, Any] = {
        "n": len(rows),
        "n_eligible": len(eligible),
        "n_selected": len(selected),
        "n_not_pass": n_wrong,
        "n_junk": n_junk,
        "unique_behaviors": len(buckets),
        "behaviors_covered": len({behavior_signature(r) for r in selected}),
        "target": goal,
        "selection": select,
        "min_reward": float(min_reward),
        "reward_mean_eligible": _mean(eligible),
        "reward_mean_selected": _mean(selected),
        "eval_sourced": eval_sourced(selected),
    }
    if k is not None:
        report["k"] = k
    if report["eval_sourced"]:
        report["warning"] = _eval_sourced_warning(report["eval_sourced"], "selected row(s)")
    # Rejection sampling picks the best of N completions per prompt, and
    # the published recipes use 10 to 30 (rlhf-book ch. 9); fewer makes
    # the pick biased or noisy. With k=1 there is no pick at all, only a
    # pass/fail filter, which is fine for a first SFT set but worth saying.
    per_prompt: dict[str, int] = {}
    for row in rows:
        if isinstance(row, dict):
            key = " ".join(str(row.get("prompt") or "").lower().split())
            per_prompt[key] = per_prompt.get(key, 0) + 1
    max_k = max(per_prompt.values(), default=0)
    report["completions_per_prompt_max"] = max_k
    if 0 < max_k < 10:
        report["note"] = (
            f"at most {max_k} completion(s) per prompt; rejection-sampling selection "
            "wants 10 to 30 so the pick is not biased (rlhf-book ch. 9). Raise "
            "repeats= if you mean to choose among completions rather than filter."
        )
    return report


def trim_out_of_band(
    rows: Sequence[dict],
    *,
    lo: float = DEFAULT_BAND[0],
    hi: float = DEFAULT_BAND[1],
    min_k: int = 2,
) -> tuple[list[dict], dict[str, Any]]:
    """Drop asks whose pass rate over k >= ``min_k`` rollouts sits outside
    [``lo``, ``hi``]. The offline difficulty filter: an ask the policy
    almost always or almost never passes carries little gradient per
    rollout. Unanimous asks are ``trim_unanimous_groups``'s job and are
    left alone here; singles always stay.
    """
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError(f"band must satisfy 0 <= lo <= hi <= 1, got ({lo}, {hi})")
    groups = _group_label_lists(rows)
    too_easy: set[str] = set()
    too_hard: set[str] = set()
    for prompt, labels in groups.items():
        if len(labels) < max(2, int(min_k)):
            continue
        p = sum(labels) / len(labels)
        if not 0.0 < p < 1.0:
            continue
        if p > hi:
            too_easy.add(prompt)
        elif p < lo:
            too_hard.add(prompt)
    dead = too_easy | too_hard
    kept = [row for row in rows if str(row.get("prompt") or "") not in dead]
    return kept, {
        "n": len(rows),
        "n_kept": len(kept),
        "n_dropped": len(rows) - len(kept),
        "n_groups_dropped": len(dead),
        "too_easy": len(too_easy),
        "too_hard": len(too_hard),
        "band": [lo, hi],
    }


def select_for_rl(
    rows: Sequence[dict],
    *,
    target: int = 1000,
    lo: float = DEFAULT_BAND[0],
    hi: float = DEFAULT_BAND[1],
    enforce_band: bool = True,
    has_tools: bool = True,
    dedupe: bool = True,
    drop_truncated: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    """Whole mixed groups up to roughly ``target`` rows. Groups never split.

    After the row gates, duplicate and truncated rollouts (``dedupe``,
    ``drop_truncated``), the unanimous trim, and (``enforce_band``) the
    difficulty band, remaining asks are ranked by distance from p = 0.5
    and taken round-robin across observed fault kinds, so the dataset
    keeps a grounded spread of no-fault, miss, timeout, and already-done
    situations rather than one over-represented failure. The last group
    may overshoot ``target``; an RL update wants the complete group or
    none of it. ``enforce_band=False`` keeps out-of-band asks and only
    ranks them last. The report's ``correlations`` block is the
    reward-hack scan over the selection: reward tracking reply length or
    tool count is a judge problem, flagged in ``hygiene_warnings``.
    """
    from .hygiene import (
        dedupe_groups,
        hygiene_warnings,
        length_report,
        reward_correlations,
    )
    from .hygiene import drop_truncated as _drop_truncated

    kept, base_report = filter_rl_rows(rows, has_tools=has_tools)
    original_sizes: dict[str, int] = {}
    for row in kept:
        key = str(row.get("prompt") or "")
        original_sizes[key] = original_sizes.get(key, 0) + 1
    dup_report: dict[str, Any] = {"n_dropped": 0, "groups_affected": 0, "conflicting_rewards": 0}
    if dedupe:
        kept, dup_report = dedupe_groups(kept)
    trunc_report: dict[str, Any] = {"n_dropped": 0}
    if drop_truncated:
        kept, trunc_report = _drop_truncated(kept)
    kept, trim_report = trim_unanimous_groups(kept)
    # A group that hygiene shrank to one rollout, or to rollouts that all
    # agree, has no contrast left. It is not a single (which always stays,
    # since it never had a group) but a dead group, and goes the same way.
    collapsed = {
        prompt
        for prompt, labels in _group_label_lists(kept).items()
        if original_sizes.get(prompt, 0) >= 2 and len(set(labels)) == 1
    }
    if collapsed:
        kept = [row for row in kept if str(row.get("prompt") or "") not in collapsed]
        trim_report["n_groups_dropped"] += len(collapsed)
    trim_report["collapsed_groups_dropped"] = len(collapsed)
    band_report: dict[str, Any] = {"n_groups_dropped": 0, "too_easy": 0, "too_hard": 0}
    if enforce_band:
        kept, band_report = trim_out_of_band(kept, lo=lo, hi=hi)
    groups: dict[str, list[dict]] = {}
    for row in kept:
        groups.setdefault(str(row.get("prompt") or ""), []).append(row)

    def _score(prompt: str) -> tuple:
        labels = [lbl for lbl in (_binary_label(r) for r in groups[prompt]) if lbl is not None]
        p = sum(labels) / len(labels) if labels else 0.0
        in_band = lo <= p <= hi
        return (0 if in_band else 1, abs(p - 0.5), _stable_key(prompt))

    fault_buckets: dict[str, list[str]] = {}
    for prompt, members in groups.items():
        fault = trace_fault(members[0])
        fault_buckets.setdefault(fault, []).append(prompt)
    for fault in fault_buckets:
        fault_buckets[fault].sort(key=_score)
    fault_order = sorted(fault_buckets, key=lambda f: (-len(fault_buckets[f]), f))
    selected: list[dict] = []
    picked_groups = 0
    goal = max(1, int(target))
    round_i = 0
    while len(selected) < goal:
        took = False
        for fault in fault_order:
            prompts = fault_buckets[fault]
            if round_i < len(prompts):
                selected.extend(groups[prompts[round_i]])
                picked_groups += 1
                took = True
                if len(selected) >= goal:
                    break
        if not took:
            break
        round_i += 1
    report = {
        "n": len(rows),
        "n_after_gates": base_report["n_kept"],
        "n_after_trim": len(kept),
        "unanimous_groups_dropped": trim_report["n_groups_dropped"],
        "collapsed_groups_dropped": trim_report["collapsed_groups_dropped"],
        "n_selected": len(selected),
        "groups_selected": picked_groups,
        "fault_kinds": {fault: len(prompts) for fault, prompts in fault_buckets.items()},
        "target": goal,
        "band": [lo, hi],
        "enforce_band": bool(enforce_band),
        "band_groups_dropped": band_report["n_groups_dropped"],
        "band_dropped": {"too_easy": band_report["too_easy"], "too_hard": band_report["too_hard"]},
        "duplicates": dup_report,
        "truncated_dropped": trunc_report["n_dropped"],
        "length": length_report(selected),
        "correlations": reward_correlations(selected),
        "signal": group_signal(selected, lo=lo, hi=hi),
        "eval_sourced": eval_sourced(selected),
    }
    report["hygiene_warnings"] = hygiene_warnings(
        duplicates=dup_report,
        lengths=report["length"],
        correlations=report["correlations"],
    )
    if report["eval_sourced"]:
        report["hygiene_warnings"].append(
            _eval_sourced_warning(report["eval_sourced"], "selected row(s)")
        )
    # A selection with no mixed group has no within-group contrast: GRPO
    # advantage is zero everywhere and the run trains nothing. That is a
    # grading or difficulty problem upstream, and it must not exit this
    # function looking like a dataset.
    if report["signal"].get("n_mixed", 0) == 0:
        report["warning"] = (
            "no_mixed_groups: every selected group is unanimous or single, "
            "so group-relative advantages are all zero. Regrade with a "
            "stricter rubric or raise difficulty (fault_rate, harder asks) "
            "before training on this."
        )
    return selected, report


def _default_output(src: str) -> str:
    path = Path(src)
    return str(path.with_name(path.stem + ".rl" + (path.suffix or ".jsonl")))


def optimize_for_rl(
    source, *, output: str | None = None, trim_unanimous: bool = True
) -> dict[str, Any]:
    """Filter a JSONL path or a row list. Writes kept rows when given a path.

    Given a path and no ``output``, kept rows land next to the source as
    ``<name>.rl.jsonl``; the source file is never overwritten. Passing
    ``output`` equal to the source is an explicit overwrite and allowed.

    Drops three row classes, then (``trim_unanimous=True``) whole asks
    whose k rollouts all landed 0 or all landed 1, which are dead
    gradient for a grouped update:

    * ``do_nothing``: the situation wanted a tool (``ask_family=tool``,
      known tool, world/fault, or an actionable opener) and the agent
      never called one. Those rows teach the policy to stop using tools.
    * ``incomplete_junk``: empty or infra stub, degenerate text, leaked
      internal test text, raw tool markup, truncated reply, or a
      conversation that ends on the user or a tool result.
    * ``unusable_label``: ``reward`` (or ``qwen_reward``) missing or not 0/1.

    An injected missed tool call is not a drop by itself. A row that
    called the tool and reacted stays if the 0/1 label is present.
    The report's ``signal`` block is the within-ask contrast after
    filtering; rerun the simulator if ``mixed_rate`` is low.
    """
    if isinstance(source, (str, Path)):
        rows = load_jsonl(source)
        src = str(source)
    else:
        rows = list(source)
        src = ""
    kept, report = filter_rl_rows(rows)
    if trim_unanimous:
        kept, trim_report = trim_unanimous_groups(kept)
        report["n_kept"] = len(kept)
        report["n_dropped"] = report["n"] - len(kept)
        report["unanimous_groups_dropped"] = trim_report["n_groups_dropped"]
        report["unanimous_rows_dropped"] = trim_report["n_dropped"]
        report["signal"] = trim_report["signal"]
    else:
        report["signal"] = group_signal(kept)
    dest = output or (_default_output(src) if src else "")
    written = None
    if dest:
        written = write_jsonl(dest, kept)
    if isinstance(source, list):
        source[:] = kept
    report["path"] = written or src
    report["n_written"] = len(kept) if dest else 0
    return report


def recommend(
    tools: Sequence[dict] | None = None,
    policy: str = "",
    *,
    mode: str = "sft",
    target: int | None = None,
    mixed_rate: float = 0.5,
) -> dict[str, Any]:
    """How much data this agent needs, from its own grid. No guessing.

    Grounded two ways: the agent's measured covering grid (every cell wants
    ``SATURATION_COPIES`` visits, and selection wants surplus to choose
    from), and published post-training practice (curated agent SFT lands at
    500 to 2,000 trajectories: FireAct 500, LIMA 1,000, AgentTuning 1,866;
    agent RL uses 8 to 16 rollouts per prompt and drops all-pass/all-fail
    groups: DAPO 2025, Skywork-OR1 2025).

    Returns the numbers plus ``simulate_kwargs`` ready to splat, and
    ``reasoning`` lines that show the arithmetic.
    """
    from ..generate.coverage import SATURATION_COPIES
    from ..generate.scenarios import scenario_regions

    kind = "sft" if str(mode).lower() == "sft" else "rl"
    cells = len(scenario_regions(list(tools or []), policy, mode=str(mode).lower()))
    reasoning = [f"covering grid: {cells} cells for this agent"]
    if kind == "sft":
        goal = int(target or 800)
        by_grid = cells * SATURATION_COPIES
        raw = max(by_grid, 3 * goal)
        raw = int(-(-raw // 100) * 100)
        reasoning += [
            f"saturation wants {SATURATION_COPIES} visits per cell = {by_grid} rows",
            f"selection wants about 3x its target of {goal} to choose from",
            f"generate {raw}, select {goal} diverse 1-labeled rows",
            "time_budget off: a sized run stops on rows, not the clock",
        ]
        return {
            "mode": "sft",
            "grid_cells": cells,
            "budget": raw,
            "optimize_target": goal,
            "simulate_kwargs": {"mode": "sft", "budget": raw, "time_budget": None},
            "reasoning": reasoning,
        }
    goal = int(target or 800)
    k = 8
    # Whole-group selection keeps only asks whose k rollouts disagree.
    # The surviving fraction is the agent's, not ours: a struggling agent
    # mixes on half its asks; a competent agent on a cold-start grid
    # measured 4% (field test, 48x8, hosted Qwen). Probe first: 12 asks,
    # grade, group_signal, then pass the measured rate back in here.
    rate = min(1.0, max(0.02, float(mixed_rate)))
    # Expected mixed rows = situations * k * rate, so situations =
    # goal / (k * rate). Rounding k * rate to an integer first (the old
    # form) collapsed to 1 below rate 1/16 and under-provisioned by 3x
    # at the 4% rate measured in the field.
    situations = max(2 * cells, math.ceil(goal / (k * rate)))
    raw = situations * k
    reasoning += [
        f"k={k} rollouts per ask; assumed mixed-group rate {rate:.0%} "
        "(measure it with a 12-ask probe and group_signal, then recompute)",
        f"{situations} asks x {k} = {raw} rows to select about {goal} mixed-group rows",
        "a low measured rate means the grid is too easy for this agent: "
        "use traces=, harder cells, or a stricter judge prompt before "
        "buying more rollouts",
        "in post-training terms the mixed rate is the pass@k - pass@1 headroom "
        "(group_signal reports both); pass@k near pass@1 means nothing to learn",
    ]
    return {
        "mode": "rl",
        "grid_cells": cells,
        "budget": raw,
        "situations": situations,
        "rollouts_per_request": k,
        "mixed_rate": rate,
        "optimize_target": goal,
        "simulate_kwargs": {
            "mode": "rl",
            "situations": situations,
            "budget": raw,
            "time_budget": None,
        },
        "reasoning": reasoning,
    }


def optimize(
    source,
    *,
    mode: str | None = None,
    target: int = 1000,
    output: str | None = None,
    band: tuple[float, float] = DEFAULT_BAND,
    enforce_band: bool = True,
    select: str = "top_per_prompt",
    min_reward: float = 1.0,
) -> tuple[list[dict], dict[str, Any]]:
    """One call after grading: concentrate for the post-training target.

    ``source`` is a ``SimulationData``, a row list, or a JSONL path.
    ``mode`` defaults to the data's own mode: ``"sft"`` picks diverse
    correct demonstrations (``select`` and ``min_reward`` as in
    ``select_for_sft``), anything else keeps whole mixed RL groups
    inside the difficulty ``band`` (default 20%-80% pass rate;
    ``enforce_band=False`` only ranks out-of-band asks last).
    Returns ``(rows, report)``; writes ``output`` when given, or
    ``<name>.<mode>.jsonl`` next to a path source. Never overwrites the
    source file unless ``output`` names it explicitly.
    """
    resolved = mode
    src = ""
    has_tools = True
    if hasattr(source, "trajectories"):
        rows = list(source.trajectories)
        resolved = resolved or getattr(source, "mode", None)
        profile = getattr(source, "profile", None)
        if profile is not None:
            has_tools = bool(getattr(profile, "tools", None))
    elif isinstance(source, (str, Path)):
        rows = load_jsonl(source)
        src = str(source)
    else:
        rows = list(source)
    resolved = "sft" if str(resolved or "").lower() == "sft" else "rl"
    if resolved == "sft":
        picked, report = select_for_sft(rows, target=target, select=select, min_reward=min_reward)
    else:
        picked, report = select_for_rl(
            rows,
            target=target,
            lo=float(band[0]),
            hi=float(band[1]),
            enforce_band=enforce_band,
            has_tools=has_tools,
        )
    report["mode"] = resolved
    dest = output
    if not dest and src:
        path = Path(src)
        dest = str(path.with_name(path.stem + f".{resolved}" + (path.suffix or ".jsonl")))
    if dest:
        report["path"] = write_jsonl(dest, picked)
        report["n_written"] = len(picked)
    return picked, report


optimize_rows = optimize_for_rl
