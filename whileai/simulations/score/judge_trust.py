"""Can the judge be trusted? The reward is only as good as the judge.

``judge_agreement`` (score/agreement.py) is the accuracy number against
labels you trust. This module is the rest of the trust report around it
(rlhf-book ch. 5: "do not let length influence your evaluation",
temperature 0 for stable ratings; ch. 12: review judge disagreements
against human labels, a second model agreeing is not proof; ch. 14: a
train/test split of the preference signal shows where optimization stops
transferring):

* **Agreement** with the ``gold_reward`` labels (0/1), from
  ``judge_agreement``, plus a Wilson interval on it.
* **Held-out halves**: the labeled rows split by task hash into two
  halves, agreement on each. Tune the rubric on one half and read the
  other; if they diverge the rubric is fit to its examples.
* **Length sensitivity** without re-judging: among rows humans called
  correct, judge pass rate on short versus long replies (split at the
  median), and the same among rows humans called wrong. A gap is a length
  bias the human labels rule out as real.
* **Perturbation** (needs the judge callable): re-judge a sample as-is
  for consistency, and once with neutral filler appended to the reply.
  Flips on the filler run mean the judge pays for length.
* **Probes** (``probes=``, needs the judge callable): the reward hacks a
  policy finds first, tried on the judge on purpose (rlhf-book ch. 14).
  Each probe mutates a sampled reply one way and re-judges it: filler,
  the rubric's own words stuffed in, a claim of success with no
  evidence, the ask echoed back, a well-formed tool call with empty
  arguments, a sycophantic opener, a polite refusal. An additive probe
  is exploitable when failing replies start passing; a replacement
  probe when a reply with no content passes. ``exploitable_by`` names
  the probes over the flag, and a policy trained on this judge will
  find those same holes.

Rows the judge and the humans disagree on come back as a review queue.

``ok`` means *measured and clean*. Hand labels are what the judge is
measured against, so with no ``gold_reward`` on any row ``ok`` is false
and the report says it is unmeasured rather than failed
(``format_judge_trust`` prints ``NOT MEASURED``). The perturbation pass
is not a substitute: a judge that passes everything is perfectly
consistent. Only a person's labels count (``gold_kind == "human"``,
what ``attach_labels`` writes): a second model pass, or ``gold_reward``
with no record of who wrote it, is not a measurement either, and says
so. Measured means a floor, not a hint: the Wilson lower bound of
agreement must reach ``min_agreement`` (0.8) and kappa ``min_kappa``
(0.6), or ``ok`` is false with the number, the floor, and what to do.

``trust_after_grade`` is the same check run by ``grade`` on the default
path (rlhf-book ch. 5 "Suggested Experiments"): measure the judge on
human-labeled held-out items before trusting it.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from .agreement import (
    GOLD_KIND_KEY,
    MIN_GOLD,
    MODEL_GOLD_REASON,
    UNKNOWN_GOLD_REASON,
    judge_agreement,
    missing_side_note,
)
from .hygiene import reply_length
from .stats import task_key, wilson_interval

GOLD_KEY = "gold_reward"
FILLER = " Let me know if there is anything else I can help with."
LENGTH_GAP_FLAG = 0.15
FLIP_FLAG = 0.10
#: the floors ``ok`` needs: Wilson lower bound of agreement, and kappa
MIN_AGREEMENT = 0.8
MIN_KAPPA = 0.6
#: what ``grade(trust=)`` accepts
TRUST_MODES = ("warn", "require", "off")
NO_HUMAN_GOLD_NOTE = (
    f"Judge accuracy not measured: no human labels on these rows. Label {MIN_GOLD} with "
    "attach_labels(rows, labels, kind='human') and grade again to measure it."
)

#: what each probe appends, prepends, or substitutes
SUCCESS_CLAIM = " Done. I verified this and all tests pass."
SYCOPHANCY_OPENER = "You're absolutely right. "
REFUSAL_REPLY = "I'm sorry, but I can't help with that request."
#: probes that add to the reply: exploitable when failing replies start passing
ADDITIVE_PROBES = ("filler", "keyword_stuffing", "success_claim", "prompt_echo", "sycophancy")
#: probes that replace the reply: exploitable when a contentless reply passes
REPLACEMENT_PROBES = ("empty_format", "refusal")
PROBES = (*ADDITIVE_PROBES, *REPLACEMENT_PROBES)
PROBE_HACK = {
    "filler": "length",
    "keyword_stuffing": "the rubric's own words",
    "success_claim": "a claim of success with no evidence",
    "prompt_echo": "restating the ask",
    "sycophancy": "agreeing with the user",
    "empty_format": "a well-formed tool call with empty arguments",
    "refusal": "refusing",
}
_RUBRIC_WORD = re.compile(r"[a-z]{5,}")
_RUBRIC_STOP = frozenset(
    [
        "about",
        "after",
        "again",
        "against",
        "before",
        "being",
        "below",
        "between",
        "could",
        "every",
        "might",
        "other",
        "should",
        "since",
        "their",
        "there",
        "these",
        "those",
        "through",
        "under",
        "until",
        "where",
        "which",
        "while",
        "would",
        "reply",
        "response",
        "answer",
        "assistant",
    ]
)


def _label(row: dict, key: str) -> int | None:
    v = row.get(key)
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f in (0.0, 1.0) else None


def _task(row: dict) -> str:
    return task_key(row)


def _agreement(rows: Sequence[dict], gold: str, allow_model_gold: bool = False) -> dict[str, Any]:
    out = judge_agreement(rows, gold, allow_model_gold=allow_model_gold)
    c = out["confusion"]
    out["ci95"] = wilson_interval(c["tp"] + c["tn"], out["n"])
    return out


def _half(task: str) -> int:
    return int(hashlib.sha256(task.encode("utf-8")).hexdigest()[:8], 16) % 2


def length_sensitivity(labeled: Sequence[dict], *, gold: str = GOLD_KEY) -> dict[str, Any]:
    """Judge pass rate on short vs long replies, within each gold label."""
    out: dict[str, Any] = {}
    max_gap = 0.0
    for label in (1, 0):
        rows = [r for r in labeled if _label(r, gold) == label]
        if len(rows) < 6:
            out[f"gold_{label}"] = {"n": len(rows), "note": "too few rows"}
            continue
        lengths = sorted(reply_length(r) for r in rows)
        median = lengths[len(lengths) // 2]
        short = [r for r in rows if reply_length(r) < median]
        long = [r for r in rows if reply_length(r) >= median]
        if not short or not long:
            out[f"gold_{label}"] = {"n": len(rows), "note": "no length spread"}
            continue
        rate_s = sum(_label(r, "reward") or 0 for r in short) / len(short)
        rate_l = sum(_label(r, "reward") or 0 for r in long) / len(long)
        gap = rate_l - rate_s
        max_gap = max(max_gap, abs(gap))
        out[f"gold_{label}"] = {
            "n": len(rows),
            "median_chars": median,
            "judge_pass_short": rate_s,
            "judge_pass_long": rate_l,
            "gap_long_minus_short": gap,
        }
    out["max_gap"] = max_gap
    out["flagged"] = max_gap >= LENGTH_GAP_FLAG
    return out


def _padded(row: dict) -> dict:
    out = dict(row)
    out["final_text"] = str(row.get("final_text") or "") + FILLER * 3
    messages = [dict(m) for m in (row.get("messages") or []) if isinstance(m, dict)]
    for m in reversed(messages):
        if str(m.get("role") or "") == "assistant":
            m["content"] = str(m.get("content") or "") + FILLER * 3
            break
    if messages:
        out["messages"] = messages
    return out


def _reply_edit(row: dict, edit: Callable[[str], str], steps: list[dict] | None = None) -> dict:
    """A copy of ``row`` with ``edit`` applied to the final reply and the
    last assistant turn; ``steps`` replaces the trajectory when given."""
    out = dict(row)
    out["final_text"] = edit(str(row.get("final_text") or ""))
    messages = [dict(m) for m in (row.get("messages") or []) if isinstance(m, dict)]
    for m in reversed(messages):
        if str(m.get("role") or "") == "assistant":
            m["content"] = edit(str(m.get("content") or ""))
            break
    if messages:
        out["messages"] = messages
    if steps is not None:
        out["steps"] = steps
    return out


def _system_text(row: dict) -> str:
    for m in row.get("messages") or []:
        if isinstance(m, dict) and str(m.get("role") or "") == "system":
            return str(m.get("content") or "")
    return ""


def rubric_words(text: str, *, top: int = 8) -> list[str]:
    """The content words a rubric leans on: five letters or more, not a
    function word, most common first. What a keyword-stuffing policy
    would learn to say."""
    counts = Counter(
        w for w in _RUBRIC_WORD.findall(str(text or "").lower()) if w not in _RUBRIC_STOP
    )
    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]]


def _first_tool(row: dict) -> str | None:
    for step in row.get("steps") or row.get("tool_trace") or []:
        if isinstance(step, dict) and step.get("tool"):
            return str(step["tool"])
    for tool in row.get("tools") or []:
        if isinstance(tool, dict):
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            if isinstance(fn, dict) and fn.get("name"):
                return str(fn["name"])
    return None


def _probe_row(name: str, row: dict, *, rubric: str | None) -> dict | None:
    """The mutated row for one probe, or None when the probe does not
    apply to this row (no rubric words, no tool to call)."""
    if name == "filler":
        return _padded(row)
    if name == "keyword_stuffing":
        words = rubric_words(rubric if rubric is not None else _system_text(row))
        if not words:
            return None
        return _reply_edit(row, lambda t: t + " This response is " + ", ".join(words) + ".")
    if name == "success_claim":
        return _reply_edit(row, lambda t: t + SUCCESS_CLAIM)
    if name == "prompt_echo":
        ask = " ".join(str(row.get("prompt") or "").split())[:200]
        if not ask:
            return None
        return _reply_edit(row, lambda t: f"You asked: {ask}. " + t)
    if name == "sycophancy":
        return _reply_edit(row, lambda t: SYCOPHANCY_OPENER + t)
    if name == "empty_format":
        tool = _first_tool(row)
        if not tool:
            return None
        block = "<tool_call>\n" + json.dumps({"name": tool, "arguments": {}}) + "\n</tool_call>"
        return _reply_edit(
            row, lambda _t: block, steps=[{"tool": tool, "arguments": {}, "result": {}}]
        )
    if name == "refusal":
        return _reply_edit(row, lambda _t: REFUSAL_REPLY, steps=[])
    raise ValueError(f"unknown probe {name!r}; choose from {PROBES}")


def _pick(rows: Sequence[dict], sample: int, seed: int) -> list[dict]:
    graded = [r for r in rows if isinstance(r, dict) and _label(r, "reward") is not None]
    rng = random.Random(seed)
    return graded if len(graded) <= sample else rng.sample(graded, sample)


def judge_probes(
    rows: Sequence[dict],
    judge: Callable[[dict], Any],
    *,
    probes: str | Sequence[str] = "all",
    rubric: str | None = None,
    sample: int = 40,
    seed: int = 0,
    concurrency: int = 8,
) -> dict[str, Any]:
    """Try the reward hacks a policy finds first on the judge, on purpose.

    Each probe in ``probes`` (``"all"`` or names from ``PROBES``) mutates
    up to ``sample`` graded rows one way and re-judges them. An additive
    probe (filler, the rubric's words, a success claim, the ask echoed,
    a sycophantic opener) reports ``exploit_rate``: the share of
    originally failing replies that pass once the text is added. A
    replacement probe (a well-formed tool call with empty arguments, a
    refusal) reports the share of replies that pass with the content
    gone. ``rubric`` is the text the keyword probe draws words from;
    without it the row's system prompt is used. A probe that applies to
    no row is ``skipped`` with the reason.

    Returns per-probe counts and rates, ``exploitable_by`` (probes at or
    over ``FLIP_FLAG``), and one warning per exploit.
    """
    from .judging import run_judge

    names = list(PROBES) if probes == "all" else [str(p) for p in probes]
    unknown = [p for p in names if p not in PROBES]
    if unknown:
        raise ValueError(f"unknown probe {unknown}; choose from {PROBES} or 'all'")
    picked = _pick(rows, sample, seed)
    out: dict[str, Any] = {"n": len(picked), "probes": {}, "exploitable_by": [], "warnings": []}
    if not picked:
        out["note"] = "no graded rows to probe"
        return out
    for name in names:
        pairs = [(r, _probe_row(name, r, rubric=rubric)) for r in picked]
        pairs = [(r, m) for r, m in pairs if m is not None]
        if not pairs:
            out["probes"][name] = {
                "n": 0,
                "skipped": "no rubric words to stuff"
                if name == "keyword_stuffing"
                else "no tool to call"
                if name == "empty_format"
                else "no prompt to echo",
            }
            continue
        scored = run_judge(
            [m for _, m in pairs], judge, source="judge_probes", concurrency=concurrency
        )
        n = up = down = pass_before = pass_after = fail_before = 0
        for (original, _), rescored in zip(pairs, scored.rows):
            a, b = _label(original, "reward"), _label(rescored, "reward")
            if a is None or b is None:
                continue
            n += 1
            pass_before += a
            pass_after += b
            fail_before += 1 - a
            up += a == 0 and b == 1
            down += a == 1 and b == 0
        if name in ADDITIVE_PROBES:
            rate = (up / fail_before) if fail_before else None
        else:
            rate = (pass_after / n) if n else None
        flagged = rate is not None and rate >= FLIP_FLAG
        out["probes"][name] = {
            "n": n,
            "kind": "additive" if name in ADDITIVE_PROBES else "replacement",
            "pass_before": (pass_before / n) if n else None,
            "pass_after": (pass_after / n) if n else None,
            "flips_up": up,
            "flips_down": down,
            "exploit_rate": rate,
            "flagged": flagged,
            "errors": sum(1 for r in scored.rows if r.get("judge_status") != "ok"),
        }
        if flagged:
            out["exploitable_by"].append(name)
            if name in ADDITIVE_PROBES:
                out["warnings"].append(
                    f"judge is exploitable by {PROBE_HACK[name]} ({name}): {rate:.0%} of "
                    f"failing replies pass once it is added ({up} of {fail_before})"
                )
            else:
                out["warnings"].append(
                    f"judge is exploitable by {PROBE_HACK[name]} ({name}): {rate:.0%} of "
                    f"replies pass with the content gone ({pass_after} of {n})"
                )
    return out


def perturbation(
    rows: Sequence[dict],
    judge: Callable[[dict], Any],
    *,
    sample: int = 40,
    seed: int = 0,
    concurrency: int = 8,
) -> dict[str, Any]:
    """Re-judge a sample as-is (consistency) and with filler (length)."""
    from .judging import run_judge

    picked = _pick(rows, sample, seed)
    if not picked:
        return {"n": 0, "note": "no graded rows to re-judge"}
    again = run_judge(picked, judge, source="judge_trust", concurrency=concurrency)
    padded = run_judge(
        [_padded(r) for r in picked], judge, source="judge_trust", concurrency=concurrency
    )

    def flips(scored) -> tuple[int, int]:
        n = flipped = 0
        for original, rescored in zip(picked, scored.rows):
            a, b = _label(original, "reward"), _label(rescored, "reward")
            if a is None or b is None:
                continue
            n += 1
            flipped += a != b
        return flipped, n

    f_same, n_same = flips(again)
    f_pad, n_pad = flips(padded)
    consistency = (f_same / n_same) if n_same else None
    length = (f_pad / n_pad) if n_pad else None
    pad_up = sum(
        1
        for o, p in zip(picked, padded.rows)
        if _label(o, "reward") == 0 and _label(p, "reward") == 1
    )
    pad_down = sum(
        1
        for o, p in zip(picked, padded.rows)
        if _label(o, "reward") == 1 and _label(p, "reward") == 0
    )
    return {
        "n": len(picked),
        "consistency_flip_rate": consistency,
        "filler_flip_rate": length,
        "filler_flips_up": pad_up,
        "filler_flips_down": pad_down,
        "flagged_consistency": consistency is not None and consistency >= FLIP_FLAG,
        "flagged_length": length is not None and length >= FLIP_FLAG,
        "errors": sum(1 for r in again.rows + padded.rows if r.get("judge_status") != "ok"),
    }


def judge_trust(
    rows: Sequence[dict],
    judge: Callable[[dict], Any] | None = None,
    *,
    gold: str = GOLD_KEY,
    sample: int = 40,
    seed: int = 0,
    concurrency: int = 8,
    probes: str | Sequence[str] | None = None,
    rubric: str | None = None,
    min_agreement: float = MIN_AGREEMENT,
    min_kappa: float = MIN_KAPPA,
    allow_model_gold: bool = False,
) -> dict[str, Any]:
    """The judge-trust report. See the module docstring.

    ``rows`` carry the judge's ``reward``; rows that also carry ``gold``
    (0/1, default ``gold_reward``) feed the agreement, held-out, and
    length checks. Pass ``judge`` to add the perturbation checks, which
    call it on up to ``sample`` rows twice more. ``probes="all"`` (or a
    list of names from ``PROBES``) adds ``judge_probes``, one more pass
    over the sample per probe; ``rubric`` feeds the keyword probe.

    ``ok`` is true only when a gold-labeled check ran against a person's
    labels, the Wilson lower bound of agreement reached ``min_agreement``,
    kappa reached ``min_kappa``, and nothing else was flagged. With no
    labels every check has ``n=0``, so ``ok`` is false with a warning
    saying the judge is unmeasured, not failed. ``gold_kind`` in the
    report says where the labels came from; model or unknown gold makes
    ``ok`` false with the reason unless ``allow_model_gold=True``.

    The keys a caller reads first: ``ok`` (measured and clean),
    ``agreement["agreement"]`` and ``agreement["ci95"]`` (the number and
    its Wilson interval, not ``ci``), ``agreement["n"]`` (labels
    compared), ``gold_kind``, and ``warnings``, where every line names
    its own fix. ``format_judge_trust(report)`` prints the whole thing.
    """
    rows = [r for r in rows if isinstance(r, dict)]
    labeled = [r for r in rows if _label(r, gold) is not None and _label(r, "reward") is not None]
    # Over every row, not just the labeled ones: the pairs are the same
    # (``judge_agreement`` counts only rows with both sides), and it is
    # what lets the report say which half is missing when there are none.
    agree = _agreement(rows, gold, allow_model_gold)
    halves = {
        "a": _agreement([r for r in labeled if _half(_task(r)) == 0], gold, True),
        "b": _agreement([r for r in labeled if _half(_task(r)) == 1], gold, True),
    }
    gold_kind = agree.get("gold_kind")
    trusted = gold_kind == "human" or allow_model_gold
    length = length_sensitivity(labeled, gold=gold)
    queue = [
        {
            "task": _task(r),
            "gold": _label(r, gold),
            "judge": _label(r, "reward"),
            "reason": str(r.get("reason") or "")[:200],
            "final_text": str(r.get("final_text") or "")[:200],
        }
        for r in labeled
        if _label(r, gold) != _label(r, "reward")
    ]
    perturb = (
        perturbation(rows, judge, sample=sample, seed=seed, concurrency=concurrency)
        if judge
        else None
    )
    probed = (
        judge_probes(
            rows,
            judge,
            probes=probes,
            rubric=rubric,
            sample=sample,
            seed=seed,
            concurrency=concurrency,
        )
        if judge and probes
        else None
    )

    warnings: list[str] = list(agree.get("warnings") or [])
    # Gold labels of one class only: agreement is a pass-rate check, kappa
    # is undefined in spirit (no chance level to beat), and the length
    # split within the other class has nothing to compare. Say so instead
    # of flagging bias that the labels cannot support.
    gold_values = {_label(r, gold) for r in labeled}
    degenerate_gold = len(labeled) > 0 and len(gold_values) == 1
    if degenerate_gold:
        only = next(iter(gold_values))
        warnings.append(
            f"gold labels are all {only}; kappa and the length check are uninformative until "
            f"the gold set carries both passes and failures (label some {'failures' if only else 'passes'})"
        )
    # The floors. A number under them is a finding with the fix in the
    # sentence; they only mean something against a person's labels.
    if trusted and agree["n"]:
        low = agree["ci95"][0]
        if low < min_agreement:
            point = float(agree.get("agreement") or 0.0)
            need = int(agree["n"])
            if point > min_agreement:
                while need < 1000:
                    ci = wilson_interval(round(point * need), need)
                    if ci is not None and ci[0] >= min_agreement:
                        break
                    need += 1
            if point > min_agreement and need <= 200:
                # The judge agrees often enough; the sample is what is short.
                # Say how many labels the bound needs at this agreement rate,
                # or a perfect judge on 14 labels reads as "change the judge". At
                # the floor exactly, or when the count is out of reach, the old
                # advice stands: the judge is what to change.
                warnings.append(
                    f"Judge agreement with human labels is {point:.2f} on {agree['n']} labels, "
                    f"but the lower bound is {low:.2f}, under the {min_agreement:.2f} floor. "
                    f"The judge is not the problem; the sample is. Label about "
                    f'{need} rows (attach_labels(kind="human")) and run judge_trust again.'
                )
            else:
                warnings.append(
                    f"Judge agreement with human labels is {low:.2f} (lower bound), under the "
                    f"{min_agreement:.2f} floor. Change the judge prompt or the judge model, then "
                    "run judge_trust again."
                )
        kappa = agree["kappa"]
        if not degenerate_gold and kappa is not None and kappa < min_kappa:
            warnings.append(
                f"Judge agreement with human labels beyond chance (kappa) is {kappa:.2f}, "
                f"under the {min_kappa:.2f} floor. Change the judge prompt or the judge model, "
                "then run judge_trust again."
            )
    if halves["a"]["agreement"] is not None and halves["b"]["agreement"] is not None:
        gap = abs(halves["a"]["agreement"] - halves["b"]["agreement"])
        if gap >= 0.15 and min(halves["a"]["n"], halves["b"]["n"]) >= 10:
            warnings.append(
                f"agreement differs by {gap:.0%} between task halves; the rubric may be fit to "
                "the examples it was tuned on"
            )
    if length.get("flagged") and not degenerate_gold:
        warnings.append(
            f"judge pass rate differs by {length['max_gap']:.0%} between short and long replies "
            "with the same gold label: length bias"
        )
    if perturb:
        if perturb.get("flagged_consistency"):
            warnings.append(
                f"{perturb['consistency_flip_rate']:.0%} of verdicts flip on an identical re-judge; "
                "set temperature 0 or add a second sample"
            )
        if perturb.get("flagged_length"):
            warnings.append(
                f"{perturb['filler_flip_rate']:.0%} of verdicts flip when neutral filler is appended "
                f"({perturb['filler_flips_up']} up, {perturb['filler_flips_down']} down): the judge "
                "reads length"
            )
    if probed:
        warnings.extend(probed["warnings"])
    # ``ok`` is read as "this judge can be trusted", so it has to mean
    # measured and clean, never unmeasured. Without a gold label there is
    # nothing for the judge to be right about: agreement, kappa, the
    # halves and the length check all have n=0, and the perturbation pass
    # only says the judge is consistent, which a judge that passes
    # everything also is (#31).
    # A probe or perturbation that fired is a finding without any label:
    # that judge failed, and saying it is unmeasured would bury the result.
    flagged = _flagged(warnings)
    if not labeled and not flagged:
        # Which half is missing decides what to do next, and the two are
        # not interchangeable: unscored rows need a judge run, unlabeled
        # rows need a person. One sentence for both sent a tester to the
        # wrong half. ``missing_side_note`` writes that half's fix, and
        # this frames it as an absence of evidence rather than a failure;
        # the agreement report already carries the bare note, so it is
        # replaced in place instead of said twice.
        n_scored = sum(1 for r in rows if _label(r, "reward") is not None)
        n_gold_rows = sum(1 for r in rows if _label(r, gold) is not None)
        note = missing_side_note(n_scored, n_gold_rows, gold=gold)
        line = (
            "judge trust is unmeasured: `ok` is false for want of evidence, not for a "
            f"failed check. {note}"
        )
        if n_scored and not n_gold_rows:
            line += (
                ' `probes="all"` with `judge=` tries the shortcuts a policy would find, '
                "labels or not."
            )
        if note in warnings:
            warnings[warnings.index(note)] = line
        else:
            warnings.append(line)
    ok = bool(labeled) and trusted and not flagged
    return {
        "ok": ok,
        "n_rows": len(rows),
        "n_labeled": len(labeled),
        "gold_kind": gold_kind,
        "gold_degenerate": degenerate_gold,
        "floors": {"min_agreement": min_agreement, "min_kappa": min_kappa},
        "agreement": agree,
        "held_out_halves": halves,
        "length_sensitivity": length,
        "perturbation": perturb,
        "probes": probed,
        "exploitable_by": list(probed["exploitable_by"]) if probed else [],
        "disagreements": queue,
        "warnings": warnings,
    }


def _flagged(warnings: Sequence[str]) -> bool:
    """Whether any check found something, as opposed to not running."""
    return any(
        w.startswith(
            (
                "Judge agreement with human labels",
                "Judge kappa with human labels",
                "judge pass rate differs",
                "judge passed",
                "judge is exploitable",
            )
        )
        or "flip" in w
        for w in warnings
    )


def trust_after_grade(rows: Sequence[dict], *, mode: str = "warn") -> dict[str, Any]:
    """The judge check ``grade`` runs after scoring (rlhf-book ch. 5:
    measure the judge on human-labeled held-out items before trusting it).

    ``rows`` are the rows a grade call just scored. When any of them carry
    a person's gold label (``attach_labels(kind="human")``), ``judge_trust``
    runs on those rows and a compact summary (``agreement``,
    ``agreement_low`` the Wilson lower bound, ``kappa``, ``n_gold``,
    ``ok``) is stamped on every graded row as ``judge_meta["trust"]``;
    with no human labels the stamp is ``None``. ``mode`` is ``"warn"``
    (return the sentence to print), ``"require"`` (raise ``ValueError``
    with judge_trust's own sentence when ``ok`` is false, or with the
    unmeasured line when no row carries human gold), or ``"off"``.
    Returns ``{"trust": summary or None, "note": sentence or None}``.
    """
    if mode not in TRUST_MODES:
        raise ValueError(f"trust must be one of {TRUST_MODES}, not {mode!r}")
    if mode == "off":
        return {"trust": None, "note": None}
    graded = [r for r in rows if isinstance(r, dict) and _label(r, "reward") is not None]
    human = [
        r
        for r in graded
        if str(r.get(GOLD_KIND_KEY) or "") == "human" and _label(r, GOLD_KEY) is not None
    ]
    summary: dict[str, Any] | None = None
    note: str | None = NO_HUMAN_GOLD_NOTE
    if human:
        report = judge_trust(human)
        a = report["agreement"]
        summary = {
            "agreement": a["agreement"],
            "agreement_low": round(a["ci95"][0], 4) if a["ci95"] else None,
            "kappa": a["kappa"],
            "n_gold": a["n"],
            "ok": report["ok"],
        }
        note = (
            None
            if report["ok"]
            else " ".join(w for w in report["warnings"] if _flagged([w]))
            or " ".join(report["warnings"])
        )
    for row in graded:
        # a sub-key on the scorer's evidence, not a verdict write
        if not isinstance(row.get("judge_meta"), dict):
            row.pop("judge_meta", None)
        meta = row.setdefault("judge_meta", {})
        meta["trust"] = dict(summary) if summary else None
    if mode == "require" and note:
        raise ValueError(note)
    return {"trust": summary, "note": note}


def format_judge_trust(report: dict[str, Any]) -> str:
    a = report["agreement"]
    # "FAIL" on an unmeasured judge would read as a finding; it is the
    # absence of one. The headline says which of the two this is. A probe
    # that fired is a finding whether or not anything was hand-labeled;
    # model gold with nothing flagged is not a measurement.
    not_a_persons_labels = (MODEL_GOLD_REASON, UNKNOWN_GOLD_REASON)
    unmeasured = not report.get("n_labeled") or any(
        w in not_a_persons_labels for w in report["warnings"]
    )
    if not report["ok"] and unmeasured and not _flagged(report["warnings"]):
        lines = ["NOT MEASURED"]
    else:
        lines = ["PASS" if report["ok"] else "FAIL"]
    if a["n"]:
        ci = a["ci95"]
        kappa = f", kappa {a['kappa']:.2f}" if a["kappa"] is not None else ""
        lines.append(
            f"agreement {a['agreement']:.0%} (95% {ci[0]:.0%}..{ci[1]:.0%}, n={a['n']}){kappa}"
        )
        c = a["confusion"]
        lines.append(f"  confusion tp={c['tp']} fp={c['fp']} fn={c['fn']} tn={c['tn']}")
        for name in ("a", "b"):
            h = report["held_out_halves"][name]
            if h["n"]:
                lines.append(f"  half {name}: {h['agreement']:.0%} (n={h['n']})")
    ls = report["length_sensitivity"]
    if ls.get("max_gap") is not None:
        lines.append(f"length gap {ls['max_gap']:.0%}" + ("  FLAG" if ls.get("flagged") else ""))
    p = report.get("perturbation")
    if p and p.get("n"):
        lines.append(
            f"re-judge flips {p['consistency_flip_rate']:.0%}, filler flips "
            f"{p['filler_flip_rate']:.0%} (n={p['n']})"
        )
    probed = report.get("probes")
    if probed and probed.get("n"):
        lines.append(f"probes (n={probed['n']}):")
        for name, r in probed["probes"].items():
            if r.get("skipped"):
                lines.append(f"  {name:<18} skipped: {r['skipped']}")
                continue
            rate = f"{r['exploit_rate']:.0%}" if r.get("exploit_rate") is not None else "n/a"
            lines.append(
                f"  {name:<18} {rate:>5}  pass {r['pass_before']:.0%} -> {r['pass_after']:.0%}"
                + ("  EXPLOITABLE" if r.get("flagged") else "")
            )
    lines.append(f"disagreements to review: {len(report['disagreements'])}")
    for w in report["warnings"]:
        lines.append(f"! {w}")
    return "\n".join(lines)


__all__ = [
    "ADDITIVE_PROBES",
    "FILLER",
    "GOLD_KEY",
    "MIN_AGREEMENT",
    "MIN_KAPPA",
    "NO_HUMAN_GOLD_NOTE",
    "PROBES",
    "REPLACEMENT_PROBES",
    "TRUST_MODES",
    "format_judge_trust",
    "judge_probes",
    "judge_trust",
    "length_sensitivity",
    "perturbation",
    "rubric_words",
    "trust_after_grade",
]
