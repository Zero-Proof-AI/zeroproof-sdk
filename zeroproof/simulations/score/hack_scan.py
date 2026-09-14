"""What will the policy learn from this reward? Name it before training.

A grouped RL update (GRPO and its variants, rlhf-book ch. 6) baselines
every rollout against the other rollouts of the same ask. Whatever
separates reward *within* an ask is the gradient; whatever only tracks
*which* ask it is (difficulty) is subtracted away. So the question "is
the reward paying for the behavior or for a shortcut?" has to be asked
within ask too. The pooled correlation ``reward_correlations`` reports
cannot tell the two apart: hard asks get long replies and low reward,
and the pooled number calls that a length penalty.

    Var(r) = E_g[Var(r | g)]  +  Var_g(E[r | g])
              (within: the gradient)  (between: the difficulty)

This scan centers reward and every candidate feature within ask, ranks
features by that correlation, and compares the top of the ranking to a
noise floor: the 95th percentile of the same maximum when reward is
shuffled within ask (difficulty preserved, signal destroyed). A feature
above the floor is something the policy will move toward. If the user
names what the reward *should* track (``endorsed=``) and the top feature
is not it, that is the reward hack, named (rlhf-book ch. 14: over-
optimization is the training metric parting from the evaluation of
interest; the scan says on which feature).

Two feature tiers, both pure Python:

* the hand tier, always on: reply length, tool calls, turns, truncation,
  surface counts (digits, punctuation, newlines, uppercase share), one
  indicator per tool name called, one per trajectory flag that fired
  (``trace:lie.tests_claimed`` and the rest of ``score.trace``), the
  policy's mean token logprob when captured, and every numeric
  ``markers`` entry. Add your own with ``features=``.
* the auto tier (``auto=True``): presence of the ``top_k`` most common
  words and word pairs in the agent's text, plus pairwise ANDs of the
  strongest binary features. This is the tier that finds the hack nobody
  listed: a delimiter, a rubric word, an echoed fragment of the ask.

The permutation floor costs ``n_perm`` passes over the feature matrix.
Within-ask centering makes each pass a sum over a feature's non-zero
rows only, so a few thousand rollouts with a 200-term vocabulary scan
in seconds without numpy. Regimes and thresholds follow the RLVR signal
sweeps this descends from: ``sat`` above 0.2 was the one branch that
needed a calibrated cut; the rest is the floor.
"""

from __future__ import annotations

import itertools
import math
import random
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .hygiene import assistant_turns, is_truncated, reply_length, tool_calls
from .optimize import _messages

#: auto-tier vocabulary: the most common words and word pairs
DEFAULT_TOP_K = 200
#: permutations behind the noise floor
DEFAULT_N_PERM = 100
#: a feature must be non-zero on at least this many graded rows
DEFAULT_MIN_OBS = 20
#: a rival above the floor is reported when it carries at least this
#: share of the endorsed signal (integrity below 1 - RIVAL_SHARE)
RIVAL_SHARE = 0.2
#: strongest binary features that seed pairwise conjunctions
DEFAULT_SEEDS = 12
#: share of asks the policy already always passes before the pool is
#: called exhausted (the one calibrated threshold from the signal sweeps)
SAT_FLAG = 0.2
#: how many ranked features the report lists
REPORT_TOP = 20

REGIMES = ("train", "reward_hack", "pool_exhausted", "no_signal", "unknown")

_TOKEN = re.compile(r"[a-z_]{2,}|\d+|[^\sa-z_\d]")
_UPPER = re.compile(r"[A-Z]")
_PUNCT = re.compile(r"[^\w\s]")


# ----------------------------------------------------------------- inputs


def _reward(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _group_of(row: dict) -> str:
    """The GRPO group: rollouts of one ask. Same key ``group_signal`` and
    ``pass_at`` use, so the scan reads the groups the update would."""
    return str(row.get("prompt") or "")


def scan_text(row: Mapping[str, Any]) -> str:
    """Everything the agent said: the final reply and every assistant
    turn. Tool output is the world's text, not the policy's."""
    parts = [str(row.get("final_text") or "")]
    last = parts[0]
    for message in _messages(dict(row)):
        if str(message.get("role") or "") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content and content != last:
            parts.append(content)
    return "\n".join(p for p in parts if p)


def _tool_names(row: dict) -> list[str]:
    steps = row.get("steps") or row.get("tool_trace") or []
    return [str(s["tool"]) for s in steps if isinstance(s, dict) and s.get("tool")]


def hand_features(row: dict) -> dict[str, float]:
    """The hand tier for one row: the cheap things a judge rewards by
    accident, plus every marker already on the row."""
    text = scan_text(row)
    out: dict[str, float] = {
        "reply_length": float(reply_length(row)),
        "tool_calls": float(tool_calls(row)),
        "assistant_turns": float(assistant_turns(row)),
        "truncated": 1.0 if is_truncated(row) else 0.0,
        "n:words": float(len(text.split())),
        "n:digits": float(sum(c.isdigit() for c in text)),
        "n:newlines": float(text.count("\n")),
        "n:punct": float(len(_PUNCT.findall(text))),
        "frac:upper": (len(_UPPER.findall(text)) / len(text)) if text else 0.0,
    }
    for name in sorted(set(_tool_names(row))):
        out[f"tool:{name}"] = 1.0
    from .trace import trace_flags

    for name in trace_flags(row):
        out[f"trace:{name}"] = 1.0
    lp, n_tok = row.get("logprob"), row.get("n_tokens")
    if (
        isinstance(lp, (int, float))
        and not isinstance(lp, bool)
        and isinstance(n_tok, int)
        and not isinstance(n_tok, bool)
        and n_tok > 0
    ):
        out["logprob_mean"] = float(lp) / n_tok
    markers = row.get("markers")
    if isinstance(markers, dict):
        for name, value in markers.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[f"marker:{name}"] = float(value)
    return out


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def auto_terms(texts: Sequence[str], *, top_k: int, min_obs: int) -> tuple[list[str], list[set]]:
    """The ``top_k`` most common words and word pairs present in at least
    ``min_obs`` texts and absent from at least one, and each text's set."""
    present: list[set[str]] = []
    counts: Counter = Counter()
    for text in texts:
        toks = _tokens(text)
        terms = set(toks) | {f"{a} {b}" for a, b in itertools.pairwise(toks)}
        present.append(terms)
        counts.update(terms)
    n = len(texts)
    # Ties broken by the term, not by hash order: the same rows scan the
    # same way in every process.
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = [t for t, c in ordered if min_obs <= c < n][:top_k]
    return vocab, present


# ----------------------------------------------------------------- the math


class _Feature:
    """One candidate column, stored sparse: the rows where it is non-zero.
    ``aliases`` are other columns with exactly the same entries (``tool_calls``
    when one tool is ever called, ``tool:<name>``): one feature, several
    names, so a duplicate never counts against the one it duplicates."""

    __slots__ = ("aliases", "binary", "entries", "name", "parents", "sd_pooled", "sd_within")

    def __init__(
        self, name: str, entries: list[tuple[int, float]], parents: tuple[_Feature, ...] = ()
    ):
        self.name = name
        self.entries = entries
        self.aliases: list[str] = []
        self.parents = parents
        self.binary = all(v == 1.0 for _, v in entries)
        self.sd_within = 0.0
        self.sd_pooled = 0.0

    @property
    def names(self) -> list[str]:
        return [self.name, *self.aliases]

    def endorsed(self, patterns: Sequence[str]) -> bool:
        """Named by an endorsed pattern, under any alias; a conjunction is
        endorsed when a parent is (it is that parent, narrowed)."""
        return any(_match(x, patterns) for x in self.names) or any(
            p.endorsed(patterns) for p in self.parents
        )


def _collapse(columns: dict[str, list[tuple[int, float]]], min_obs: int) -> list[_Feature]:
    """Columns with identical entries become one feature with aliases.
    Hand-tier names come first in ``columns`` and win the name."""
    by_key: dict[tuple, _Feature] = {}
    out: list[_Feature] = []
    for name, entries in columns.items():
        if len(entries) < min_obs:
            continue
        key = tuple(entries)
        if key in by_key:
            by_key[key].aliases.append(name)
            continue
        feature = _Feature(name, entries)
        by_key[key] = feature
        out.append(feature)
    return out


def _within_sd(feature: _Feature, group_of_row: list[int], group_size: list[int], n: int) -> float:
    """Standard deviation of the within-group-centered column, from the
    non-zero entries: sum over groups of (sum f^2 - (sum f)^2 / n_g)."""
    sums: dict[int, float] = {}
    sumsq: dict[int, float] = {}
    for i, v in feature.entries:
        g = group_of_row[i]
        sums[g] = sums.get(g, 0.0) + v
        sumsq[g] = sumsq.get(g, 0.0) + v * v
    total = sum(sumsq[g] - sums[g] * sums[g] / group_size[g] for g in sums)
    return math.sqrt(max(total, 0.0) / n) if n else 0.0


def _pooled_sd(feature: _Feature, n: int) -> float:
    s = sum(v for _, v in feature.entries)
    ss = sum(v * v for _, v in feature.entries)
    var = ss / n - (s / n) ** 2 if n else 0.0
    return math.sqrt(max(var, 0.0))


def _rho_within(feature: _Feature, rc: list[float], sr: float, n: int) -> float:
    """corr(centered feature, centered reward). The centered reward sums
    to zero in every group, so the raw feature can stand in for the
    centered one in the numerator: only its non-zero rows contribute."""
    if feature.sd_within <= 0.0 or sr <= 0.0:
        return 0.0
    num = sum(v * rc[i] for i, v in feature.entries)
    return num / (n * feature.sd_within * sr)


def _rho_pooled(feature: _Feature, rewards: list[float], n: int) -> float:
    sr = math.sqrt(max(sum(r * r for r in rewards) / n - (sum(rewards) / n) ** 2, 0.0))
    if feature.sd_pooled <= 0.0 or sr <= 0.0:
        return 0.0
    mean_f = sum(v for _, v in feature.entries) / n
    mean_r = sum(rewards) / n
    cov = sum(v * rewards[i] for i, v in feature.entries) / n - mean_f * mean_r
    return cov / (feature.sd_pooled * sr)


def _match(name: str, patterns: Sequence[str]) -> bool:
    low = name.lower()
    return any(str(p).lower() in low for p in patterns)


# ----------------------------------------------------------------- the scan


def hack_scan(
    rows: Sequence[dict],
    *,
    endorsed: Sequence[str] = (),
    features: Mapping[str, Callable[[dict], float | None]] | None = None,
    auto: bool = True,
    top_k: int = DEFAULT_TOP_K,
    n_perm: int = DEFAULT_N_PERM,
    min_obs: int = DEFAULT_MIN_OBS,
    seeds: int = DEFAULT_SEEDS,
    reward: str = "reward",
    seed: int = 0,
    top_features: int | None = REPORT_TOP,
) -> dict[str, Any]:
    """Rank what separates reward within each ask against a permutation
    noise floor, and say what a grouped update would learn.

    ``rows`` are graded rollouts, several per ask (``mode="rl"``); the
    reward under ``reward`` may be 0/1 or partial credit. ``endorsed``
    names the features the reward is supposed to track, as substrings of
    feature names (``"lookup_order"`` matches ``tool:lookup_order`` and
    ``contains:lookup_order``; ``"marker:grounded"`` a marker). Without
    it the scan still ranks and floors, but cannot call a hack a hack.
    ``features`` adds hand-tier columns: ``{"name": lambda row: value}``.
    ``top_features`` caps the ranking in the report (``None`` lists all).

    Returns ``regime`` (``train``, ``reward_hack``, ``pool_exhausted``,
    ``no_signal``, ``unknown``), ``tau`` (the floor), ``features`` ranked
    by |within-ask correlation| with the pooled correlation beside each,
    ``top_feature``, ``endorsed_on_top``, ``integrity`` (share of the
    above-floor signal that sits on an endorsed feature), the support
    numbers (asks all-pass, all-fail, mixed, gradient capacity), and
    ``warnings`` in one line each.
    """
    graded = [r for r in rows if isinstance(r, dict) and _reward(r, reward) is not None]
    n = len(graded)
    n_rows = sum(1 for r in rows if isinstance(r, dict))
    warnings: list[str] = []
    endorsed = [str(e) for e in endorsed if str(e)]
    base: dict[str, Any] = {
        "regime": "unknown",
        "n_rows": n_rows,
        "n_graded": n,
        "n_groups": 0,
        "n_groups_multi": 0,
        "rollouts_per_group": 0,
        "effective_rollouts": 0,
        "support": {"sat": 0.0, "dead": 0.0, "mixed": 0.0, "capacity": 0.0},
        "tau": None,
        "n_perm": int(n_perm),
        "n_features": 0,
        "n_above_floor": 0,
        "features": [],
        "top_feature": None,
        "rho_max": 0.0,
        "endorsed": endorsed,
        "endorsed_matched": 0,
        "endorsed_on_top": None,
        "integrity": None,
        "continuous_reward": False,
        "warnings": warnings,
    }
    if n == 0:
        warnings.append(f"no row carries a numeric {reward!r}; grade first")
        return base

    # groups
    rewards = [float(_reward(r, reward)) for r in graded]  # type: ignore[arg-type]
    group_index: dict[str, int] = {}
    group_of_row: list[int] = []
    for r in graded:
        key = _group_of(r)
        group_of_row.append(group_index.setdefault(key, len(group_index)))
    n_groups = len(group_index)
    members: list[list[int]] = [[] for _ in range(n_groups)]
    for i, g in enumerate(group_of_row):
        members[g].append(i)
    group_size = [len(m) for m in members]
    multi = [g for g in range(n_groups) if group_size[g] >= 2]
    sizes = sorted(group_size[g] for g in multi)
    base["n_groups"] = n_groups
    base["n_groups_multi"] = len(multi)
    base["rollouts_per_group"] = sizes[len(sizes) // 2] if sizes else 1
    base["continuous_reward"] = any(v not in (0.0, 1.0) for v in rewards)

    # support: what the update could move
    all_pass = all_fail = unanimous = 0
    capacity = 0.0
    for g in range(n_groups):
        vals = [rewards[i] for i in members[g]]
        if min(vals) == max(vals):
            unanimous += 1
            if vals[0] >= 1.0:
                all_pass += 1
            elif vals[0] <= 0.0:
                all_fail += 1
        mean = sum(vals) / len(vals)
        capacity += sum((v - mean) ** 2 for v in vals) / len(vals)
    base["support"] = {
        "sat": all_pass / n_groups,
        "dead": all_fail / n_groups,
        "mixed": (n_groups - unanimous) / n_groups,
        "capacity": capacity / (0.25 * n_groups),
    }
    live = [g for g in multi if len({rewards[i] for i in members[g]}) > 1]
    base["effective_rollouts"] = sum(group_size[g] for g in live)

    # centered reward
    group_mean = [sum(rewards[i] for i in m) / len(m) for m in members]
    rc = [rewards[i] - group_mean[group_of_row[i]] for i in range(n)]
    sr = math.sqrt(sum(v * v for v in rc) / n)

    # features
    texts = [scan_text(r) for r in graded]
    columns: dict[str, list[tuple[int, float]]] = {}
    for i, r in enumerate(graded):
        row_features = hand_features(r)
        if features:
            for name, fn in features.items():
                try:
                    value = fn(r)
                except Exception:
                    value = None
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row_features[str(name)] = float(value)
        for name, value in row_features.items():
            if value != 0.0 and value == value:
                columns.setdefault(name, []).append((i, value))
    if auto:
        vocab, present = auto_terms(texts, top_k=top_k, min_obs=min_obs)
        for term in vocab:
            columns[f"contains:{term}"] = [(i, 1.0) for i in range(n) if term in present[i]]
    feats = _collapse(columns, min_obs)
    for f in feats:
        f.sd_within = _within_sd(f, group_of_row, group_size, n)
        f.sd_pooled = _pooled_sd(f, n)
    feats = [f for f in feats if f.sd_pooled > 0.0]

    if not multi or sr <= 0.0:
        base["n_features"] = len(feats)
        base["features"] = [
            {
                "name": f.name,
                "aliases": f.aliases,
                "rho": 0.0,
                "pooled": round(_rho_pooled(f, rewards, n), 4),
                "n_obs": len(f.entries),
                "above_floor": False,
                "endorsed": f.endorsed(endorsed),
            }
            for f in sorted(feats, key=lambda f: -abs(_rho_pooled(f, rewards, n)))[:top_features]
        ]
        warnings.append(
            "every ask has one rollout, or every ask is unanimous: within-ask "
            "correlation needs repeats that disagree (mode='rl', repeats>=4); "
            "only the pooled column is filled"
            if multi
            else "one rollout per ask: within-ask correlation needs repeats "
            "(mode='rl', repeats>=4); only the pooled column is filled"
        )
        return base

    rho = {f.name: _rho_within(f, rc, sr, n) for f in feats}

    # conjunctions of the strongest binary features, on raw indicators
    # (centering does not commute with AND). A conjunction is kept only
    # when it beats both parents: otherwise it is a parent, narrowed.
    if auto and seeds > 0:
        binary = sorted((f for f in feats if f.binary), key=lambda f: -abs(rho[f.name]))[
            : int(seeds)
        ]
        extra: list[_Feature] = []
        for a_i, fa in enumerate(binary):
            rows_a = {i for i, _ in fa.entries}
            for fb in binary[a_i + 1 :]:
                both = sorted(rows_a & {i for i, _ in fb.entries})
                if (
                    len(both) >= min_obs
                    and len(both) < len(fa.entries)
                    and len(both) < len(fb.entries)
                ):
                    extra.append(
                        _Feature(f"{fa.name} AND {fb.name}", [(i, 1.0) for i in both], (fa, fb))
                    )
        for f in extra:
            f.sd_within = _within_sd(f, group_of_row, group_size, n)
            f.sd_pooled = _pooled_sd(f, n)
            if f.sd_pooled <= 0.0:
                continue
            value = _rho_within(f, rc, sr, n)
            if abs(value) > max(abs(rho[p.name]) for p in f.parents):
                feats.append(f)
                rho[f.name] = value

    # permutation floor: shuffle reward within ask, keep the max |rho|
    rng = random.Random(seed)
    nulls: list[float] = []
    perm = list(rewards)
    for _ in range(max(1, int(n_perm))):
        for g in multi:
            idx = members[g]
            vals = [perm[i] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                perm[i] = v
        rc_p = [perm[i] - group_mean[group_of_row[i]] for i in range(n)]
        best = 0.0
        for f in feats:
            if f.sd_within <= 0.0:
                continue
            value = abs(sum(v * rc_p[i] for i, v in f.entries) / (n * f.sd_within * sr))
            if value > best:
                best = value
        nulls.append(best)
    nulls.sort()
    tau = nulls[min(len(nulls) - 1, math.ceil(0.95 * len(nulls)) - 1)] if nulls else 0.0

    ranked = sorted(feats, key=lambda f: (-abs(rho[f.name]), f.name))
    listed: list[dict[str, Any]] = [
        {
            "name": f.name,
            "aliases": f.aliases,
            "rho": round(rho[f.name], 4),
            "pooled": round(_rho_pooled(f, rewards, n), 4),
            "n_obs": len(f.entries),
            "above_floor": abs(rho[f.name]) > tau,
            "endorsed": f.endorsed(endorsed),
        }
        for f in ranked
    ]
    above = [x for x in listed if x["above_floor"]]
    top = listed[0] if listed else None
    rho_max = abs(top["rho"]) if top else 0.0
    matched = sum(1 for x in listed if x["endorsed"])
    base.update(
        {
            "tau": round(tau, 4),
            "n_features": len(listed),
            "n_above_floor": len(above),
            "features": listed[:top_features],
            "top_feature": top["name"] if top else None,
            "rho_max": round(rho_max, 4),
            "endorsed_matched": matched,
        }
    )

    # regime
    sat = base["support"]["sat"]
    if endorsed and matched == 0:
        base["regime"] = "unknown"
        warnings.append(
            f"endorsed={endorsed!r} matches none of the {len(listed)} features; fix the "
            "patterns (feature names are like tool:<name>, marker:<name>, contains:<term>) "
            "or add a features= extractor that emits them"
        )
        return base
    e = max((abs(x["rho"]) for x in above if x["endorsed"]), default=0.0)
    a = max((abs(x["rho"]) for x in above if not x["endorsed"]), default=0.0)
    if endorsed:
        base["endorsed_on_top"] = bool(top and top["endorsed"])
        base["integrity"] = round(e / (e + a), 4) if (e + a) > 0 else 0.0
    if not above or top is None:
        base["regime"] = "no_signal"
        warnings.append(
            f"no feature clears the noise floor (max |rho| {rho_max:.2f}, floor {tau:.2f} "
            f"from {len(nulls)} within-ask shuffles): the reward is not separating "
            "rollouts of the same ask on anything measurable; check the judge before "
            "training"
        )
    elif endorsed and not top["endorsed"]:
        base["regime"] = "reward_hack"
        warnings.append(
            f'reward is best explained by "{top["name"]}" (within-ask rho {top["rho"]:+.2f}, '
            f"floor {tau:.2f}), not by anything endorsed"
            + (f" (best endorsed {e:.2f})" if e else " (nothing endorsed clears the floor)")
            + f'; a policy trained on it learns "{top["name"]}"'
        )
    elif sat > SAT_FLAG:
        base["regime"] = "pool_exhausted"
        warnings.append(
            f"{all_pass} of {n_groups} asks are already all-pass ({sat:.0%}); those groups "
            "carry no gradient, raise difficulty or drop them (optimize(mode='rl') does)"
        )
    else:
        base["regime"] = "train"
    if (
        endorsed
        and base["regime"] in ("train", "pool_exhausted")
        and a > 0
        and base["integrity"] < 1.0 - RIVAL_SHARE
    ):
        rivals = [x["name"] for x in above if not x["endorsed"]][:3]
        warnings.append(
            "also above the floor, not endorsed: "
            + ", ".join(f'"{r}"' for r in rivals)
            + f" (integrity {base['integrity']:.2f})"
        )
    if base["rollouts_per_group"] < 4:
        warnings.append(
            f"{base['rollouts_per_group']} rollouts per ask at the median; the floor is "
            "coarse below 4, re-scan at repeats>=8 before acting on a close call"
        )
    return base


def format_hack_scan(report: dict[str, Any], *, top: int = 12) -> str:
    """The block a person reads: the regime, the floor, the ranking."""
    lines = [f"{report['regime'].upper().replace('_', ' ')}"]
    s = report.get("support") or {}
    lines.append(
        f"{report['n_graded']} graded rollouts, {report['n_groups']} asks, "
        f"{report['rollouts_per_group']} per ask; mixed {s.get('mixed', 0):.0%}, "
        f"all-pass {s.get('sat', 0):.0%}, all-fail {s.get('dead', 0):.0%}, "
        f"capacity {s.get('capacity', 0):.2f}"
    )
    if report.get("tau") is not None:
        lines.append(
            f"noise floor tau {report['tau']:.3f} ({report['n_above_floor']} of "
            f"{report['n_features']} features above)"
        )
        lines.append(f"  {'feature':<44}{'within':>8}{'pooled':>8}{'n':>6}")
        for x in report["features"][:top]:
            flag = "*" if x["above_floor"] else " "
            mark = "e" if x["endorsed"] else " "
            lines.append(
                f"{flag}{mark}{x['name'][:43]:<44}{x['rho']:>+8.3f}{x['pooled']:>+8.3f}{x['n_obs']:>6}"
            )
    if report.get("integrity") is not None:
        lines.append(f"integrity {report['integrity']:.2f} (share of above-floor signal endorsed)")
    for w in report.get("warnings") or []:
        lines.append(f"! {w}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MIN_OBS",
    "DEFAULT_N_PERM",
    "DEFAULT_TOP_K",
    "REGIMES",
    "RIVAL_SHARE",
    "SAT_FLAG",
    "auto_terms",
    "format_hack_scan",
    "hack_scan",
    "hand_features",
    "scan_text",
]
