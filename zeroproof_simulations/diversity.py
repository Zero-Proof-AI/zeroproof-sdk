"""Sparse generic writer knobs and annealing helpers."""
from __future__ import annotations

import hashlib
import random
import re
import threading
from typing import Any

_LENGTHS = ("short prompt", "medium prompt", "long prompt")
_VAGUENESS = ("specific", "vague", "underspecified")
_WEIRD = ("incomplete", "specific", "rambling")
# Generic search tiers. Domain comes from tools+policy, not these names.
_TIERS = ("ordinary", "ambiguous", "boundary", "adversarial")
_TIER_ALIASES = {
    "ordinary": "ordinary",
    "vague": "ambiguous",
    "underspecified": "ambiguous",
    "ambiguous": "ambiguous",
    "boundary": "boundary",
    "policy-push": "boundary",
    "conflicting": "boundary",
    "adversarial": "adversarial",
    "malicious": "adversarial",
    "forbidden": "adversarial",
    "hurried": "ordinary",
    "retry": "ordinary",
    "mistaken": "ordinary",
    "exploratory": "ordinary",
    "unsure": "ambiguous",
}
# Ordinary majority; other tiers stay in the bag so a short run still hits them.
_TIER_BAG = (("ordinary",) * 6 + ("ambiguous",) + ("boundary",) + ("adversarial",) * 2)
ORDINARY_SHARE = 0.60
# Human texture: how the message is typed, independent of what it asks.
_TEXTURES = ("lowercase", "abbreviations", "typo", "no_punctuation",
             "run_on", "clipped")
_TONES = ("impatient", "frustrated", "chatty", "polite", "curt", "sarcastic")
# User-side only. Most tagged cells are one question or one ask.
# Rare compound: several asks, or tell the agent to do several things.
_ASKS = ("question", "question", "ask")
_COMPOUND = ("several asks", "do several things")
_PRESSURES = ("rushed", "insistent", "repeat")
_USER_TYPES = ("first time", "returning", "in a hurry", "careful", "brief")
DEFAULT_TEXTURE_RATE = 0.35

NOVELTY_RESTART_FLOOR = 0.025
MAX_NOVELTY_RESTARTS = 5

_FAMILY_STOP = {
    "a", "about", "again", "all", "an", "and", "any", "are", "be", "can",
    "could", "do", "for", "from", "get", "good", "help", "hey", "i", "in",
    "is", "it", "just", "latest", "like", "looking", "me", "my", "new",
    "of", "on", "open", "please", "quick", "recent", "show", "some",
    "something", "that", "the", "this", "to", "under", "want", "what",
    "with", "you", "your", "item", "product", "request", "issue", "pr",
    "order", "project", "repo", "account", "message", "thing", "status",
    "human", "scenario",
}
_INTENT_WORDS = {
    "search": {"find", "search", "recommend", "browse", "looking", "show"},
    "inspect": {"check", "look", "status", "track", "review", "read"},
    "create": {"create", "open", "add", "file", "book", "schedule"},
    "change": {"change", "update", "edit", "rename", "move", "assign"},
    "send": {"send", "comment", "reply", "email", "post", "share"},
    "finish": {"merge", "buy", "checkout", "place", "submit", "approve"},
    "undo": {"cancel", "delete", "remove", "return", "refund"},
}


def scenario_family(text: str) -> tuple[str, frozenset[str]]:
    """Coarse intent + salient words for cross-request near-duplicate caps."""
    words = re.findall(r"[a-z][a-z0-9'-]{2,}", str(text).lower())
    intent = "other"
    for name, markers in _INTENT_WORDS.items():
        if any(word in markers or any(
                len(marker) >= 4 and word.startswith(marker)
                for marker in markers) for word in words):
            intent = name
            break
    salient = frozenset(
        word for word in words
        if word not in _FAMILY_STOP
        and not any(word in markers or any(
            len(marker) >= 4 and word.startswith(marker)
            for marker in markers) for markers in _INTENT_WORDS.values()))
    return intent, salient


def cap_scenario_families(rows: list[dict], history: list[tuple[str, frozenset[str]]],
                          *, cap: int = 2) -> tuple[list[dict], list[dict]]:
    """Keep at most ``cap`` messages sharing intent and a salient subject."""
    kept: list[dict] = []
    rejected: list[dict] = []
    for row in rows:
        family = scenario_family(str(row.get("text") or ""))
        intent, words = family
        similar = sum(1 for old_intent, old_words in history
                      if intent == old_intent and words and old_words
                      and bool(words & old_words))
        if similar >= max(1, int(cap)):
            rejected.append(row)
            continue
        kept.append(row)
        history.append(family)
    return kept, rejected


def _draw(seed: int, round_index: int, key: str, salt: str) -> int:
    digest = hashlib.sha256(
        f"{seed}:{round_index}:{key}:{salt}".encode()).hexdigest()
    return int(digest[:16], 16)


WRITER_TEMP_LO = 0.45
WRITER_TEMP_HI = 1.05

_LENGTH_PROSE = {
    "short": ("You keep it brief.", "You write one short line.",
              "You use only a few words."),
    "medium": ("You write a couple of sentences.",
               "You give enough context, not an essay.",
               "You write a short paragraph."),
    "long": ("You use more words and add what led here.",
             "You write it out with the situation.",
             "You add a full paragraph of circumstances."),
}
_NICE_PROSE = (
    (0.14, "You are mean and impatient."),
    (0.28, "You are frustrated."),
    (0.40, "You are confused."),
    (0.52, "You are curt."),
    (0.66, "You are calm."),
    (0.80, "You are being nice."),
    (0.90, "You are in a hurry."),
    (1.01, "You are being chatty."),
)
_TONE_PROSE = {
    "impatient": "You are in a hurry.",
    "frustrated": "You are frustrated.",
    "chatty": "You are being chatty.",
    "curt": "You are curt.",
    "polite": "You are being nice.",
    "sarcastic": "You are sarcastic about how this is going.",
}


def _unit(seed: int, round_index: int, key: str, salt: str) -> float:
    return (_draw(int(seed), int(round_index), str(key), salt) % 10007) / 10007.0


def sample_writer_temperature(seed: int, round_index: int) -> float:
    """Continuous writer temperature in a sane band. One draw per batch."""
    u = _unit(int(seed), int(round_index), "batch", "temp")
    return round(WRITER_TEMP_LO + u * (WRITER_TEMP_HI - WRITER_TEMP_LO), 3)


_WRITER_N_CAP = 8


def sample_writer_n(seed: int, round_index: int, *,
                    elapsed: float | None = None,
                    time_budget: float | None = None,
                    max_n: int | None = None) -> int:
    """Draw writer completions for one batch from remaining wall-clock.

    Early (first ~30%): 3–6 to fill the pool. Mid: 1–4. Late (last ~20%):
    1–2 so leftover time buys distinct cards. Unknown budget: 1–4.
    A per-batch hash jitter means two writers at the same timestamp
    need not share n. ``max_n`` is a ceiling (default 8).
    """
    ceiling = _WRITER_N_CAP if max_n is None else max(1, min(_WRITER_N_CAP, int(max_n)))
    if time_budget is None or float(time_budget) <= 0 or elapsed is None:
        lo, hi = 1, 4
    else:
        frac = max(0.0, min(1.0, float(elapsed) / float(time_budget)))
        if frac < 0.30:
            lo, hi = 3, 6
        elif frac >= 0.80:
            lo, hi = 1, 2
        else:
            lo, hi = 1, 4
    hi = min(hi, ceiling)
    lo = min(lo, hi)
    u = _unit(int(seed), int(round_index), "batch", "n")
    return lo + int(u * (hi - lo + 1))


def sample_writer_vars(seed: int, round_index: int, key: str, *,
                       tags: dict | None = None, ask_family: str = "",
                       assignment: dict | None = None) -> dict[str, Any]:
    """Continuous-ish writer slots as prose. No tag labels."""
    tags = dict(tags or {})
    assignment = dict(assignment or {})
    length = str(tags.get("length") or assignment.get("length") or "")
    if "short" in length:
        bucket = "short"
    elif "long" in length:
        bucket = "long"
    else:
        u_len = _unit(seed, round_index, key, "wlen")
        bucket = "short" if u_len < 0.12 else ("long" if u_len >= 0.82 else "medium")
    variants = _LENGTH_PROSE[bucket]
    u_lp = _unit(seed, round_index, key, "lprose")
    length_prose = variants[int(u_lp * len(variants)) % len(variants)]
    tone = str(tags.get("tone") or "")
    if tone in _TONE_PROSE:
        niceness_prose = _TONE_PROSE[tone]
    else:
        u_n = _unit(seed, round_index, key, "nice")
        niceness_prose = next(prose for cut, prose in _NICE_PROSE if u_n < cut)
    u_c = _unit(seed, round_index, key, "conf")
    if ask_family == "vague":
        conf = 0.12 + 0.28 * u_c
    elif ask_family == "general":
        conf = 0.42 + 0.28 * u_c
    else:
        conf = 0.62 + 0.38 * u_c
    if conf >= 0.70:
        confidence_prose = "You know exactly what you want done."
    elif conf >= 0.40:
        confidence_prose = "You know what you want but one detail is fuzzy."
    else:
        confidence_prose = "You haven't settled on a specific action yet."
    return {
        "length_prose": length_prose,
        "niceness_prose": niceness_prose,
        "confidence_prose": confidence_prose,
        "confidence": conf,
        "length_bucket": bucket,
    }


def behavior_tier(assignment: dict | None) -> str:
    """Map a cell onto a generic search tier. Missing → ordinary."""
    raw = str((assignment or {}).get("stance")
              or (assignment or {}).get("user_behavior")
              or "")
    return _TIER_ALIASES.get(raw, "ordinary")


# Trainer-facing conversation labels. Only keys the sampler already uses.
_CONVERSATION_OPTIONAL = (
    "stance", "tone", "length", "ask", "vagueness", "phrasing",
    "pressure", "user", "texture", "history",
)


def conversation_features(assignment: dict | None = None,
                          tags: dict | None = None, *,
                          ask_family: str | None = None,
                          tool: str | None = None) -> dict[str, Any]:
    """User/conversation labels for a training row. Omit unsampled tags."""
    assignment = dict(assignment or {})
    tags = dict(tags or {})
    out: dict[str, Any] = {"tier": behavior_tier(assignment)}
    family = str(ask_family or "").strip()
    if family in {"tool", "general", "vague"}:
        out["ask_family"] = family
        out["intent_known"] = family != "vague"
        named = bool(tool) and str(tool) not in {"unrelated", "multi_tool"}
        if not named:
            named = bool(assignment.get("tool")) and str(
                assignment.get("tool")) not in {"unrelated", "multi_tool"}
        out["tool_known"] = family == "tool" and named
    for key in _CONVERSATION_OPTIONAL:
        val = tags.get(key)
        if val not in (None, ""):
            out[key] = val
    return out


def mix_items_by_tier(items: list, n: int, tier_of, *,
                      ordinary_share: float = ORDINARY_SHARE) -> list:
    """Breadth-first across tiers, then fill ordinary-majority.

    First items hit ordinary plus a hard case. A 24-cell or 100-row slice
    is not one tier. Unknown tiers count as ordinary.
    """
    if not items or n <= 0:
        return []
    n = min(int(n), len(items))
    buckets: dict[str, list] = {tier: [] for tier in _TIERS}
    for item in items:
        tier = tier_of(item)
        buckets[tier if tier in buckets else "ordinary"].append(item)

    picked: list = []
    seen: set[int] = set()

    def take(tier: str) -> bool:
        for item in buckets.get(tier) or []:
            marker = id(item)
            if marker in seen:
                continue
            seen.add(marker)
            picked.append(item)
            return True
        return False

    take("ordinary")
    if n >= 2:
        for tier in ("adversarial", "boundary", "ambiguous"):
            if take(tier):
                break
    if n >= 8:
        have = {tier_of(item) for item in picked}
        for tier in _TIERS:
            if len(picked) >= n:
                break
            if tier not in have:
                take(tier)

    target_ordinary = max((n + 1) // 2, int(round(n * ordinary_share)))
    while len(picked) < n:
        ordinary_count = sum(1 for item in picked if tier_of(item) == "ordinary")
        if ordinary_count < target_ordinary and take("ordinary"):
            continue
        progressed = False
        for tier in ("ambiguous", "boundary", "adversarial", "ordinary"):
            if take(tier):
                progressed = True
                break
        if not progressed:
            break
    return picked


_HINT_STOP = {
    "the", "a", "an", "to", "of", "or", "and", "if", "is", "do", "not",
    "you", "your", "must", "should", "with", "for", "on", "in", "be",
}


def _private_hint(text: Any, *, limit: int = 24) -> str:
    """Compact tag. Full policy / world sentences stay out of the cell."""
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if not raw:
        return ""
    if len(raw) <= limit and " " in raw and raw.endswith("."):
        return raw.rstrip(".")[:limit]
    if len(raw) <= limit:
        return raw
    words = [w for w in re.findall(r"[A-Za-z0-9_]+", raw.lower())
             if w not in _HINT_STOP]
    slug = " ".join(words[:3])
    return (slug or raw)[:limit]


def _world_hint(raw: Any) -> str:
    from .scenarios import WORLD_HINTS
    text = str(raw or "").strip()
    if text in WORLD_HINTS:
        return WORLD_HINTS[text]
    return _private_hint(text, limit=16)


def _length_hint(raw: Any, n_len: int) -> str:
    text = str(raw or "").strip().lower()
    if "short" in text:
        return "short prompt"
    if "long" in text:
        return "long prompt"
    if "medium" in text or "mid" in text:
        return "medium prompt"
    if text in _LENGTHS:
        return text
    # Right-skew: most medium, some short, long tail.
    bucket = n_len % 100
    if bucket < 12:
        return "short prompt"
    if bucket < 82:
        return "medium prompt"
    return "long prompt"


def sample_cell_tags(seed: int, round_index: int, key: str,
                     assignment: dict | None = None,
                     texture_rate: float = DEFAULT_TEXTURE_RATE,
                     **_unused) -> dict[str, Any]:
    """Most cells are tools and policy. Length and phrasing are rare hints."""
    assignment = dict(assignment or {})
    n = _draw(seed, round_index, key, "sparse")
    n_len = _draw(seed, round_index, key, "length")
    n_tier = _draw(seed, round_index, key, "tier")
    situation: dict[str, Any] = {}
    if assignment.get("tool"):
        situation["tool"] = assignment["tool"]
    if assignment.get("rule"):
        hint = _private_hint(assignment["rule"])
        if hint:
            situation["rule"] = hint

    def add_stance() -> None:
        raw = str(assignment.get("stance") or assignment.get("user_behavior") or "")
        if not raw:
            from .scenarios import STANCES
            raw = STANCES[n_tier % len(STANCES)]
        if raw == "ordinary":
            return
        situation["stance"] = raw

    if assignment.get("length") or (n % 5 != 0):
        situation["length"] = _length_hint(assignment.get("length"), n_len)

    n_ask = _draw(seed, round_index, key, "ask")
    if assignment.get("ask"):
        situation["ask"] = assignment["ask"]
    elif n_ask % 23 == 0:
        situation["ask"] = _COMPOUND[(n_ask // 23) % len(_COMPOUND)]
    elif n_ask % 5 == 0:
        situation["ask"] = _ASKS[(n_ask // 5) % len(_ASKS)]

    mode = n % 20
    if mode == 14:
        situation["vagueness"] = (
            assignment.get("vagueness") or _VAGUENESS[(n // 20) % 3])
    elif mode == 16:
        situation["phrasing"] = _WEIRD[(n // 20) % len(_WEIRD)]
    raw_stance = assignment.get("stance") or assignment.get("user_behavior")
    mapped_stance = _TIER_ALIASES.get(str(raw_stance)) if raw_stance else None
    if raw_stance and str(raw_stance) != "ordinary" and mode < 10:
        add_stance()
    elif mapped_stance is None and mode == 15:
        add_stance()

    n_extra = _draw(seed, round_index, key, "axis")
    if assignment.get("pressure") or n_extra % 19 == 0:
        situation["pressure"] = (
            assignment.get("pressure") or _PRESSURES[(n_extra // 19) % len(_PRESSURES)])
    if assignment.get("user") or n_extra % 17 == 0:
        situation["user"] = (
            assignment.get("user") or _USER_TYPES[(n_extra // 17) % len(_USER_TYPES)])
    hist = assignment.get("history")
    if hist and hist != "fresh" and mode == 11:
        situation["history"] = hist
    world = assignment.get("world_state")
    if world and world not in {"unspecified", "unknown"} and mode % 5 == 2:
        situation["world_state"] = _world_hint(world)
    cond = assignment.get("tool_condition")
    if cond and cond != "success" and n_extra % 11 == 0:
        situation["tool_condition"] = cond

    t = _draw(seed, round_index, key, "texture")
    if texture_rate > 0 and (t % 1000) < int(min(1.0, texture_rate) * 1000):
        situation["texture"] = _TEXTURES[(t // 1000) % len(_TEXTURES)]
        if (t // 7919) % 3 == 0:
            situation["tone"] = _TONES[(t // 104729) % len(_TONES)]
    elif (t // 977) % 5 < 3:
        # The writer model collapses to lowercase texting on its own, so
        # ordinary prose (capitals, end marks) must be an explicit style too.
        situation["texture"] = "standard"

    return situation


def sampling_plan(time_budget: float | None) -> dict[str, Any]:
    """Wider search when they give more wall-clock. Still BFS at 60s."""
    seconds = 600.0 if time_budget is None or float(time_budget) <= 0 else float(time_budget)
    scale = min(3.0, max(0.5, seconds / 60.0))
    return {
        "seconds": seconds,
        "scale": scale,
        "shape_limit": max(8, int(round(12 * scale))),
        "enum_cap": max(200, int(round(200 * scale))),
        "max_shape_len": 2 if seconds < 120 else 3,
        "ordinary_share": ORDINARY_SHARE,
    }


def adaptive_allocator(time_budget: float | None, until: str = "compute",
                       *, elapsed: float | None = None) -> dict[str, Any]:
    """Adaptive mix. Short remaining clock is messier; saturation walks more cards.

    Shares are explore / expand / verify. n_req and k are caps so expand and
    verify can actually run. Not a pinned n=1 k=1 policy.
    """
    until_key = str(until or "compute").strip().lower()
    if until_key in {"first", "saturation"}:
        until_key = "saturation"
    elif until_key in {"compute", "budget_only", "budget", "time"}:
        until_key = "compute"
    sat = until_key == "saturation"
    if time_budget is None or float(time_budget) <= 0:
        total = 600.0
        left = 600.0
    else:
        total = float(time_budget)
        spent = 0.0 if elapsed is None else max(0.0, float(elapsed))
        left = max(0.0, total - spent)
    # Saturation keeps covering the grid even on a short clock.
    clock = max(left, 120.0) if sat else left
    scale = min(1.0, max(0.0, (clock - 15.0) / 165.0))
    explore = 0.30 + 0.50 * scale
    rest = 1.0 - explore
    expand = rest * 0.55
    verify = rest * 0.45
    return {
        "explore": explore,
        "expand": expand,
        "verify": verify,
        "n_req": 3,
        "k": 3 if sat or total >= 90.0 else 2,
        "until": until_key,
        "seconds": total,
        "remaining": left,
    }


def allocator_slot_counts(take: int, plan: dict | None) -> dict[str, int]:
    """Integer explore/expand/verify slots from mix shares."""
    take = max(0, int(take))
    if take <= 0 or not plan:
        return {"explore": take, "expand": 0, "verify": 0}
    expand_s = float(plan.get("expand") or 0.0)
    verify_s = float(plan.get("verify") or 0.0)
    messy_s = expand_s + verify_s
    messy_n = int(round(take * messy_s))
    if take >= 2:
        messy_n = max(1, min(take - 1, messy_n))
    explore_n = take - messy_n
    if messy_n <= 0:
        return {"explore": take, "expand": 0, "verify": 0}
    v_part = verify_s / messy_s if messy_s else 0.5
    verify_n = int(round(messy_n * v_part))
    verify_n = min(messy_n, max(0, verify_n))
    if messy_n >= 2 and verify_n == 0 and v_part > 0:
        verify_n = 1
    expand_n = messy_n - verify_n
    return {"explore": explore_n, "expand": expand_n, "verify": verify_n}


def new_turn_stats() -> dict[str, Any]:
    """Shared running mean of observed conversation turns."""
    return {"n": 0, "sum": 0, "lock": threading.Lock()}


def observed_turn_count(row: dict | None) -> int:
    """User + agent utterances. Opening prompt counts as the first user turn."""
    n = 1
    for step in (row or {}).get("steps") or []:
        if not isinstance(step, dict):
            continue
        if step.get("user"):
            n += 1
        if step.get("text") or step.get("tool"):
            n += 1
    return n


def record_turns(stats: dict | None, row: dict | None) -> None:
    if not stats:
        return
    n = observed_turn_count(row)
    lock = stats.get("lock")
    if lock is not None:
        lock.acquire()
    try:
        stats["n"] = int(stats.get("n") or 0) + 1
        stats["sum"] = int(stats.get("sum") or 0) + n
    finally:
        if lock is not None:
            lock.release()


def running_turn_mean(stats: dict | None) -> float | None:
    if not stats:
        return None
    lock = stats.get("lock")
    if lock is not None:
        lock.acquire()
    try:
        n = int(stats.get("n") or 0)
        if n <= 0:
            return None
        return float(stats.get("sum") or 0) / n
    finally:
        if lock is not None:
            lock.release()


def sample_turn_budget(seed: int, key: str, max_turns: int,
                       avg_turns: float | None = None,
                       running_mean: float | None = None) -> int:
    """15/75/10 around ``avg_turns``, shifted by the live mean, snapped even."""
    cap = max(2, int(max_turns))
    target = 6.0 if avg_turns is None else float(avg_turns)
    center = target
    if running_mean is not None:
        center = target + (target - float(running_mean))
    center = max(2, int(round(center)))
    mid_lo = max(2, center - 2)
    mid_hi = min(cap, max(mid_lo, center + 2))
    short_hi = mid_lo - 1
    tail_lo = min(cap, mid_hi + 1)
    u = _draw(int(seed), 0, str(key), "turns") % 1000
    if target <= 3.5:
        # Old short mix: 60% 2–4, 30% 5–8, 10% long tail.
        if u < 600:
            want = 2 + (u % 3)
        elif u < 900:
            want = 5 + (u % 4)
        else:
            want = 9 + (u % max(1, cap - 8))
    elif u < 150 and short_hi >= 2:
        want = 2 + (u % (short_hi - 1))
    elif u < 900:
        want = mid_lo + (u % (mid_hi - mid_lo + 1))
    else:
        want = tail_lo + (u % max(1, cap - tail_lo + 1))
    # Even length: user, agent, user, agent. Steer the odd snap toward the target.
    if want % 2:
        want = want + 1 if (running_mean is None or running_mean <= target) else want - 1
    if want % 2:
        want += 1
    even_cap = cap if cap % 2 == 0 else cap - 1
    return max(2, min(even_cap if even_cap >= 2 else cap, want))


def sample_request_axes(seed: int, round_index: int, key: str, *,
                        assignment: dict | None = None, **_unused) -> dict[str, str]:
    """Optional extras only. Often empty."""
    tags = sample_cell_tags(seed, round_index, key, assignment)
    return {k: str(v) for k, v in tags.items() if k not in {"tool", "rule"}}


def anneal_temperature(round_index: int, *, start: float = 1.0,
                       end: float = 0.12, decay: float = 0.93) -> float:
    return max(end, start * (decay ** max(0, int(round_index))))


def explore_slot_count(batch_size: int, round_index: int) -> int:
    need = max(1, int(batch_size))
    temp = anneal_temperature(round_index)
    frac = 0.04 + 0.36 * temp
    return max(0, min(need // 2, int(round(need * frac))))


def accept_anneal_candidate(novelty: float, *, temperature: float,
                            rng: random.Random | None = None) -> bool:
    if temperature <= 0.01:
        return False
    rng = rng or random.Random()
    uplift = max(0.0, 0.55 - float(novelty))
    prob = min(1.0, temperature * (0.15 + uplift * 2.0))
    return rng.random() < prob


def apply_annealing_explore(batch_idx: list[int], texts_len: int,
                            novelty_of: dict[int, float], *, need: int,
                            round_index: int, seed: int) -> list[int]:
    explore_n = explore_slot_count(need, round_index)
    if explore_n <= 0 or texts_len <= need:
        return batch_idx
    temp = anneal_temperature(round_index)
    rng = random.Random(int(seed) + int(round_index) * 31 + 7)
    in_batch = set(batch_idx)
    pool = [i for i in range(texts_len) if i not in in_batch]
    pool.sort(key=lambda i: novelty_of.get(i, 0.0))
    accepted: list[int] = []
    for index in pool:
        if len(accepted) >= explore_n:
            break
        if accept_anneal_candidate(novelty_of.get(index, 0.0),
                                   temperature=temp, rng=rng):
            accepted.append(index)
    if not accepted:
        return batch_idx
    sorted_batch = sorted(batch_idx, key=lambda i: novelty_of.get(i, 0.0))
    drop = set(sorted_batch[: len(accepted)])
    merged = [i for i in batch_idx if i not in drop] + accepted
    return list(dict.fromkeys(merged))[:need]
