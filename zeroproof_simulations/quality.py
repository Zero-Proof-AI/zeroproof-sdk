"""Second-pass conversation quality ranker. Scores rows; does not rewrite them."""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Sequence

from .agents import _ID_FOLLOW, _echoes_agent
from .generator import (
    _ACK_ONLY,
    _DIRECTIVE_ANY,
    _META_LINE,
    agent_voice_user,
    usable_user_message,
)
from .grading import _DEGENERATE, _INFRA_STUB

DIMENSIONS = ("opener", "ping_pong", "leak", "complexity", "structure")
FAIL = 0.5

_DESK_OPENER = re.compile(
    r"^hello,?\s+i am writing because\b|"
    r"^thank you for (your patience|contacting|reaching out)\b|"
    r"^how (may|can) i (help|assist) you\b|"
    r"^please (be advised|note that|allow me to)\b|"
    r"^i hope this (email|message) finds you\b",
    re.I,
)
_GIBBERISH = re.compile(
    r"^(asdf|qwerty|zxcv|huh|\.+|!+|asdf qwerty zxcv)(\s+\S+){0,3}$",
    re.I,
)
_HARD_LEAK = re.compile(
    r"("
    r"<USER_TURN>|"
    r"\b(llm_guided|behavior_targeted|failure_mutation)\b|"
    r"\b(scene brief|situation card|region_id|talk_about|"
    r"how_people_ask|usually_want|private scene)\b|"
    r"\b(entity missing|duplicate entity|prior_partial_action|"
    r"clearly_allowed|tool_condition|ask_family)\b|"
    r"\b(world_state|scenario_dimensions|search arm)\b"
    r")",
    re.I,
)
_CALL_SYNTAX = re.compile(r"\b[a-z][a-z0-9]*_[a-z0-9_]*\s*\(")
_STRONG_ACTION = re.compile(
    r"\b(merge|refund|schedule|commit|deploy|rebase|checkout)\b", re.I)
_ACTION = re.compile(
    r"\b(merge|refund|schedule|search|look(?:ing)? up|lookup|cancel|"
    r"create|delete|comment|close|open|move|run|deploy|send|book|"
    r"update|fix|trace|status|commit|checkout|rebase)\b",
    re.I,
)
_IDISH = re.compile(
    r"\b([A-Z]{2,}[-_]\d+|#\d+|asin\b|ord[-_]\w+|pr\s*#?\s*\d+)\b",
    re.I,
)
_QUESTION_END = re.compile(r"\?\s*$")
_WORD = re.compile(r"[A-Za-z0-9']+")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _words(text: str) -> list[str]:
    return _WORD.findall(str(text or ""))


def _messages(row: dict) -> list[dict]:
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        return [m for m in msgs if isinstance(m, dict)]
    from zeroproof_simulations import conversation
    return conversation(row)


def _user_texts(messages: list[dict], prompt: str) -> list[str]:
    texts = [str(m.get("content") or "") for m in messages if m.get("role") == "user"]
    if texts:
        return texts
    first = str(prompt or "").strip()
    return [first] if first else []


def _assistant_spoken(messages: list[dict]) -> list[str]:
    out = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        text = _norm(m.get("content") or "")
        if text:
            out.append(text)
    return out


def _tool_names(row: dict, messages: list[dict]) -> set[str]:
    names: set[str] = set()
    for step in row.get("steps") or []:
        if isinstance(step, dict) and step.get("tool"):
            names.add(str(step["tool"]))
    for m in messages:
        for call in m.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("name"):
                names.add(str(call["name"]))
    names.discard("")
    return names


def _substantial(text: str) -> bool:
    return len(_norm(text)) >= 12


def _same_beat(prev: dict, content: str, has_tools: bool) -> bool:
    if has_tools:
        return True
    prev_text = " ".join(prev.get("texts") or [])
    if prev["tools"] and not _substantial(prev_text):
        return True
    if prev["tools"] and _substantial(content) and not has_tools:
        return True
    if (_substantial(prev_text) and _substantial(content)
            and not has_tools and not prev["tools"]):
        return False
    if not _substantial(content):
        return True
    return False


def _beats(messages: list[dict]) -> list[dict]:
    beats: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            continue
        content = str(m.get("content") or "")
        if role == "user":
            beats.append({"role": "user", "text": content})
            continue
        if role != "assistant":
            continue
        has_tools = bool(m.get("tool_calls"))
        if (beats and beats[-1]["role"] == "assistant"
                and _same_beat(beats[-1], content, has_tools)):
            beats[-1]["texts"].append(content)
            beats[-1]["tools"] = beats[-1]["tools"] or has_tools
        else:
            beats.append({"role": "assistant", "texts": [content],
                          "tools": has_tools})
    return beats


def _leak_hits(text: str, *, speaker: str = "user") -> list[str]:
    blob = str(text or "")
    found: list[str] = []
    match = _HARD_LEAK.search(blob)
    if match:
        found.append(match.group(0).strip()[:40])
    if "<USER_TURN>" in blob:
        found.append("USER_TURN")
    if speaker == "user":
        if _META_LINE.search(blob):
            found.append("tag caption")
        if _DIRECTIVE_ANY.search(blob):
            found.append("writer directive")
        if _CALL_SYNTAX.search(blob):
            found.append("tool call syntax")
    return found


def _score_opener(text: str) -> tuple[float, str]:
    raw = str(text or "").strip()
    if not raw:
        return 0.0, "empty opener"
    if _GIBBERISH.match(raw):
        return 0.1, "gibberish opener"
    if agent_voice_user(raw):
        return 0.0, "agent-voice opener"
    if _ACK_ONLY.match(raw):
        return 0.0, "ack-only opener"
    leaks = _leak_hits(raw, speaker="user")
    if leaks:
        return 0.0, "leaked " + leaks[0]
    if _DESK_OPENER.search(raw):
        return 0.25, "desk-voice opener"
    if not usable_user_message(raw):
        return 0.2, "non-human opener"
    n = len(_words(raw))
    if n <= 2 and not _IDISH.search(raw) and not _ACTION.search(raw):
        return 0.3, "one-word opener"
    if raw.endswith("'") or len(raw) < 8:
        return 0.55, "truncated opener"
    return 1.0, ""


def _score_ping_pong(beats: list[dict]) -> tuple[float, str]:
    if not beats:
        return 0.0, "empty conversation"
    if beats[0]["role"] != "user":
        return 0.2, "does not start with user"
    stacked = 0
    adjacent_user = 0
    for i in range(1, len(beats)):
        if beats[i]["role"] == beats[i - 1]["role"] == "assistant":
            stacked += 1
        if beats[i]["role"] == beats[i - 1]["role"] == "user":
            adjacent_user += 1
    n_user = sum(1 for b in beats if b["role"] == "user")
    n_asst = sum(1 for b in beats if b["role"] == "assistant")
    if stacked:
        return 0.0, "stacked assistant variants"
    if n_user == 1 and n_asst >= 4:
        return 0.0, "one user and 4+ assistant turns"
    if n_asst == 0:
        return 0.1, "no agent turn"
    if adjacent_user:
        return 0.3, "adjacent user turns"
    alternating = all(beats[i]["role"] != beats[i - 1]["role"]
                      for i in range(1, len(beats)))
    if not alternating:
        return 0.3, "roles do not alternate"
    if beats[-1]["role"] == "user":
        return 0.4, "ends on user"
    if n_user >= 2 and n_asst >= 2:
        return 1.0, ""
    return 0.7, ""


def _score_leak(user_texts: list[str], assistant_texts: list[str]) -> tuple[float, str]:
    hits: list[str] = []
    for text in user_texts:
        hits.extend(_leak_hits(text, speaker="user"))
    for text in assistant_texts:
        hits.extend(_leak_hits(text, speaker="assistant"))
    if not hits:
        return 1.0, ""
    uniq = list(dict.fromkeys(hits))
    return 0.0, "leaked " + uniq[0]


def _score_complexity(row: dict, messages: list[dict], beats: list[dict],
                      tools: set[str], opener: str, final: str) -> tuple[float, str]:
    n_user = sum(1 for b in beats if b["role"] == "user")
    n_words = len(_words(opener))
    final_n = len(_words(final))
    stub = bool(_INFRA_STUB.search(final.strip())) if final.strip() else not tools
    strong = bool(_IDISH.search(opener) or _STRONG_ACTION.search(opener))
    clarify = (n_user <= 1 and not tools and _QUESTION_END.search(final))
    if _DEGENERATE.search(final):
        return 0.2, "degenerate reply"
    if stub and not tools and n_user <= 1 and n_words < 8:
        return 0.1, "empty or infra stub"
    if n_words <= 2 and not tools and n_user <= 1:
        return 0.15, "one-line dead end"
    if clarify:
        return 0.25, "clarify and stop"
    if strong and not tools:
        return 0.3, "actionable ask unused tools"
    score = 0.2
    if tools:
        score += 0.35
    if n_user >= 2:
        score += 0.3
    if final_n >= 12 and not stub:
        score += 0.15
    if n_words >= 12:
        score += 0.1
    score = min(1.0, score)
    if score < FAIL:
        return score, "thin conversation"
    return score, ""


def _score_structure(messages: list[dict], beats: list[dict],
                     final: str, user_texts: list[str],
                     assistant_texts: list[str]) -> tuple[float, str]:
    notes: list[str] = []
    score = 1.0
    last_role = messages[-1].get("role") if messages else None
    if last_role == "user":
        score -= 0.6
        notes.append("ends on user")
    elif last_role == "tool":
        score -= 0.55
        notes.append("ends on tool")
    elif last_role != "assistant":
        score -= 0.4
        notes.append("does not end on agent")
    last_spoken = assistant_texts[-1] if assistant_texts else ""
    if not _norm(final):
        score -= 0.4
        notes.append("empty final_text")
    elif last_spoken and _norm(final) != _norm(last_spoken):
        stale = any(_norm(final) == _norm(t) for t in assistant_texts[:-1])
        score -= 0.55
        notes.append("stale final_text" if stale else "final_text mismatch")
    elif last_spoken and _norm(final) == _norm(last_spoken):
        pass
    elif _norm(final) and not last_spoken:
        score -= 0.2
        notes.append("final_text not in messages")
    for i, follow in enumerate(user_texts[1:], start=1):
        prior_agent = assistant_texts[min(i - 1, len(assistant_texts) - 1)] if assistant_texts else ""
        if not usable_user_message(follow):
            score -= 0.35
            notes.append("follow-up left world")
            break
        if (_ID_FOLLOW.search(follow) or len(_words(follow)) <= 8):
            continue
        if prior_agent and _echoes_agent(follow, prior_agent):
            score -= 0.35
            notes.append("follow-up echoes agent")
            break
    if beats and beats[0]["role"] != "user":
        score -= 0.2
        notes.append("starts on agent")
    return max(0.0, min(1.0, score)), notes[0] if notes else ""


def score_row(row: dict) -> dict[str, Any]:
    """Score one row. Returns quality, quality_reason, quality_scores. No mutate."""
    messages = _messages(row)
    opener = str(row.get("prompt") or "")
    users = _user_texts(messages, opener)
    if users and not opener:
        opener = users[0]
    elif users:
        opener = users[0]
    final = str(row.get("final_text") or "")
    assistant = _assistant_spoken(messages)
    beats = _beats(messages)
    tools = _tool_names(row, messages)

    opener_s, opener_n = _score_opener(opener)
    ping_s, ping_n = _score_ping_pong(beats)
    leak_s, leak_n = _score_leak(users, assistant)
    comp_s, comp_n = _score_complexity(row, messages, beats, tools, opener, final)
    struct_s, struct_n = _score_structure(messages, beats, final, users, assistant)

    scores = {
        "opener": round(opener_s, 3),
        "ping_pong": round(ping_s, 3),
        "leak": round(leak_s, 3),
        "complexity": round(comp_s, 3),
        "structure": round(struct_s, 3),
    }
    quality = round(sum(scores[k] for k in DIMENSIONS) / len(DIMENSIONS), 3)
    notes = list(dict.fromkeys(
        n for n in (opener_n, ping_n, leak_n, comp_n, struct_n) if n))
    fails = [k for k in DIMENSIONS if scores[k] < FAIL]
    if not fails:
        reason = "conforms"
    else:
        reason = "; ".join(notes) if notes else "below rubric"
    return {
        "quality": quality,
        "quality_reason": reason,
        "quality_scores": scores,
    }


def apply_quality(row: dict) -> dict:
    """Write quality fields onto row. Returns the same dict."""
    scored = score_row(row)
    row["quality"] = scored["quality"]
    row["quality_reason"] = scored["quality_reason"]
    row["quality_scores"] = scored["quality_scores"]
    return row


def rank_rows(rows: Sequence[dict]) -> list[dict]:
    """Score each row in place. Returns the same list when given a list."""
    for row in rows:
        apply_quality(row)
    return rows if isinstance(rows, list) else list(rows)


def _clip(text: str, n: int = 88) -> str:
    blob = _norm(text)
    return blob if len(blob) <= n else blob[: n - 1] + "…"


def summarize(rows: Sequence[dict], *, top: int = 3) -> dict[str, Any]:
    """Short report over already-scored rows."""
    n = len(rows)
    values = [float(r.get("quality") or 0.0) for r in rows]
    fail_rates = {}
    for dim in DIMENSIONS:
        if n == 0:
            fail_rates[dim] = 0.0
            continue
        misses = 0
        for r in rows:
            scores = r.get("quality_scores") or {}
            if float(scores.get(dim, 0.0)) < FAIL:
                misses += 1
        fail_rates[dim] = round(misses / n, 3)
    buckets = {"<0.50": 0, "0.50-0.69": 0, "0.70-0.84": 0, ">=0.85": 0}
    for q in values:
        if q < 0.50:
            buckets["<0.50"] += 1
        elif q < 0.70:
            buckets["0.50-0.69"] += 1
        elif q < 0.85:
            buckets["0.70-0.84"] += 1
        else:
            buckets[">=0.85"] += 1
    ranked = sorted(rows, key=lambda r: (
        float(r.get("quality") or 0.0), str(r.get("prompt") or "")))
    def card(r):
        return {
            "quality": r.get("quality"),
            "quality_reason": r.get("quality_reason"),
            "prompt": _clip(str(r.get("prompt") or "")),
            "scores": r.get("quality_scores") or {},
        }

    def unique(seq, n):
        seen: set[str] = set()
        out = []
        for r in seq:
            key = _norm(str(r.get("prompt") or ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(card(r))
            if len(out) >= n:
                break
        return out
    mean = round(statistics.fmean(values), 3) if values else 0.0
    median = round(statistics.median(values), 3) if values else 0.0
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "min": round(min(values), 3) if values else 0.0,
        "max": round(max(values), 3) if values else 0.0,
        "distribution": buckets,
        "fail_rates": fail_rates,
        "n_pass": sum(1 for q in values if q >= FAIL),
        "best": unique(reversed(ranked), top) if ranked else [],
        "worst": unique(ranked, top) if ranked else [],
    }


def _write_jsonl(path: str | Path, rows: Sequence[dict]) -> str:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with open(tmp, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str) + "\n")
        tmp.replace(dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return str(dest)


def _load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def rank(source, *, output: str | None = None,
         min_quality: float | None = None) -> dict[str, Any]:
    """Score a JSONL path or a list of rows. Writes scored JSONL when given a path.

    Existing keys stay. ``quality``, ``quality_reason``, and ``quality_scores``
    are added. ``min_quality`` filters the written file only when ``output`` is
    a different path; the in-place rewrite keeps every scored row.
    """
    if isinstance(source, (str, Path)):
        rows = _load_jsonl(source)
        src = str(source)
    else:
        rows = list(source)
        src = ""
    rank_rows(rows)
    dest = output or src
    written = None
    kept = rows
    if dest:
        if (min_quality is not None and output
                and src and Path(output).resolve() != Path(src).resolve()):
            kept = [r for r in rows if float(r.get("quality") or 0.0) >= min_quality]
        written = _write_jsonl(dest, kept)
    report = summarize(rows)
    report["path"] = written or src
    report["n_written"] = len(kept) if dest else 0
    if min_quality is not None:
        report["min_quality"] = min_quality
        report["n_kept"] = sum(
            1 for r in rows if float(r.get("quality") or 0.0) >= min_quality)
    return report


def format_report(report: dict) -> str:
    """Plain-text summary for a CLI or notebook."""
    lines = [
        f"n={report.get('n', 0)}  mean={report.get('mean')}  "
        f"median={report.get('median')}  "
        f"min={report.get('min')}  max={report.get('max')}",
        "distribution: " + ", ".join(
            f"{k}={v}" for k, v in (report.get("distribution") or {}).items()),
        "fail rates: " + ", ".join(
            f"{k}={v:.0%}" for k, v in (report.get("fail_rates") or {}).items()),
    ]
    if report.get("path"):
        lines.append(f"wrote {report.get('n_written', 0)} rows to {report['path']}")
    for label, key in (("best", "best"), ("worst", "worst")):
        items = report.get(key) or []
        if not items:
            continue
        lines.append(f"{label}:")
        for item in items:
            lines.append(
                f"  {item.get('quality'):.3f}  {item.get('quality_reason')}  "
                f"{item.get('prompt')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Score generated conversation JSONL with the quality rubric.")
    parser.add_argument("path", help="JSONL from simulate().save()")
    parser.add_argument("-o", "--output", help="Scored JSONL path. Default: rewrite path.")
    parser.add_argument("--min-quality", type=float, default=None,
                        help="If set with --output, write only rows at or above this score.")
    args = parser.parse_args(argv)
    report = rank(args.path, output=args.output, min_quality=args.min_quality)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
