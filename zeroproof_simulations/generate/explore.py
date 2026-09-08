"""Offline-only mutations of existing user messages. Live path never wraps Qwen text."""
from __future__ import annotations
import hashlib
import json
import re

_SPECIFIC = re.compile(r"\b([A-Z]{2,}-?\d{2,}|\d+\.\d{2}|\$\d+|\b\d{3,}\b)")
_TURN = "\n<USER_TURN>\n"


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def crossover(a: str, b: str) -> str:
    sa, sb = _sentences(a), _sentences(b)
    if not sa or not sb:
        return a
    return " ".join(sa[: max(1, len(sa) // 2)] + sb[max(1, len(sb) // 2):])


def truncate(a: str) -> str:
    sa = _sentences(a)
    return sa[0] if sa else a


def strip_specifics(a: str) -> str:
    return _SPECIFIC.sub("that one", a)


def duplicate_request(a: str) -> str:
    sa = _sentences(a)
    return a if not sa else a + " " + sa[0]


def contradict(a: str) -> str:
    return a + " Actually no, cancel that — but still get it sorted today."


def reverse_sentences(a: str) -> str:
    sentences = _sentences(a)
    return " ".join(reversed(sentences)) if len(sentences) > 1 else a


def list_structure(a: str) -> str:
    sentences = _sentences(a)
    parts = sentences if len(sentences) > 1 else [p.strip() for p in re.split(r",|;", a) if p.strip()]
    return "Please handle all of these:\n" + "\n".join(
        f"{index + 1}. {part}" for index, part in enumerate(parts))


def punctuation_free(a: str) -> str:
    return re.sub(r"[^\w\s]", " ", a).replace("_", " ")


def token_delete(a: str, i: int) -> str:
    words = a.split()
    if len(words) < 4:
        return a
    index = (i * 7 + len(words) // 2) % len(words)
    return " ".join(words[:index] + words[index + 1:])


def token_swap(a: str, i: int) -> str:
    words = a.split()
    if len(words) < 3:
        return a
    index = (i * 5 + len(words) // 3) % (len(words) - 1)
    words[index], words[index + 1] = words[index + 1], words[index]
    return " ".join(words)


def typo_transpose(a: str, i: int) -> str:
    words = a.split()
    eligible = [index for index, word in enumerate(words) if len(word.strip(".,!?")) >= 5]
    if not eligible:
        return a
    index = eligible[i % len(eligible)]
    word = words[index]
    pos = max(1, min(len(word) - 2, len(word) // 2))
    words[index] = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
    return " ".join(words)


def quote_payload(a: str) -> str:
    return f'The exact request I received was: "{a}" Please handle it.'


def structured_payload(a: str) -> str:
    compact = " ".join(a.split())
    return "User request payload:\n```json\n" + json.dumps({"request": compact}) + "\n```"


def follow_up_correction(a: str) -> str:
    return (a + _TURN +
            "Wait, use the other one. Check what already happened before acting again.")


MUTATORS = (
    ("truncate", lambda a, b, i: truncate(a)),
    ("strip_specifics", lambda a, b, i: strip_specifics(a)),
    ("crossover", lambda a, b, i: crossover(a, b)),
    ("duplicate", lambda a, b, i: duplicate_request(a)),
    ("contradict", lambda a, b, i: contradict(a)),
    ("reverse_sentences", lambda a, b, i: reverse_sentences(a)),
    ("list_structure", lambda a, b, i: list_structure(a)),
    ("punctuation_free", lambda a, b, i: punctuation_free(a)),
    ("token_delete", lambda a, b, i: token_delete(a, i)),
    ("token_swap", lambda a, b, i: token_swap(a, i)),
    ("typo_transpose", lambda a, b, i: typo_transpose(a, i)),
    ("quote_payload", lambda a, b, i: quote_payload(a)),
    ("structured_payload", lambda a, b, i: structured_payload(a)),
    ("follow_up_correction", lambda a, b, i: follow_up_correction(a)),
)


def mutate_pool(texts: list[str], rounds: int = 1,
                limit: int = 20_000) -> list[tuple[str, str]]:
    """Apply every mutator across the pool. Returns (mutator_name, text)."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    frontier = list(dict.fromkeys(texts))
    for r in range(rounds):
        if not frontier or len(out) >= limit:
            break
        stride = 1 + r
        next_frontier: list[str] = []
        for i, a in enumerate(frontier):
            b = frontier[(i + stride) % len(frontier)]
            for name, fn in MUTATORS:
                try:
                    text = fn(a, b, i).strip()
                except Exception:
                    continue
                if len(text) < 12:
                    continue
                key = hashlib.blake2b(text.lower().encode(), digest_size=8).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                out.append((name, text))
                next_frontier.append(text)
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        frontier = next_frontier
    return out
