"""Records for a spec that has tools and a policy but no executable world.

tau-bench ships a database, so a question about reservation ABC123 can be
grounded in a real row. Most specs do not. Rather than hand-write records,
which would put one author's imagination into every row, the records are
SIMULATED from the spec's own tool schemas: the model is shown what the tools
return and asked to invent one internally-consistent record.

That keeps the property the voice lane actually needs, which is that the
answer is decidable and the identifiers it must name are known in advance. It
does not claim the records are real, and the card says so.
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import endpoint as RA

RECORD_SYSTEM = (
    "You invent ONE realistic record for a system, given the tools that operate on it. "
    "Return a single JSON object and nothing else. It must contain an obvious unique "
    "identifier field whose value is an opaque code, and at least six other fields with "
    "concrete values a support agent would be asked about: statuses, timestamps, names, "
    "numbers, categories. Make the fields internally consistent. Do not include commentary."
)

_ID = re.compile(r"^[A-Z0-9][A-Z0-9_\-]{4,}$")


def _balanced(text: str) -> str | None:
    """The first complete {...} object, by brace depth rather than by regex."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _first_id(rec: dict) -> str | None:
    """The opaque code a question will have to name."""
    for v in rec.values():
        if isinstance(v, str) and _ID.match(v.strip()):
            return v.strip()
    for k, v in rec.items():
        if "id" in k.lower() and isinstance(v, (str, int)) and len(str(v)) >= 4:
            return str(v)
    return None


def build_records(spec: dict, n: int, seed: int = 0, workers: int = 10) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    tools = spec.get("tools") or []
    names = [t.get("function", t).get("name") for t in tools]
    brief = json.dumps(tools, default=str)[:2500]

    def one(i: int) -> dict | None:
        try:
            out = RA.chat(
                [
                    {"role": "system", "content": RECORD_SYSTEM},
                    {
                        "role": "user",
                        "content": f"System policy:\n{str(spec.get('policy'))[:900]}\n\n"
                        f"Tools that read or change this record: {', '.join(names[:8])}\n"
                        f"Tool schemas:\n{brief}\n\n"
                        f"Invent record number {i + 1}. JSON only.",
                    },
                ],
                temperature=1.0,
                max_tokens=420,
            )
        except Exception:
            return None
        # A greedy {...} match swallows prose between two objects and parses
        # as nothing. Strip fences, then try the whole thing, then the
        # outermost balanced object found by scanning.
        text = re.sub(r"^```(?:json)?|```$", "", out.strip(), flags=re.MULTILINE).strip()
        for cand in (text, _balanced(text)):
            if not cand:
                continue
            try:
                rec = json.loads(cand)
            except Exception:
                continue
            if isinstance(rec, dict) and _first_id(rec):
                return rec
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        recs = [r for r in pool.map(one, range(n)) if r]
    # an id must be unique or two questions collide on the same answer
    seen, out = set(), []
    for r in recs:
        rid = _first_id(r)
        if rid in seen:
            continue
        seen.add(rid)
        out.append(r)
    return out


def record_id(rec: dict) -> str | None:
    return _first_id(rec)
