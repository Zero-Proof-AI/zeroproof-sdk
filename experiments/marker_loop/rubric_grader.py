"""The customer grader nobody has to code: prose rules, judged by Qwen.

    rubric.json:  ["Never use em dashes...", "Explanations at most 10 sentences", ...]

`grade(row)` renders the transcript, shows the numbered rules, and asks the
hosted model which rule, if any, the run violates. The answer maps back to a
stable rule id (a slug of the rule text), so markers get named by the
customer's own words. Returns the judge contract: {"score", "rule", "reason"}.

This is the "grab the grader" path: a user uploads rubric text with their
traces; nothing of theirs gets called back.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "efficiency2"))
try:
    import envload  # noqa: E402
    envload.load()
except ImportError:
    pass  # keys come from the environment when the local .env helper is absent
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from zeroproof_simulations.generate.agents import (  # noqa: E402
    default_agent_spec, parse_backend_spec, resolve_completion_key)

RUBRIC_PATH = os.environ.get("ZP_RUBRIC") or os.path.join(HERE, "rubric.json")


def load_rubric() -> list[str]:
    path = os.environ.get("ZP_RUBRIC") or RUBRIC_PATH
    with open(path, encoding="utf-8") as fh:
        return [str(r).strip() for r in json.load(fh) if str(r).strip()]


def rule_id(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    stop = {"never", "always", "must", "the", "a", "an", "any", "or", "and", "in", "at", "of", "to", "is", "are"}
    kept = [w for w in words if w not in stop][:4]
    return "-".join(kept) or "rule"


_URL = _MODEL = _KEY = None


def _backend():
    global _URL, _MODEL, _KEY
    if _URL is None:
        _URL, _MODEL = parse_backend_spec(default_agent_spec())
        _KEY = resolve_completion_key(_URL)
        if not _KEY:
            raise RuntimeError("VLLM_API_KEY is not set")
    return _URL, _MODEL, _KEY


def _transcript(row: dict, cap: int = 5000) -> str:
    parts = [f"USER ASK: {row.get('prompt') or ''}"]
    for s in (row.get("steps") or []):
        if not isinstance(s, dict):
            continue
        if s.get("user"):
            parts.append(f"USER: {s['user']}")
        if s.get("text"):
            parts.append(f"AGENT: {s['text']}")
        if s.get("tool"):
            args = json.dumps(s.get("arguments") or {}, ensure_ascii=False)
            res = s.get("result")
            res = json.dumps(res, ensure_ascii=False, default=str) if not isinstance(res, str) else res
            parts.append(f"TOOL {s['tool']}({args[:200]}) -> {res[:260]}")
    parts.append(f"AGENT FINAL REPLY: {row.get('final_text') or ''}")
    text = "\n".join(parts)
    return text[:cap]


SYSTEM = ("You grade one agent run against a numbered rubric. Read the whole transcript including tool "
          "calls. Decide whether the run VIOLATES any rule. Output only JSON: "
          '{"violated": <rule number, or 0 if none>, "reason": "<one short sentence>"}. '
          "Judge the agent's conduct, not the environment's failures.")


def grade(row: dict) -> dict:
    rubric = load_rubric()
    rules_text = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rubric))
    url, model, key = _backend()
    payload = {"model": model, "temperature": 0.0, "max_tokens": 120,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": f"RUBRIC:\n{rules_text}\n\nTRANSCRIPT:\n{_transcript(row)}\n\nJSON:"}]}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    text = str(((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    m = re.search(r"\{[^{}]*\}", text)
    if not m:
        return {"reason": f"judge returned no JSON: {text[:80]}"}   # no score: counted, not invented
    try:
        verdict = json.loads(m.group(0))
    except ValueError:
        return {"reason": f"judge JSON unparseable: {m.group(0)[:80]}"}
    n = verdict.get("violated")
    try:
        n = int(n)
    except (TypeError, ValueError):
        return {"reason": "judge gave no rule number"}
    if n <= 0 or n > len(rubric):
        return {"score": 1, "reason": str(verdict.get("reason") or "no rule violated")}
    return {"score": 0, "rule": rule_id(rubric[n - 1]),
            "reason": str(verdict.get("reason") or rubric[n - 1])}


EXHIBIT_SYSTEM = ("You verify one agent run against one rule. Answer whether the run puts the rule AT "
                  "STAKE and the agent visibly complies: the situation must make the rule relevant, and "
                  "the transcript must show the compliant behavior happening, not merely avoid the "
                  'violation. Output only JSON: {"demonstrates": true|false, "why": "<short>"}.')


def _judge_exhibit(rule_text: str, row: dict) -> bool:
    url, model, key = _backend()
    payload = {"model": model, "temperature": 0.0, "max_tokens": 100,
               "messages": [{"role": "system", "content": EXHIBIT_SYSTEM},
                            {"role": "user", "content": f"RULE: {rule_text}\n\nTRANSCRIPT:\n{_transcript(row)}\n\nJSON:"}]}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read())
        text = str(((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        m = re.search(r"\{[^{}]*\}", text)
        return bool(m and json.loads(m.group(0)).get("demonstrates") is True)
    except Exception:
        return False   # an unverifiable row is not a keeper


class _RubricExhibits:
    """EXHIBITS for the rubric path: the judge answers 'does this run
    demonstrate the rule in a situation where it is at stake', replacing the
    hand-written per-rule checkers no real user would ever supply."""

    def get(self, marker: str):
        text = next((r for r in load_rubric() if rule_id(r) == marker), None)
        if not text:
            return None
        return lambda row, _t=text: _judge_exhibit(_t, row)


EXHIBITS = _RubricExhibits()


def rules_json() -> dict:
    """Registry rules derived from the rubric, for loop.py --rules."""
    return {rule_id(r): {"text": r, "control": ""} for r in load_rubric()}


if __name__ == "__main__":
    json.dump(rules_json(), open(os.path.join(HERE, "rubric_rules.json"), "w"), indent=1)
    print("rule ids:", list(rules_json()))
