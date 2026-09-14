"""Verifiers: rewards that are programs, not judges.

RLHF book ch. 13 (Tools, RLVR) and ch. 7 (Reasoning): the reward for a
verifiable task is a checker, not an opinion. A verifier reads a rollout,
decides pass or fail (or a partial score in [0, 1]), and says why.

Every verifier honors the judge contract in ``zeroproof.simulations.judging``
(``callable(row) -> {"reward", "reason", ...}``), so a verifier drops
straight into ``data.grade(judge=...)``, ``evaluate``, ``optimize`` and a
gated ``push``. Nothing here talks to a model.

A verifier reads two things from a row:

- the *candidate*: what the policy produced. ``final_text`` by default,
  else the last assistant message.
- the *reference*: the gold the checker compares against. It lives in the
  row's ``privileged`` block (``privileged.reference``), which the training
  export deliberately never projects, so the answer key cannot leak into a
  training file. Common flat fields (``reference``, ``answer``, ``target``,
  ``solution``, ``info.answer``) are read as a fallback so a dataset that
  stores the key plainly still works.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

# Where a verifier looks for the gold, in order. privileged.reference is the
# schema-native home (never exported to training rows).
REFERENCE_FIELDS = ("reference", "answer", "target", "solution", "label", "expected")


class VerifierError(Exception):
    """A verifier could not run at all (bad config), distinct from a fail."""


def candidate_text(row: dict) -> str:
    """What the policy produced: final_text, else the last assistant turn."""
    if not isinstance(row, dict):
        return str(row or "")
    text = row.get("final_text")
    if text:
        return str(text)
    for msg in reversed(row.get("messages") or []):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
            return str(msg["content"])
    return ""


def reference_value(row: dict, field: str | None = None) -> Any:
    """The gold. An explicit ``field`` wins; then privileged.reference; then
    the flat fallback fields; then ``info``/``metadata`` sub-dicts."""
    if not isinstance(row, dict):
        return None
    if field:
        if field in row:
            return row[field]
        # dotted path, e.g. "info.answer"
        cur: Any = row
        for part in field.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                cur = None
                break
        return cur
    priv = row.get("privileged")
    if isinstance(priv, dict) and priv.get("reference") is not None:
        return priv["reference"]
    for key in REFERENCE_FIELDS:
        if row.get(key) is not None:
            return row[key]
    for holder in ("info", "metadata", "privileged"):
        sub = row.get(holder)
        if isinstance(sub, dict):
            for key in REFERENCE_FIELDS:
                if sub.get(key) is not None:
                    return sub[key]
    return None


def _result(reward: float | int | None, reason: str, **meta: Any) -> dict[str, Any]:
    # Build the verdict as a literal, never a subscript write, so this stays
    # a judge-contract return and not a direct verdict-key assignment.
    base: dict[str, Any] = {"reward": reward, "reason": reason[:400]}
    return {**base, "judge_meta": meta} if meta else base


class Verifier:
    """Base class. Subclasses implement ``check(candidate, reference, row)``
    and return a float in [0, 1] (or a bool), or a ``(score, reason)`` pair.

    Instances are callables honoring the judge contract, and carry ``name``
    and ``kind`` so the scored row's ``ScorerRef`` records what graded it.
    """

    kind: str = "rule"

    def __init__(self, *, field: str | None = None, name: str | None = None):
        self.field = field
        self.name = name or self.__class__.__name__

    # subclasses override this
    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        raise NotImplementedError

    def __call__(self, row: dict) -> dict[str, Any]:
        try:
            candidate = candidate_text(row)
            reference = reference_value(row, self.field)
            outcome = self.check(candidate, reference, row)
        except VerifierError as exc:
            return _result(None, f"{self.name}: {exc}", verifier=self.name, error=str(exc))
        except Exception as exc:
            return {
                "reward": None,
                "reason": f"{self.name}: {type(exc).__name__}: {exc}"[:400],
                "judge_status": "error",
                "judge_meta": {"verifier": self.name},
            }
        if isinstance(outcome, tuple):
            score, reason = outcome[0], str(outcome[1]) if len(outcome) > 1 else ""
        else:
            score, reason = outcome, ""
        if isinstance(score, bool):
            score = 1 if score else 0
        if score is None:
            return _result(
                None, reason or f"{self.name}: no reference to check against", verifier=self.name
            )
        score = max(0.0, min(1.0, float(score)))
        score = int(score) if score in (0.0, 1.0) else score
        if not reason:
            reason = (
                f"{self.name}: {'pass' if score == 1 else 'fail' if score == 0 else f'{score:.2f}'}"
            )
        return _result(score, reason, verifier=self.name)


class FunctionVerifier(Verifier):
    """Wrap a plain ``fn(candidate, reference, row) -> score`` as a Verifier."""

    def __init__(
        self,
        fn: Callable[[str, Any, dict], Any],
        *,
        name: str | None = None,
        field: str | None = None,
        kind: str = "rule",
    ):
        super().__init__(field=field, name=name or getattr(fn, "__name__", "verifier"))
        self._fn = fn
        self.kind = kind

    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        return self._fn(candidate, reference, row)


def verifier(fn: Callable[[str, Any, dict], Any]) -> FunctionVerifier:
    """Decorator: turn ``fn(candidate, reference, row) -> score`` into a Verifier."""
    return FunctionVerifier(fn)


class All(Verifier):
    """Pass only if every verifier passes. Score is the min; the reason names
    the first failure. Use for a task with several hard constraints."""

    def __init__(self, verifiers: Sequence[Verifier], *, name: str = "All"):
        super().__init__(name=name)
        self.verifiers = list(verifiers)

    def __call__(self, row: dict) -> dict[str, Any]:
        scores, reasons = [], []
        for v in self.verifiers:
            r = v(row)
            if r.get("reward") is None:
                return r
            scores.append(r["reward"])
            reasons.append(r.get("reason", ""))
        worst = min(scores) if scores else 0
        idx = scores.index(worst) if scores else 0
        return _result(
            1 if worst == 1 else worst,
            reasons[idx] if scores else "All: no verifiers",
            verifier=self.name,
            parts=scores,
        )


class Any_(Verifier):
    """Pass if any verifier passes. Score is the max."""

    def __init__(self, verifiers: Sequence[Verifier], *, name: str = "Any"):
        super().__init__(name=name)
        self.verifiers = list(verifiers)

    def __call__(self, row: dict) -> dict[str, Any]:
        scores, reasons = [], []
        for v in self.verifiers:
            r = v(row)
            if r.get("reward") is None:
                continue
            scores.append(r["reward"])
            reasons.append(r.get("reason", ""))
        if not scores:
            return _result(None, f"{self.name}: no verifier could run", verifier=self.name)
        best = max(scores)
        idx = scores.index(best)
        return _result(best, reasons[idx], verifier=self.name, parts=scores)


class Weighted(Verifier):
    """Weighted sum of verifiers, normalized to [0, 1]. Use for a rubric with
    graded criteria rather than one hard pass/fail (book ch. 12 rubrics)."""

    def __init__(self, pairs: Sequence[tuple[Verifier, float]], *, name: str = "Weighted"):
        super().__init__(name=name)
        self.pairs = list(pairs)

    def __call__(self, row: dict) -> dict[str, Any]:
        total_w = sum(w for _, w in self.pairs) or 1.0
        acc, detail = 0.0, {}
        unjudged: str | None = None
        for v, w in self.pairs:
            r = v(row)
            s = r.get("reward")
            detail[v.name] = s
            if s is None:
                # A part that could not run (no reference, bad config) is
                # not a 0: the contract never invents a reward. Same as All.
                unjudged = unjudged or f"{self.name}: {r.get('reason') or v.name}"
                continue
            acc += float(s) * w
        if unjudged is not None:
            return _result(None, unjudged, verifier=self.name, parts=detail)
        score = round(acc / total_w, 4)
        score = int(score) if score in (0.0, 1.0) else score
        return _result(score, f"{self.name}: {score}", verifier=self.name, parts=detail)


def as_verifier(obj: Callable[..., Any]) -> Verifier:
    """Coerce a Verifier or a plain callable into a Verifier."""
    if isinstance(obj, Verifier):
        return obj
    return FunctionVerifier(lambda c, r, row: obj(row), name=getattr(obj, "__name__", "verifier"))


def extract_json(text: str) -> Any:
    """Best-effort: parse text as JSON, or the first {...}/[...] block in it."""
    text = str(text or "").strip()
    fenced = text
    if "```" in fenced:
        import re

        m = re.search(r"```(?:json)?\s*(.+?)```", fenced, re.S)
        if m:
            fenced = m.group(1).strip()
    for candidate in (fenced, text):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    import re

    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None
