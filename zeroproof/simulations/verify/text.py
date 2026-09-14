"""Text verifiers: exact match, substring, regex, multiple choice.

The table-stakes checkers. Each honors the judge contract through the
``Verifier`` base; none calls a model.
"""

from __future__ import annotations

import re
import string
from typing import Any

from .base import Verifier, VerifierError


def normalize(
    text: str, *, lower: bool = True, strip_punct: bool = True, collapse_ws: bool = True
) -> str:
    s = str(text or "")
    if lower:
        s = s.lower()
    if strip_punct:
        s = s.translate(str.maketrans("", "", string.punctuation))
    if collapse_ws:
        s = " ".join(s.split())
    return s.strip()


class ExactMatch(Verifier):
    """Candidate equals the reference after normalization. The last
    non-empty line of the candidate is used, so a chatty answer that ends
    in the value still matches; pass ``whole=True`` to match the full text."""

    def __init__(
        self,
        *,
        field: str | None = None,
        lower: bool = True,
        strip_punct: bool = True,
        whole: bool = False,
        name: str | None = None,
    ):
        super().__init__(field=field, name=name)
        self.lower, self.strip_punct, self.whole = lower, strip_punct, whole

    def _norm(self, t: str) -> str:
        return normalize(t, lower=self.lower, strip_punct=self.strip_punct)

    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        if reference is None:
            return None
        refs = reference if isinstance(reference, (list, tuple)) else [reference]
        cand = candidate if self.whole else (candidate.strip().splitlines() or [""])[-1]
        cn = self._norm(cand)
        for r in refs:
            if cn == self._norm(str(r)):
                return 1, f"exact match: {str(r)[:60]!r}"
        # also allow full-text match as a fallback when last-line missed
        if not self.whole and any(self._norm(candidate) == self._norm(str(r)) for r in refs):
            return 1, "exact match (full text)"
        return 0, f"no match; got {cand.strip()[:60]!r}, want {str(refs[0])[:60]!r}"


class Includes(Verifier):
    """The reference appears somewhere in the candidate (case-insensitive by
    default). With a list reference, ``mode='all'`` needs every item present,
    ``mode='any'`` needs one."""

    def __init__(
        self,
        *,
        field: str | None = None,
        lower: bool = True,
        mode: str = "all",
        name: str | None = None,
    ):
        super().__init__(field=field, name=name)
        self.lower, self.mode = lower, mode

    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        if reference is None:
            return None
        hay = candidate.lower() if self.lower else candidate
        needles = reference if isinstance(reference, (list, tuple)) else [reference]
        hits = [str(n) for n in needles if (str(n).lower() if self.lower else str(n)) in hay]
        ok = len(hits) > 0 if self.mode == "any" else len(hits) == len(needles)
        return (1 if ok else 0), f"{len(hits)}/{len(needles)} present"


class Regex(Verifier):
    """A regex over the candidate. Three modes:

    - no reference: pass iff the pattern matches (a format check).
    - reference given, ``group`` set: the captured group must equal the
      reference (extract-then-compare).
    - reference given, no group: the pattern is built to require the
      reference literally is not needed; pass iff it matches.
    """

    def __init__(
        self,
        pattern: str,
        *,
        group: int | str | None = None,
        field: str | None = None,
        flags: int = re.I,
        lower: bool = True,
        name: str | None = None,
    ):
        super().__init__(field=field, name=name)
        try:
            self.re = re.compile(pattern, flags)
        except re.error as exc:
            raise VerifierError(f"bad regex: {exc}") from exc
        self.group, self.lower = group, lower

    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        m = self.re.search(candidate or "")
        if not m:
            return 0, "pattern did not match"
        if reference is None or self.group is None:
            return 1, "pattern matched"
        got = m.group(self.group)
        a, b = (got or ""), str(reference)
        if self.lower:
            a, b = a.lower().strip(), b.lower().strip()
        return (1 if a == b else 0), f"captured {got!r}, want {reference!r}"


_CHOICE = re.compile(r"\b([A-J])\b")
_CHOICE_ANSWER = re.compile(r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-J])\)?", re.I)


class MultipleChoice(Verifier):
    """Extract a single choice letter (A-J) from the candidate and compare to
    the reference letter. Prefers an explicit 'the answer is X', else the
    last standalone letter."""

    def __init__(self, *, field: str | None = None, name: str | None = None):
        super().__init__(field=field, name=name)

    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        if reference is None:
            return None
        want = str(reference).strip().upper().lstrip("(").rstrip(")")[:1]
        m = _CHOICE_ANSWER.search(candidate or "")
        got = m.group(1).upper() if m else None
        if not got:
            letters = _CHOICE.findall((candidate or "").upper())
            got = letters[-1] if letters else None
        if not got:
            return 0, "no choice letter found"
        return (1 if got == want else 0), f"chose {got}, answer {want}"
