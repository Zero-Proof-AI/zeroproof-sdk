"""``Report``: a measurement that prints itself.

    print(wai.judge_trust(scored.rows))   # the report, not its dict literal

Rule 5 of ``docs/reference/style.md``: a measurement returns an object
with ``__str__`` for the terminal and ``_repr_html_`` for a notebook, and
no public call hands back a bare dict the user has to know the keys of.
The reports in this package were dicts first, and code in the wild reads
their keys, so ``Report`` *is* a dict: every key, ``.get``, ``json.dumps``
and ``==`` against a plain dict keep working. What it adds is the
printing. Each report subclasses it next to its own formatter and says
how it reads:

    class JudgeTrustReport(Report):
        def __str__(self) -> str:
            return format_judge_trust(self)

The ``format_*`` functions stay: they take a dict, and a report is one.
"""

from __future__ import annotations

from typing import Any


class Report(dict):
    """A measurement. A dict for reading keys, an object for printing.

    Subclasses override ``__str__`` with the report's own formatter and
    set ``_summary_keys`` to the one or two fields that belong in a
    ``repr``.

    Reference: docs/reference/style.md rule 5 (results are objects that
    print themselves).
    """

    #: the fields a one-line ``repr`` shows, in order.
    _summary_keys: tuple[str, ...] = ("ok",)

    def _repr_html_(self) -> str:
        """The same text, as a notebook cell."""
        escaped = str(self).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<pre>{escaped}</pre>"

    def __repr__(self) -> str:
        bits = ", ".join(f"{k}={self[k]!r}" for k in self._summary_keys if k in self)
        return f"{type(self).__name__}({bits})"

    def dict(self) -> dict[str, Any]:
        """The plain dict, for callers that want to serialize or mutate it."""
        return dict(self)
