"""Behavior packs: a transform that provokes, a marker that judges.

Each pack targets one trainable behavior (sycophancy resistance,
abstention, tool-call discipline, injection resistance, ...) and ships as
one module exposing:

* ``SPEC``: {"name", "description"} — what the pack provokes and checks.
* ``transform(rows, *, seed=0) -> list[dict]``: derive provocation rows
  from existing conversations (append the wrong pushback, delete the
  supporting passage, plant the override...). Pure and deterministic for
  a given seed; never mutates its input.
* ``marker(row) -> dict[str, int]``: deterministic 0/1 measurements named
  like ``sycophancy.answer_flipped``. String rules, not judges: a marker a
  model grades is a judge wearing a marker's name.

Packs register by existing: any module in this package with a SPEC is
discovered, so adding a pack never edits shared files.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Any


def packs() -> dict[str, Any]:
    """Every behavior pack in the package, keyed by SPEC name."""
    found: dict[str, Any] = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        spec = getattr(module, "SPEC", None)
        if isinstance(spec, dict) and spec.get("name"):
            found[str(spec["name"])] = module
    return found


__all__ = ["packs"]
