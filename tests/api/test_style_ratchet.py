"""The style ratchet: the public surface may shrink, never grow.

``docs/reference/style.md`` sets the coding standard (PyTorch/DSPy
ergonomics: few names, objects carry configuration, calls carry data,
reports print themselves). This test pins today's counts for the shapes
the standard retires and fails when any of them grows. Lower a pin when
you retire a name; never raise one without saying why in the PR body.

Run ``uv run pytest tests/api/test_style_ratchet.py -q -rA`` to print
the current counts next to the pins.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

import whileai.simulations as wai

# Rule 3: a public call takes at most this many parameters.
MAX_PARAMS = 8

# Today's counts. Each is a ceiling; the test names the rule it guards.
PINS = {
    "exports": 212,  # rule 1: names in whileai.simulations.__all__
    "wide_calls": 27,  # rule 3: public calls with more than MAX_PARAMS parameters
    "format_twins": 13,  # rule 5: format_* functions instead of __str__ on a report
    "in_place_mutators": 6,  # rule 4: attach_* / stamp_* free functions over rows
    "implementation_names": 17,  # rule 6: build_/load_/run_ prefixes, _of/_rows suffixes
}

IMPLEMENTATION_PREFIXES = ("build_", "load_", "run_")
IMPLEMENTATION_SUFFIXES = ("_of", "_rows")


def _public_callables() -> dict[str, object]:
    """Public functions and classes a user calls. Dataclasses are records
    (a row schema, a result, a typed options object); their field count is
    not a call's parameter count, so they are not held to rule 3."""
    out: dict[str, object] = {}
    for n in wai.__all__:
        obj = getattr(wai, n)
        if callable(obj) and not dataclasses.is_dataclass(obj):
            out[n] = obj
    return out


def _param_count(obj: object) -> int:
    try:
        sig = inspect.signature(obj)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return sum(
        1
        for p in sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and p.name != "self"
    )


def _counts() -> dict[str, int]:
    names = list(wai.__all__)
    calls = _public_callables()
    return {
        "exports": len(names),
        "wide_calls": sum(1 for obj in calls.values() if _param_count(obj) > MAX_PARAMS),
        "format_twins": sum(1 for n in names if n.startswith("format_")),
        "in_place_mutators": sum(1 for n in names if n.startswith(("attach_", "stamp_"))),
        "implementation_names": sum(
            1
            for n in names
            if n.startswith(IMPLEMENTATION_PREFIXES) or n.endswith(IMPLEMENTATION_SUFFIXES)
        ),
    }


@pytest.mark.parametrize("key", sorted(PINS))
def test_surface_does_not_grow(key: str) -> None:
    now = _counts()[key]
    pin = PINS[key]
    assert now <= pin, (
        f"{key}: {now} public names, pin is {pin}. docs/reference/style.md retires this "
        f"shape; put the new name one dot down, or make it a method on the rows object."
    )
    if now < pin:
        pytest.fail(
            f"{key}: {now} < pin {pin}. Good: lower PINS[{key!r}] to {now} in this PR "
            f"so the ratchet holds the gain.",
            pytrace=False,
        )


def test_wide_calls_are_named() -> None:
    """The calls over the cap, so a reviewer sees which ones a PR should split."""
    wide = sorted(
        (n, _param_count(obj))
        for n, obj in _public_callables().items()
        if _param_count(obj) > MAX_PARAMS
    )
    assert len(wide) <= PINS["wide_calls"], wide
