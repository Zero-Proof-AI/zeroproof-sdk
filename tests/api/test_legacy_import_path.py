"""``zeroproof_simulations`` is an alias of ``zeroproof.simulations``, not a copy."""
from __future__ import annotations

import importlib
import subprocess
import sys
import warnings


def test_legacy_top_level_name_is_the_same_module():
    import zeroproof.simulations as new
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("zeroproof_simulations")
    assert old is new


def test_legacy_submodule_paths_are_the_same_objects():
    from zeroproof.simulations.run import engine as new_engine
    from zeroproof.simulations.score import judging as new_judging
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_engine = importlib.import_module("zeroproof_simulations.run.engine")
        old_judging = importlib.import_module("zeroproof_simulations.score.judging")
    assert old_engine is new_engine
    assert old_judging is new_judging


_FRESH = """
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import zeroproof_simulations as zps
messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
assert messages and "zeroproof.simulations" in messages[0], messages
import zeroproof.simulations as new
assert zps is new and callable(zps.simulate)
print("warned-and-aliased")
"""


def test_fresh_interpreter_warns_and_aliases():
    out = subprocess.run([sys.executable, "-c", _FRESH],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-1500:]
    assert "warned-and-aliased" in out.stdout
