"""Deprecated import path. The package is ``zeroproof.simulations``.

``import zeroproof_simulations`` and every submodule path under it keep
working for now and resolve to the very same module objects, so
``zeroproof_simulations.run.engine is zeroproof.simulations.run.engine``.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

_OLD = "zeroproof_simulations"
_NEW = "zeroproof.simulations"


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve ``zeroproof_simulations.x`` from ``zeroproof.simulations.x``."""

    def find_spec(self, name, path=None, target=None):
        if name == _OLD or name.startswith(_OLD + "."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(_NEW + spec.name[len(_OLD):])

    def exec_module(self, module):
        return None


warnings.warn(
    "zeroproof_simulations moved to zeroproof.simulations; the old name is "
    "kept for now and goes away in a later release. Change "
    "`import zeroproof_simulations as zps` to `import zeroproof.simulations as zps`.",
    DeprecationWarning, stacklevel=2)

if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
sys.modules[__name__] = importlib.import_module(_NEW)
