"""The installed SDK is zeroproof_simulations only. Specs and examples stay out."""
from __future__ import annotations

import pkgutil
from pathlib import Path

import zeroproof_simulations as zps
from tests.helpers import FIXTURES, REPO_ROOT


def test_sdk_package_excludes_tests_examples_and_specs():
    root = Path(zps.__file__).resolve().parent
    assert root.name == "zeroproof_simulations"
    names = {module.name for module in pkgutil.iter_modules(zps.__path__)}
    assert "tests" not in names
    assert "examples" not in names
    assert "specs" not in names
    assert not (root / "tests").exists()
    assert not (root / "examples").exists()
    assert not (root / "specs").exists()
    assert (REPO_ROOT / "tests" / "helpers.py").is_file()
    assert (FIXTURES / "github" / "spec.json").is_file()
