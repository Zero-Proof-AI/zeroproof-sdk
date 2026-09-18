#!/usr/bin/env python3
"""Fail when a number that steers behaviour is written inline.

Every threshold, share, cap and budget in ``whileai/simulations`` has one
home: ``defaults.py`` (values more than one module reads, or that a
``simulate()`` key moves) or a named module constant with a comment of
the form ``# NAME = value: why (source)``. This check reads every module
under ``whileai/simulations`` except ``defaults.py`` and reports each
numeric literal that is

* an operand of a comparison (``x < 0.35``), or
* the value of an assignment (``rate = 0.35``, ``x: float = 0.35``,
  ``n += 4``),

other than ``0``, ``1``, ``2`` and ``-1`` (loop seeds, counts, the
structural minimums) and other than indices and slices, which are
positions, not thresholds. A module-level ``NAME = value`` is a named
constant and passes when a comment says why: the ``# NAME = value: why
(source)`` form, a comment block directly above it (one block may explain
a group of constants), or a trailing comment. ``defaults.py`` itself is
held to the strict form by ``tests/grade/test_no_hardcoding_score.py``.
A line that must keep a literal ends with ``# literal: <reason>`` and
passes too; the reason is required.

    uv run python scripts/check_no_hardcoding.py          # exit 1 on a finding
    uv run python scripts/check_no_hardcoding.py --list   # print, exit 0

Tests, recipes and scripts are not scanned: a test writes the number it
checks, and a recipe is a worked example.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "whileai" / "simulations"
SKIP_FILES = {"defaults.py"}

# The values that are structure, not tuning: an empty count, a unit step,
# a pair, and the last index.
FREE_VALUES = {0, 1, 2, -1}

ALLOW_MARK = re.compile(r"#\s*literal:\s*\S")
CONSTANT_LINE = re.compile(r"^_?[A-Z][A-Z0-9_]*(?:, _?[A-Z][A-Z0-9_]*)*(?::[^=]+)? = ")
NAMED_CONSTANT_LINE = re.compile(r"^#.*(?<![A-Za-z0-9_])(?P<name>[A-Z_][A-Z0-9_]*) = ")


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    text: str

    def __str__(self) -> str:
        rel = self.path.relative_to(REPO_ROOT).as_posix()
        return f"{rel}:{self.line}: {self.text}"


def _literal_value(node: ast.AST) -> int | float | None:
    """The numeric value of a plain literal or ``-literal``; None otherwise."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal_value(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            return None
        return node.value
    return None


def _named_constant_lines(source_lines: list[str]) -> set[str]:
    """Names that a comment line introduces as ``# NAME = value``."""
    names: set[str] = set()
    for line in source_lines:
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        for m in re.finditer(r"(?<![A-Za-z0-9_])([A-Z_][A-Z0-9_]*) = ", stripped):
            names.add(m.group(1))
    return names


def _target_names(node: ast.stmt) -> list[str]:
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Tuple):
            names.extend(e.id for e in target.elts if isinstance(e, ast.Name))
    return names


def _assigned_literals(value: ast.expr) -> list[ast.expr]:
    """The literal nodes an assignment's value is made of: the value
    itself, or each element of a tuple/list of literals."""
    if _literal_value(value) is not None:
        return [value]
    if isinstance(value, (ast.Tuple, ast.List)):
        return [e for e in value.elts if _literal_value(e) is not None]
    return []


def check_file(path: Path) -> list[Finding]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    named = _named_constant_lines(lines)
    findings: list[Finding] = []
    module_level = {id(stmt) for stmt in tree.body}
    # a constant block written as ``A, B = 1, 2`` on one line counts as
    # module level for each name

    # statement spans, innermost first, so a ``# literal:`` mark counts
    # anywhere in the statement the literal sits in (the formatter may
    # wrap a marked line and leave the comment on its last line)
    spans = sorted(
        (
            (node.lineno, node.end_lineno or node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.stmt)
        ),
        key=lambda span: span[1] - span[0],
    )

    def allowed_line(lineno: int) -> bool:
        if ALLOW_MARK.search(lines[lineno - 1]):
            return True
        for start, end in spans:
            if start <= lineno <= end:
                return any(ALLOW_MARK.search(lines[i - 1]) for i in range(start, end + 1))
        return False

    def _documented(name: str, lineno: int) -> bool:
        """A module constant is documented when a comment says why: the
        ``# NAME = value: why (source)`` form anywhere in the module, a
        comment block directly above it (one block may explain a group of
        constants), or a trailing comment on its line."""
        bare = name.lstrip("_")
        if bare in named:
            return True
        # the constant block: contiguous comment lines and sibling
        # ``NAME = value`` lines above, so one comment can explain a group
        block: list[str] = []
        i = lineno - 2
        while i >= 0:
            stripped = lines[i].strip()
            if stripped.startswith("#"):
                block.append(stripped)
            elif not CONSTANT_LINE.match(lines[i]):
                break
            i -= 1
        trailing = lines[lineno - 1].partition("#")[2].strip()
        # a comment that names it, or any why at all: the name is the
        # documentation, the comment is the reason
        return bool(trailing) or any(line.strip("#: ") for line in block)

    def report(node: ast.AST) -> None:
        if allowed_line(node.lineno):
            return
        findings.append(Finding(path, node.lineno, lines[node.lineno - 1].strip()))

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for operand in [node.left, *node.comparators]:
                value = _literal_value(operand)
                if value is not None and value not in FREE_VALUES:
                    report(operand)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if node.value is None:
                continue
            literals = _assigned_literals(node.value)
            if not literals:
                continue
            if any(_literal_value(lit) in FREE_VALUES for lit in literals) and len(literals) == 1:
                continue
            names = _target_names(node)
            is_module_constant = id(node) in module_level and all(
                n.lstrip("_").isupper() for n in names
            )
            if is_module_constant and all(_documented(n, node.lineno) for n in names):
                continue
            for lit in literals:
                value = _literal_value(lit)
                if value is not None and value not in FREE_VALUES:
                    report(node)
                    break
    return findings


def main(argv: list[str]) -> int:
    list_only = "--list" in argv
    findings: list[Finding] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name in SKIP_FILES:
            continue
        findings.extend(check_file(path))
    for finding in findings:
        print(finding)
    if findings:
        print(
            f"\n{len(findings)} inline number(s). Move each to defaults.py or a named "
            "module constant with a `# NAME = value: why (source)` comment, or end the "
            "line with `# literal: <reason>`."
        )
        return 0 if list_only else 1
    print("no inline thresholds in whileai/simulations")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
