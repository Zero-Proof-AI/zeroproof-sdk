"""No tracked file carries a git conflict marker.

A bad merge left ``<<<<<<< HEAD`` in CHANGELOG.md and it shipped on main
(709eb70 removed it). Nothing caught it: the lint job runs ruff and mypy,
which only read Python, so markers in Markdown, JSON or YAML pass CI
untouched. This test reads every tracked text file instead.
"""

from __future__ import annotations

import subprocess

from tests.helpers import REPO_ROOT

# Built rather than written literally, so this file does not match itself.
# Only the three unambiguous markers: a bare "=======" is also a Markdown
# setext underline, and every real conflict carries an opener anyway.
MARKERS = tuple(char * 7 for char in "<>|")


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert out.returncode == 0, out.stderr[-500:]
    return [name for name in out.stdout.split("\0") if name]


def test_no_tracked_file_has_a_conflict_marker():
    names = _tracked_files()
    assert len(names) > 50, f"git ls-files returned only {len(names)} paths"
    offenders: list[str] = []
    scanned = 0
    for name in names:
        path = REPO_ROOT / name
        if not path.is_file():
            continue  # submodule, or a symlink to nowhere
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary; the SVG reads fine, images do not
        scanned += 1
        for number, line in enumerate(text.splitlines(), 1):
            if any(line.startswith(marker) for marker in MARKERS):
                offenders.append(f"{name}:{number}: {line[:70]}")
    assert scanned > 50, f"only {scanned} text files were readable"
    assert not offenders, "conflict markers left in tracked files:\n" + "\n".join(offenders)
