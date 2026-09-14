"""No tracked file carries merge-conflict markers.

Several sessions rebase onto a fast-moving main; one committed an
unresolved CHANGELOG once and the build stayed green for an hour. A
marker at the start of a line is never legitimate content here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MARKERS = ("<<<<<<< ", ">>>>>>> ")
BINARY = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".woff", ".woff2", ".pdf", ".lock"}


def test_no_tracked_file_has_conflict_markers():
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, cwd=str(REPO), check=True
    ).stdout.decode("utf-8", "replace")
    offenders: list[str] = []
    for rel in filter(None, tracked.split("\0")):
        path = REPO / rel
        if path.suffix.lower() in BINARY or not path.is_file() or path.name == Path(__file__).name:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if line.startswith(MARKERS):
                offenders.append(f"{rel}:{number}: {line[:40]}")
    assert not offenders, "conflict markers committed:\n" + "\n".join(offenders)
