"""Gate the release version before anything is published.

House rule: versions move in hundredths and only ever by one step.

    1.01 -> 1.02 -> 1.03 ... 1.99 -> 2.00

PEP 440 strips leading zeros, so PyPI stores "1.01" as "1.1" and treats the two
as equal. Ordering still behaves (1.10 > 1.9 > 1.2), so the scheme works, but
the rule is enforced on the normalized release tuple rather than the string:
(1, 1) -> (1, 2) -> ... -> (1, 99) -> (2, 0).

Exit codes:
  0  version is a valid single step, publish
  0  version unchanged, nothing to cut (prints SKIP)
  1  version is invalid or skips ahead, block the release
"""
from __future__ import annotations

import json
import os
import sys
import tomllib
import urllib.error
import urllib.request

PYPI = "https://pypi.org/pypi/{name}/json"


def local_version(path: str = "pyproject.toml") -> tuple[str, str]:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    project = data["project"]
    return project["name"], project["version"]


def published(name: str) -> list[tuple[int, ...]]:
    """Every release already on PyPI, as normalized tuples."""
    try:
        with urllib.request.urlopen(PYPI.format(name=name), timeout=30) as r:
            data = json.load(r)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []           # first ever release
        raise
    from packaging.version import InvalidVersion, Version
    out = []
    for raw in data.get("releases", {}):
        try:
            out.append(Version(raw).release)
        except InvalidVersion:
            continue
    return sorted(out)


def next_allowed(prev: tuple[int, ...]) -> tuple[int, ...]:
    major, minor = prev[0], (prev[1] if len(prev) > 1 else 0)
    return (major + 1, 0) if minor >= 99 else (major, minor + 1)


def fail(msg: str) -> None:
    print(f"::error::{msg}")
    sys.exit(1)


def main() -> int:
    name, version = local_version()
    from packaging.version import InvalidVersion, Version
    try:
        current = Version(version)
    except InvalidVersion:
        fail(f"{version!r} is not a valid PEP 440 version")
        return 1

    if len(current.release) != 2:
        fail(f"version must be MAJOR.MINOR in hundredths (e.g. 1.02), got {version!r}. "
             f"Three-part versions are not part of this scheme.")
    if current.pre or current.post or current.dev or current.local:
        fail(f"{version!r} has a pre/post/dev/local segment; releases must be plain.")

    prior = published(name)
    print(f"package        : {name}")
    print(f"local version  : {version}  (normalized {current})")
    print(f"published      : {[".".join(map(str, p)) for p in prior[-5:]] or 'none'}")

    if not prior:
        # First release. Anything sane is fine; require it to start at x.1 or x.0.
        if current.release[1] not in (0, 1):
            fail(f"first release should be x.00 or x.01, got {version!r}")
        print(f"::notice::first release of {name} {current}")
        return emit(publish=True, version=str(current))

    if current.release in prior:
        # Already on PyPI. Not an error: main moves for reasons other than a
        # release, and re-running CI on an unchanged version must not fail.
        print(f"::notice::{current} is already published, nothing to cut")
        return emit(publish=False, version=str(current))

    latest = prior[-1]
    if current.release < latest:
        fail(f"{current} is older than the published {'.'.join(map(str, latest))}")

    allowed = next_allowed(latest)
    if current.release != allowed:
        fail(f"version must step by exactly one hundredth. "
             f"published {'.'.join(map(str, latest))}, "
             f"expected {'.'.join(map(str, allowed))}, got {current}")

    print(f"::notice::cutting {name} {current}")
    return emit(publish=True, version=str(current))


def emit(*, publish: bool, version: str) -> int:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"publish={str(publish).lower()}\n")
            fh.write(f"version={version}\n")
    print(f"publish={publish} version={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
