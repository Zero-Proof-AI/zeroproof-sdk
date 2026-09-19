"""Publish one shields.io endpoint badge to the `badges` branch.

    python .github/scripts/badge.py coverage "91.4%"

writes `coverage.json` on the `badges` branch, which the README renders as

    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/whilehq/whileai-sdk/badges/coverage.json

shields.io and raw.githubusercontent.com each cache for five minutes, so a
number published here is on the README within ten. The branch is an orphan
holding nothing but these JSON files. If it is ever deleted, recreate it by
hand once (`git checkout --orphan badges`, commit one file, push); this
script updates files, it does not create branches.

Needs GH_TOKEN with `contents: write` and GITHUB_REPOSITORY, which Actions
sets on every job. An unchanged value is not committed, so a green main does
not grow a commit per run.

Usage: badge.py NAME MESSAGE [LABEL] [COLOR]
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

BRANCH = "badges"
GREEN = "5cb08a"  # the brand green the other README badges use


def api(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    repo = os.environ["GITHUB_REPOSITORY"]
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/{path}",
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def publish(name: str, message: str, label: str | None = None, color: str = GREEN) -> int:
    label = label or name
    badge = {"schemaVersion": 1, "label": label, "message": message, "color": color}
    content = json.dumps(badge, indent=2) + "\n"
    path = f"contents/{name}.json"
    for attempt in range(4):
        status, current = api("GET", f"{path}?ref={BRANCH}")
        sha = current.get("sha") if status == 200 else None
        if sha and base64.b64decode(current["content"]).decode() == content:
            print(f"{name}.json already says {message}; nothing to publish")
            return 0
        body = {
            "message": f"{label} {message}",
            "content": base64.b64encode(content.encode()).decode(),
            "branch": BRANCH,
        }
        if sha:
            body["sha"] = sha
        status, out = api("PUT", path, body)
        if status in (200, 201):
            print(f"{name}.json -> {message}")
            return 0
        # 409: another job committed to `badges` between the GET and the PUT.
        if status in (409, 422) and attempt < 3:
            time.sleep(2**attempt)
            continue
        print(f"publish failed ({status}): {out.get('message')}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(publish(*sys.argv[1:5]))
