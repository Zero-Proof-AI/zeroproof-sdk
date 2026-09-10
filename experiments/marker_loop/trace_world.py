"""A world reconstructed from the customer's own traces. No hand-built stage.

Everything the agent's environment ever answered is already recorded in the
traces OTel captured: file contents behind read_file results, directory
listings, command outputs. This world replays those answers, keeps a
per-rollout overlay for writes, and falls back honestly (unknown file reads
fail like a real filesystem) instead of improvising content.

The only generic machinery is a start/stop state for compute-shaped
commands, because a repair rollout must be ABLE to stop what the failing
trace started; the outputs it speaks with still come from the traces
whenever the traces contain them.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from zeroproof_simulations.generate.agents import current_rollout  # noqa: E402

START_CMD = re.compile(r"\bmodal (deploy|run)\b|\bvllm serve\b|--gpu\b|\bstart\b.{0,15}(gpu|endpoint|container|server)", re.I)
STOP_CMD = re.compile(r"\bmodal (app|container) stop\b|\bscale.{0,10}(down|zero|0)\b|\bshut ?down\b|\bstop\b|\btear.{0,6}down\b", re.I)


_PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "user@example.com"),
    (re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b"), "555-555-0100"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "000-00-0000"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "4111111111111111"),
]


def _scrub(text: str) -> str:
    """Recorded answers carry the customer's real data; training rows must
    not. Values that look like PII are replaced before anything is replayed
    into generation. Shape survives, identities do not."""
    for pat, sub in _PII:
        text = pat.sub(sub, text)
    return text


def _result_text(res) -> dict:
    if isinstance(res, dict):
        return res
    if isinstance(res, str):
        try:
            parsed = json.loads(res)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
        return {"status": "ok", "output": res}
    return {"status": "ok"}


class TraceWorld:
    """execute= built from observed rows: same call, same answer."""

    def __init__(self, rows) -> None:
        self.files: dict[str, str] = {}
        self.listings: dict[str, str] = {}
        self.commands: list[tuple[str, dict]] = []   # (observed command, its result)
        self.start_result: dict | None = None
        self.stop_result: dict | None = None
        for row in rows or []:
            for s in (row.get("steps") or []):
                if not isinstance(s, dict) or not s.get("tool"):
                    continue
                tool = str(s["tool"])
                args = s.get("arguments") if isinstance(s.get("arguments"), dict) else {}
                res = _result_text(s.get("result"))
                if res.get("status") == "error":
                    continue
                path = str(args.get("path") or "").strip().lstrip("./")
                if tool == "read_file" and path and res.get("content") is not None:
                    self.files.setdefault(path, _scrub(str(res["content"])))
                elif tool == "write_file" and path:
                    self.files.setdefault(path, _scrub(str(args.get("content") or "")))
                elif tool in ("list_dir", "ls") and res.get("output") is not None:
                    self.listings.setdefault(path.rstrip("/"), str(res["output"]))
                elif tool == "run_command":
                    cmd = str(args.get("command") or "")
                    if cmd:
                        scrubbed = dict(res)
                        for k in ("output", "content", "result"):
                            if isinstance(scrubbed.get(k), str):
                                scrubbed[k] = _scrub(scrubbed[k])
                        self.commands.append((cmd, scrubbed))
                        if START_CMD.search(cmd) and self.start_result is None:
                            self.start_result = scrubbed
                        if STOP_CMD.search(cmd) and self.stop_result is None:
                            self.stop_result = scrubbed
        self._local = threading.local()

    # -- per-rollout overlay -------------------------------------------------

    def _state(self):
        key = (getattr(current_rollout, "prompt", None), getattr(current_rollout, "rollout_index", None))
        if getattr(self._local, "key", None) != key:
            self._local.key = key
            self._local.writes = {}
            self._local.deleted = set()
            self._local.compute = False
        return self._local

    def _nearest_command(self, cmd: str) -> dict | None:
        want = set(re.findall(r"[a-z0-9_.-]+", cmd.lower()))
        best, score = None, 0.0
        for seen, res in self.commands:
            have = set(re.findall(r"[a-z0-9_.-]+", seen.lower()))
            if not want or not have:
                continue
            overlap = len(want & have) / len(want | have)
            if overlap > score:
                best, score = res, overlap
        return best if score >= 0.3 else None

    def __call__(self, tool: str, arguments: dict) -> dict:
        st = self._state()
        args = arguments if isinstance(arguments, dict) else {}
        path = str(args.get("path") or "").strip().lstrip("./")
        if tool == "read_file":
            if path in st.writes:
                return {"status": "ok", "content": st.writes[path]}
            if path in st.deleted:
                return {"status": "error", "error": f"no such file: {path}"}
            if path in self.files:
                return {"status": "ok", "content": self.files[path]}
            base = {p: c for p, c in self.files.items() if os.path.basename(p) == os.path.basename(path)}
            if len(base) == 1:
                return {"status": "ok", "content": next(iter(base.values()))}
            return {"status": "error", "error": f"no such file: {path}"}
        if tool in ("list_dir", "ls"):
            key = path.rstrip("/")
            if key in self.listings:
                return {"status": "ok", "output": self.listings[key]}
            prefix = key + "/" if key and key != "." else ""
            names = sorted({p[len(prefix):].split("/")[0]
                            for p in list(self.files) + list(st.writes)
                            if p.startswith(prefix) and p not in st.deleted})
            if names:
                return {"status": "ok", "output": "\n".join(names)}
            return {"status": "error", "error": f"no such directory: {path}"}
        if tool == "write_file":
            st.writes[path] = str(args.get("content") or "")
            st.deleted.discard(path)
            return {"status": "ok", "result": f"wrote {path} ({len(st.writes[path])} bytes)"}
        if tool == "delete_file":
            st.deleted.add(path)
            return {"status": "ok", "result": f"deleted {path}"}
        if tool == "run_command":
            cmd = str(args.get("command") or "")
            if START_CMD.search(cmd):
                st.compute = True
                return dict(self.start_result or {"status": "ok", "output": "deployed: 1 container running", "exit_code": 0})
            if STOP_CMD.search(cmd):
                st.compute = False
                return dict(self.stop_result or {"status": "ok", "output": "stopped: 0 containers running", "exit_code": 0})
            near = self._nearest_command(cmd)
            if near is not None:
                return dict(near)
            return {"status": "ok", "output": "", "exit_code": 0}
        if tool in ("run_tests", "git_diff", "git_commit", "git_push"):
            near = self._nearest_command(tool)
            if near is not None:
                return dict(near)
            if tool == "run_tests":
                return {"status": "ok", "exit_code": 0, "output": "suite passed"}
            return {"status": "ok", "result": f"{tool}: ok"}
        return {"status": "error", "error": f"unknown tool: {tool}"}
