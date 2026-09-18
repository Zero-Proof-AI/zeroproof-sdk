"""Report what you trained to the While platform, so a person can decide.

The platform draws one screen per agent: the held-out score by version
with the frontier model as the line to beat, the training curve, what
moved on the behaviors you did not train, the judge checks, live traffic
on the served version, and cost. This module is how a coding agent fills
that screen. The person reads it and presses Promote.

Four objects, in the order they happen::

    from whileai.platform import Agent

    agent = Agent("refund-bot", model="Qwen/Qwen3-4B", harness="h2",
                  frontier={"name": "Sonnet 5", "score": 81, "cost_per_1k": 18.0})
    agent.behavior("refunds", test_version="v2", n=240,
                   judge={"agreement": 0.86, "human_n": 60, "length_bias": 0.08},
                   noise_floor=2.4, contamination=0, reward_is_judge=False)

    run = agent.run("v4", base="Qwen/Qwen3-4B", method="GRPO",
                    targets=["refunds"], trained_on=["refunds-grpo", "character-sft"])
    run.log(step=10, reward=0.41, kl=0.01)        # or trainer.add_callback(wai.TrainerCallback(run))
    run.score("refunds", 83, ci=2.7, n=240)       # every behavior, not only the targets
    run.score("length", 76, ci=2.8, n=120)
    run.finish(hours=2.1, gpu="1xH100", cost_usd=31)

    print(agent.verdict())                        # "refunds: v4 beats v3 by 5 (interval excludes zero)"
    agent.promote("v4")                           # usually the person does this on the platform

An *agent* is a model plus a harness (prompt, tools, skills). A *behavior*
is one thing you measure, with its own frozen held-out test and its own
judge. A *run* is one training job that produces a version and is scored
on every behavior. *Live* rows are daily traffic on the served version.

Rules the platform holds you to: a version is scored on every behavior
(regressions are the point), the test for a behavior does not change
under you (bump ``test_version`` when it does), and the interval matters
as much as the score (``ci`` is the half-width of the 95% interval).

Logging never raises into a training loop: points are buffered, sent in
batches, and a failed send is retried on the next flush. Stdlib only.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from whileai._env import getenv
from whileai.auth import resolve_api_key

log = logging.getLogger("whileai.platform")

DEFAULT_PLATFORM_URL = "https://mbxp83jd48.execute-api.us-east-1.amazonaws.com"
PLATFORM_URL_ENV = "WHILEAI_PLATFORM_URL"

FLUSH_EVERY = 25
FLUSH_SECONDS = 15.0
MAX_BATCH = 2000


class PlatformError(RuntimeError):
    """The platform said no. ``status`` is the HTTP code."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def platform_url() -> str:
    return (getenv("PLATFORM_URL", DEFAULT_PLATFORM_URL) or DEFAULT_PLATFORM_URL).rstrip("/")


def _key(explicit: str | None) -> str:
    key = resolve_api_key(explicit)
    if not key:
        raise PlatformError(
            401,
            "No API key. Run `whileai login`, or set WHILEAI_API_KEY, or pass api_key=... "
            "(keys are on while.ai under Account).",
        )
    return key


def _request(
    method: str,
    path: str,
    *,
    api_key: str,
    body: Any = None,
    timeout: float = 30.0,
) -> Any:
    """One call. Returns the parsed JSON; raises PlatformError on 4xx/5xx."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(platform_url() + path, data=data, method=method)
    req.add_header("X-Api-Key", api_key)
    req.add_header("User-Agent", "whileai-sdk")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            message = json.loads(raw or b"{}").get("error") or e.reason
        except (ValueError, AttributeError):
            message = e.reason
        raise PlatformError(e.code, f"{method} {path}: {message}") from None
    except urllib.error.URLError as e:
        raise PlatformError(
            0, f"{method} {path}: could not reach {platform_url()} ({e.reason})"
        ) from None
    return json.loads(raw or b"{}")


Transport = Callable[..., Any]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _camel(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """snake_case keyword arguments to the API's camelCase, dropping None."""
    names = {
        "test_version": "testVersion",
        "noise_floor": "noiseFloor",
        "reward_is_judge": "rewardIsJudge",
        "trained_on": "trainedOn",
        "cost_usd": "costUsd",
        "cost_per_1k": "costPer1k",
        "p50_s": "p50s",
        "human_n": "humanN",
        "length_bias": "lengthBias",
    }
    out: dict[str, Any] = {}
    for k, v in mapping.items():
        if v is None:
            continue
        if isinstance(v, Mapping):
            v = _camel(v)
        out[names.get(k, k)] = v
    return out


class Run:
    """One training job on one agent, producing one version.

    ``log`` buffers and ``flush`` sends, so a training loop never waits on
    the network and never sees an exception from it. ``score`` records an
    eval on one behavior. ``finish`` flushes and closes the run.

    Works with ``wai.TrainerCallback(run)`` on any Transformers or TRL
    trainer: the callback calls ``progress``, ``log`` and ``finish``.
    """

    def __init__(
        self,
        agent: Agent,
        run_id: str,
        version: str,
        *,
        flush_every: int = FLUSH_EVERY,
        flush_seconds: float = FLUSH_SECONDS,
    ):
        self.agent = agent
        self.id = run_id
        self.version = version
        self.status = "running"
        self.step = 0
        self.total_steps: int | None = None
        self.errors = 0
        self.scores: dict[str, dict[str, Any]] = {}
        self._flush_every = max(1, int(flush_every))
        self._flush_seconds = float(flush_seconds)
        self._buffer: list[dict[str, Any]] = []
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()
        self._warned = False
        self._ci_warned: set[str] = set()

    # ------------------------------------------------------------ logging

    def log(self, step: int, **metrics: Any) -> None:
        """Record one point of the training curve.

        ``reward`` (RL), ``loss`` (SFT/DPO) and ``kl`` are what the
        platform draws; any other finite number is kept too.
        """
        point: dict[str, Any] = {"step": int(step)}
        for key, value in metrics.items():
            number = _number(value)
            if number is not None:
                point[key] = number
        if len(point) == 1:
            return
        with self._lock:
            self._buffer.append(point)
            self.step = max(self.step, point["step"])
            due = len(self._buffer) >= self._flush_every or (
                time.monotonic() - self._last_flush >= self._flush_seconds
            )
        if due:
            self.flush()

    def progress(self, step: int, total: int | None = None) -> None:
        """Where the run is; the callback reads this from the trainer."""
        self.step = max(self.step, int(step))
        if total:
            self.total_steps = int(total)

    def flush(self) -> None:
        """Send buffered points now. Failures are counted, never raised."""
        with self._lock:
            batch, self._buffer = self._buffer[:MAX_BATCH], self._buffer[MAX_BATCH:]
            self._last_flush = time.monotonic()
        if not batch:
            return
        try:
            self.agent._call("POST", f"/runs/{self.id}/train", batch)
        except PlatformError as e:
            self.errors += 1
            with self._lock:
                self._buffer = batch + self._buffer
            if not self._warned:
                self._warned = True
                log.warning(
                    "Could not send %d training points (%s); will retry on the next flush.",
                    len(batch),
                    e,
                )
            return
        if self._buffer:
            self.flush()

    # ------------------------------------------------------------ evals

    def score(
        self,
        behavior: str,
        score: float,
        *,
        ci: float | None = None,
        n: int | None = None,
        test_version: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Record this version's score on one behavior's held-out test.

        ``ci`` is the half-width of the 95% interval, ``n`` the number of
        held-out tasks. Score every behavior the agent has, not only the
        ones this run trained: the ones you did not train are the check.
        """
        if ci is None and not self._warned_ci(behavior):
            log.warning(
                "score(%r) has no interval; pass ci=<half-width of the 95%% interval> so the "
                "platform can say whether the change is real.",
                behavior,
            )
        body = _camel(
            {
                "behavior": behavior,
                "score": score,
                "ci": ci,
                "n": n,
                "test_version": test_version,
                "version": version,
            }
        )
        out = self.agent._call("POST", f"/runs/{self.id}/evals", [body])
        recorded = (out.get("evals") or [body])[0]
        self.scores[behavior] = recorded
        return recorded

    def _warned_ci(self, behavior: str) -> bool:
        seen = behavior in self._ci_warned
        self._ci_warned.add(behavior)
        return seen

    # ------------------------------------------------------------ lifecycle

    def finish(
        self,
        status: str = "evaluated",
        *,
        hours: float | None = None,
        gpu: str | None = None,
        cost_usd: float | None = None,
        steps: int | None = None,
        summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Flush, then close the run with a status and what it cost."""
        self.flush()
        patch = _camel(
            {
                "status": status,
                "hours": hours,
                "gpu": gpu,
                "cost_usd": cost_usd,
                "steps": steps if steps is not None else (self.total_steps or self.step or None),
            }
        )
        if summary and "train_loss" in summary and _number(summary["train_loss"]) is not None:
            patch.setdefault("summary", {})["train_loss"] = summary["train_loss"]
        self.status = status
        return self.agent._call("PATCH", f"/runs/{self.id}", patch)

    def fail(self, error: str) -> dict[str, Any]:
        self.flush()
        self.status = "failed"
        return self.agent._call(
            "PATCH", f"/runs/{self.id}", {"status": "failed", "error": str(error)[:400]}
        )

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None and self.status == "running":
            self.fail(f"{exc_type.__name__}: {exc}")
        elif self.status == "running":
            self.flush()

    @property
    def url(self) -> str:
        return f"https://while.ai/platform/runs?agent={self.agent.id}"

    def __repr__(self) -> str:
        return f"Run({self.id!r}, version={self.version!r}, status={self.status!r}, points={self.step})"


class Agent:
    """A model plus its harness, as the platform tracks it.

    Creating the object registers (or updates) the agent on first use.
    ``behavior`` declares what you measure, ``run`` opens a training run,
    ``live`` reports a day of traffic, ``dashboard`` reads the screen back.
    """

    def __init__(
        self,
        id: str,
        *,
        name: str | None = None,
        model: str | None = None,
        harness: str | Mapping[str, Any] | None = None,
        frontier: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        transport: Transport | None = None,
        register: bool = True,
    ):
        self.id = id
        self.name = name or id
        self.model = model
        self.harness = (
            {"label": harness} if isinstance(harness, str) else (dict(harness) if harness else None)
        )
        self.frontier = _camel(frontier) if frontier else None
        self._api_key = api_key
        self._transport = transport
        self.record: dict[str, Any] | None = None
        if register:
            self.register()

    # ------------------------------------------------------------ plumbing

    def _call(self, method: str, path: str, body: Any = None) -> Any:
        if self._transport is not None:
            return self._transport(method, path, body)
        return _request(method, path, api_key=_key(self._api_key), body=body)

    # ------------------------------------------------------------ objects

    def register(self) -> dict[str, Any]:
        """Create or update the agent record. Safe to call every run."""
        body = _camel(
            {
                "id": self.id,
                "name": self.name,
                "model": self.model,
                "harness": self.harness,
                "frontier": self.frontier,
            }
        )
        self.record = self._call("POST", "/agents", body)
        return self.record

    def behavior(
        self,
        name: str,
        *,
        test_version: str | None = None,
        n: int | None = None,
        judge: Mapping[str, Any] | None = None,
        noise_floor: float | None = None,
        contamination: int | None = None,
        reward_is_judge: bool | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Declare one thing you measure.

        ``test_version`` names the frozen held-out set; bump it when the
        set changes. ``noise_floor`` is the spread you saw scoring the same
        model twice. ``judge`` carries ``agreement`` (with people),
        ``human_n``, ``length_bias`` (correlation of score with length) and
        ``name``. ``reward_is_judge`` should be False: a judge that is also
        the reward cannot check the reward.
        """
        body = _camel(
            {
                "test_version": test_version,
                "n": n,
                "judge": judge,
                "noise_floor": noise_floor,
                "contamination": contamination,
                "reward_is_judge": reward_is_judge,
                "description": description,
            }
        )
        if reward_is_judge:
            log.warning(
                "behavior(%r): the judge is also the reward, so it cannot catch reward hacking; "
                "score with a different model or a verifier (reward_is_judge=False).",
                name,
            )
        return self._call("PUT", f"/agents/{self.id}/behaviors/{name}", body)

    def behaviors(self) -> list[dict[str, Any]]:
        return list(self._call("GET", f"/agents/{self.id}/behaviors").get("behaviors") or [])

    def run(
        self,
        version: str,
        *,
        base: str | None = None,
        method: str | None = None,
        targets: Sequence[str] | None = None,
        trained_on: Sequence[str] | None = None,
        harness: str | Mapping[str, Any] | None = None,
        gpu: str | None = None,
        id: str | None = None,
        flush_every: int = FLUSH_EVERY,
        flush_seconds: float = FLUSH_SECONDS,
    ) -> Run:
        """Open a training run that will produce ``version``.

        ``targets`` are the behaviors this run trains; ``trained_on`` the
        dataset names. Train once on a mixture of datasets rather than one
        behavior after another: sequential fine-tunes forget.
        """
        body = _camel(
            {
                "id": id,
                "agent": self.id,
                "version": version,
                "base": base or self.model,
                "method": method,
                "targets": list(targets or []),
                "trained_on": list(trained_on or []),
                "harness": harness if harness is not None else (self.harness or {}).get("label"),
                "gpu": gpu,
            }
        )
        out = self._call("POST", "/runs", body)
        return Run(
            self,
            out["id"],
            out.get("version", version),
            flush_every=flush_every,
            flush_seconds=flush_seconds,
        )

    def runs(self) -> list[dict[str, Any]]:
        return list(self._call("GET", f"/runs?agent={self.id}").get("runs") or [])

    def promote(self, version: str) -> dict[str, Any]:
        """Make ``version`` the served one. Usually the person's button."""
        return self._call("POST", f"/agents/{self.id}/promote", {"version": version})

    def live(
        self,
        day: str,
        *,
        version: str,
        replies: int,
        flagged: int = 0,
        p50_s: float | None = None,
        cost_usd: float | None = None,
    ) -> dict[str, Any]:
        """One day of traffic on the served version (``day`` is YYYY-MM-DD).

        ``flagged`` is how many replies failed a check. The served endpoint
        writes these itself when the platform serves the model; call this
        when you serve it elsewhere.
        """
        body = _camel(
            {
                "agent": self.id,
                "day": day,
                "version": version,
                "replies": replies,
                "flagged": flagged,
                "p50_s": p50_s,
                "cost_usd": cost_usd,
            }
        )
        return self._call("POST", "/live", [body])

    # ------------------------------------------------------------ reading

    def dashboard(self, behavior: str | None = None) -> dict[str, Any]:
        """The Runs screen as JSON: versions, train, deltas, judge, live, verdict."""
        q = f"?behavior={behavior}" if behavior else ""
        return self._call("GET", f"/agents/{self.id}/dashboard{q}")

    def verdict(self, behavior: str | None = None) -> str:
        """One line: does the candidate beat the served version, and is it real?"""
        d = self.dashboard(behavior)
        v = d.get("verdict") or {}
        b = (d.get("behavior") or {}).get("name") or behavior or "?"
        if v.get("delta") is None:
            served = v.get("serving") or "a served version"
            return f"{b}: no candidate scored against {served} yet"
        delta = v["delta"]
        word = "beats" if delta >= 0 else "trails"
        real = v.get("excludesZero")
        if real is None:
            interval = "no interval"
        elif real:
            interval = "interval excludes zero"
        else:
            interval = "inside the noise"
        regress = v.get("regressions") or 0
        tail = f"; {regress} regression{'' if regress == 1 else 's'}" if regress else ""
        return f"{b}: {v['candidate']} {word} {v['serving']} by {abs(delta)} ({interval}){tail}"

    def __repr__(self) -> str:
        return f"Agent({self.id!r}, model={self.model!r})"


def agents(api_key: str | None = None) -> list[dict[str, Any]]:
    """Every agent on the account."""
    out = _request("GET", "/agents", api_key=_key(api_key))
    return list(out.get("agents") or [])


__all__ = [
    "DEFAULT_PLATFORM_URL",
    "Agent",
    "PlatformError",
    "Run",
    "agents",
    "platform_url",
]
