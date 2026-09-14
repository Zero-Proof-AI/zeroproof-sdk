"""Training runs: the loss curve and the progress bar on the platform.

Two ways to train, one record. ``train`` starts SFT, GRPO, DPO or a reward
model (``rm``) on the platform's trainer and returns the run handle;
``serve`` puts a finished run's adapter on an OpenAI-compatible endpoint
and ``reward_model`` turns a finished ``rm`` run into a judge. Or your own trainer runs
wherever it runs and reports through the same handle: a run is created,
points are logged as it goes, and it is finished with a status. The
platform draws the curve and the progress bar at
zeroproofai.com/platform/training.

Three ways in for your own trainer:

* Engineer, one line. ``trainer.add_callback(zps.TrainerCallback(run))``
  on a Transformers or TRL trainer logs every ``on_log`` (loss, learning
  rate, eval loss, epoch, grad norm), sets the step count from the
  trainer, and finishes the run when training ends or crashes.
* Data scientist with a loop. ``run = zps.training_run("sft-v3",
  dataset="ds_...")``, then ``run.log(step, loss=...)`` wherever the loop
  has a number, ``run.finish()`` at the end. Points are buffered and sent
  in batches; logging never raises into the training loop.
* Researcher with a stack. Plain HTTP: ``POST /runs``, ``POST
  /runs/{id}/log`` with ``{"points": [{"step": 10, "loss": 1.2}]}``,
  ``POST /runs/{id}/finish``. The README lists the bodies.
"""

from __future__ import annotations

import logging
import threading
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .ingest.platform import _call

log = logging.getLogger("zeroproof.simulations")

SITE_URL = "https://www.zeroproofai.com"


def _json_safe(value: Any) -> Any:
    """Tuples to lists, NaN/inf to None, so a report survives JSON."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


FLUSH_EVERY = 25
FLUSH_SECONDS = 15.0
MAX_BATCH = 500


class TrainingRun:
    """One fine-tune, as the platform sees it. Create with ``training_run``.

    ``log`` buffers; ``flush`` sends. A send that fails is retried on the
    next flush and counted in ``errors``; the training loop is never
    interrupted by the dashboard. ``finish`` flushes first.
    """

    def __init__(
        self,
        run_id: str,
        *,
        name: str,
        api_key: str | None = None,
        total_steps: int | None = None,
        flush_every: int = FLUSH_EVERY,
        flush_seconds: float = FLUSH_SECONDS,
        transport: Callable[..., Any] | None = None,
    ):
        self.run_id = run_id
        self.name = name
        self.total_steps = total_steps
        self.status = "running"
        self.step = 0
        self.errors = 0
        self._api_key = api_key
        self._flush_every = max(1, int(flush_every))
        self._flush_seconds = float(flush_seconds)
        self._call = transport or _call
        self._buffer: list[dict[str, Any]] = []
        self._pending_total: int | None = None
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()
        self._warned = False
        self._delta: dict[str, Any] | None = None
        self._summary: dict[str, Any] = {}
        #: where the trained weights landed, once known (``finish(adapter=)``
        #: or a hosted run that reached ``done``)
        self.adapter: str | None = None
        # Hosted runs (``train``): the platform's trainer owns the lifecycle,
        # so ``refresh``/``wait`` read it and the context manager never
        # finishes it from here.
        self.dataset_id: str | None = None
        self.call_id: str | None = None
        self.method: str | None = None
        self.holdout_id: str | None = None
        self.training: dict[str, Any] = {}
        self.error: str | None = None
        self._hosted = False

    @property
    def url(self) -> str:
        return f"{SITE_URL}/platform/training/{self.run_id}"

    # ------------------------------------------------------------ logging

    def log(self, step: int, **metrics: float) -> None:
        """Record one point. Any finite numeric keyword is a metric
        (``loss``, ``eval_loss``, ``lr``, ``epoch``, ``grad_norm``, ...)."""
        point: dict[str, Any] = {"step": int(step), "ts": time.time()}
        for key, value in metrics.items():
            if isinstance(value, bool) or value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number != number or number in (float("inf"), float("-inf")):
                continue
            point[str(key)] = number
        with self._lock:
            self._buffer.append(point)
            self.step = max(self.step, int(step))
            due = (
                len(self._buffer) >= self._flush_every
                or time.monotonic() - self._last_flush >= self._flush_seconds
            )
        if due:
            self.flush()

    def progress(self, step: int, total_steps: int | None = None) -> None:
        """Advance the bar without a metric. ``total_steps`` (re)sets the
        denominator; a trainer that learns its length late can call this."""
        if total_steps:
            with self._lock:
                self.total_steps = int(total_steps)
                self._pending_total = int(total_steps)
        self.log(step)

    def flush(self) -> bool:
        """Send buffered points. Returns True when nothing is left unsent."""
        with self._lock:
            batch = self._buffer[:MAX_BATCH]
            total = self._pending_total
        if not batch and total is None:
            return True
        body: dict[str, Any] = {"points": batch or [{"step": self.step, "ts": time.time()}]}
        if total is not None:
            body["total_steps"] = total
        try:
            self._call("POST", f"/runs/{self.run_id}/log", self._api_key, body)
        except Exception as exc:  # the dashboard must never stop the trainer
            self.errors += 1
            if not self._warned:
                self._warned = True
                warnings.warn(
                    f"training run {self.run_id}: could not send points ({exc}); "
                    "will retry on the next flush",
                    stacklevel=2,
                )
            log.debug("training run %s flush failed: %s", self.run_id, exc)
            return False
        with self._lock:
            del self._buffer[: len(batch)]
            if total is not None and self._pending_total == total:
                self._pending_total = None
            self._last_flush = time.monotonic()
            remaining = bool(self._buffer)
        if remaining:
            return self.flush()
        return True

    # ------------------------------------------------------------ lifecycle

    def finish(
        self,
        status: str = "done",
        *,
        summary: Mapping[str, Any] | None = None,
        adapter: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Flush, then mark the run ``done``, ``failed``, or ``stopped``."""
        self.flush()
        body: dict[str, Any] = {"status": status}
        # A second finish (an eval after the callback already finished the
        # run) adds to what was sent, never replaces it.
        merged = {**self._summary, **dict(summary or {})}
        if self._delta is not None:
            merged["delta"] = self._delta
        if merged:
            body["summary"] = _json_safe(merged)
            self._summary = dict(merged)
        if adapter:
            body["adapter"] = str(adapter)
            self.adapter = str(adapter)
        if error:
            body["error"] = str(error)[:2000]
        try:
            out = self._call("POST", f"/runs/{self.run_id}/finish", self._api_key, body)
        except Exception as exc:
            self.errors += 1
            warnings.warn(f"training run {self.run_id}: could not finish ({exc})", stacklevel=2)
            out = {"runId": self.run_id, "status": status, "unsent": True}
        self.status = status
        return out if isinstance(out, dict) else {"runId": self.run_id, "status": status}

    def fail(self, error: str) -> dict[str, Any]:
        return self.finish("failed", error=error)

    def note(self, **fields: Any) -> None:
        """Put fields on the run's summary ahead of ``finish``: whichever
        callback finishes the run, the summary carries them. Sent right
        away when the run is already finished."""
        self._summary.update(_json_safe(dict(fields)))
        if self.status == "running":
            return
        if self._delta is not None:
            self._send_delta()
        else:
            self._resend_summary()

    def _resend_summary(self) -> None:
        try:
            self._call(
                "POST",
                f"/runs/{self.run_id}/finish",
                self._api_key,
                {"status": self.status, "summary": dict(self._summary)},
            )
        except Exception as exc:
            self.errors += 1
            warnings.warn(
                f"training run {self.run_id}: could not send summary ({exc})", stacklevel=2
            )

    def delta(
        self,
        before: Sequence[dict],
        after: Sequence[dict],
        *,
        target: str | None = "pass_at_1",
        must_not_regress: Sequence[str] = (),
        by: str | Callable[[dict], Any] | None = None,
        proxy: str | None = None,
    ) -> dict[str, Any]:
        """Did the training move the behavior? ``delta_report`` over the
        rollouts before and after, kept on the run and sent with
        ``finish`` under ``summary["delta"]`` (sent right away when the run
        is already finished). The run page draws it, including the
        per-group table when ``by`` names a row key or marker. ``proxy``
        names the training reward's marker so the report can call the
        run over-optimized when the proxy moved and the target did not."""
        from .score.delta import delta_report

        report = delta_report(
            before,
            after,
            target=target,
            must_not_regress=list(must_not_regress),
            by=by,
            proxy=proxy,
        )
        self._delta = _json_safe(report)
        if self.status != "running":
            self._send_delta()
        return report

    def _send_delta(self) -> None:
        try:
            self._call(
                "POST",
                f"/runs/{self.run_id}/finish",
                self._api_key,
                {"status": self.status, "summary": {**self._summary, "delta": self._delta}},
            )
        except Exception as exc:
            self.errors += 1
            warnings.warn(f"training run {self.run_id}: could not send delta ({exc})", stacklevel=2)

    # ------------------------------------------------------------ hosted runs

    @property
    def hosted(self) -> bool:
        """True when the platform's trainer runs this (``train``)."""
        return self._hosted

    def refresh(self) -> str:
        """Read a hosted run's state from the platform: ``running``,
        ``done`` or ``failed``. Fills ``adapter``, ``training`` (before,
        after, seconds, rows) and ``error`` once it has ended."""
        if not self._hosted or not self.dataset_id:
            return self.status
        out = self._call("GET", f"/datasets/{self.dataset_id}/train", self._api_key)
        state = dict((out or {}).get("training") or {}) if isinstance(out, dict) else {}
        if not state:
            return self.status
        self._absorb(state)
        return self.status

    def wait(self, *, timeout: float | None = None, poll: float = 15.0) -> str:
        """Block until a hosted run ends. Returns the final status;
        raises ``TimeoutError`` when ``timeout`` seconds pass first."""
        started = time.monotonic()
        while self.refresh() == "running":
            if timeout is not None and time.monotonic() - started >= timeout:
                raise TimeoutError(
                    f"training run {self.run_id} still running after {timeout:.0f}s; "
                    f"watch it at {self.url}"
                )
            time.sleep(max(1.0, float(poll)))
        return self.status

    def _absorb(self, state: Mapping[str, Any]) -> None:
        self.training = dict(state)
        status = str(state.get("status") or self.status)
        self.status = status if status in ("running", "done", "failed", "stopped") else self.status
        if state.get("runId"):
            self.run_id = str(state["runId"])
        if state.get("callId"):
            self.call_id = str(state["callId"])
        if state.get("method"):
            self.method = str(state["method"])
        if state.get("holdoutId"):
            self.holdout_id = str(state["holdoutId"])
        if state.get("adapter"):
            self.adapter = str(state["adapter"])
        if state.get("error"):
            self.error = str(state["error"])

    def __enter__(self) -> TrainingRun:
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        if self.status != "running" or self._hosted:
            return
        if exc is not None:
            self.finish("failed", error=f"{exc_type.__name__}: {exc}")
        else:
            self.finish("done")

    def __repr__(self) -> str:
        return (
            f"TrainingRun({self.run_id!r}, {self.name!r}, status={self.status!r}, step={self.step})"
        )


def training_run(
    name: str,
    *,
    dataset: str | None = None,
    after_dataset: str | None = None,
    base_model: str | None = None,
    trainer: str | None = None,
    total_steps: int | None = None,
    config: Mapping[str, Any] | None = None,
    api_key: str | None = None,
    flush_every: int = FLUSH_EVERY,
    flush_seconds: float = FLUSH_SECONDS,
    transport: Callable[..., Any] | None = None,
) -> TrainingRun:
    """Create a run on the platform and return the handle to log into.

    ``dataset`` is the ``ds_...`` id trained on; ``after_dataset`` the set
    of post-training rollouts, when you have one, so the run page can show
    the before/after. ``config`` is anything JSON-shaped you want on the
    run page (hyperparameters, the command). ``api_key`` defaults to the
    usual credential chain.
    """
    body: dict[str, Any] = {"name": name}
    if dataset:
        body["dataset_id"] = dataset
    if after_dataset:
        body["after_dataset_id"] = after_dataset
    if base_model:
        body["base_model"] = base_model
    if trainer:
        body["trainer"] = trainer
    if total_steps:
        body["total_steps"] = int(total_steps)
    if config:
        body["config"] = dict(config)
    call = transport or _call
    created = call("POST", "/runs", api_key, body)
    run = TrainingRun(
        str(created["runId"]),
        name=name,
        api_key=api_key,
        total_steps=int(total_steps) if total_steps else None,
        flush_every=flush_every,
        flush_seconds=flush_seconds,
        transport=transport,
    )
    log.info("training run %s: %s", run.run_id, run.url)
    return run


METHODS = ("sft", "grpo", "dpo", "rm")
#: The bases the serving app runs. An adapter trained on any other base is a
#: file on a volume that ``serve`` cannot host; the trainer's defaults
#: (Qwen2.5-0.5B for SFT, 1.5B for GRPO and DPO) are not on this list.
SERVED_BASES = ("Qwen/Qwen3-4B", "microsoft/phi-4")


def train(
    dataset: str,
    *,
    method: str = "sft",
    steps: int | None = None,
    epochs: float | None = None,
    holdout: str | None = None,
    base_model: str | None = None,
    generations: int | None = None,
    learning_rate: float | None = None,
    beta: float | None = None,
    seed: int | None = None,
    max_completion_length: int | None = None,
    loss_type: str | None = None,
    config: Mapping[str, Any] | None = None,
    wait: bool = False,
    timeout: float | None = None,
    poll: float = 15.0,
    api_key: str | None = None,
    transport: Callable[..., Any] | None = None,
) -> TrainingRun:
    """Start a hosted fine-tune on a pushed dataset and return the run.

    ``method`` is ``"sft"`` (LoRA on the passing rows), ``"grpo"`` (the
    reference-first-action reward over the graded rows), ``"dpo"`` (a
    pass against a fail per prompt, length matched) or ``"rm"`` (a reward
    model on those same pairs; ``reward_model(run)`` is then a judge).
    ``steps`` sets the optimizer steps for GRPO, DPO and RM, ``epochs``
    the SFT epochs; each
    method has a default. ``holdout`` names the eval set; it defaults to
    the train set's split sibling from ``datasets.cut``. ``base_model``
    overrides the trainer's base; only ``SERVED_BASES`` can be served
    afterwards, and ``train`` warns when the run will not be.

    The run is the same record ``training_run`` makes, so ``run.url`` is
    the loss curve, ``run.delta`` and ``get_run`` work unchanged, and the
    trainer finishes it. ``run.refresh()`` reads where it is;
    ``run.wait()`` (or ``wait=True``) blocks until ``done`` or ``failed``,
    after which ``run.adapter`` names the weights and ``run.training``
    carries before, after, rows and seconds. ``serve`` puts the adapter on
    an endpoint.

    The knobs a run is reproduced and compared by (rlhf-book ch. 6, 7):
    ``generations`` is the group size per prompt for GRPO (the ``k`` the
    advantage is taken over; a pushed set's ``repeats`` is the natural
    value), ``beta`` the KL coefficient for GRPO and DPO, ``learning_rate``
    the optimizer step for every method, ``seed`` the sampling and data
    order seed, ``max_completion_length`` the token cap on a sampled reply
    (GRPO, DPO), ``loss_type`` the objective variant (GRPO: ``bnpo``,
    ``grpo``, ``dr_grpo``; DPO: any TRL loss). Each has a trainer default
    when left ``None``. ``config`` passes further host keys as given
    (``epsilonHigh``, ``scaleRewards``, ``maskTruncated``, ``balance``).
    Every knob lands on the run's ``config`` so the run page shows it.

    A dataset already training answers with that run instead of a second.
    """
    method = str(method or "sft").lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}; got {method!r}")
    if not dataset:
        raise ValueError("dataset: the ds_... id of a pushed dataset")
    if base_model not in SERVED_BASES:
        which = f"base_model={base_model!r}" if base_model else "the trainer's default base"
        warnings.warn(
            f"hosted {method} run on {dataset} uses {which}, which zps.serve cannot host "
            f"(served bases: {', '.join(SERVED_BASES)}); pass base_model={SERVED_BASES[0]!r} "
            "if the goal is an endpoint",
            stacklevel=2,
        )
    body: dict[str, Any] = {"method": method}
    if steps:
        body["steps"] = int(steps)
    if epochs:
        body["epochs"] = float(epochs)
    if holdout:
        body["holdoutId"] = str(holdout)
    if base_model:
        body["base"] = str(base_model)
    if generations is not None:
        if method != "grpo":
            raise ValueError("generations is the GRPO group size; other methods do not sample")
        if not 2 <= int(generations) <= 32:
            raise ValueError("generations: 2 to 32 rollouts per prompt")
        body["generations"] = int(generations)
    if learning_rate is not None:
        if not 0 < float(learning_rate) < 1:
            raise ValueError(
                "learning_rate: a positive step below 1 (5e-6 for RL, 2e-4 for SFT LoRA)"
            )
        body["lr"] = float(learning_rate)
    if beta is not None:
        if method not in ("grpo", "dpo"):
            raise ValueError("beta is the KL coefficient; it applies to grpo and dpo only")
        if float(beta) < 0:
            raise ValueError("beta: 0 or more")
        body["beta"] = float(beta)
    if seed is not None:
        body["seed"] = int(seed)
    if max_completion_length is not None:
        if method not in ("grpo", "dpo"):
            raise ValueError(
                "max_completion_length caps a sampled reply; it applies to grpo and dpo only"
            )
        if not 16 <= int(max_completion_length) <= 4096:
            raise ValueError("max_completion_length: 16 to 4096 tokens")
        body["maxCompletionLength"] = int(max_completion_length)
    if loss_type is not None:
        if method not in ("grpo", "dpo"):
            raise ValueError("loss_type picks the grpo or dpo objective variant")
        body["lossType"] = str(loss_type)
    for key, value in dict(config or {}).items():
        if key in body:
            raise ValueError(f"config[{key!r}] collides with a named argument")
        body[str(key)] = value
    call = transport or _call
    out = call("POST", f"/datasets/{dataset}/train", api_key, body)
    state = dict((out or {}).get("training") or {}) if isinstance(out, dict) else {}
    run_id = str(state.get("runId") or "")
    if not run_id:
        raise RuntimeError(
            f"the platform started training on {dataset} but reported no run id: {out!r}"
        )
    run = TrainingRun(run_id, name=f"{dataset} · {method}", api_key=api_key, transport=transport)
    run._hosted = True
    run.dataset_id = str(dataset)
    run.method = method
    run._absorb(state)
    if isinstance(out, dict) and out.get("alreadyRunning"):
        warnings.warn(
            f"dataset {dataset} is already training ({run.run_id}); returning that run",
            stacklevel=2,
        )
    log.info("hosted %s run %s on %s: %s", method, run.run_id, dataset, run.url)
    if wait:
        run.wait(timeout=timeout, poll=poll)
    return run


class RewardModel:
    """A finished ``method="rm"`` run as a judge (rlhf-book ch. 5).

    Calling it with one rollout row honors the judge contract: ``reward``
    is 1 when the model's score clears the run's pass threshold, 0
    otherwise, and ``rm_score`` carries the raw number so ``judge_trust``,
    ``build_preference_pairs(min_margin=)`` and a margin-aware loss can
    use it. ``score(rows)`` scores a batch in one call. Rows are rendered
    on the platform exactly as the model was trained: the run's system
    prompt, the user prompt, the first assistant turn.
    """

    def __init__(
        self,
        run: TrainingRun | str,
        *,
        threshold: float | None = None,
        api_key: str | None = None,
        transport: Callable[..., Any] | None = None,
    ):
        self.run_id = run.run_id if isinstance(run, TrainingRun) else str(run or "").strip()
        if not self.run_id:
            raise ValueError("run: a finished reward-model run (zps.train(method='rm')) or its id")
        self.threshold = threshold
        self.__name__ = f"reward_model:{self.run_id}"
        self._api_key = api_key
        self._call = transport or _call
        self.base: str | None = None

    def score(self, rows: Sequence[dict]) -> list[dict[str, Any]]:
        """``[{"rm_score", "reward", "threshold"}]`` for each row, in order.
        Up to 256 rows a call; more are sent in batches."""
        src = [r for r in rows if isinstance(r, dict)]
        out: list[dict[str, Any]] = []
        for i in range(0, len(src), 256):
            chunk = src[i : i + 256]
            res = self._call("POST", f"/runs/{self.run_id}/score", self._api_key, {"rows": chunk})
            res = res if isinstance(res, dict) else {}
            scores = list(res.get("scores") or [])
            if len(scores) != len(chunk):
                raise RuntimeError(
                    f"reward model {self.run_id} answered {len(scores)} scores for {len(chunk)} rows"
                )
            t = float(self.threshold if self.threshold is not None else res.get("threshold", 0.0))
            self.base = self.base or res.get("base")
            out += [
                {"rm_score": float(s), "reward": int(float(s) >= t), "threshold": t} for s in scores
            ]
        return out

    def __call__(self, trajectory: dict) -> dict[str, Any]:
        one = self.score([trajectory])[0]
        verdict = ">=" if one["reward"] else "<"
        return {
            "reward": one["reward"],
            "rm_score": one["rm_score"],
            "threshold": one["threshold"],
            "reason": f"reward model {self.run_id}: {one['rm_score']:.3f} {verdict} {one['threshold']:.3f}",
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"RewardModel({self.run_id!r}, threshold={self.threshold})"


def reward_model(
    run: TrainingRun | str,
    *,
    threshold: float | None = None,
    api_key: str | None = None,
    transport: Callable[..., Any] | None = None,
) -> RewardModel:
    """A judge backed by a finished reward-model run.

    ``run = zps.train("ds_...", method="rm", wait=True)`` trains a
    sequence-classification head on the set's pass-vs-fail pairs and
    picks the score threshold that best separates the held-out pairs.
    ``judge = zps.reward_model(run)`` then scores any rollout row:
    ``data.grade(judge=judge)``, ``zps.evaluate(rollouts, judge)``,
    ``zps.judge_trust(scored.rows, judge=judge)``. Pass ``threshold=`` to
    override the run's own cut. The scores are the model's; a reward
    model trained on one agent's pairs says nothing about another agent.
    """
    return RewardModel(run, threshold=threshold, api_key=api_key, transport=transport)


def models(*, api_key: str | None = None) -> list[dict[str, Any]]:
    """The account's hosted models: ``name``, ``baseModel``, ``adapter``,
    ``adapterRunId``, ``version``, ``endpoint`` (an OpenAI-compatible
    base URL; send the account key as the bearer and ``name`` as the
    model)."""
    out = _call("GET", "/models", api_key)
    return list(out.get("models") or []) if isinstance(out, dict) else []


def serve(
    name: str,
    run: TrainingRun | str | None = None,
    *,
    base_model: str | None = None,
    api_key: str | None = None,
    transport: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Host a finished run's adapter under ``name``. Returns the model
    row; ``endpoint`` is the OpenAI-compatible base URL and ``name`` the
    model id to send. Posting an existing name bumps ``version``.

    ``run`` is a ``TrainingRun`` or its id; the adapter and base model
    come from the run record unless ``base_model`` is given. No ``run``
    serves the bare base (``base_model`` required).
    """
    call = transport or _call
    adapter: str | None = None
    base = base_model
    if isinstance(run, TrainingRun):
        adapter = run.adapter
        run_id: str | None = run.run_id
    else:
        run_id = str(run) if run else None
    if run_id and (adapter is None or base is None):
        meta = call("GET", f"/runs/{run_id}", api_key)
        meta = meta if isinstance(meta, dict) else {}
        adapter = adapter or meta.get("adapter")
        base = base or meta.get("baseModel") or meta.get("base_model")
        if not adapter:
            status = str(meta.get("status") or "?")
            if status in ("failed", "stopped"):
                raise ValueError(
                    f"run {run_id} {status} and produced no adapter: "
                    f"{meta.get('error') or 'no error recorded'}"
                )
            raise ValueError(f"run {run_id} has no adapter yet (status {status}); wait for it")
    if not base:
        raise ValueError("base_model: which served base the adapter was trained on")
    if base not in SERVED_BASES:
        raise ValueError(
            f"{base} is not a served base ({', '.join(SERVED_BASES)}); the adapter"
            + (f" from run {run_id}" if run_id else "")
            + f" cannot be hosted. Train with base_model={SERVED_BASES[0]!r} for an endpoint"
        )
    body: dict[str, Any] = {"name": str(name).strip().lower(), "baseModel": str(base)}
    if adapter:
        body["adapter"] = str(adapter)
    out = call("POST", "/models", api_key, body)
    row = dict(out) if isinstance(out, dict) else {"name": body["name"]}
    log.info("hosted model %s v%s at %s", row.get("name"), row.get("version"), row.get("endpoint"))
    return row


def list_runs(*, api_key: str | None = None) -> list[dict[str, Any]]:
    out = _call("GET", "/runs", api_key)
    return list(out.get("runs") or []) if isinstance(out, dict) else []


def get_run(run_id: str, *, api_key: str | None = None) -> dict[str, Any]:
    """The run plus ``series``: its points, oldest first."""
    return _call("GET", f"/runs/{run_id}", api_key)


def delete_run(run_id: str, *, api_key: str | None = None) -> dict[str, Any]:
    return _call("DELETE", f"/runs/{run_id}", api_key)


def attach_delta(
    run_id: str,
    before: Sequence[dict],
    after: Sequence[dict],
    *,
    target: str | None = "pass_at_1",
    must_not_regress: Sequence[str] = (),
    by: str | Callable[[dict], Any] | None = None,
    proxy: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Compute ``delta_report`` for a finished run and put it on the run
    page: the summary is re-sent with ``delta`` added, status unchanged."""
    from .score.delta import delta_report

    run = _call("GET", f"/runs/{run_id}", api_key)
    report = delta_report(
        before,
        after,
        target=target,
        must_not_regress=list(must_not_regress),
        by=by,
        proxy=proxy,
    )
    summary = dict(run.get("summary") or {})
    summary["delta"] = _json_safe(report)
    status = str(run.get("status") or "done")
    if status == "running":
        status = "done"
    _call("POST", f"/runs/{run_id}/finish", api_key, {"status": status, "summary": summary})
    return report


# ---------------------------------------------------------------- Transformers / TRL


_EVENTS = (
    "on_init_end",
    "on_train_begin",
    "on_train_end",
    "on_epoch_begin",
    "on_epoch_end",
    "on_step_begin",
    "on_substep_end",
    "on_step_end",
    "on_optimizer_step",
    "on_pre_optimizer_step",
    "on_evaluate",
    "on_predict",
    "on_save",
    "on_log",
    "on_prediction_step",
)


class _NoOpCallback:
    """Stand-in base when transformers is not installed: every event the
    Trainer fires is a no-op, so the duck-typed callback still fits."""


for _event in _EVENTS:
    setattr(_NoOpCallback, _event, lambda self, *a, **k: None)


def _callback_base() -> type:
    try:
        from transformers import TrainerCallback as HFCallback

        return HFCallback
    except Exception:  # transformers is not a dependency of this package
        return _NoOpCallback


# Timing and throughput keys the Trainer logs alongside eval metrics; not
# learning signal, so not drawn.
_SKIP_KEYS = {
    "eval_runtime",
    "eval_samples_per_second",
    "eval_steps_per_second",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    "total_flos",
    "step",
}

_LOG_KEYS = {
    "loss": "loss",
    "eval_loss": "eval_loss",
    "learning_rate": "lr",
    "epoch": "epoch",
    "grad_norm": "grad_norm",
    "train_loss": "train_loss",
    "mean_token_accuracy": "token_accuracy",
    "eval_mean_token_accuracy": "eval_token_accuracy",
    "num_tokens": "tokens",
    # TRL RL trainers (GRPO, PPO, RLOO, online DPO): the curves an RL run is
    # read by. Any ``rewards/<name>`` key is kept under ``reward_<name>``.
    "reward": "reward",
    "reward_std": "reward_std",
    "kl": "kl",
    "objective/kl": "kl",
    "objective/rlhf_reward": "reward",
    "objective/scores": "score",
    "objective/entropy": "entropy",
    "entropy": "entropy",
    "completion_length": "completion_length",
    "completions/mean_length": "completion_length",
    "clip_ratio": "clip_ratio",
    "policy_loss": "policy_loss",
    "value_loss": "value_loss",
}


class TrainerCallback(_callback_base()):  # type: ignore[misc]
    """One line on a Transformers or TRL trainer:
    ``trainer.add_callback(zps.TrainerCallback(run))``.

    Logs every ``on_log`` to the run (loss, lr, eval loss, epoch, grad
    norm, token accuracy), takes the step count from the trainer at
    ``on_train_begin``, and finishes the run at ``on_train_end``. If the
    trainer raises, finish the run yourself with ``run.fail(...)`` or use
    the run as a context manager around ``trainer.train()``.
    """

    def __init__(self, run: TrainingRun, *, finish: bool = True):
        super().__init__()
        self.run = run
        #: finish the run at on_train_end. Pass False when the script
        #: evaluates after training and calls run.finish itself.
        self.finish_on_end = finish

    def on_train_begin(self, args=None, state=None, control=None, **kwargs):
        total = getattr(state, "max_steps", None)
        if total:
            self.run.progress(int(getattr(state, "global_step", 0) or 0), int(total))
        return control

    def on_log(self, args=None, state=None, control=None, logs=None, **kwargs):
        if not isinstance(logs, dict):
            return control
        metrics = {}
        for key, value in logs.items():
            if key in _SKIP_KEYS:
                continue
            name = _LOG_KEYS.get(key)
            if name is None and key.startswith("rewards/"):
                name = "reward_" + key[len("rewards/") :].replace("/", "_")
            if name is None and key.startswith("eval_") and isinstance(value, (int, float)):
                name = key
            if name is not None:
                metrics[name] = value
        step = int(getattr(state, "global_step", 0) or 0)
        if metrics:
            self.run.log(step, **metrics)
        return control

    def on_train_end(self, args=None, state=None, control=None, **kwargs):
        if self.finish_on_end and self.run.status == "running":
            summary: dict[str, Any] = {}
            history = getattr(state, "log_history", None) or []
            for entry in reversed(history):
                if isinstance(entry, dict) and "train_loss" in entry:
                    summary["train_loss"] = entry["train_loss"]
                    break
            self.run.finish("done", summary=summary or None)
        return control


__all__ = [
    "TrainerCallback",
    "TrainingRun",
    "attach_delta",
    "delete_run",
    "get_run",
    "list_runs",
    "models",
    "serve",
    "train",
    "training_run",
]
