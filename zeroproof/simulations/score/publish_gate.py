"""The gate a dataset passes before it leaves for the platform.

Two jobs, both from the RLVR playbook (rlhf-book ch. 6 and 7: dynamic
sampling drops all-pass / all-fail groups; offline difficulty filtering
keeps prompts the start policy solves 20-80% of the time, measured with
N samples; curricula need that per-prompt difficulty stored with the data).

* ``calibrate`` writes the measured difficulty on every graded row: the
  per-task pass rate over its k rollouts, the sample count, and the
  policy that produced them. That is the schema's ``Calibration`` record,
  flattened onto the row as ``calibration``.
* ``publish_gate`` refuses an RL-shaped dataset that could not train
  anything: ungraded rows, or no mixed group anywhere. It reports what a
  grouped update would see (pass@1, headroom, band counts) and warns
  when out-of-band groups are still present, because ``push`` uploads
  rows as they are and the pruning lives in ``optimize``.

Explore-shaped runs (one rollout per ask) pass through with a calibration
stamp and a report; they are not RL data and are not judged as such.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from ..schema import Calibration, PolicyRef
from .optimize import DEFAULT_BAND, _binary_label, _group_label_lists, group_signal
from .passat import pass_at


class PublishGateError(ValueError):
    """The dataset must not be published as it stands. The message says why."""


def policy_ref(policy: PolicyRef | dict | str | None, *, model: str | None = None) -> PolicyRef:
    """Coerce whatever the caller has into a ``PolicyRef``. A bare string
    is the system prompt and gets hashed; a dict is field-by-field."""
    if isinstance(policy, PolicyRef):
        return policy
    if isinstance(policy, dict):
        return PolicyRef(
            name=str(policy.get("name") or ""),
            model=policy.get("model") or model,
            prompt_hash=policy.get("prompt_hash"),
            version=policy.get("version"),
        )
    prompt_hash = None
    if isinstance(policy, str) and policy:
        prompt_hash = hashlib.sha256(policy.encode("utf-8")).hexdigest()[:16]
    return PolicyRef(name="", model=model, prompt_hash=prompt_hash)


def _task_id(row: dict) -> str:
    return str(row.get("task_id") or row.get("prompt") or "")


def calibrate(
    rows: Sequence[dict],
    *,
    policy: PolicyRef | dict | str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Stamp ``calibration`` on every graded row, in place.

    The per-task pass rate is over the binary rewards grouped by prompt,
    the same grouping ``group_signal`` and ``pass_at`` use. Rows without a
    0/1 reward are left alone and counted. Returns a report with the
    number of tasks and rows stamped plus the ``pass_at`` summary.
    """
    student = policy_ref(policy, model=model)
    groups = _group_label_lists(rows)
    stamped = 0
    for row in rows:
        if not isinstance(row, dict) or _binary_label(row) is None:
            continue
        labels = groups.get(str(row.get("prompt") or ""))
        if not labels:
            continue
        record = Calibration(
            task_id=_task_id(row),
            student=student,
            n=len(labels),
            pass_rate=sum(labels) / len(labels),
        )
        row["calibration"] = asdict(record)
        stamped += 1
    rates = pass_at(rows)
    return {
        "n_tasks": len(groups),
        "n_rows": len(rows),
        "n_stamped": stamped,
        "n_unstamped": len(rows) - stamped,
        "student": asdict(student),
        "pass_at": rates.to_dict(),
    }


def is_rl_shaped(rows: Sequence[dict], *, mode: str | None = None) -> bool:
    """RL data means repeats of one ask. ``mode="rl"`` says so outright;
    otherwise any prompt seen more than once counts."""
    if str(mode or "").lower() == "rl":
        return True
    seen: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("prompt") or "")
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            return True
    return False


def publish_gate(
    rows: Sequence[dict],
    *,
    mode: str | None = None,
    band: tuple[float, float] = DEFAULT_BAND,
    policy: PolicyRef | dict | str | None = None,
    model: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Check, calibrate, and report. Raises ``PublishGateError`` when
    ``strict`` and the rows are RL-shaped but ungraded or carry no mixed
    group. Never mutates anything except the ``calibration`` stamp.
    """
    lo, hi = float(band[0]), float(band[1])
    rl = is_rl_shaped(rows, mode=mode)
    calibration = calibrate(rows, policy=policy, model=model)
    signal = group_signal(rows, lo=lo, hi=hi)
    graded = calibration["n_stamped"]
    warnings: list[str] = []
    refusal: str | None = None

    if rl and graded == 0:
        refusal = (
            "ungraded_rl_rows: RL data needs a 0/1 reward on every rollout so a "
            "grouped update has contrast; grade first (zps.grade or data.grade)"
        )
    elif rl and signal.get("n_mixed", 0) == 0:
        refusal = (
            "no_mixed_groups: every ask is unanimous, so group-relative advantages "
            "are zero everywhere and the run would train nothing; regrade with a "
            "stricter rubric or raise difficulty before publishing"
        )
    if rl and not refusal:
        out_of_band = signal["n_mixed"] - signal["n_in_band"]
        if out_of_band:
            warnings.append(
                f"{out_of_band} mixed ask(s) fall outside the {lo:.0%}-{hi:.0%} band and "
                "were not pruned; run optimize(mode='rl') before push to enforce it"
            )
        if signal["n_all_zero"] or signal["n_all_one"]:
            warnings.append(
                f"{signal['n_all_zero'] + signal['n_all_one']} unanimous ask(s) still "
                "present; optimize(mode='rl') drops them"
            )
        if calibration["n_unstamped"]:
            warnings.append(f"{calibration['n_unstamped']} row(s) have no 0/1 reward")
    report = {
        "ok": refusal is None,
        "rl_shaped": rl,
        "mode": mode,
        "band": [lo, hi],
        "signal": signal,
        "calibration": calibration,
        "warnings": warnings,
        "refusal": refusal,
    }
    if refusal and strict:
        raise PublishGateError(refusal)
    return report


__all__ = ["PublishGateError", "calibrate", "is_rl_shaped", "policy_ref", "publish_gate"]
