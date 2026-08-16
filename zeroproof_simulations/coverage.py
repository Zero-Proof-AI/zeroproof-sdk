"""Run-level coverage curve and saturation prediction (cheap, no dashboard)."""
from __future__ import annotations

import json
from typing import Any

NEW_SIGNATURE_FLOOR = 0.02
# A short unique-signature blip is not a plateau. Need a long stall.
FLAT_BATCHES = 12
SATURATION_COPIES = 5


def space_saturated(
    cell_counts: dict[str, int],
    shape_counts: dict[str, int],
    *,
    uncovered_shapes: int = 0,
    walked_shapes: bool = False,
    copies: int = SATURATION_COPIES,
) -> bool:
    """Whole walked grid, enough copies. Not a one-hash flatline."""
    if not cell_counts:
        return False
    if min(cell_counts.values()) < copies:
        return False
    if walked_shapes:
        if uncovered_shapes > 0:
            return False
        if shape_counts and min(shape_counts.values()) < copies:
            return False
    return True


def copies_remaining(counts: dict[str, int], *, copies: int = SATURATION_COPIES) -> int:
    if not counts:
        return copies
    return sum(max(0, copies - n) for n in counts.values())


def coverage_point(
    trajectories: list[dict],
    *,
    cells: set[str],
    shape_keys: set[str],
    arm_weights: dict[str, float] | None = None,
    batch_fresh_rate: float | None = None,
    mean_batch_novelty: float | None = None,
    stopped_because: str | None = None,
) -> dict[str, Any]:
    """One snapshot after a rollout batch."""
    prompts = {t.get("prompt") for t in trajectories if t.get("prompt")}
    sids = {t.get("scenario_id") for t in trajectories if t.get("scenario_id")}
    sigs = {t.get("behavior_signature") for t in trajectories
            if t.get("behavior_signature")}
    return {
        "n_rows": len(trajectories),
        "unique_prompts": len(prompts),
        "unique_scenario_ids": len(sids),
        "unique_behavior_signatures": len(sigs),
        "unique_cells": len(cells),
        "unique_action_shapes": len(shape_keys),
        "arm_weights": dict(arm_weights or {}),
        "batch_fresh_rate": batch_fresh_rate,
        "mean_batch_novelty": mean_batch_novelty,
        "stopped_because": stopped_because,
    }


def predict_to_saturation(
    curve: list[dict],
    *,
    budget: int,
    flat_streak: int = 0,
    last_batch_size: int = 1,
    copy_deficit: int = 0,
) -> dict[str, Any]:
    """Rows left until 5-copy grid + plateau, capped by budget."""
    if not curve:
        return {"predicted_to_saturation": None, "predicted_remaining": None}
    current = curve[-1]["n_rows"]
    batch_sz = max(1, int(last_batch_size))
    if curve[-1].get("stopped_because") == "saturation":
        return {"predicted_to_saturation": current, "predicted_remaining": 0}
    plateau_left = max(0, FLAT_BATCHES - int(flat_streak)) * batch_sz
    remaining = max(int(copy_deficit), plateau_left)
    remaining = min(max(0, budget - current), remaining)
    predicted = min(budget, current + remaining)
    return {
        "predicted_to_saturation": predicted,
        "predicted_remaining": max(0, predicted - current),
    }


def build_coverage_summary(
    curve: list[dict],
    *,
    budget: int,
    stopped_because: str,
    flat_streak: int = 0,
    last_batch_size: int = 1,
    copy_deficit: int = 0,
) -> dict[str, Any]:
    """Final run-level coverage summary."""
    if not curve:
        point = coverage_point([], cells=set(), shape_keys=set())
    else:
        point = dict(curve[-1])
    pred = predict_to_saturation(
        curve, budget=budget, flat_streak=flat_streak,
        last_batch_size=last_batch_size, copy_deficit=copy_deficit)
    return {
        "rows": point.get("n_rows", 0),
        "unique_prompts": point.get("unique_prompts", 0),
        "unique_scenario_ids": point.get("unique_scenario_ids", 0),
        "unique_signatures": point.get("unique_behavior_signatures", 0),
        "unique_cells": point.get("unique_cells", 0),
        "unique_shapes": point.get("unique_action_shapes", 0),
        "budget": int(budget),
        "stopped_because": stopped_because,
        "saturation": stopped_because == "saturation",
        "predicted_to_saturation": pred["predicted_to_saturation"],
        "predicted_remaining": pred["predicted_remaining"],
    }


def cell_key(row: dict) -> str:
    dims = row.get("scenario_dimensions")
    if isinstance(dims, dict) and dims:
        return json.dumps(dims, sort_keys=True, default=str)
    return str(row.get("scenario_id") or "")
