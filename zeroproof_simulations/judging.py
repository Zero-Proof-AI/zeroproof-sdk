"""One judge contract for grading and evaluation, and the loop around it.

A judge is any callable that takes a trajectory dict and returns a verdict.
LLM judge, rules engine, reward model, human-label lookup, API call: the
SDK does not care how the reward was produced, only that the result honors
the contract below. The same engine scores generated simulation data and
held-out model rollouts, so training-time grading and eval-time judging
share one schema.

Judge contract (minimal now, extensible later)::

    judge(trajectory) -> {"reward": 0 or 1}            # minimum
    judge(trajectory) -> {"reward": 0.7,               # floats allowed
                          "reason": "...",             # optional
                          "failure_class": "...",      # optional
                          ...anything else}            # kept as metadata
    judge(trajectory) -> 0 or 1 or 0.7                 # bare number works

Anything else — missing reward, unsupported type, an exception — marks the
row (``judge_status`` of ``missing_reward`` / ``invalid_result`` /
``error`` / ``timeout``) with ``reward=None``. Nothing is silently zero.

The five-line loop::

    import zeroproof_simulations as zps
    judge = lambda t: {"reward": int("sorry" not in t["final_text"])}
    scored = zps.run_judge(data.trajectories, judge)     # or data.grade(judge=judge)
    zps.export_training(scored.passes(), output="train.jsonl",
                        system_prompt=POLICY, tools=TOOLS)
    # ...train externally, roll the model on a holdout...
    evald = zps.evaluate(rollouts, judge, model="my-tuned-v1")
    nxt = zps.simulate(tools=TOOLS, system_prompt=POLICY,
                       traces=evald.failed_traces())     # loop closed
"""
from __future__ import annotations

import concurrent.futures
import json
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

_VALID_STATUSES = ("ok", "missing_reward", "invalid_result", "error",
                   "timeout")


def normalize_judge_result(raw: Any) -> dict[str, Any]:
    """Coerce one judge return into the contract; never invent a reward."""
    if isinstance(raw, bool):
        return {"reward": int(raw), "reason": "", "judge_status": "ok",
                "judge_meta": {}}
    if isinstance(raw, (int, float)):
        value = float(raw)
        # Scalar lane: 1 pass, 0 fail, (0, 1) partial. Anything outside
        # [0, 1] fits no lane and is a contract break, not a reward.
        if not 0.0 <= value <= 1.0:
            return {"reward": None, "reason": "",
                    "judge_status": "invalid_result",
                    "judge_meta": {"reward_out_of_range": value}}
        reward = int(value) if value in (0.0, 1.0) else value
        return {"reward": reward, "reason": "", "judge_status": "ok",
                "judge_meta": {}}
    if isinstance(raw, dict):
        if "reward" not in raw and "score" not in raw:
            return {"reward": None, "reason": "",
                    "judge_status": "missing_reward",
                    "judge_meta": {"returned_keys": sorted(map(str, raw))}}
        value = raw.get("reward", raw.get("score"))
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, (int, float)):
            return {"reward": None, "reason": str(raw.get("reason") or ""),
                    "judge_status": "invalid_result",
                    "judge_meta": {"reward_type": type(value).__name__}}
        if not 0.0 <= float(value) <= 1.0:
            return {"reward": None, "reason": str(raw.get("reason") or ""),
                    "judge_status": "invalid_result",
                    "judge_meta": {"reward_out_of_range": float(value)}}
        reward = int(value) if float(value) in (0.0, 1.0) else float(value)
        meta = {k: v for k, v in raw.items()
                if k not in {"reward", "score", "reason"}}
        return {"reward": reward, "reason": str(raw.get("reason") or ""),
                "judge_status": "ok", "judge_meta": meta}
    return {"reward": None, "reason": "", "judge_status": "invalid_result",
            "judge_meta": {"returned_type": type(raw).__name__}}


class ScoredData:
    """Scored trajectories: the one representation grade and eval share.

    Iterates as plain dicts, so it feeds ``simulate(traces=...)``,
    ``mine_traces``, ``export_training`` and JSONL writers directly —
    no conversion scripts.
    """

    def __init__(self, rows: list[dict], *, run_id: str, source: str,
                 judge_name: str, model: str | None = None):
        self.rows = rows
        self.run_id = run_id
        self.source = source
        self.judge_name = judge_name
        self.model = model

    def __iter__(self) -> Iterator[dict]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

    def partials(self) -> list[dict]:
        """Rows with a continuous reward strictly between 0 and 1. The
        scalar lane: 1 pass, 0 fail, partials here, None unjudged - every
        contract-legal reward is visible in exactly one view."""
        return [r for r in self.rows
                if isinstance(r.get("reward"), float)
                and 0.0 < r["reward"] < 1.0]

    def select_by_reward_range(self, lo: float, hi: float, *,
                               inclusive: bool = True) -> list[dict]:
        """Rows whose numeric reward falls in [lo, hi] (or (lo, hi))."""
        out = []
        for r in self.rows:
            value = r.get("reward")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if (lo <= value <= hi if inclusive else lo < value < hi):
                out.append(r)
        return out

    def select_by_reward(self, reward) -> list[dict]:
        return [r for r in self.rows if r.get("reward") == reward]

    def passes(self) -> list[dict]:
        return self.select_by_reward(1)

    def failures(self) -> list[dict]:
        return self.select_by_reward(0)

    @property
    def traces(self) -> list[dict]:
        """The doctrine-sketch spelling of the loop-closing move:
        ``simulate(traces=results.traces)``. Failed trajectories, ready
        to aim the next generation round."""
        return self.failed_traces()

    def failed_traces(self) -> list[dict]:
        """Failures, ready to hand to ``simulate(traces=...)``."""
        return self.failures()

    def unjudged(self) -> list[dict]:
        """Rows the judge could not score. Never treated as failures."""
        return [r for r in self.rows if r.get("judge_status") != "ok"]

    def select_for_sft(self, *, target: int = 1000
                       ) -> tuple[list[dict], dict[str, Any]]:
        """Diverse correct demonstrations: 1-labeled, deduped by behavior."""
        from .optimize import select_for_sft
        return select_for_sft(self.rows, target=target)

    def select_for_preference(self, *, max_pairs_per_prompt: int = 1
                              ) -> tuple[list[dict], dict[str, Any]]:
        """Chosen/rejected pairs from same-task contrast. Failures earn here."""
        return build_preference_pairs(
            self.rows, max_pairs_per_prompt=max_pairs_per_prompt)

    def select_for_rl(self, *, target: int = 1000, lo: float = 0.3,
                      hi: float = 0.7, has_tools: bool = True
                      ) -> tuple[list[dict], dict[str, Any]]:
        """Whole mixed-reward groups for RL; groups never split."""
        from .optimize import select_for_rl
        return select_for_rl(self.rows, target=target, lo=lo, hi=hi,
                             has_tools=has_tools)

    def report(self, *, tools: Sequence[dict] | None = None,
               system_prompt: str = "") -> dict[str, Any]:
        from .preflight import dataset_report
        out = dataset_report(self.rows, tools=tools,
                             system_prompt=system_prompt)
        rewards = [r.get("reward") for r in self.rows
                   if isinstance(r.get("reward"), (int, float))]
        out["reward_distribution"] = {
            str(k): rewards.count(k) for k in sorted(set(rewards), key=float)}
        out["unjudged"] = len(self.unjudged())
        out["scoring_run_id"] = self.run_id
        out["judge"] = self.judge_name
        return out

    def save(self, path: str) -> str:
        with open(path, "w") as fh:
            for row in self.rows:
                fh.write(json.dumps(row, default=str) + "\n")
        return path


def _score_one(judge: Callable, row: dict) -> dict[str, Any]:
    try:
        return normalize_judge_result(judge(row))
    except Exception as exc:  # judge bugs mark the row, never crash the run
        return {"reward": None, "reason": "", "judge_status": "error",
                "judge_meta": {"error": f"{type(exc).__name__}: {exc}"}}


def run_judge(rows: Sequence[dict] | Any, judge: Callable[[dict], Any], *,
              judge_name: str | None = None, source: str = "grade",
              model: str | None = None, run_id: str | None = None,
              concurrency: int = 8,
              timeout: float | None = None) -> ScoredData:
    """Score trajectories with any judge. Originals are left unmodified.

    Each scored row is a copy of the input row plus ``reward``, ``reason``,
    ``judge_status``, ``judge_meta``, and a ``lineage`` record naming the
    scoring run, its source (grade or eval), the judged model, and the
    parent trajectory. Rows whose judge result breaks the contract keep
    ``reward=None`` and a non-ok status; they are counted, not hidden.
    """
    src_rows = [r for r in rows if isinstance(r, dict)]
    rid = run_id or f"score_{uuid.uuid4().hex[:12]}"
    name = judge_name or getattr(judge, "__name__", "") or "judge"
    if name == "<lambda>":
        name = "lambda_judge"
    verdicts: list[dict[str, Any]]
    if concurrency > 1 and len(src_rows) > 1:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=concurrency) as pool:
            futures = [pool.submit(_score_one, judge, r) for r in src_rows]
            verdicts = []
            for future in futures:
                try:
                    verdicts.append(future.result(timeout=timeout))
                except concurrent.futures.TimeoutError:
                    verdicts.append({"reward": None, "reason": "",
                                     "judge_status": "timeout",
                                     "judge_meta": {"timeout_s": timeout}})
    else:
        verdicts = [_score_one(judge, r) for r in src_rows]
    scored: list[dict] = []
    for i, (row, verdict) in enumerate(zip(src_rows, verdicts)):
        out = dict(row)
        out["reward"] = verdict["reward"]
        if verdict["reason"]:
            out["reason"] = verdict["reason"]
        out["judge_status"] = verdict["judge_status"]
        out["judge_name"] = name
        if verdict["judge_meta"]:
            out["judge_meta"] = verdict["judge_meta"]
        fc = (verdict["judge_meta"] or {}).get("failure_class")
        if fc:
            out["failure_class"] = str(fc)
        parent = row.get("scenario_id") or row.get("id") or f"row_{i}"
        lineage = dict(row.get("lineage") or {})
        lineage.update({"scoring_run_id": rid, "source": source,
                        "judge": name, "parent": str(parent)})
        if model:
            lineage["model"] = model
        if row.get("lineage", {}).get("scoring_run_id"):
            lineage["prior_scoring_run_id"] = row["lineage"]["scoring_run_id"]
        out["lineage"] = lineage
        scored.append(out)
    return ScoredData(scored, run_id=rid, source=source, judge_name=name,
                      model=model)


def evaluate(rows: Sequence[dict] | Any = None,
             judge: Callable[[dict], Any] | None = None, *,
             grader: Callable[[dict], Any] | None = None,
             model: str | None = None,
             eval_set: Sequence[Any] | None = None,
             judge_name: str | None = None,
             run_id: str | None = None, concurrency: int = 8,
             timeout: float | None = None) -> ScoredData:
    """Judge held-out rollouts under the exact contract ``grade`` uses.

    Same engine, same schema; only the lineage source differs. Feeding
    ``evaluate(...).traces`` to ``simulate(traces=...)`` is the
    loop-closing move. ``grader=`` is the doctrine-sketch name for the
    judge callable; either spelling works, not both. ``eval_set=``
    (prompt strings or rows) checks that the rollouts actually cover the
    frozen evaluation set and reports the gap on the result's
    ``eval_coverage`` instead of letting a silent partial eval pass as a
    full one.
    """
    if grader is not None and judge is not None:
        raise ValueError("pass judge= or grader=, not both")
    judge = judge if judge is not None else grader
    if judge is None:
        raise ValueError("evaluate() needs a judge (judge= or grader=)")
    scored = run_judge(rows, judge, judge_name=judge_name, source="eval",
                       model=model, run_id=run_id, concurrency=concurrency,
                       timeout=timeout)
    if eval_set is not None:
        wanted = {" ".join(str(item.get("prompt") if isinstance(item, dict)
                               else item or "").lower().split())
                  for item in eval_set}
        wanted.discard("")
        got = {" ".join(str(r.get("prompt") or "").lower().split())
               for r in scored.rows}
        missing = sorted(wanted - got)
        scored.eval_coverage = {"eval_set": len(wanted),
                                "covered": len(wanted) - len(missing),
                                "missing": missing[:20]}
    return scored


def _pair_key(row: dict) -> str:
    return " ".join(str(row.get("prompt") or "").lower().split())


def build_preference_pairs(rows: Sequence[dict], *,
                           max_pairs_per_prompt: int = 1
                           ) -> tuple[list[dict], dict[str, Any]]:
    """Same-task chosen/rejected pairs for preference training (DPO-style).

    A pair exists only where the same prompt has both a 1-labeled and a
    0-labeled trajectory — the contrast is the training signal, so
    failures are supply here, not waste. Rows without a valid judge result
    never pair. Returns (pairs, report); each pair carries both parents'
    lineage.
    """
    groups: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("judge_status", "ok") != "ok":
            continue
        if row.get("reward") not in (0, 1):
            continue
        key = _pair_key(row)
        if not key:
            continue
        slot = groups.setdefault(key, {"pass": [], "fail": []})
        slot["pass" if row["reward"] == 1 else "fail"].append(row)
    pairs: list[dict] = []
    contrast_prompts = 0
    for key, slot in groups.items():
        if not slot["pass"] or not slot["fail"]:
            continue
        contrast_prompts += 1
        for i in range(min(max_pairs_per_prompt,
                           len(slot["pass"]), len(slot["fail"]))):
            chosen, rejected = slot["pass"][i], slot["fail"][i]
            pairs.append({
                "prompt": chosen.get("prompt"),
                "chosen": chosen,
                "rejected": rejected,
                "chosen_reason": str(chosen.get("reason") or ""),
                "rejected_reason": str(rejected.get("reason") or ""),
                "rejected_failure_class": rejected.get("failure_class"),
                "lineage": {"chosen": chosen.get("lineage"),
                            "rejected": rejected.get("lineage")},
            })
    report = {"pairs": len(pairs), "prompts_seen": len(groups),
              "prompts_with_contrast": contrast_prompts,
              "note": ("pairs require the same prompt to have both a pass "
                       "and a fail; raise rollouts_per_request to create "
                       "contrast" if not pairs else "")}
    return pairs, report


__all__ = ["run_judge", "evaluate", "ScoredData", "normalize_judge_result",
           "build_preference_pairs"]
