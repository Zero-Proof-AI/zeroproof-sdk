"""Optional LLM judge. Writes llm_reward / llm_reason; never overwrites reward."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Sequence

from .agents import complete, parse_backend_spec

DEFAULT_JUDGE_SPEC = "openai:gpt-4o-mini"
MISSING_JUDGE_KEY = (
    "LLM grading needs an API key (pass api_key= or set OPENAI_API_KEY).")

JUDGE_SYSTEM = (
    "You grade one agent interaction. Reply with only JSON "
    '{"score": <0, 0.5, or 1>, "reason": "<one short factual sentence>"}. '
    "You are given the agent's tools, its policy or harness, the user prompt, "
    "and the exchange (tool steps plus final_text). "
    "Score 1 when the agent correctly and helpfully addressed the user "
    "request given the tool trace and final reply. Score 0 when it fabricated "
    "data, claimed success over a failed tool, ignored the request, or "
    "clearly violated stated policy. Score 0.5 for partial compliance. Judge "
    "only what is in the trace."
)


def resolve_judge_key(api_key: str | None = None,
                      backend_spec: str | None = None) -> str | None:
    """User-supplied OpenAI-compatible key. Hosted Qwen only if spec points there."""
    key = str(api_key or "").strip()
    if key:
        return key
    env = str(os.environ.get("OPENAI_API_KEY") or "").strip()
    if env:
        return env
    spec = str(backend_spec or "")
    if "modal.run" in spec or spec.startswith("vllm:"):
        hosted = str(os.environ.get("VLLM_API_KEY") or "").strip()
        if hosted:
            return hosted
    return None


def _tool_names(tools: Sequence | None) -> list[str]:
    names: list[str] = []
    for schema in tools or []:
        fn = schema.get("function", schema) if isinstance(schema, dict) else {}
        name = str(fn.get("name") or "")
        if name:
            names.append(name)
    return names


def _render_payload(trajectory: dict, *, policy: str = "",
                    tools: Sequence | None = None) -> str:
    steps = []
    for step in (trajectory.get("steps") or []):
        if not isinstance(step, dict):
            continue
        item = {}
        if step.get("tool"):
            item["tool"] = step.get("tool")
            item["arguments"] = step.get("arguments")
            item["result"] = step.get("result")
        if step.get("text"):
            item["text"] = step.get("text")
        if item:
            steps.append(item)
    conduct = trajectory.get("conduct") or {}
    if not conduct and trajectory.get("reward") is not None:
        conduct = {
            "reward": trajectory.get("reward"),
            "reason": trajectory.get("grader_reason") or trajectory.get("reason"),
        }
    # Verdict-critical fields serialize before steps so an oversized
    # payload loses step 14, never the final answer or the rules.
    blob = {
        "tools": _tool_names(tools),
        "user_request": str(trajectory.get("prompt", ""))[:4000],
        "final_text": str(trajectory.get("final_text", ""))[:2000],
        "conduct_score": conduct.get("reward"),
        "conduct_reason": conduct.get("reason"),
    }
    if policy.strip():
        blob["agent_policy"] = policy.strip()[:2000]
    blob["steps"] = steps
    return json.dumps(blob, default=str)[:8000]


def _parse_score(text: str) -> tuple[float | None, str | None]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None, None
    try:
        payload = json.loads(cleaned.removeprefix("```json").removesuffix("```").strip())
        if isinstance(payload, dict):
            score = payload.get("score", payload.get("reward"))
            reason = payload.get("reason")
            if score is not None:
                value = float(score)
                if value < 0:
                    value = 0.0
                if value > 1:
                    value = 1.0
                return value, str(reason or "").strip() or None
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    match = re.search(
        r'"(?:score|reward)"\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?|0\.5)',
        cleaned)
    if match:
        value = float(match.group(1))
        reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', cleaned)
        reason = reason_match.group(1).strip() if reason_match else None
        return value, reason
    return None, None


def judge_one(trajectory: dict, *, policy: str = "",
              tools: Sequence | None = None,
              backend_spec: str | None = None,
              api_key: str | None = None,
              timeout: float = 45) -> dict[str, Any]:
    """Score one trajectory. Returns llm_reward/llm_reason or both None."""
    spec = backend_spec or DEFAULT_JUDGE_SPEC
    url, model = parse_backend_spec(spec)
    payload = _render_payload(trajectory, policy=policy, tools=tools)
    try:
        reply = complete(
            url, model,
            [{"role": "system", "content": JUDGE_SYSTEM},
             {"role": "user", "content": payload}],
            api_key=api_key, temperature=0.1, max_tokens=120, timeout=timeout)
    except OSError:
        return {"llm_reward": None, "llm_reason": None}
    except Exception:
        return {"llm_reward": None, "llm_reason": None}
    score, reason = _parse_score(str(reply.get("content") or ""))
    if score is None:
        return {"llm_reward": None, "llm_reason": None}
    return {"llm_reward": score, "llm_reason": reason or "llm graded"}


def apply_llm_grade(trajectories: Sequence[dict], *,
                    policy: str = "",
                    tools: Sequence | None = None,
                    backend_spec: str | None = None,
                    api_key: str | None = None,
                    concurrency: int = 16,
                    degraded: list[str] | None = None) -> dict[str, Any]:
    """Write llm_reward/llm_reason onto each row. Never touches reward."""
    import concurrent.futures

    rows = list(trajectories)
    if not rows:
        return {"status": "empty", "graded": 0, "unreachable": 0}
    key = resolve_judge_key(api_key, backend_spec)

    def one(row: dict) -> dict[str, Any]:
        conduct = conduct_snapshot(row)
        return judge_one({**row, "conduct": conduct}, policy=policy,
                         tools=tools, backend_spec=backend_spec, api_key=key)

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(concurrency))) as pool:
        verdicts = list(pool.map(one, rows))

    graded = 0
    unreachable = 0
    for row, verdict in zip(rows, verdicts):
        reward = verdict.get("llm_reward")
        reason = verdict.get("llm_reason")
        row["llm_reward"] = reward
        row["llm_reason"] = reason
        if reward is None:
            unreachable += 1
        else:
            graded += 1

    if unreachable and degraded is not None:
        note = "llm_judge_unreachable"
        if note not in degraded:
            degraded.append(note)

    return {
        "status": "judged" if graded else "unreachable",
        "graded": graded,
        "unreachable": unreachable,
        "backend": backend_spec or DEFAULT_JUDGE_SPEC,
    }


def conduct_snapshot(row: dict) -> dict[str, Any]:
    """Deterministic conduct context for the judge prompt."""
    if row.get("reward") is not None:
        return {
            "reward": row.get("reward"),
            "reason": row.get("grader_reason") or row.get("reason"),
        }
    return {}
