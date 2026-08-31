"""Simulate tab. Start a zps.simulate job and poll *.progress.json.

Hosted Qwen reads VLLM_API_KEY from process env (.env at boot). The
browser does not paste that key. Own-model jobs may send base_url,
model, and an optional api_key in the JSON body. Nothing here writes
keys to disk or logs.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from collections import defaultdict
from pathlib import Path

STUDIO = Path(__file__).resolve().parent.parent
ROOT = STUDIO.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeroproof_simulations import simulate  # noqa: E402
from zeroproof_simulations.agents import (  # noqa: E402
    HOSTED_DROPPED, hosted_model, local_model, public_llm_error,
)

ENDPOINT_DOWN = "The endpoint isn't up"

AGENTS = ROOT / "outputs" / "studio-runs" / "agents"
JOBS: dict[str, dict] = {}
SECRETS: dict[str, dict] = {}
LOCK = threading.Lock()
SECRET_KEYS = ("vllm_key", "api_key", "VLLM_API_KEY", "key", "authorization")
DEFAULT_CONCURRENCY = 16
HOSTED_MAX_CONCURRENCY = 32
MODES = {"explore", "sft", "rl", "adaptive"}

__all__ = ["start_job", "start", "job_status", "studio_error", "ENDPOINT_DOWN"]


def _agent_name(raw) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(raw or "").strip())
    s = s.strip("-._")[:64]
    return s or ""


def _norm_tags(raw) -> list[str]:
    if isinstance(raw, str):
        parts = re.split(r"[,;\n]+", raw)
    elif isinstance(raw, list):
        parts = [str(x) for x in raw]
    else:
        parts = []
    out, seen = [], set()
    for part in parts:
        tag = re.sub(r"\s+", " ", part.strip())[:40]
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out[:24]


def _agent_dir(name: str) -> Path:
    return AGENTS / name


def _harness_path(name: str) -> Path:
    return _agent_dir(name) / "harness.json"


def _runs_dir(name: str) -> Path:
    return _agent_dir(name) / "runs"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _redact(text, secrets: list[str]) -> str:
    out = str(text or "")
    for secret in secrets:
        s = str(secret or "").strip()
        if len(s) >= 4:
            out = out.replace(s, "[redacted]")
    return out


def _tools_and_policy_hint(raw) -> tuple[list, str, str | None]:
    """Return tools, optional policy found inside a spec blob, and an error."""
    policy = ""
    if raw is None or raw == "":
        return [], "", "paste tools JSON"
    parsed = raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return [], "", "paste tools JSON"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return [], "", f"tools JSON: {exc}"
    if isinstance(parsed, dict):
        policy = str(parsed.get("policy") or parsed.get("system_prompt") or "")
        if "tools" in parsed:
            tools = parsed.get("tools") or []
        else:
            return [], policy, "tools must be a JSON list or a spec with a tools key"
    elif isinstance(parsed, list):
        tools = parsed
    else:
        return [], "", "tools must be a JSON list or a spec with a tools key"
    if not isinstance(tools, list) or not tools:
        return [], policy, "tools list is empty"
    return list(tools), policy, None


def _load_harness(name: str) -> dict:
    path = _harness_path(name)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _write_harness(name: str, tools: list, policy: str) -> Path:
    folder = _agent_dir(name)
    folder.mkdir(parents=True, exist_ok=True)
    _runs_dir(name).mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "tools": tools,
        "policy": policy,
    }
    path = _harness_path(name)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _session_hosted_key() -> str:
    try:
        from api import auth
        key = str(auth.hosted_key() or "").strip()
        if key:
            return key
    except Exception:
        pass
    return str(os.environ.get("VLLM_API_KEY") or "").strip()


def _hosted_ready(*, fresh: bool = False) -> bool:
    try:
        from api import auth
        return bool(auth.hosted_ready(fresh=fresh))
    except Exception:
        return False


def studio_error(exc: BaseException | str | None) -> str:
    """Studio headline. Modal-down is never a stack trace or 'lost track of input'."""
    text = public_llm_error(exc)
    low = str(text or "").lower()
    if ENDPOINT_DOWN.lower() in low:
        return ENDPOINT_DOWN
    needles = (
        "lost track of input",
        "internalfailure",
        "modal-http",
        "hosted qwen dropped",
        "hosted not connected",
        "vllm_api_key",
        "stressd-vllm",
        "connection refused",
        "connection reset",
        "timed out",
        "timeout",
        "name or service not known",
        "nodename nor servname",
        "failed to establish",
        "temporarily unavailable",
        "max retries exceeded",
        "network is unreachable",
    )
    if any(n in low for n in needles):
        return ENDPOINT_DOWN
    if "modal.run" in low and any(
            code in low for code in (" 500", " 502", " 503", " 504", "returned 5")):
        return ENDPOINT_DOWN
    if text == HOSTED_DROPPED:
        return ENDPOINT_DOWN
    return text


def _lookup_agent(name: str) -> dict | None:
    try:
        from api import agents
        row = agents.get_agent(name)
        if isinstance(row, dict) and row.get("error") == "not found":
            return None
        if isinstance(row, dict) and (row.get("tools") or row.get("policy")):
            return row
    except Exception:
        pass
    harness = _load_harness(name)
    return harness or None


def _save_new_agent(name: str, tools: list, policy: str) -> dict | None:
    try:
        _write_harness(name, tools, policy)
        return None
    except OSError as exc:
        return {"error": f"could not write harness: {exc}"}


def _agent_exists(name: str) -> bool:
    if _lookup_agent(name):
        return True
    if _harness_path(name).is_file():
        return True
    if _runs_dir(name).is_dir() and any(_runs_dir(name).glob("*.jsonl")):
        return True
    return False


def _strip_secrets(meta: dict) -> dict:
    out = dict(meta)
    for key in SECRET_KEYS + ("tools", "system_prompt", "policy", "api_key"):
        out.pop(key, None)
    return out


def _read_meta(path: Path) -> dict:
    mp = path.with_suffix(".run.json")
    if not mp.is_file():
        return {}
    try:
        data = json.loads(mp.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_meta(path: Path, extra: dict) -> dict:
    meta = _read_meta(path)
    meta.update(extra)
    meta = _strip_secrets(meta)
    if path.is_file():
        meta["mtime"] = int(path.stat().st_mtime)
        meta["path"] = _rel(path)
    meta["tags"] = _norm_tags(meta.get("tags"))
    name = _agent_name(meta.get("agent"))
    if name:
        meta["agent"] = name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".run.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _bin(row: dict) -> int | None:
    try:
        r = float(row.get("reward"))
    except (TypeError, ValueError):
        return None
    if r <= 0:
        return 0
    if r >= 1:
        return 1
    return None


def _stats(path: Path) -> dict:
    n = n0 = n1 = 0
    total = 0.0
    by: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    if not path.is_file():
        return {"n": 0, "n0": 0, "n1": 0, "mean_reward": None, "prompts": 0, "n_split": 0}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {"n": 0, "n0": 0, "n1": 0, "mean_reward": None, "prompts": 0, "n_split": 0}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        n += 1
        prompt = str(row.get("prompt") or "")
        b = _bin(row)
        if b == 0:
            n0 += 1
            by[prompt][0] += 1
        elif b == 1:
            n1 += 1
            by[prompt][1] += 1
        else:
            by[prompt]
        try:
            total += float(row.get("reward") or 0)
        except (TypeError, ValueError):
            pass
    return {
        "n": n,
        "n0": n0,
        "n1": n1,
        "mean_reward": round(total / n, 4) if n else None,
        "prompts": len(by),
        "n_split": sum(1 for a, b in by.values() if a > 0 and b > 0),
    }


def _read_progress(out_path: Path) -> dict:
    p = Path(str(out_path) + ".progress.json")
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _tail(path: Path, n: int = 12) -> list[dict]:
    if not path.is_file():
        return []
    try:
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    except OSError:
        return []
    out = []
    for line in lines[-n:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        out.append({
            "prompt": str(row.get("prompt") or "")[:200],
            "reward": row.get("reward"),
            "reason": str(row.get("reason") or "")[:120],
            "final_text": str(row.get("final_text") or "")[:160],
        })
    return out


def _int(raw, default: int | None = None, lo: int | None = None,
         hi: int | None = None) -> int | None:
    if raw is None or raw == "":
        return default
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return default
    if lo is not None:
        n = max(lo, n)
    if hi is not None:
        n = min(hi, n)
    return n


def _float(raw, default: float | None = None, lo: float | None = None,
           hi: float | None = None) -> float | None:
    if raw is None or raw == "":
        return default
    try:
        n = float(raw)
    except (TypeError, ValueError):
        return default
    if lo is not None:
        n = max(lo, n)
    if hi is not None:
        n = min(hi, n)
    return n


def _execute(job_id: str, spec: dict) -> None:
    with LOCK:
        held = dict(SECRETS.pop(job_id, {}) or {})
    secrets = [str(v) for v in held.values() if v] + [_session_hosted_key()]
    out_path = Path(spec["output"])
    try:
        with LOCK:
            if job_id in JOBS:
                JOBS[job_id]["status"] = "running"
        tools = spec["tools"]
        policy = spec["system_prompt"]
        mode = spec["mode"]
        budget = spec.get("budget")
        time_budget = spec.get("time_budget")
        concurrency = int(spec.get("concurrency") or DEFAULT_CONCURRENCY)
        if (spec.get("brain") or "hosted") == "hosted":
            concurrency = min(concurrency, HOSTED_MAX_CONCURRENCY)
        advanced = {
            "concurrency": concurrency,
            "fault_rate": float(spec.get("fault_rate") if spec.get("fault_rate") is not None else 0.5),
            "avg_turns": float(spec.get("avg_turns") or 4),
            "embedder": "hash",
        }
        extra: dict = {}
        if spec.get("phrasings"):
            extra["phrasings"] = int(spec["phrasings"])
        if spec.get("repeats"):
            extra["repeats"] = int(spec["repeats"])
        elif mode == "rl":
            extra["repeats"] = 4
        brain = spec.get("brain") or "hosted"
        if brain == "hosted":
            if not _hosted_ready(fresh=True):
                raise ValueError(ENDPOINT_DOWN)
            key = _session_hosted_key()
            if not key:
                raise ValueError(ENDPOINT_DOWN)
            agent = hosted_model(tools, system=policy, api_key=key, timeout=60)
        else:
            url = str(spec.get("base_url") or "").strip()
            model = str(spec.get("model") or "model").strip() or "model"
            key = str(held.get("api_key") or "").strip()
            if not url:
                raise ValueError("base URL is required for your model")
            agent = local_model(
                url, model, tools=tools, system=policy,
                api_key=key or None, timeout=60)
        simulate(
            agent=agent,
            tools=tools,
            system_prompt=policy,
            budget=budget,
            time_budget=time_budget,
            mode=mode,
            grade=False,
            output=str(out_path),
            advanced=advanced,
            **extra,
        )
        stats = _stats(out_path)
        _write_meta(out_path, {
            "agent": spec.get("agent") or "",
            "tags": spec.get("tags") or [],
            "mode": mode,
            "status": "done",
            "brain": brain,
            "grade": bool(spec.get("grade", True)),
            "concurrency": concurrency,
            **stats,
        })
        with LOCK:
            job = JOBS.get(job_id)
            if job:
                job["status"] = "done"
                job["error"] = None
                job.update(stats)
        # Record token usage: estimate from number of completed simulation rows.
        # Each row represents one agent rollout consuming model tokens.
        # Replace with real token counts when zeroproof_simulations exposes usage.
        api_key = str(spec.get("_api_key") or "").strip()
        n_rows = stats.get("n") or 0
        if api_key and n_rows > 0:
            try:
                from token_gate import record_usage
                in_tokens = n_rows * 500
                out_tokens = n_rows * 200
                record_usage(api_key, input_tokens=in_tokens, output_tokens=out_tokens)
                print(f"usage-recorded rows={n_rows} input={in_tokens} output={out_tokens}", flush=True)
            except Exception as usage_exc:
                print(f"usage-record-failed rows={n_rows} error={usage_exc}", flush=True)
        else:
            print(f"usage-record-skipped api_key={bool(api_key)} rows={n_rows}", flush=True)
    except Exception as exc:
        err = studio_error(_redact(str(exc), secrets))
        trace = _redact(traceback.format_exc()[-2000:], secrets)
        with LOCK:
            job = JOBS.get(job_id)
            if job:
                job["status"] = "error"
                job["error"] = err
                job["trace"] = trace
        try:
            _write_meta(out_path, {
                "agent": spec.get("agent") or "",
                "tags": spec.get("tags") or [],
                "mode": spec.get("mode"),
                "status": "error",
                "error": err,
            })
        except OSError:
            pass


def start_job(body: dict | None = None) -> dict:
    body = body if isinstance(body, dict) else {}
    source = str(body.get("source") or "existing").strip().lower()
    if source not in {"existing", "new"}:
        source = "existing"

    agent = _agent_name(body.get("agent") or body.get("name") or body.get("id"))
    if not agent:
        return {"error": "agent name is required"}

    tools, spec_policy, tools_err = _tools_and_policy_hint(body.get("tools"))
    policy = str(body.get("system_prompt") or body.get("policy") or "").strip()
    if not policy:
        policy = str(spec_policy or "").strip()

    stop = str(body.get("stop") or "time").strip().lower()
    if stop not in {"time", "rows"}:
        stop = "time"
    if stop == "rows":
        budget = _int(body.get("budget"), 200, lo=1, hi=100_000)
        if budget is None:
            return {"error": "rows must be a number"}
        time_budget = None
    else:
        time_budget = _float(body.get("time_budget"), 60.0, lo=1.0, hi=86_400.0)
        if time_budget is None:
            return {"error": "seconds must be a number"}
        budget = None

    mode = str(body.get("mode") or "rl").strip().lower()
    if mode not in MODES:
        return {"error": "mode must be explore, sft, rl, or adaptive"}

    brain = str(body.get("brain") or "hosted").strip().lower()
    if brain not in {"hosted", "own"}:
        brain = "hosted"
    request_api_key = str(body.get("_api_key") or "").strip()
    api_key = str(body.get("api_key") or "").strip() if brain == "own" else ""
    base_url = str(body.get("base_url") or "").strip() if brain == "own" else ""
    model = str(body.get("model") or "").strip() if brain == "own" else ""
    if "vllm_key" in body:
        body = dict(body)
        body.pop("vllm_key", None)
    if brain == "hosted" and not _hosted_ready(fresh=True):
        return {"error": ENDPOINT_DOWN}
    if brain == "own" and not base_url:
        return {"error": "base URL is required for your model"}

    if source == "existing":
        found = _lookup_agent(agent)
        if not found:
            return {"error": f"no saved agent named {agent}. create one first."}
        if tools_err:
            harness_tools = found.get("tools") or []
            if harness_tools:
                tools, tools_err = list(harness_tools), None
        if not policy:
            policy = str(found.get("policy") or found.get("system_prompt") or "").strip()
        if tools_err:
            return {"error": tools_err}
        if not policy:
            return {"error": "system prompt is required"}
    else:
        if _agent_exists(agent):
            return {"error": f"agent {agent} already exists. use existing agent to add a run."}
        if tools_err:
            return {"error": tools_err}
        if not policy:
            return {"error": "system prompt is required"}
        save_err = _save_new_agent(agent, tools, policy)
        if save_err:
            return save_err

    concurrency = _int(body.get("concurrency"), DEFAULT_CONCURRENCY, lo=1, hi=192)
    if concurrency is None:
        concurrency = DEFAULT_CONCURRENCY
    if brain == "hosted":
        concurrency = min(concurrency, HOSTED_MAX_CONCURRENCY)
    fault_rate = _float(body.get("fault_rate"), 0.5, lo=0.0, hi=1.0)
    avg_turns = _float(body.get("avg_turns"), 4.0, lo=1.0, hi=32.0)
    repeats = _int(body.get("repeats") if body.get("repeats") is not None else body.get("k"),
                   None, lo=1, hi=32)
    phrasings = _int(body.get("phrasings") if body.get("phrasings") is not None else body.get("n"),
                     None, lo=1, hi=32)
    if mode == "rl" and repeats is None:
        repeats = 4
    if mode != "rl":
        repeats = None
    if mode != "sft":
        phrasings = None
    grade = body.get("grade")
    if isinstance(grade, str):
        grade = grade.strip().lower() not in {"0", "false", "no", "off"}
    elif grade is None:
        grade = False
    else:
        grade = bool(grade)

    tags = _norm_tags(body.get("tags"))
    runs = _runs_dir(agent)
    runs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    job_id = uuid.uuid4().hex[:10]
    stem = f"{stamp}_{mode}_{job_id}"
    out_path = runs / f"{stem}.jsonl"
    output_rel = _rel(out_path)

    spec = {
        "tools": tools,
        "system_prompt": policy,
        "mode": mode,
        "agent": agent,
        "tags": tags,
        "stop": stop,
        "budget": budget,
        "time_budget": time_budget,
        "brain": brain,
        "base_url": base_url,
        "model": model,
        "concurrency": concurrency,
        "fault_rate": fault_rate,
        "avg_turns": avg_turns,
        "phrasings": phrasings,
        "repeats": repeats,
        "output": str(out_path),
        "grade": grade,
        "source": source,
        "_api_key": request_api_key,
    }
    _write_meta(out_path, {
        "agent": agent,
        "tags": tags,
        "mode": mode,
        "status": "queued",
        "n_tools": len(tools),
        "stop": stop,
        "budget": budget,
        "time_budget": time_budget,
        "path": output_rel,
        "mtime": int(time.time()),
        "n": 0,
        "n0": 0,
        "n1": 0,
        "mean_reward": None,
        "prompts": 0,
        "n_split": 0,
        "brain": brain,
        "grade": grade,
        "concurrency": concurrency,
        "source": source,
    })
    with LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "error": None,
            "output": output_rel,
            "agent": agent,
            "mode": mode,
            "stop": stop,
            "budget": budget,
            "time_budget": time_budget,
            "grade": grade,
            "started": time.time(),
            "n_tools": len(tools),
            "concurrency": concurrency,
            "repeats": repeats,
            "source": source,
        }
        if brain == "own" and api_key:
            SECRETS[job_id] = {"api_key": api_key}
    threading.Thread(target=_execute, args=(job_id, spec), daemon=True).start()
    return {
        "id": job_id,
        "output": output_rel,
        "agent": agent,
        "mode": mode,
        "status": "queued",
        "source": source,
    }


def start(body: dict | None = None) -> dict:
    return start_job(body)


def _recover_job(job_id: str) -> dict:
    """Reload a job from disk if the process forgot it (container recycle)."""
    if not AGENTS.is_dir():
        return {}
    needle = f"_{job_id}.jsonl"
    matches = sorted(AGENTS.glob(f"*/runs/*{needle}"))
    if not matches:
        return {}
    path = matches[-1]
    meta = _read_meta(path)
    stats = _stats(path) if path.is_file() else {}
    status = str(meta.get("status") or "running")
    if status == "queued" and stats.get("n"):
        status = "running"
    job = {
        "id": job_id,
        "status": status,
        "error": meta.get("error"),
        "output": _rel(path),
        "agent": meta.get("agent") or path.parent.parent.name,
        "mode": meta.get("mode"),
        "stop": meta.get("stop"),
        "budget": meta.get("budget"),
        "time_budget": meta.get("time_budget"),
        "grade": meta.get("grade"),
        "concurrency": meta.get("concurrency"),
        "repeats": meta.get("repeats"),
        "started": meta.get("mtime") or time.time(),
    }
    with LOCK:
        JOBS[job_id] = dict(job)
    return job


def job_status(job_id: str = "") -> dict:
    job_id = str(job_id or "").strip()
    if not job_id:
        return {"status": "idle"}
    with LOCK:
        job = dict(JOBS.get(job_id) or {})
    if not job:
        job = _recover_job(job_id)
    if not job:
        return {"status": "not_found", "id": job_id}
    out = ROOT / job["output"] if not Path(job["output"]).is_absolute() else Path(job["output"])
    progress = _read_progress(out)
    n = 0
    if out.is_file():
        try:
            n = sum(1 for line in out.open() if line.strip())
        except OSError:
            n = 0
    rows = progress.get("rows", n)
    if rows is None:
        rows = n
    payload = {
        "id": job_id,
        "status": job.get("status"),
        "error": studio_error(job.get("error")) if job.get("error") else None,
        "output": job.get("output"),
        "agent": job.get("agent"),
        "mode": job.get("mode"),
        "stop": job.get("stop"),
        "budget": job.get("budget"),
        "time_budget": job.get("time_budget"),
        "grade": job.get("grade"),
        "rows": rows,
        "stage": progress.get("stage") or job.get("status"),
        "total_s": progress.get("total_s"),
        "inflight": progress.get("inflight"),
        "writers": progress.get("writers"),
        "unused": progress.get("unused"),
        "scenario_s": progress.get("scenario_s"),
        "rollout_s": progress.get("rollout_s"),
        "tail": _tail(out),
        "concurrency": job.get("concurrency"),
        "repeats": job.get("repeats"),
    }
    if job.get("status") in {"done", "error"}:
        stats = _stats(out) if out.is_file() else {
            "n": rows, "n0": 0, "n1": 0, "mean_reward": None,
            "prompts": 0, "n_split": 0,
        }
        payload.update(stats)
        payload["rows"] = stats.get("n", rows)
    return payload
