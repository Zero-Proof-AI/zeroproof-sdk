# Changelog

Versions move in hundredths (`0.04` then `0.05`). PyPI normalizes them, so
`pip install zeroproof==0.4` is the `0.04` line below.

## Unreleased

- `simulate(reproducible=True)`: round-synchronous scheduling. Same seed,
  same concurrency, same agent gives the same rows at any concurrency.
  Costs throughput under uneven latency and needs the clock off.
- README leads with bring-your-own-model; hosted Qwen is the fallback.
- This changelog.

## 0.04 (2026-09-10)

- `zeroproof` is trace ingest only. The pre-0.3 encrypted agent-to-agent
  messaging client (`ZeroProof`, `send_encrypted`, reputation, approval
  workflows) is gone. Pin `zeroproof<0.3` to keep it.
- `simulate()` is a `run/` package with named phases (inputs, build, loop,
  finish). Behavior unchanged; verified row-for-row on a 17-configuration
  serial harness.
- A seeded serial run reproduces bit-for-bit across processes. Scenario ids
  and the axis-gap hint no longer depend on Python's per-process hash.
- A stop (clock, cap, saturation) cancels queued rollouts, waits up to
  `advanced["stop_grace"]` (5 s) for running ones, keeps what finishes, and
  reports the rest as `rollouts_abandoned`. Nothing calls the agent after
  `simulate()` returns.
- Progress goes through the `zeroproof_simulations` logger instead of
  `print()`.
- `zeroproof_simulations` ships `py.typed`.
- Trace ingest defaults to `https://api.zeroproofai.com`.
- README parameter tables match the code (`concurrency` 32, `time_budget`
  `None`) and a test keeps them matching.
- A failed tool draft for a prompt-only agent is a `tool_draft_unavailable`
  degraded note instead of a silent tool-free run.
- Two timing-dependent tests made deterministic. `zeroproof.__version__`
  reads package metadata.

## 0.3 (2026-08-31)

- The `zeroproof` package absorbed `zeroproof-simulations`, which is
  deprecated on PyPI. Releases of `zeroproof` before 0.3 were an unrelated
  encrypted messaging client.
