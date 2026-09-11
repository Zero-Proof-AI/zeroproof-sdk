# Changelog

Versions move in hundredths (`0.04` then `0.05`). PyPI normalizes them, so
`pip install zeroproof==0.4` is the `0.04` line below.

## 0.08 (2026-09-11)

- Typed row schema, additive half. `zeroproof_simulations.schema` defines
  the four objects every row projects from (`Task`, `Rollout`, `Judgment`,
  `Marker`) plus `Dataset` and `Calibration`, with `from_row` / `to_row`
  between them and the flat JSONL row. Every row the engine, `save()`,
  the exporters, and `rows_from_otel` write now carries
  `schema_version: "1"`; `.meta.json` carries it too. The wire contract is
  `zeroproof_simulations/schemas/row-v1.json`, shipped in the wheel.
- Rows without a stamp are version 0 and are read by shape: engine rows
  (by `scenario_id`), platform trace pulls (by `tool_trace`), OTel ingest
  (by `conversation_id`). Nothing that loaded before is rejected.
- Validators run at the boundaries: `push_rows`, `training_rows` /
  `export_training`, `export_preference`, `rows_from_otel`, and the row
  writer behind both the streamed file and `save()`. In this version they
  check the stamp and the required field types only.
- A test freezes the count of direct verdict-key writes per module, so
  new grading paths go through `attach` once it lands.
- `SimulationData.trajectories` stays the source of truth; the objects
  are a view until the store moves.

## 0.07 (2026-09-11)

- A bring-your-own run with no key fails at setup with a message naming
  `OPENAI_API_KEY`, instead of spending its whole time budget on 401s and
  returning zero rows. Loopback and plain-http endpoints (ollama, local
  vLLM) still need no key.
- A stop raises a flag that every rollout and writer wave checks before
  starting, so nothing begins work after the stop is declared. Closes a
  race where a wave marked running could still start its body after
  `simulate()` returned.
- The two stop tests give the hanging agent and writer six seconds and
  assert a four-second return, so a loaded CI box cannot trip them.

## 0.06 (2026-09-10)

- A stop also settles writer waves: queued waves are cancelled, running
  ones get the same `stop_grace`, and any still running are reported as
  `writer_waves_abandoned`. Found by an end-to-end run of the 0.5 wheel
  against hosted Qwen, where four writer threads outlived `simulate()`.

## 0.05 (2026-09-10)

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
