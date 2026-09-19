---
title: "Package layout and development"
sidebarTitle: "Development"
description: "Where the code lives, how to run the tests, lint and type checks, and what CI runs."
---

## Package layout

The public surface is the package itself: `import whileai.simulations as wai`. Internals are grouped by stage and may move between releases.

| folder or file | what lives there |
|---|---|
| `generate/` | situation grid, writer, diversity selection, agent runners and adapters |
| `score/` | conduct checks, judges, quality ranking, selection for SFT and RL, the trust and delta reports |
| `verify/` | verifiers: programmatic rewards (`MathEqual`, `CodeExec`, `JSONSchema`, ...) that honor the judge contract |
| `ingest/` | trace loading, OpenTelemetry rows (`gen_ai.usage.*` sums into `row["usage"]`), platform push and pull |
| `world/` | the mock tool environment (`WorldOptions` in `sandbox.py`) |
| `run/` | the engine behind `simulate()`: knob resolution (`config.py`), spec loading (`spec.py`), row helpers (`rows.py`), and the scheduler itself (`engine.py`: inputs, build, loop, finish) |
| `simulation.py`, `data.py`, `export.py` | the `simulate()` entry point, its result object, and training export |
| `schema.py`, `schemas/` | the typed row (`Task`, `Rollout`, `Judgment`, `Marker`) and the `row-v1.json` wire contract |
| `defaults.py` | every default the engine and the reports use, each with the reason it is what it is |
| `environment.py` | `export_environment` and `load_environment`: a run as an installable RL environment |
| `training.py` | hosted training runs, `training_run`, `serve`, `reward_model` |
| `monitor.py` | `HackMonitor`, the during-training reward-hacking watch |

Source: [github.com/whilehq/whileai-sdk/tree/main/whileai/simulations](https://github.com/whilehq/whileai-sdk/tree/main/whileai/simulations).

## Development

```bash
uv sync --extra dev
uv run pytest           # under a minute on four cores, no network; -n0 runs it serially
uv run ruff check .     # lint; --fix for the mechanical ones
uv run mypy             # type check (the gate)
uv run ty check         # same check, under a second; mypy stays the gate until ty is 1.0
pre-commit install      # optional: ruff and whitespace hooks on commit
```

CI runs the suite on Python 3.10 through 3.13, every recipe's `smoke.sh`, ruff, mypy, ty, line coverage, a plain-pip install of the built wheel into a clean venv, and a version-scheme check.

## License

Apache-2.0
