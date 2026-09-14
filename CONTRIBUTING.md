# Contributing

## Setup

```bash
uv sync --extra dev
```

## The checks CI runs

```bash
uv run pytest -q            # the suite, on 3.10 / 3.11 / 3.12 / 3.13
uv run ruff check .
uv run ruff format --check .
uv run mypy
python -m build && python -m twine check dist/*   # if you touched packaging or imports
```

## Golden harness: proving an engine change left `simulate()` alone

Any change under `zeroproof/simulations/` that could move `simulate()` output
has to be shown to be output-preserving — or its diff has to be stated and
justified. `scripts/golden.py` runs 13 offline configurations at
`concurrency=1` on fixed seeds, scrubs the keys that cannot be reproducible
(wall-clock timings, and the `uuid4` scoring run id that `run_judge` stamps on
every call), and writes one JSON snapshot per configuration.

```bash
python scripts/golden.py capture /tmp/golden/before
# ... make the change ...
python scripts/golden.py capture /tmp/golden/after
python scripts/golden.py diff /tmp/golden/before /tmp/golden/after
```

`diff` exits 0 when every configuration is byte-identical and 1 when any key
differs, naming the dotted path of each change. A whole capture takes about
five seconds and needs no API key or GPU. If the output does change and that
change is the point of the PR, say in the PR body exactly which keys moved and
why.

`scripts/` is otherwise gitignored — local helpers live there and stay local.
`golden.py` is the one committed exception.
