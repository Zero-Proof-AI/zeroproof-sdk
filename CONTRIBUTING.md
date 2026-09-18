# Contributing

## Setup

```bash
uv sync --extra dev
```

## The checks CI runs

```bash
uv run pytest -q            # the suite, on 3.10 / 3.11 / 3.12 / 3.13; add -n0 for serial or --pdb
uv run ruff check .
uv run ruff format --check .
uv run mypy
python -m build && python -m twine check dist/*   # if you touched packaging or imports
```

## Contributing a recipe

Recipes are the part of this repo most worth adding to, and the easiest to
start with. A recipe is one runnable script plus a README that says what you
learn, what you need, and how long it takes. The conventions are in
[`recipes/README.md`](recipes/README.md#conventions); the shortest way in is to
copy [`recipes/_template/`](recipes/_template) into the step it belongs to and
replace the parts in angle brackets.

```bash
cp -r recipes/_template recipes/03-select/my-recipe
sh recipes/03-select/my-recipe/smoke.sh     # what CI will run
```

Two rules on top of the conventions:

- **Every recipe has an offline path.** `smoke.sh` runs the whole script with
  no key, no GPU and no spend, in under a minute — `--dry-run`, `--limit`,
  `--steps`, whatever fits. CI runs every `smoke.sh` in `recipes/` on every
  pull request, so a recipe that needs an A10G still gets its wiring checked
  by a machine.
- **Every measured claim is paired, with an interval, on a held-out set**
  (`pass_at`, `delta_report`). A mean alone is not a result.

A recipe that trains on Modal we verify ourselves on our own account before
merging, because GitHub does not give a fork's pull request access to a
repository's secrets — by design, and we are not working around it. Say in the
PR body what you ran and what it cost, and we will run it.

Found a recipe that does not work? Open an issue with the recipe path, the
command, and what happened. Bad recipes are bugs.

## Golden harness: proving an engine change left `simulate()` alone

Any change under `whileai/simulations/` that could move `simulate()` output
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
