# Working in whileai-sdk

## Shipping a release

One command, run from any checkout, by anyone with write access:

```bash
gh workflow run release.yml
```

It cuts the next hundredth from whatever sits under `## Unreleased` in
CHANGELOG.md, lands the bump on main, and starts publish.yml. Runs queue
on a concurrency group, so two agents shipping in the same minute get two
releases in order, or one release and one "nothing to ship". Watch it with
`gh run watch` and confirm with `pip index versions whileai`.

Rules the tooling enforces (CI fails otherwise):

- Never bump `version` by hand. A PR that bumps it may touch only
  CHANGELOG.md, both pyprojects and uv.lock.
- Every change ships with an entry under `## Unreleased`. Keep that header;
  the cut renames it and puts a fresh one above.
- To see what a cut would ship without doing it:
  `uv run python .github/scripts/release.py --dry-run`

## Pull requests

- `gh pr create` fails on this repo ("must be a collaborator"); open PRs
  through the API: `gh api -X POST repos/whilehq/whileai-sdk/pulls -f title=.. -f head=<branch> -f base=main -F body=@body.md`
- A PR that touches `whileai/` must touch `docs/` too, or carry the
  `no-docs` label.
- Add files by path. Never `git add -A`; other sessions share this checkout.

## Checks

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run mypy            # the gate
uv run ty check        # same, under a second; mypy stays the gate until ty is 1.0
uv run python scripts/check_no_hardcoding.py
```
