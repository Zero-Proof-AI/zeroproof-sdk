# Supplying the agent's world

Read this reference when the agent's tools touch code, files, a database, or
another stateful system whose real behavior determines correctness.

## Choose the world

The SDK's built-in world invents schema-shaped records and is useful for
conduct evals over ordinary business tools. It is not a source of truth for
code: invented source files and invented test output cannot establish whether
a patch works.

For a coding agent, reuse the customer's existing eval harness. Give every
rollout its own disposable checkout and pass a callable through `execute=`:

```python
import inspect
import zeroproof_simulations as zps

if "execute" not in inspect.signature(zps.simulate).parameters:
    raise SystemExit("upgrade zeroproof: coding evals require simulate(execute=...)")

def execute(tool_name: str, arguments: dict):
    return harness.call(tool_name, arguments)

data = zps.simulate(
    agent="openai:<model>",
    tools=tools,
    system_prompt=policy,
    traces=traces,
    execute=execute,
    mode="explore",
    budget=40,
    time_budget=150,
    grade=False,
    output="new_evals/trace_guided.jsonl",
)
```

The callable receives the declared tool name and its argument dictionary. It
should return the exact result the agent would receive. Exceptions are recorded
as tool errors, but repeated exceptions indicate a broken adapter and should
stop the run.

## Required properties for coding evals

- Create one isolated checkout per rollout. Never share mutable files between
  concurrent rollouts.
- Resolve all file paths beneath that checkout and reject traversal outside it.
- Return real file contents, stdout, stderr, exit codes, and timeouts.
- Allow only the commands the customer's harness already permits. Do not add
  network access, package installation, credentials, or host-level writes just
  to make a generated row succeed.
- Keep hidden tests and expected outcomes unavailable to the rollout model.
- Determine correctness with the customer's real tests or end-state checker.
- Preserve the tool calls and results in the row so the verdict is auditable.
- Clean up checkouts after grading, including failed and timed-out rollouts.

The SDK exposes `zeroproof_simulations.generate.agents.current_rollout`, a
thread-local containing `prompt`, `rollout_index`, and `seed`. A concurrent
adapter can use `(prompt, rollout_index)` to select the correct checkout.

## Before keeping the data

Run a small smoke test first. Confirm that two rollouts of the same prompt use
different checkouts, a missing file returns a real not-found result, path escape
is refused, a failing command reports its nonzero exit code, and a known repair
passes the hidden checker.

For generated output:

1. Reject blocking generation failures described in `SKILL.md`.
2. Confirm every tool call was answered by the intended checkout.
3. Reject rows that claim tests passed without a recorded successful test run.
4. Reject rows that modify tests or other protected files.
5. Review every generated failure and at least 20 passing rows.

If the customer has no isolated execution harness, report that as the missing
prerequisite. Do not silently fall back to invented code or canned test output.
