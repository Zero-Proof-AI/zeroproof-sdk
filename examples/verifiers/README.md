# Verifiers: verifiable rewards

A verifier is a reward that is a program, not an opinion (RLHF book ch. 7
Reasoning, ch. 13 Tools/RLVR). It reads a rollout and returns pass, fail, or
a partial score, with no model call. Every verifier honors the judge
contract, so it plugs into `data.grade(judge=...)`, `evaluate`, `optimize`
and a gated `push` exactly where an LLM judge would.

```
python examples/verifiers/run.py
```

No key, no model. The script shows math (`MathEqual`), an answer-and-format
gate (`All([...])`), code execution against hidden tests (`CodeExec`), and a
JSON-schema check (`JSONSchema`).

## The pieces

```python
from zeroproof.simulations.verify import (
    ExactMatch,
    Includes,
    Regex,
    MultipleChoice,  # text
    Numeric,
    MathEqual,  # math
    JSONValid,
    JSONSchema,
    JSONField,  # structured output
    CodeExec,  # run code against tests
    All,
    Any,
    Weighted,
    verifier,  # compose / wrap
)
```

- **Where the answer comes from.** The candidate is the rollout's
  `final_text` (or the last assistant turn). The gold is read from the row's
  `privileged.reference` — which the training export never projects, so the
  answer key cannot leak into a training file — with flat fields (`answer`,
  `target`, `solution`, ...) as a fallback. Point any verifier at another
  column with `field=`.
- **Compose.** `All` needs every check to pass (right answer *and* right
  format), `Any` needs one, `Weighted` is a graded rubric in [0, 1].
- **Your own.** `@verifier def f(candidate, reference, row): ...` returns a
  bool or a score, or a `(score, reason)` pair.

## In the loop

```python
import zeroproof.simulations as zps
from zeroproof.simulations.verify import MathEqual

data = zps.simulate(spec="specs/math", mode="rl", situations=200, repeats=8)
scored = data.grade(judge=MathEqual())  # the verifier IS the reward
rows, _ = zps.optimize(scored, mode="rl")  # GRPO data, gradient checked
zps.push_rows(rows, "math-rl-v1", gate=True, mode="rl")
```

## Code execution safety

`CodeExec` runs the candidate in a fresh subprocess with isolated mode, a
private temp directory, a wall-clock timeout, and CPU/memory caps on POSIX.
That stops runaway loops and accidents. It is **not** a security boundary
against hostile code — for untrusted policies, run the verifier inside a
container or the hosted sandbox.
