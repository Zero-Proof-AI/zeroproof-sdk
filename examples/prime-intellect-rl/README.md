# Synthetic RL data for Prime Intellect

Generate a GRPO-ready dataset for a coding agent, then check it carries gradient
before spending GPU time on it.

The agent in this example is a quant research assistant working in a Python repo
that uses pandas and yfinance. Swap `spec.json` for your own tools and policy and
the rest of the pipeline is unchanged.

## Run it

```bash
export VLLM_API_KEY=...            # ask ZeroProof for a key
uv run python generate.py --situations 100 --k 8
uv run python diagnose.py data/rl.jsonl
uv run python export_prompts.py data/rl.jsonl
```

About three minutes for 800 rollouts.

## What each step does

**`generate.py`** runs `simulate(mode="rl")`. That topology gives every prompt the
same number of rollouts, which is what GRPO needs: it scores a rollout against the
other rollouts of the same prompt, so the training unit is a group of k, not a row.

Pass `situations=` explicitly. Without it the generator seeds probes from the
spec's `situations` list and can starve well short of the row cap.

**`diagnose.py`** is the gate. Mean reward is close to useless on its own, so it
reports four things instead:

| Check | Why it matters |
|---|---|
| uniform groups | every prompt needs the same k, or advantages are not comparable |
| live groups | a group whose k rollouts all score the same has zero advantage and contributes no gradient |
| within-group std | how much signal the live groups actually carry |
| effort correlation | `corr(tool calls, reward)`. Negative means the reward pays the policy to do less |

**`export_prompts.py`** writes the dataset in the shape `verifiers` reads:

```python
TASK_INPUT_FIELDS = {"prompt", "answer", "info", "example_id"}
```

`answer` is optional. `info` is a free-form dict passed to every reward function,
which is where scenario ground truth goes.

It exports prompts only. The trainer regenerates rollouts from the policy being
trained via `rollouts_per_example`; the rollouts in `rl.jsonl` came from a
different model and are off-policy. They are useful as a baseline eval or a
rejection-sampling SFT warm start, not for GRPO.

It keeps one phrasing per situation. The generator writes several phrasings of
some situations and one of others, so keeping all of them would give a few
situations several times the gradient weight of the rest.

It also drops seed probes. Every entry in the spec's `situations` list becomes a
probe, and some reach the output as the seed text itself. Those are third-person
scenario descriptions ("the user asks to fill missing bars forward"), not things a
user would type, and they make broken tasks. On the run below this removed 7 of
100 prompts. Pass `--keep-seeds` to leave them in.

## Measured on this spec

100 prompts, k=8, 800 rollouts, `Qwen3-4B-Instruct` on both roles.

| `fault_rate` | live groups | within-group std | mean reward | effort corr |
|---|---|---|---|---|
| 0.50 | 71% | 0.345 | 0.526 | -0.465 |
| 0.15 | **77%** | **0.378** | 0.576 | -0.451 |

Groups were uniform at k=8 in both runs. 77% live is a healthy dataset: roughly
three of every four prompts produce a usable advantage.

`export_prompts.py` turned the 100 prompts into 79 tasks: 7 dropped as seed
probes, 14 as duplicate phrasings of a situation already covered.

## Read the effort correlation before you train

Both runs score around -0.45. Mean reward by tool calls, at `fault_rate=0.5`:

```
0 tools  0.82      3 tools  0.33
1 tool   0.88      4+ tools 0.33
```

The default `conduct_grade` is a process reward. It checks whether the agent
invented an identifier, claimed success after a tool failed, or repeated itself.
It does not check whether the agent accomplished anything, so a reply with no tool
calls scores 1.0. Meanwhile every additional tool call is another chance to meet an
injected fault, and an unacknowledged fault is a hard zero. Effort buys risk and
earns nothing.

Lowering `fault_rate` does not fix this (-0.465 to -0.451). It is structural.

`conduct_grade` is doing its job. It is an honesty floor, and it is good at that:
it catches invented file paths and fabricated test results, which is most of what
you want to rule out in a coding agent. It is not a task reward, and GRPO needs a
task reward.

## Add an outcome term in the environment

Build the reward on Prime Intellect's side, where the task is known:

```python
import verifiers as vf

def load_environment(**kwargs):
    return vf.ToolEnv(
        dataset=load_prompts("data/prompts.jsonl"),
        tools=mock_tools(),                     # wrap sandbox.MockEnvironment
        rubric=vf.Rubric(
            funcs=[found_defect, task_complete, conduct],
            weights=[0.5, 0.3, 0.2],
        ),
        max_turns=10,
    )
```

`found_defect` and `task_complete` read the seeded ground truth out of `info`.
`conduct` is a port of `conduct_grade` with the fault penalty clipped to a floor
rather than a hard zero, so meeting a broken tool costs less than lying about one.

Then rerun `diagnose.py` against a rollout dump from the new rubric and ship only
when the effort correlation is positive. That is the gate this example exists to
make cheap.

`sandbox.MockEnvironment` is seeded on a hash of the tool name and arguments, so
all k rollouts of a prompt see an identical world. That determinism is what makes
the within-group comparison meaningful.

## Files

| | |
|---|---|
| `spec.json` | tools, policy rules, and situation seeds for the quant coding agent |
| `generate.py` | `simulate(mode="rl")` with uniform group size |
| `diagnose.py` | gradient and support report, exits non-zero on a bad dataset |
| `export_prompts.py` | verifiers-shaped prompt set |
