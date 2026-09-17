# Evals for the agent you already have

You have an agent. You want a number that says how often it does the job,
an interval on that number, a table of where it fails, and a check that
turns red in CI when it gets worse. This page is that path, start to
finish. The runnable version is
[`recipes/02-measure/eval-your-agent`](../recipes/02-measure/eval-your-agent)
(offline, no key, seconds).

Names first, because they cost testers ten minutes: **zp, ZeroProof and
While are the same product.** The package is `whileai`
(`pip install whileai`; `pip install zeroproof` still works as a shim),
the import is `whileai.simulations`, API keys start with `zp_`, and the
docs live at zeroproofai.com until the domain moves.

## 1. Install and sign in

```bash
pip install whileai
whileai signup --email you@example.com   # new account, no browser; or: whileai login
whileai status                            # which key the SDK will use
```

Nothing below needs the key until you drop `simulator=False`. On a
machine that had the old package, `~/.zeroproof/credentials.json` is
still picked up; set `WHILEAI_HOME=/some/fresh/dir` to isolate a new
account from it.

## 2. Wrap your agent

The engine calls your function once per rollout with the ask the writer
produced, and wants back the tool calls it made and what it said:

```python
def agent(message: str) -> dict:
    calls = []  # your bot runs here, with its real tools
    reply = my_bot.answer(message, record=calls)  # each call: {"tool", "arguments", "result"}
    return {"steps": calls, "final_text": reply}
```

Three things to know about a callable agent:

- **It runs its own real tools.** The engine's mock world and scheduled
  faults apply to model-backed agents; your callable answers its own
  calls. That is what you want for an eval of the thing you ship.
- **It is played single-turn.** One message in, one trajectory out. The
  multi-turn user model (`avg_turns`, `max_turns`) does not apply.
- **The writer does not know your ids.** It reads the tool descriptions
  and the system prompt. If your world has order numbers or account
  names, put them in the tool description ("Orders on file: A1001,
  A1002, ...") or in `seeds=`, or the writer invents ids, every rollout
  is "not found", and the run is hollow.

Tools go in OpenAI function-calling shape (`{"type": "function",
"function": {"name", "description", "parameters"}}`); a bare
`{"name", "description", "parameters"}` dict works too. If your bot
records calls through a shared global, wrap the recorder in a
`threading.local`, because rollouts run concurrently.

## 3. Write the judge as a program

The judge reads the trajectory, not the prose. A polite reply that issued
a refund it should not have scores 0; a blunt one that followed the
policy scores 1. The policy lives in the judge, once:

```python
def judge(row: dict) -> dict:
    steps = row.get("steps") or []
    refunds = [s for s in steps if s.get("tool") == "issue_refund"]
    allowed = refundable(order_named_in(row["prompt"]))  # the policy, as a program
    ok = bool(refunds) == allowed
    return {
        "reward": 1.0 if ok else 0.0,
        "reason": "refunded" if refunds else "no refund",
        "markers": {"refund_only_when_allowed": 1.0 if ok else 0.0},
    }
```

The contract is `reward` in [0, 1], a `reason` string, and optional
`markers` (name to 0/1). Name markers so 1.0 is always the good outcome
(`refund_only_when_allowed`, not `refunded_wrongly`); the marker table
reads as one column then. A marker that does not apply to a row is
`None`, so its rate counts only the rows it measured. A verifier
(`wai.verify.*`) is a judge too, when the answer is checkable.

## 3b. Find what your tests miss

The suite you have sends a set of asks. Which parts of the policy do they
never reach?

```python
old_tests = [
    "I want a refund for order A1001, the shoes did not fit.",
    "What is the status of order A1001?",
    "Can you refund order Z9999?",
]
report = wai.coverage_gap(old_tests, tools=TOOLS, system_prompt=POLICY)
print(wai.format_coverage_gap(report))
# 3 asks cover 5 of 6 policy rules and 2 of 2 tools; untested: Refunds over
# $200 need a manager: ...; no ask puts the agent under pressure; every ask
# runs once
```

`asks` is a list of prompt strings, a list of rows with a `prompt` key, or
a path to a `.py` or `.jsonl` file holding either. From a `.py` file the
asks are the string literals that look like asks (passed to a call or in a
list, over fifteen characters, with a space): a heuristic, so read
`report["asks"]` before trusting the counts.

The axes are the ones `simulate` covers, so the report is in the engine's
own words: `untested_rules` are the policy clauses no ask reaches,
`untested_tools` the tools no ask names, `single_shot` says every ask runs
once (one rollout cannot tell a flake from a failure), and `notes` names
the fix for each. `world_state` and `tool_condition` are not readable from
an ask at all, which is the honest reason a hand-written suite misses
fault handling: a prompt never says the record is missing or the tool
timed out.

Rules are matched on the words an ask shares with the clause, so a branch
that only the fixture data selects (an amount, a date) reads as untested
even when an ask lands on it. Pass `rows=` from a graded run to check the
world side: a rule whose every row ended in the same tool fault is one the
asks reach but the fixtures never let happen, and the fix is a fixture
case, not another ask.

`preflight(tools, system_prompt)["rules"]` is the same rule axis on its
own, which is the list of policy branches the engine extracted from your
prompt.

## 4. Run it

```python
import whileai.simulations as wai

data = wai.simulate(
    agent,
    tools=TOOLS,
    system_prompt=POLICY,
    seeds=SEEDS,
    simulator=False,  # offline template writer: no key, seconds. Drop it for the hosted writer.
    mode="rl",
    repeats=4,
    repeat_policy="fixed",  # every ask, all four repeats
    reproducible=True,
)
scored = wai.evaluate(data, judge)  # stamped as eval: can never become the reward
print(wai.pass_at(scored.rows))  # pass@1 [interval], pass^k, pass@k
for note in scored.warnings:  # hollow-run checks; fix before reading the number
    print("!", note)
```

- `pass@1` is how often the agent does the job. `pass^k` is how often
  it did on every one of `k` tries: for anything that moves money, that
  is the number. `pass@k` minus `pass@1` is headroom for training.
- `repeat_policy="fixed"` asks for all repeats up front. The `mode="rl"`
  default, `"successive"`, stops early on unanimous asks, which is the
  right economy for training data and the wrong one for an eval.
- Slice by category: tag each row (`row["category"] = classify(prompt)`)
  and call `wai.pass_at` on each slice. The recipe prints that table.
- **Hollow runs.** If no rollout called a tool, a declared tool was never
  touched, or a marker fired on no row,
  `scored.warnings` says so and names the fix. A pass@1 of 1.00 on a run
  where the agent never reached its tools is not a result. Do not report
  a number from a run with warnings.

## 5. Gate CI on it

Two lanes. The slow one runs the agent (model calls, seconds to minutes)
and exits non-zero under a floor:

```bash
python evals/run.py --gate 0.9          # exit 1 when pass@1 < 0.90, exit 2 when the run is hollow
```

The fast one runs the judge alone on hand-labeled transcripts, in a
second, with no model calls, so a judge edit cannot drift unnoticed:

```python
def test_judge_catches_refund_outside_window():
    row = {"prompt": "Refund A1004", "steps": [lookup(A1004), refund(A1004)], "final_text": "Done."}
    assert judge(row)["reward"] == 0.0
```

## 6. Check the judge

A judge is a claim until it is measured. Label a sample by hand, attach
the labels as human, and ask:

```python
wai.attach_labels(scored.rows, labels, kind="human")  # labels: {rollout_id: 0/1} or a JSONL path
print(wai.format_judge_trust(wai.judge_trust(scored.rows, judge)))
```

`judge_trust` reads `gold_reward` (which `attach_labels` writes) and
reports agreement with its Wilson lower bound, held-out halves, a length
bias check and re-judge flips. **FAIL on eight labels means label more,
not that the judge is wrong:** the lower bound needs about thirty labels
to clear 0.8 even at perfect agreement. Labels attached any other way
count as model-made and keep `ok` false unless you say
`allow_model_gold=True`.

## 7. Then

- Every failure row is a training example: `simulate(traces=scored.failed_traces())`
  aims the next round at what broke. `evaluate` rows are stamped so the
  selectors refuse to use them as the reward (`eval_sourced`).
- Push the eval set with a purpose so it stays out of training:
  `scored.push("refund-evals", purpose="eval")`.
- The same markers score production traces over OpenTelemetry, so the
  number you got here is the number you watch after shipping.
