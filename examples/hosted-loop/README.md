# The hosted loop: push, train, serve, call

Four calls from graded rows to a chat completion from the trained model,
all on the platform. One key, one A10G run, no GPU of your own.

```bash
pip install zeroproof
zeroproof login                 # or export ZEROPROOF_API_KEY=...
python run.py                   # data -> train -> serve -> call
python run.py train --method sft --epochs 2   # any step alone; state is in hosted-loop.json
```

## What each step does

| step | call | what comes back |
|---|---|---|
| `data` | `simulate` (template writer, scripted agent), `run_judge`, `split_pseudo_production`, `push_rows` x2 | a train set and a task-disjoint holdout on the platform, with the publish gate's warnings |
| `train` | `zps.train(train_id, method="sft", base_model="Qwen/Qwen3-4B", holdout=..., wait=True)` | a finished run: held-out loss before and after, the adapter's location, the curve at `run.url` |
| `serve` | `zps.serve("hosted-loop", run)` | a model row: `endpoint` (OpenAI-compatible base URL) and `name` (the model id to send) |
| `call` | `POST {endpoint}/chat/completions` with the account key as bearer | the trained model's reply |

A run from today, 24 train rows over 7 tasks:

```
== train
started sft run run_726d53b769141506: https://www.zeroproofai.com/platform/training/run_726d53b769141506
done in 47s: loss 5.0094 -> 4.1311 on 72 held-out rows
adapter: volume zeroproof-train-runs:/run_726d53b769141506/adapter
== serve
serving hosted-loop v1 on Qwen/Qwen3-4B
endpoint https://zeroproofai--zeroproof-serve-qwen3-4b.modal.run/v1
== call
HTTP 200 in 287s
I cannot process your request. The order ID "88213" is not valid or does not exist in our system. Please provide a valid order ID, and I will assist you accordingly.
```

The rows here are small on purpose (a scripted agent, template situations)
so the loop finishes in minutes. The loss drop shows the wiring works; it
says nothing about the agent. Replace `scripted_agent` and `judge` with
yours, or point `data` at rows you already graded.

## What to know before you run it

- **Only two bases serve.** `Qwen/Qwen3-4B` and `microsoft/phi-4`. The
  trainer's defaults (Qwen2.5-0.5B for SFT, 1.5B for GRPO and DPO) train
  faster but cannot be hosted; `zps.train` warns and `zps.serve` refuses.
  SFT runs on an A10G and takes about a minute here; GRPO and DPO run
  on an L40S (`--method grpo --steps 10` took 137 s on Qwen3-4B).
- **Cold starts.** The serving GPU scales to zero. The first call after
  idle can take a few minutes; `call` waits up to fifteen.
- **Thinking mode.** Qwen3 reasons before it answers unless told not to.
  `call` sends `chat_template_kwargs: {"enable_thinking": false}` so the
  reply is the answer, not the reasoning.
- **Cost.** SFT here is about a minute of A10G, GRPO a few minutes of L40S. Serving bills while the
  GPU is awake; the endpoint idles back to zero on its own.
- **Holdout.** `split_pseudo_production` moves whole tasks and seeds the
  held-out side with one task per failure signature first, so on a tiny
  set (7 tasks here) the holdout ends up larger than the fraction asks.
  That is fine for a wiring check; a real set has hundreds of tasks.

## Where it shows up

`run.url` is the loss curve and the before/after on the platform.
`zps.models()` lists what the account hosts, `zps.get_run(run_id)` returns
the points, and the dataset cards link to the run. Docs:
https://www.zeroproofai.com/docs/training
