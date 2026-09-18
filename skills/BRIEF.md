# Writing a While skill

A skill is the playbook a coding agent reads to do one kind of post-training
job end to end, with the While SDK (`whileai`, import `whileai.simulations as
wai`) and the While platform (`whileai.platform`). One folder per skill:

```
skills/<name>/SKILL.md   the playbook: frontmatter, steps, python blocks
skills/<name>/check.py   the same steps, runnable offline, no key, no network
```

Rules, all enforced by `tests/skills/test_skills.py`:

1. **Frontmatter** like the existing skills (`skills/whileai-simulations/SKILL.md`):
   `name`, `description` (when to use it, one paragraph), `metadata.version`.
2. **Every ```python block in SKILL.md appears verbatim in check.py.** The
   playbook is the tested code. Prose may differ; code may not. Blocks can
   reference names defined in an earlier block or in check.py's setup
   (fixtures, a scripted agent, an offline transport), and the SKILL.md must
   say what those names are in one line before the block.
3. **check.py runs offline** under `uv run python skills/<name>/check.py`
   in under 60 seconds with no `WHILEAI_API_KEY`, no model, no GPU. Use a
   scripted agent (see `recipes/02-measure/eval-your-agent/run.py`:
   `agent(message) -> {"steps": [...], "final_text": ...}`), the offline
   situation writer (`wai.simulate(..., simulator=False, seeds=[...],
   repeats=k)`), a judge that is a program (`wai.evaluate(rows, judge)` or a
   `wai.verifier`), and `whileai.platform.track(..., transport=fake)` where
   `fake(method, path, body)` records calls and answers like the API (copy
   `recipes/04-train/report-run/run.py::printing_transport`). Exit non-zero
   on any assertion that the step did what the playbook claims.
4. **Every skill ends the same way**: build the frozen held-out test first,
   score every behavior the agent has (not only the one you trained), then
   report with `track(...)`, `tracked.behavior(...)`, `run = tracked.run(...)`,
   `run.score(...)`, `run.finish(...)`, and print `str(tracked.verdict())`.
   Read `whileai/platform.py` for the exact signatures and field names.
5. **Grounded.** Each step that rests on the literature names the
   rlhfbook.com chapter BY TITLE ("Evaluation", "Over-Optimization",
   "Reward Modeling", "Policy Gradients", "Direct Alignment", "Instruction
   Fine-Tuning", "Rejection Sampling", "Synthetic Data and Distillation")
   or a paper, in one clause, the way `whileai/platform.py` docstrings do.
   Do not invent a claim the book does not make. If unsure, leave the cite
   out and say the step is engineering, not research.
6. **Voice** (Jacob's bar): plain English, short declarative sentences,
   field terms are fine (pass@1, GRPO, DPO, KL, held-out, CI), no filler, no
   jargon used as decoration. Bold label then one or two sentences. Under
   1,200 words per SKILL.md. Every warning or check says what happened and
   the one call that fixes it.
7. **Names.** `wai` for `whileai.simulations`; `from whileai.platform import
   track, Behavior, Judge, Harness, Frontier, Score, LiveDay`. Never write
   `zps` or `zeroproof`. The handle is `tracked`, never `agent` (the user's
   framework owns that word).
8. Lint: `uv run ruff check skills/<name>/check.py && uv run ruff format
   skills/<name>/check.py`. Line length 100, py310.

Where the pieces live (read before writing; signatures are ground truth):

- `whileai/platform.py` (track, Behavior, Judge, Score, Run, LiveDay).
- `recipes/02-measure/eval-your-agent/run.py`: scripted agent + `simulate`
  offline + `evaluate` with a programmatic judge + pass@1 with interval.
- `recipes/01-simulate/verifiers/run.py`: `wai.verifier`, `verify.*`.
- `recipes/04-train/grpo/reward.py`: a reward that reads tool calls.
- `recipes/03-select/character/`: constitution -> traits -> judge checked
  against spec labels -> pairs and SFT rows -> `delta_report(must_not_regress=)`.
- `whileai.simulations`: `load_traces`, `trace_report`, `mine_traces`,
  `tools_from_traces`, `simulate_from_traces`, `select_for_sft`,
  `export_dataset` (= `export_training`), `build_preference_pairs`,
  `select_for_rl`, `export_environment`, `decontaminate`, `judge_trust`,
  `delta_report`, `holdout_size`, `pass_at`, `length_report`, `style_report`,
  `hack_scan`, `to_trl`, `loss_mask`. `inspect.getdoc` any of them.
- `whileai.simulations.score.hygiene.tool_calls(row)` counts tool calls.
