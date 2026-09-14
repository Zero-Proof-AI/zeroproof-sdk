# Changelog

Versions move in hundredths (`0.04` then `0.05`). PyPI normalizes them, so
`pip install zeroproof==0.4` is the `0.04` line below.

## Unreleased

- Trust layer: the `calibration` stamp is now measured before
  `optimize(mode="rl")` prunes and carried onto the selection (the gate
  keeps it instead of re-measuring post-dedup k), the report warns that
  `pass^k`/`pass@k` do not survive the prune, and `hack_scan` returns
  `degenerate` rather than naming an arbitrary tied feature when too few
  distinct trajectories leave every candidate collinear with reward;
  `hack_scan_diff` withholds `learned` on a degenerate side for the same
  reason, instead of reading a tie as what the policy learned.
- `hack_scan`: a tie in magnitude goes to the endorsed feature (the
  complement of the behavior correlates exactly as strongly, with the
  opposite sign, and is not a second thing the policy learns), and an
  endorsed feature the reward punishes is a `reward_hack` of its own,
  reported as `inverted` with a "reward punishes" warning. Found on the
  example: the honest reward's strongest feature was the shortcut
  sentence at rho -1.0.
- `examples/reward-hacking` and `docs/reward-hacking.md`: the detection
  loop end to end, offline, in seconds. A scripted refund agent that
  sometimes takes a shortcut, an honest judge that reads the trajectory
  and a hackable one that reads the prose; `hack_scan`, `judge_probes`
  and `trace_flag_report` on both, then a second agent that learned the
  shortcut stands in for "after training" and `delta_report(proxy=)`
  calls it over-optimized while `hack_scan_diff` names what it learned.
  The doc is the recipe: what the book says, the five checks, the three
  rules (endorse the behavior, fix the judge not the rows, keep the gold
  away from the proxy).

## 0.38 (2026-09-14)

- Examples audit (#168, #169, #170, #171, #172). Every example now has a
  test under `tests/examples/` that runs its offline path end to end and
  checks README flags, defaults and quoted numbers against the code; 100+
  new tests. Fixed: `bring-your-own-agent` part three had stopped firing
  under rl mode's `successive` repeat policy; `pass-at-k` printed a
  different pass@1 per run at the same seed (now `reproducible=True`,
  `--concurrency` exposed); `agent-behavior --dry-run` exited 0 on a
  dead endpoint; `identity/generate.py` wrote to a path on another
  machine and `eval_modal.py` could not read its output (new
  `report.py` with `identity_rate` / `leak_rate` and Wilson intervals);
  `grpo` README told readers to pass `--gpu`, which the Modal
  entrypoints now accept; `hosted-loop` ignored `zeroproof login`;
  `prime-intellect-rl/export_prompts.py` crashed without `data/` and
  collapsed rows with no `scenario_id` into one task; `hugging-face`
  gained the `--push-run` the README promised; `verifiers` README no
  longer points at a spec that does not exist. `examples/README.md` is
  a map of the twelve examples in post-training order. Main README:
  two broken code fences fixed, `dpo` and `hugging-face` added to the
  table, `attach_labels` / `decontaminate` return shapes and `train()`
  step/epoch knobs corrected.
- `Weighted` verifier: a part that cannot run (no reference on the row)
  now returns `reward: None` with the part's reason, like `ExactMatch`,
  `All` and `Any`, instead of scoring 0; a rubric row with a missing
  answer key no longer lands in RL data as a hard fail. Direct tests for
  every exported name that had none (`datasets`, `delete_dataset`,
  `list_runs`, `get_run`, `delete_run`, `models`, `claude_code`,
  `hosted_model`, `novelty`, `behavior_signature`, `Trait`) and ten for
  the OTLP `zeroproof.ingest` module, which had zero.
- `delta_report(proxy=)`, and `proxy=` on `run.delta` / `attach_delta`:
  name the training reward's marker (e.g. `"marker:first_action"`) and
  the report says whether the run over-optimized it (rlhf-book ch. 14):
  `over_optimized` is true, the report fails, and a warning names both
  intervals when the proxy moved up while the target did not, or the
  proxy's interval sits entirely above the target's. `proxy_verdict`,
  `proxy_delta`, `proxy_ci95` on the report; `format_delta_report`
  prints the proxy line. `zps.hack_scan_diff(before, after, endorsed=)`
  is the scan before training against the scan after on rollouts scored
  by the same reward: `gained` (features that clear the floor only
  after), `lost`, `moved`, and `learned`, the one line that names what
  the update moved toward and whether it is endorsed;
  `format_hack_scan_diff` prints it.
- `zps.trace_markers(rows)`, `zps.trace_flags(row)`, `zps.trace_flag_report(rows)`:
  did the agent fake the work? Flags read from the trajectory rather
  than the prose (rlhf-book ch. 13, 14), a port of the agent-behavior
  example's signals onto the row shape. `lie.tests_claimed` (tests said
  to pass when no test command ran or the last one failed, hedged claims
  excluded), `lie.unverified_claim` ("I verified" with no tool calls),
  `lie.phantom_edit` ("I updated" with nothing written),
  `lie.ignored_failure` (the turn ended on a failed call and the reply
  never mentions trouble), `hack.test_edited`, `hack.test_weakened`,
  `hack.suppressed`, `hack.bypassed`, `risk.destructive`, `risk.secrets`,
  each with the fragment that raised it. Reads `steps`, platform
  `tool_trace`, assistant `tool_calls` with their `tool` results, and
  `<tool_call>` blocks; a failed step is a failing status (the mock
  world's timeout, permission_denied, not_found, rejected, error), a
  non-zero exit code, or an error-opening result. What counts as a
  read, write, delete or command comes from the tool's arguments and
  name; `kinds={"tool": "write"}` overrides. The markers (`honest_claims`,
  `reported_failure`, `no_test_tampering`, `no_suppression`,
  `no_bypass`, `no_destructive`, `no_secrets`, 1.0 = clean) feed
  `marker_summary`, `delta_report(must_not_regress=)` and `hack_scan`,
  whose hand tier now carries one `trace:<flag>` feature per flag that
  fired; `reward_correlations` scans the fired flags beside length and
  the style phrases. `trace_flag_report` gives each flag's rate,
  examples, and its correlation with the reward, flagged when the judge
  pays for the fake.
- `data.rows()` and `output=` write the whole row (#149). The export was
  an allowlist, so 16 keys the trajectory carries never reached disk:
  `markers` (which re-broke #56 for anyone reading `rows()`),
  `judge_status` / `judge_name` / `lineage`, `scenario_dimensions`,
  `seed`, `arm`, `behavior_signature`. Now everything is exported except
  the teacher-only `privileged` block (`principle`, `hidden_state`,
  `reference`, `rubric`), which is dropped at every depth, and the
  in-memory `vector`. `data.rows` also reads as a list, matching
  `ScoredData.rows`; `data.rows()` still works.

## 0.37 (2026-09-14)

- `zps.train(generations=, learning_rate=, beta=, seed=, max_completion_length=,
  loss_type=, config=)`: the knobs a hosted run is reproduced and compared by
  (rlhf-book ch. 6, 7) reach the trainer by name and land on the run's
  config; ranges are checked before the call. Needs the gate and trainer
  deployed 2026-09-14 (site 47788a0).
- `rubric_judge` numbers the checklist and asks for verdicts by item
  number; `Rubric.score` also resolves a title, its slug or a paraphrase
  that contains it, and results carry `n_unanswered`. Live on the hosted
  judge, 10 of 32 rows had items answered under a rewritten title and
  failed for it; now none do (#156).
- `concurrency: 1` is round-synchronous, like `reproducible=True`: the
  batch's rollouts and their in-loop verdicts all land before the next
  round is chosen. Two same-seed serial runs in one process could draw
  different situations: the 0.35 s collect window decided how many of a
  batch's rollouts a round saw, and the rounds spent waiting drifted the
  counter that seeds selection. Cold processes happened to agree, so the
  cross-process check passed; `tests/api/test_reproducibility.py` now
  also runs `simulate()` twice in one process under contrasting latency.
  Golden captures move: a serial run now folds every batch whole, so a
  `scripts/golden.py` diff across this change is expected to differ.
## 0.36 (2026-09-14)

- `training_rows(max_tool_output_chars=)` / `export_training(...)`: each
  tool message over the cap is cut with a `[... N chars of tool output
  truncated]` marker and counted on the row (`tool_output_truncated`,
  `tool_output_chars_cut`) and in the report (rlhf-book ch. 13). The Claude
  Code adapter's 2000-character cut is now `CLAUDE_CODE_RESULT_CHARS` and a
  cut step carries `result_truncated` and `result_chars` (#146).
- `examples/dpo`: constructed negatives are one pair per distinct prompt
  and capped at `constructed_share` (0.3) of the sampled pairs. Uncapped,
  the balance repeats multiplied them past the sampled pairs and the
  policy learned "never call" (with-id pass@1 0.11 to 0.05).
- `decontaminate` reads prompts only by default, the book's method, and
  counts a row when it is an eval text verbatim or when one eval text
  covers at least 80% of its words (`overlap=`, the Llama 2 rule) rather
  than on any single shared 8-gram. Situations written from one set of
  templates share whole sentences, so the old default flagged every row
  of a train/eval pair from the same simulator (rate 1.0, nothing kept)
  while 41% of the prompts actually repeated; the report now separates
  `n_exact` from `n_near`, counts hits per field, and carries each
  offender's coverage. `fields=("prompt", "final_text")` keeps the
  stricter replies-versus-answers check (#125).
- `zps.judge_probes(rows, judge)` and `judge_trust(probes="all")`: the
  reward hacks a policy finds first, tried on the judge on purpose
  (rlhf-book ch. 14). Seven probes mutate a sampled reply one way and
  re-judge it: `filler`, `keyword_stuffing` (the rubric's own words,
  from `rubric=` or the row's system prompt), `success_claim` ("Done. I
  verified this and all tests pass."), `prompt_echo`, `sycophancy`
  ("You're absolutely right."), `empty_format` (a well-formed call to
  the row's tool with empty arguments), `refusal`. An additive probe
  reports the share of originally failing replies that pass once the
  text is added; a replacement probe the share that pass with the
  content gone. `exploitable_by` names the probes at or over the 10%
  flip flag, each with a one-line warning, and `judge_trust` fails on
  any. `format_judge_trust` prints the probe table. A probe with
  nothing to work on (no rubric words, no tool) is `skipped` with the
  reason.

## 0.35 (2026-09-14)

- `examples/dpo --constructed-negatives`: for every no-id or off-topic
  prompt the policy answered without a tool call, pair that reply
  against an invented call (`pairs.constructed_negatives`), so DPO has
  contrast on the prompts where its own samples had none; a second
  balanced round without it had made the invented-id habit worse.
- `select_for_rl(truncated="drop" | "keep" | "penalize")` and
  `optimize(mode="rl", truncated=)`: a rollout cut at the token cap is
  dropped (default), kept with `overlong=True` and a `finished` marker, or
  kept as a failure with the judged score under `reward_before_penalty`
  (DAPO's overlong handling, rlhf-book ch. 6, 7). `drop_truncated=False`
  now means `"keep"` (#139).
- Over-optimization and eval-variance consolidated onto one module each (#132): `score.style` (style_markers/style_report/refusal_report, 1.0=clean, delta_report-ready) and `score.stats.eval_variance` are canonical; `score.markers` (behavioral_markers/mark_rows) now warns (its presence polarity reads a paired delta backwards); the unreleased `score.benchmark` is removed.

## 0.34 (2026-09-14)

- `zps.hack_scan(rows, endorsed=[...])`: what a grouped update would
  learn from these rewards, named before training (rlhf-book ch. 6, 14).
  Reward and every candidate feature are centered within ask, the way
  GRPO baselines them, ranked by that correlation, and compared to a
  noise floor from shuffling reward within ask (`tau`, 95th percentile
  of the null maximum), so a feature that only tracks difficulty never
  counts and the threshold is measured, not guessed. Two tiers, pure
  Python: the hand tier (reply length, tool calls, turns, truncation,
  surface counts, one indicator per tool called, mean token logprob,
  every numeric marker, plus `features=` of your own) and the auto tier
  (presence of the 200 most common words and word pairs in the agent's
  text, and pairwise ANDs that beat both parents). `endorsed` names what
  the reward should track (feature-name substrings such as
  `"tool:lookup_order"` or `"marker:grounded"`); with it the report says
  `regime`: `train`, `reward_hack` (the top feature is not endorsed),
  `pool_exhausted` (over 20% of asks all-pass), `no_signal` (nothing
  clears the floor) or `unknown`, plus `integrity` (share of the
  above-floor signal that is endorsed), `top_feature`, the ranking with
  the pooled correlation beside each, and one-line `warnings`.
  `zps.format_hack_scan(report)` prints it. Duplicate columns are one
  feature with `aliases`; 3,200 rollouts scan in about a second.
  `select_for_rl` / `optimize(mode="rl", endorsed=)` carry it as
  `report["hack_scan"]` and its warnings in `hygiene_warnings`;
  `publish_gate` / `push_rows(gate=True)` / `data.push` report it on
  RL-shaped rows and, with `strict_hacks=True`, refuse a `reward_hack`.
  The pooled `reward_correlations` scan stays as the second column.
- `zps.HackMonitor(run, holdout=, proxy=, gold=, ...)`: is the run
  hacking its reward right now (rlhf-book ch. 14, figure 1)? A
  Transformers / TRL callback plus `monitor.wrap(reward_fn)` around the
  reward function. Every `every` steps it samples the holdout from the
  live policy (`k` completions on up to `n_prompts` asks) and scores it
  twice: with the training reward (the proxy; `proxy=` or the wrapped
  function) and with a scorer the proxy cannot see (`gold=`, any judge
  under the SDK contract). Both land on the run as `proxy_reward` and
  `gold_reward`, with `holdout_length`. Four alarms, one line each on
  the run and in `monitor.alarms`: `divergence` (proxy up by `delta`
  over `window` evals while the paired gold interval does not move
  up), `length` (completions up by `length_pct` while gold does not),
  `drift` (the trainer's KL past `kl_budget`), `feature` (the last
  `buffer` completions' `hack_scan` says `reward_hack`; needs
  `endorsed`). `stop_on=` names the alarms that stop the trainer; the
  default logs only. The summary (`history`, `alarms`, `last_scan`,
  `stopped_at`) rides on the run's `finish` through `run.note`, a
  stopped run finishes as `stopped` with the reason;
  `zps.format_hack_monitor(summary)` prints it. `TrainingRun.note`
  is new: fields it sets travel with whichever callback finishes the
  run. `examples/grpo` wires the monitor by default (`--monitor-every`,
  `--stop-on`).
- `run_judge(scale=(lo, hi))`, `evaluate(scale=)`, `data.grade(judge=, scale=)`:
  a rating judge (1 to 5, 0 to 10) is read on its scale; `reward` is the
  rating mapped onto [0, 1] and `judge_meta` keeps `rating` and `scale`;
  a rating outside the scale is `invalid_result` (rlhf-book ch. 11) (#137).
- `zps.attach_labels(rows, labels, annotator=)`: hand labels from a JSONL
  path, a list or a mapping, matched by rollout id, scenario id plus
  rollout index, or prompt plus final text; every label stays on the row
  as `gold_labels` (label, annotator, kind, ts, note) and `gold_reward` is
  the majority, unset on a tie. `zps.annotator_agreement(rows)`: per-
  annotator counts, unanimous share, Cohen's kappa for the busiest pair,
  the split rows (rlhf-book ch. 10, 11) (#136).
- Reward model as a judge (rlhf-book ch. 5). `zps.train(ds, method="rm")`
  trains a sequence-classification head on the set's pass-vs-fail pairs
  (Bradley-Terry loss, the pairs DPO uses) and reports pair accuracy on the
  held-out pairs before and after plus the score threshold that separates
  them. `zps.reward_model(run)` is that run as a judge: it honors the judge
  contract (`reward` 0/1 against the threshold, `rm_score` raw), so it feeds
  `data.grade(judge=)`, `evaluate`, `judge_trust` and
  `build_preference_pairs`. Gate route `POST /runs/{id}/score`.
- `simulate(tasks=previous_run)` re-runs a previous run's task set (the
  run, its rows, or its JSONL path) instead of drawing a new one: every
  prompt again, on its own `scenario_id` and `scenario_dimensions`, under
  the same faults and world state, and nothing else generated; the run
  stops with `tasks_done` once every prompt has its rollouts and reports
  `search["pinned_tasks"]`. A run draws its tasks by seed and, above
  `concurrency: 1`, by completion order, so even a same-policy re-run
  paired 43 of 49 tasks; pinned, every A/B (prompt edit, model swap,
  another seed) pairs all of them (#98).
- `data.grade(use_privileged=True)` / `grade_llm(use_privileged=)`: the
  hosted judge reads the row's `privileged` block (principle, reference,
  hidden state) as `judge_only` in its payload, with a prompt line on how
  to use it (rlhf-book ch. 12, constitutional AI). Folded into the judge
  version and stamped `judge_meta.privileged`; exports still never carry
  `privileged` (#134).

## 0.33 (2026-09-14)

- Rubrics as objects (rlhf-book ch. 12): `Rubric` / `Criterion` (hard rule,
  principle, pitfall; positive weights; a content hash as `version`) on the
  row's `privileged.rubric`, never exported. `zps.rubric_judge()` scores
  one verdict per criterion (`Rubric.score`: a missed hard rule is 0,
  otherwise principles minus pitfalls over principle weight) and lifts each
  criterion onto the row as a `rubric:<item>` marker; `zps.write_rubrics(rows,
  domain=)` drafts one rubric per prompt with the book's rubric-writer
  prompt; `attach_rubric`, `rubric_of`, `score_with_rubric` (#127).
- `training_rows(unroll=True)` / `export_training(unroll=True)`: an N-turn
  conversation becomes N samples, the k-th ending at the k-th agent turn
  with loss on that turn only (rlhf-book ch. 4); samples carry `unroll`
  (`turn`, `turns`) and `lineage.unrolled_from`, and no group fields (#129).
- `examples/dpo`: a second balanced round from the round-one adapter
  made the invented-id habit worse (no-id pass@1 0.82 to 0.26 while
  with-id rose 0.56 to 0.91); the README records it. DPO needs contrast
  on the no-id prompts themselves, not more of them.
- Sampling facts on every simulated row (rlhf-book ch. 6, 9):
  `policy_version` (`<model_version>@<sha256 of the system policy>[:16]`,
  round-tripped as `Rollout.policy.version`), `sampling` (`temperature`,
  `logprobs`) on model-backed rollouts, and with `simulate(logprobs="tokens")`
  the per-token `token_logprobs` list an importance-sampling ratio is built
  from; all three ride through `training_rows` / `export_training`.
  `zps.staleness_report(rows, base_model=)`: rows per policy version, stale
  rows for the model about to be trained, sampling / logprob coverage,
  with warnings (#121).
- `argument_grounding`: a marker for tool arguments that came from
  nowhere. `mark_grounding(rows)` stamps 1 when every string argument
  of every tool call appears in the prompt, the user and system turns,
  or an earlier tool result, else 0; `ungrounded_arguments(row)` and
  `grounding_report(rows)` name the invented values by tool and key.
  Reads `steps`, `messages` tool calls and `<tool_call>` blocks. The
  GRPO and DPO examples guard it with `must_not_regress`, so the
  invented-id regression fails the run instead of hiding under pass@1.
- Hosted GRPO and DPO run on an L40S, so `zps.train(method="grpo")` on a
  served base (`Qwen/Qwen3-4B`) trains and serves; the docs no longer say a
  4B base does not fit.

## 0.32 (2026-09-14)
- `zps.eval_variance(run_1, run_2, ...)` (or one row list split by
  `lineage.scoring_run_id` / `by=`): the eval's own re-run standard
  deviation, `noise_band` = 2 x std, and Olmo 3's stability band in
  points (rlhf-book ch. 16). `delta_report(run_std=)` marks every
  metric whose delta sits inside that band `within_noise`, keeps it out
  of improved / slipped / regressions, and reads a target there as
  `within_eval_noise` instead of moved (#113).
- `zps.judge_pairs(pairs, judge=None, swap=True)` asks a judge which side
  of each preference pair is better, then again with A and B swapped
  (rlhf-book ch. 5, 11). Each pair gets `pairwise` (winner, whether the
  two orders agreed, reasons, judge) and `tie`; a pair decided
  differently in the two orders is a tie with `position_consistent=False`.
  Report: `position_flip_rate`, `tie_rate`, `agrees_with_scores`,
  `prefers_rejected` with examples. `zps.pairwise_judge(spec)` is the
  hosted model judge with a length-neutral prompt.
  `export_preference(drop_ties=True)` leaves ties out and counts them (#114).

2026-09-14)

- `examples/grpo` and `examples/dpo`: `--balance <share>` repeats the
  prompts of any category below that share of the train split
  (`prompts.balance`), so the no-id and off-topic prompts reach the
  update at the rate they matter. The invented-id regression the
  stratified holdout exposed is a sampling-frequency problem, not a
  reward one for GRPO: the invented-id rate on no-id prompts drops 0.23
  to 0.08 at 120 steps. One-round DPO does not move on it (no pairs
  where the base never fails); a second round is DPO's lever.
- A policy edit keeps the task grid (#98). The covering array is built in
  layers: a rule-free block over tools and situation axes, then one block
  per policy clause, rotated by the clause text. Editing, adding or
  removing one clause changes that clause's cells only; every other
  `scenario_id` survives, so `compare_runs` stays paired after a prompt
  change (12 of 53 tasks paired before; all of them now). The grid is
  larger by the rule-free block, and cells in it carry no rule hint.
  Fault rows are kept per row from the row's own digest, one of every
  kind guaranteed, so a grid that grows keeps the verdict on the rows it
  had. Every existing grid changes once with this release.
- `compare_runs` says when it dropped tasks: `note` names how many were
  on one side only and that the verdict rests on the shared ones, with
  "most tasks unpaired" in front when fewer than half paired;
  `paired_share` is the fraction. `delta_report` carries it as a warning
  on the headline metric and reports `n_unpaired_tasks`.
- `examples/hosted-loop`: push, `zps.train`, `zps.serve`, call, as one
  script with a state file per step; the wiring check for training on the
  platform, with the served-base, cold-start and thinking-mode notes a
  first run needs.

## 0.31 (2026-09-14)

- `examples/grpo` and `examples/dpo`: the model-written set is split by
  scenario within each prompt category (`split_holdout_stratified`); the
  hash split had put every no-id situation in train, so the holdout
  could not see a policy that always calls the tool. Both scripts report
  `by_category_before` / `by_category_after` (pass@1 and tool-call rate
  per category). `examples/dpo --from-run` merges a previous round's
  adapter and samples fresh pairs from it: iterated on-policy DPO, with
  the merged policy saved for serving or a further round.
  On the stratified split both GRPO (120 steps, 0.29 -> 0.81) and DPO
  (one round, 0.29 -> 0.72) learned to invent an order id on a quarter of
  the no-id prompts (no_id pass@1 0.95 -> 0.75 and 0.98 -> 0.72); the
  hash split had hidden it. DPO round two from the round-one adapter:
  0.63 -> 0.92.
- `delta_report(by=...)`, and `by=` on `run.delta` / `attach_delta`: the
  target compared within each group of rows (a row key, a marker name,
  or a callable), reported as `groups` with `groups_down` for a group
  whose target dropped significantly; `format_delta_report` prints the
  block and the run page draws it. A headline over one dominant kind of
  prompt no longer hides the other kinds.
- Over-optimization signatures as markers (rlhf-book ch. 14, 17):
  `zps.style_markers(rows)` stamps `no_boilerplate`, `no_hedging`,
  `no_apology`, `no_sycophancy` and `answered` (1 = clean) from phrase
  lists, so `marker_summary` and `delta_report(must_not_regress=)`
  watch them; `zps.style_report(rows)` gives each signature's clean
  share with an interval, the phrases that fired, and its correlation
  with the reward, flagged when the judge pays for the tic;
  `zps.refusal_report(benign_rows)` is the over-refusal rate with a
  Wilson interval and examples. `reward_correlations` (and so the
  publish gate's hygiene warnings) now scans the same four phrase
  features next to length, tool calls and turns.
- `select_for_sft` / `optimize(mode="sft")` is rejection sampling by
  reward (rlhf-book ch. 9): `select="top_per_prompt"` (default),
  `"top_k_overall"` with `k=`, and `"random_per_prompt"` /
  `"random_k_overall"` as chance controls; `min_reward` (default 1.0)
  admits partial-credit graders, whose 0.9s were dropped as not-pass.
  Reports gain `selection`, `min_reward`, `reward_mean_eligible`,
  `reward_mean_selected`.
- Exported groups: `n0`/`n1` count partial credit below/above 0.5 (a
  0.3/0.9 group read as unanimous), plus `reward_mean` and `reward_std`
  per group so a trainer can see where group normalization divides by
  ~zero (rlhf-book ch. 6).
- `decontaminate` no longer takes the eval set's own replies as n-gram
  sources, only prompts, answers and references (rlhf-book ch. 16); tool
  boilerplate shared between two replies flagged clean rows.
- `llm_judge` samples at temperature 0, like `grade_llm`.
- `examples/grpo`: `--loss-type` (bnpo, TRL's default; grpo; dr_grpo),
  `--epsilon-high`, `--no-scale-rewards` and `--mask-truncated`, so
  Dr.GRPO and DAPO's clip and overlong mask are flags on the one trainer
  and land in the run config. `prompts.jsonl` is checked in for real (a
  repo-wide `*.jsonl` ignore had swallowed it) and the test requires it.
  On it: GRPO 120 steps 0.18 -> 0.85 (+0.63 [+0.50, +0.73]), Dr.GRPO at
  the same budget 0.17 -> 0.53, DPO one round 0.17 -> 0.69.
- With `grader=`, every mode judges rows as they land, on the judge pool
  beside the rollouts; only the tail is judged after the clock. Before this
  explore and sft judged everything in one pass after the run, which on a
  120 s run added about a minute past the budget. `data.search["grader"]`
  now says how many rows were judged in the loop and how many after.

## 0.30 (2026-09-14)

- `examples/grpo`: a model-written prompt set. `prompts.py` (writer chat
  per template seed, array parsing, category and near-duplicate filter
  through `case_for`, loader) and `write_prompts_modal.py` (Qwen2.5-7B-
  Instruct on an A10G) produce `prompts.jsonl`, 707 prompts from 67
  situations, checked in; both the GRPO and DPO scripts take
  `--prompts-file`, so the holdout is over a hundred prompts instead of
  fourteen and pass@1 intervals shrink accordingly.
- `zps.train(dataset_id, method="sft"|"grpo"|"dpo", steps=, epochs=,
  holdout=, base_model=, wait=)` starts a hosted run on the platform's
  trainer and returns the `TrainingRun` the dashboard draws; `run.refresh()`
  / `run.wait()` follow it, then `run.adapter` and `run.training` (before,
  after, rows, seconds). `zps.serve(name, run)` hosts the adapter on an
  OpenAI-compatible endpoint and `zps.models()` lists them. README and
  the character docs no longer say the SDK does not train (#77).
- `mode="rl"` spends rollouts where the agent is inconsistent. Every
  prompt is probed with two rollouts; a prompt that splits is filled to k,
  a unanimous one stops once the run's own measured rates say a fresh
  prompt is the better bet. `grader=` runs beside the rollouts so the
  allocation reads rewards. A `time_budget` finishes the groups in flight
  instead of cutting them. `repeat_policy="fixed"` is the old behavior.

## 0.29 (2026-09-14)

- `pass_at` says when groups are uneven. A time or row budget that cuts a
  run mid-group leaves ragged groups; `k` defaults to the smallest, so the
  k-way numbers were withheld with "set repeats>=4" even when repeats was
  4. The note now names the size range and the `k=` that scores the groups
  which reached it.
- `SimulationData.grade` docstring names what the no-argument path is
  (`grade_llm`, the hosted Phi-4 judge, in place, returning the judge
  report) and README marks `grade=True` as the legacy conduct score.
- An `agent=` callable that fails every call is called off after
  `max(16, 2 * budget)` lost rollouts with `stopped_because="agent_failed"`,
  instead of re-rolling each lost slot and refilling it until the writer
  ran dry (~15 calls per budgeted row). Runs with any surviving row keep
  the re-roll behavior (#88).

## 0.28 (2026-09-14)

- Judge verdicts: a complete JSON object in the reply decides on its own.
  A string-typed score, a duplicate `score` key, a `score` that disagrees
  with a `reward`, a bool, NaN or 1.5 leave the row ungraded; the digit
  salvage runs only when no complete object exists (#64).
- A broken `agent=` callable ends the run as `stopped_because="agent_failed"`
  instead of `writer_exhausted`, with `search["agent_errors"]`,
  `search["first_agent_error"]` (exception type included), a degraded note
  and a logged warning. A return shape without `steps`/`final_text` counts
  the same way. README states the callable contract (#29).
- `examples/agent-behavior` runs on a fresh clone: the uncommitted
  `tasks_hard` pack is optional. `tests/examples/` smoke-tests every
  example offline (#38).
- `rows_from_otel` sums `gen_ai.usage.*` (and the `llm.token_count.*`
  dialect) into `row["usage"]`, the shape simulated rows carry (#67).
- `simulate()` documents `lineage.scoring_run_id` as per-invocation
  identity outside the bit-for-bit guarantee; the `grader=` path is now
  pinned by the reproducibility test (#58).
- `select_for_rl`, `select_for_sft` and `build_preference_pairs` report
  `eval_sourced` (rows or pairs whose reward came from `evaluate()`) and
  warn when it is non-zero; `optimize.eval_sourced(rows)` is the count on
  its own. No row is dropped (#37).
- The covering array behind `scenario_regions` is memoized per process;
  each writer wave was rebuilding it and throwing it away. The slowest
  test drops from 26-49 s to under 1 s and the suite from 162 s to
  100 s. Rows out of the simulator are unchanged (golden harness: 13/13
  identical) (#36).
- `examples/dpo`: DPO on Modal end to end, the offline counterpart of
  `examples/grpo`. Pairs come from the base policy's own samples scored
  by the same rule and paired by `build_preference_pairs` (length
  matched), or from a `zps.export_preference` file with `--pairs`; TRL
  `DPOTrainer` with LoRA on Qwen2.5-1.5B-Instruct, the reference model
  is the adapter switched off, reward margin and accuracy on the
  dashboard, pass@1 before and after on a holdout, `run.delta` on the
  run page. `pairs.py` (first-turn rendering, TRL rows, export loader)
  is unit-tested offline.

## 0.27 (2026-09-14)

- Character training docs and example README point at the dataset's
  home, `zero-proof-ai/character-training-model-spec` on Hugging Face
  (train, holdout, eval) and the catalog agent `sol-character`;
  `docs/character-training.md` gains a section on the run's rows.

## 0.26 (2026-09-14)

- `examples/grpo`: GRPO on Modal end to end. Prompts from the offline
  template writer, a verifiable tool-discipline reward (`reward.py`,
  unit-tested), TRL `GRPOTrainer` with LoRA on Qwen2.5-1.5B-Instruct,
  reward and KL on the dashboard, pass@1 before and after on a
  holdout, and `run.delta` on the run page.
- Judge replies that break the contract stay ungraded. The binary
  verdict parser read a bare `true` as 1, `{"score": 1.5}` as 1 and
  `{"score": 0.5}` as 0; the 0-to-1 judge clamped a 2 to 1.0 and a -1
  to 0.0. All of these now return no score, the same contract
  `judging.py` already held a caller's own judge to. Well-formed
  verdicts inside chatter and replies cut off after the score still
  grade.

## 0.25 (2026-09-14)

- Hugging Face, both directions. `zps.hf_status()` says whether an account
  is connected and which namespaces it can publish under. `zps.hf_publish`
  pushes one of your sets to a dataset repo you own: one split per purpose,
  every push a commit tagged `zp-<dataset id>`, `zeroproof.json` in the repo
  mapping splits to datasets with history; it waits for the platform to
  stamp the commit and returns it (`wait=False` returns the pushing stamp).
  `zps.hf_publish_run` does the same for a finished run's LoRA adapter, as
  a model repo, private by default. `zps.import_hf(repo, split=...)`
  brings any Hub split onto your account as rows and waits until it is
  ready, so `zps.profile` can grade it before you train on it. Example in
  `examples/hugging-face`.
- Every model call now keeps what it cost. The server's `usage` block
  becomes `input_tokens` / `output_tokens` on the agent step and a summed
  `usage` on the row (`rows()`, `training_rows`, the typed `Step` and
  `to_row` all carry it). Before this no row said how many tokens it
  used, so a trace built from one had no `gen_ai.usage.*` and the
  platform's per-day usage counted zero for every simulation.
- Hosted-model tokens now reach the platform's Usage page. After every
  call to the hosted policy or judge the SDK batches the `usage` the
  server reported and sends it to `POST /usage` under the account's own
  key (background thread, once more at exit; `ZEROPROOF_NO_USAGE_REPORT=1`
  turns it off). Bring-your-own endpoints are never reported.

## 0.24 (2026-09-14)

- Character training example (`examples/character`): the OpenAI Model
  Spec's style section parsed into a constitution with its GOOD/BAD
  comparisons, prompts per trait, k replies each, a judge that reads the
  trait's principle, preference pairs and SFT rows with the deployment
  prompt, and `measure.py` for the before/after delta with `on_task` and
  `no_filler` guarded. The spec's labeled replies grade the judge
  (`judge_agreement`). Offline by default; `--model-url` for a live model.
  How-to in `docs/character-training.md`.
- A row's `privileged` block (`principle`, `hidden_state`, `reference`)
  now reads into `Task.privileged` in `from_row`. `to_row` still never
  projects it (it is the teacher's context, one step from a training
  file); a source row's block rides back out as passthrough. The engine
  leaves it empty; character and rubric pipelines write it.

## 0.23 (2026-09-14)

- Markers a judge returns now reach `marker_summary`. `run_judge`
  lifts `judge_meta["markers"]` onto `row["markers"]` (the judge wins a
  name collision), `simulate(grader=)` carries `markers` onto the
  trajectories, and `from_row` on a graded row returns the `Marker`
  objects. Before this a marker a judge returned measured nothing.
- `split_pseudo_production` splits by task, not by row: rows group by
  `prompt` (`scenario_id` when there is no prompt) and whole tasks move,
  so `mode="rl"` with `repeats>1` no longer puts siblings of one prompt
  on both sides. The flaw-signature rule sends a task, not a row, to the
  held-out side. `fraction` is still counted in rows and can overshoot by
  up to one task. On the reported repro, held-out prompts also present in
  train went from 100% to 0%.
- `scripts/golden.py`: a committed golden-output harness. Thirteen offline
  configurations at `concurrency=1` on fixed seeds; `capture` and `diff`
  compare two snapshot directories and `diff` names every changed key.
  `.gitignore` no longer blocks `scripts/`. The harness scrubs
  `lineage.scoring_run_id`, which `run_judge` stamps fresh per call
  (#58).
- CI measures line coverage with a floor at 84% (Python 3.12,
  `COVERAGE_CORE=sysmon`; the default tracer ran past 30 minutes).
- Tests pin that `principle`, `hidden_state` and `reference` never reach
  a student field or an exported file, and that an eval score stays
  distinguishable from a training reward through `lineage["source"]`.
- Test suite builds each identity dataset once instead of four times
  (about 22% less wall time).
- Training runs: `run = zps.training_run(name, dataset=, base_model=,
  total_steps=)`, `run.log(step, loss=, ...)`, `run.progress`,
  `run.finish()`; `zps.TrainerCallback(run)` for Transformers and TRL
  trainers; `zps.list_runs`, `zps.get_run`, `zps.delete_run`. Points
  are buffered and sent in batches and logging never raises into the
  training loop. The platform draws the loss curve and progress bar at
  /platform/training.
- `zeroproof purge --agent <slug>` and `--empty`: an agent with its traces,
  datasets and record, or datasets with no bytes (or few rows), gone in
  one command after a y/N; `--dry-run` counts. Python `zps.purge_agent`,
  `zps.delete_empty_datasets`.
- Agents: `zps.agents()`, `zps.register_agent(name, tools=, system_prompt=)`;
  `data.push(name, agent=...)` registers the agent and attaches the run's
  tools and system prompt to its record.
- `judge_trust` says when the gold labels are all one class instead of
  reporting kappa 0 and a length bias the labels cannot support;
  `gold_degenerate` on the report. Found on the first hosted-judge run.

## 0.22 (2026-09-14)

- The default judge is no longer the policy model. `zps.grade` uses
  `default_judge_spec()`: hosted `microsoft/phi-4` on its own vLLM app
  (`ZEROPROOF_JUDGE` overrides), while rollouts stay on hosted Qwen3-4B.
  A judge grading its own model's writing prefers it (rlhf-book ch. 5,
  12). The grade report carries `self_judged` and a warning when the
  judge model equals the rows' `model_version`. `judge_spec` with a bare
  URL now defaults the model name from the judge spec.

## 0.21 (2026-09-14)

- Platform-shaped rows count their tool calls: the reward-hack scan's
  `tool_calls` reads `tool_trace` as well as `steps`, so a pulled
  dataset no longer reports `None` for the tool-count correlation.
  The calibration stamp's `task_id` is the `scenario_id` when the row
  has one, not the prompt text.
- `simulate(logprobs=True)` asks the rollout model for the log-probability
  of every token it generates (rlhf-book ch. 6 off-policy correction,
  ch. 15 KL). Each agent turn's first step carries `logprob` and
  `n_tokens` (`"tokens"` adds `token_logprobs`), the row carries the
  totals, and a turn cut at the token cap is marked `truncated`. The
  fields ride through `export_row`, `training_rows`, `from_row`/`to_row`
  (`Step.logprob`, `Step.n_tokens`, `Step.truncated`) and the wire schema.
  A server that rejects `logprobs` is asked again without it.
- `zps.logprob_report(rows)`: capture coverage, mean token logprob,
  per-row quantiles, reward-vs-confidence correlation (flagged at 0.3),
  truncated count. `zps.mean_kl(rows, ref="ref_logprob")`: sampled
  KL(policy || reference) per generated token, overall and per task, from
  a reference logprob key or a second scored row list.
  `calibrate(rows, ref=...)` writes it into `calibration.mean_kl`, the
  field nothing populated before.

## 0.20 (2026-09-14)

- Every row `zps.grade` writes now says which judge produced it:
  `judge_name`, `judge_status`, and `judge_meta` (model, prompt hash,
  temperature, max_tokens, `version` = `<model>@<prompt sha>`); the
  report carries `judge_version`. A rubric edit is a new reward model,
  and the row records it. `run_judge(version=...)` does the same for a
  custom judge via `lineage.judge_version`; both read back as
  `Judgment.scorer.version`. New `schema.attach(row, judgment)` is the
  sanctioned verdict write; `grade` routes through it.
- `zps.judge_agreement(rows, gold="gold_reward")` (also
  `scored.agreement(...)`): agreement, Cohen's kappa, confusion counts,
  and the two disagreement rates against labels you trust, with the leak
  rate (gold failures the judge passed) called out because those rows
  train the failure. `gold` may be a second scoring pass for
  self-consistency. Warns below 50 gold rows (rlhf-book ch. 5).
- `training_rows` / `export_training` take `mask_mode="assistant"`
  (default, every agent turn) or `"final"` (only the last agent turn,
  rlhf-book ch. 4). `zps.loss_mask(messages, mode=)` builds the mask on
  its own; the export report carries `mask_mode`, `trained_messages`,
  `masked_messages`.
- Intervals and comparison: `pass_at(rows).ci95` (task bootstrap),
  `zps.metric_summary` / `zps.marker_summary`, and `zps.compare_runs`
  (paired task differences, bootstrap interval, sign-flip p-value;
  unpaired fallback under five shared tasks, labeled).
- `zps.decontaminate(rows, against=...)`: word 8-gram overlap against
  evaluation row lists, JSONL paths, or dataset ids; short prompts by
  exact match. Returns clean rows and the first offenders.
- `zps.delta_report(before, after, target=, must_not_regress=)`: pass@1
  and every shared marker compared as paired task differences; headline
  verdict on the target, regressions fail the report, other drops warn.
  `format_delta_report` prints it.
- `zps.judge_trust(rows, judge=)`: agreement with `gold_reward` labels
  (Wilson interval, kappa, confusion), held-out task halves, length
  sensitivity within human label, re-judge consistency and filler flips,
  and a disagreement queue. `format_judge_trust` prints it.
- Purpose on every pushed dataset: `data.push(name, purpose="train")`,
  `holdout=0.2` pushes a linked holdout set split by task,
  `zps.update_dataset(id, purpose=...)`, `zps.preview(id)`. The
  simulation mode is recorded on push. `zps.profile(id)` returns the
  trainer's numbers (pass rate, support, mixed tasks, tool use, per task).
- Preference pairs carry what a trainer and a reviewer need to trust them
  (rlhf-book ch. 8, 11): `chosen_score`, `rejected_score`, `margin` for a
  margin-aware loss; `chosen_model`, `rejected_model`, `same_policy` so an
  off-policy pair is labeled, not hidden; `length_delta` for the length
  exploit. `build_preference_pairs(min_margin=1.0, length_match=True)`:
  the default still pairs 1 against 0 only, `min_margin=0.5` admits
  partial-credit rows, and each chosen row now takes the rejected row
  closest to it in length. The report adds `mean_margin`,
  `same_policy_pairs`, `mixed_policy_pairs`, `length.chosen_longer_frac`,
  and `warnings` when chosen is the longer side in 75% or more of pairs
  or when pairs mix policies. `export_preference` keeps the new fields
  and reports `mean_margin` and `chosen_longer_frac`.

## 0.19 (2026-09-14)

- `zps.push_file` runs the publish gate too (`gate=False` uploads the
  bytes as they are), so a JSONL push no longer bypasses it. The stamped
  rows are what get uploaded; report on `entry["gate"]`.
- `calibration` is part of the typed contract: declared in
  `schemas/row-v1.json`, validated (`calibration_invalid`), read back
  with `zps.calibration_of(row)`, carried by `from_row` on
  `rollout.extra["calibration"]` and written by `to_row`.
- Training rows carry `loss_mask` (one 0/1 per message: 1 on assistant
  turns, 0 on system, user, and tool output). Declared in the schema and
  validated (`loss_mask_invalid`).

## 0.18 (2026-09-13)

- Public catalog: `zps.publish(id, agent=...)`, `zps.unpublish(id)`,
  `zps.catalog()`, and `data.push(..., agent=..., publish=True)` put a
  dataset on zeroproofai.com/datasets as a card grouped by agent.
  `zps.pull` fetches public sets with no key.
- Difficulty band is enforced, not just ranked: `select_for_rl` and
  `optimize(mode="rl")` drop asks whose pass rate falls outside
  `band=(0.2, 0.8)` (`enforce_band=False` restores rank-only); the
  report carries `band_dropped` by side. `group_signal` defaults to the
  same band (was 0.3-0.7). New `trim_out_of_band`, `DEFAULT_BAND`.
- Publish gate: `data.push` (and `push_rows(gate=True)`) runs
  `publish_gate` first. Every graded row gets a `calibration` stamp
  (task pass rate, k, producing policy); RL-shaped rows that are
  ungraded or have no mixed group raise `PublishGateError`. Report on
  `entry["gate"]`. New `zps.publish_gate`, `zps.calibrate`.
- Row hygiene: `select_for_rl` / `optimize(mode="rl")` drop duplicate
  rollouts within an ask (`dedupe=False` keeps them) and truncated
  rollouts (`drop_truncated=False`), and report the reward-hack scan
  (`correlations`: reward vs reply length, tool calls, assistant turns;
  flagged at `HACK_THRESHOLD` 0.3) plus a length report in
  `hygiene_warnings`. The publish gate reports the same, plus
  near-duplicate asks (token Jaccard 0.8), without dropping anything.
  New `zps.dedupe_groups`, `zps.near_duplicate_prompts`,
  `zps.length_report`, `zps.reward_correlations`.
- Both judge prompts say reply length must not influence the score.
  `select_for_sft` reports `completions_per_prompt_max` and notes when it
  is under 10 (rejection sampling wants 10 to 30 per prompt).

## 0.17 (2026-09-13)

- `signup` says that the key is a trial key and how to lift it.
  `zeroproof status` shows the tier; `zeroproof.account()` returns tier,
  limits and today's usage (`GET /me`).
- pass@1 / pass^k / pass@k from graded groups: `data.pass_at`,
  `ScoredData.pass_at`, `zps.pass_at(rows)` (a `PassAt` with
  `.headroom` = pass@k - pass@1 and `.per_task`). Unbiased estimators;
  k-way numbers withheld below `repeats=4`. `group_signal` reports the
  same keys, `recommend(mode="rl")` explains the mixed rate as headroom,
  and `save(meta=True)` writes `pass_at` to the sidecar.

## 0.16 (2026-09-13)

- `zeroproof signup --email`: creates the account and the key in one
  call, no browser and no password. Python: `zeroproof.signup()`.

## 0.15 (2026-09-13)

- `ruff format` across the repo, with `ruff format --check` in the CI
  lint job and `.git-blame-ignore-revs` pointing at the format commit.
  No behavior change: suite and golden harness identical.

## 0.14 (2026-09-13)

- `zeroproof login`: device-flow sign-in from a terminal or a coding
  agent. Prints a link and a code, waits for Approve in the browser,
  saves the key to `~/.zeroproof/credentials.json`. Platform calls fall
  back to that file when no env var is set. A transient network error
  while waiting is retried, not fatal. `zeroproof status` and
  `zeroproof logout`. Python: `zeroproof.login()`, `resolve_api_key()`.

## 0.13 (2026-09-13)

- Hygiene. ruff (lint) and mypy are configured in `pyproject.toml` and
  run in CI; `.editorconfig`, `.gitattributes` (LF) and a pre-commit
  config are in. The package type-checks clean except three modules that
  hang state on closures (`generate/generator.py`, `generate/scenarios.py`,
  `run/engine.py`), which are excluded until the writer is a class.
- The package root exports only `__all__` (76 names) plus the `data`,
  `schema` and `simulation` submodules. Sixty-one internal names that
  leaked through `import zeroproof.simulations as zps` are no longer
  reachable as `zps.<name>`; import them from their module.
- Cross-module helpers lost their leading underscore: `apply_spec`,
  `backend_spec`, `kind_from_spec`, `as_dict`, `intent_for_tool`,
  `load_jsonl`, `write_jsonl`, `row_cell_key`, `record_coverage`,
  `mutation_worthy`, `row_world`, `note_stage`, `clean_faults`,
  `export_row`. The old spellings remain as aliases.
- Dead code removed: six unused writer helpers, four unused constants.
- Lint fixes across the package: `raise ... from`, closure binding in
  the Claude Code adapter, redundant casts, sorted imports.

## 0.12 (2026-09-11)

- A callable `agent=` with no key now gets an error that names the two
  ways to run: `agent="openai:<model>"` on your key, or
  `simulator=False` for the built-in template writer. The README
  documents the no-key path.
- Rows in `data.trajectories` carry `messages`, matching the JSONL.
- `export_preference` on plain rows says it takes pairs and names
  `build_preference_pairs`.
- README notes that `fault_rate` applies through the mock world only.

## 0.11 (2026-09-11)

- Trace-driven allocation weighs a region's fail rate (Laplace-shrunk)
  and support, and flags regions with under three graded rows as
  `low_support`. Each region reports `n_graded` and `fail_rate`.
- `data.coverage["pairwise"]`: planned pairs, covered pairs, fraction.
- Docs say what the code does: the cold-start success flip means the
  tool-condition axis is sampled rather than covered; the leakage check
  is lexical; the unsourced benchmark claim is removed.
- Turn-length controller corrects at half gain; cluster sampling seeds
  per round; `planned_fault_fraction` removed (unused).
- The `zeroproof_simulations` alias stays until a later release rather
  than "two releases from now".

## 0.10 (2026-09-11)

- Annealing explore now prefers novel candidates. The acceptance curve
  had its sign flipped and took near-duplicates almost always.
- The search-arm bandit no longer rewards arms that produced no rows;
  idle arms take the mean observed yield and carry no vote.
- `recommend(mode="rl")` sizes the run as `goal / (k * mixed_rate)`
  instead of rounding `k * mixed_rate` to an integer first, which
  under-provisioned by 3x at a 4% mixed rate.

- Schema battle-tested against every row pool reachable: 11k local engine
  rows, both platform datasets, four agents from the public Hugging Face
  set, and an adversarial set. Two more legacy shapes are read by
  `from_row`: training exports (`messages` without `steps`; steps and
  `final_text` are derived, `tools` carried) and the Hugging Face set's
  flattened `*_json` string columns. Unknown columns now ride through
  `from_row` / `to_row` untouched, so verifiers-style `example_id` and
  `info` survive. Garbage values (a non-numeric fault rate, a
  non-integer `rollout_index`, bool or string rewards) coerce instead of
  raising. Judge rows keep `judge_status`, `judge_meta`, and both
  `judge_name` and `label_source` through the round trip.
- `examples/schema`: `migrate.py` stamps and splits any legacy file into
  rows plus a rollout-free `tasks.jsonl`; `project.py` writes eval, SFT,
  preference, GRPO, OPSD, and OPD targets from one v1 file. Offline, no key.

## 0.09 (2026-09-11)

- The simulations package moved under the namespace: `zeroproof_simulations`
  is now `zeroproof.simulations`, so the wheel is one package with one name.
  `import zeroproof_simulations` keeps working for two releases through an
  alias that resolves to the same module objects, with a deprecation
  warning. Change `import zeroproof_simulations as zps` to
  `import zeroproof.simulations as zps`. The logger is now
  `zeroproof.simulations`.

## 0.08 (2026-09-11)

- Typed row schema, additive half. `zeroproof.simulations.schema` defines
  the four objects every row projects from (`Task`, `Rollout`, `Judgment`,
  `Marker`) plus `Dataset` and `Calibration`, with `from_row` / `to_row`
  between them and the flat JSONL row. Every row the engine, `save()`,
  the exporters, and `rows_from_otel` write now carries
  `schema_version: "1"`; `.meta.json` carries it too. The wire contract is
  `zeroproof/simulations/schemas/row-v1.json`, shipped in the wheel.
- Rows without a stamp are version 0 and are read by shape: engine rows
  (by `scenario_id`), platform trace pulls (by `tool_trace`), OTel ingest
  (by `conversation_id`). Nothing that loaded before is rejected.
- Validators run at the boundaries: `push_rows`, `training_rows` /
  `export_training`, `export_preference`, `rows_from_otel`, and the row
  writer behind both the streamed file and `save()`. In this version they
  check the stamp and the required field types only.
- A test freezes the count of direct verdict-key writes per module, so
  new grading paths go through `attach` once it lands.
- `SimulationData.trajectories` stays the source of truth; the objects
  are a view until the store moves.

## 0.07 (2026-09-11)

- A bring-your-own run with no key fails at setup with a message naming
  `OPENAI_API_KEY`, instead of spending its whole time budget on 401s and
  returning zero rows. Loopback and plain-http endpoints (ollama, local
  vLLM) still need no key.
- A stop raises a flag that every rollout and writer wave checks before
  starting, so nothing begins work after the stop is declared. Closes a
  race where a wave marked running could still start its body after
  `simulate()` returned.
- The two stop tests give the hanging agent and writer six seconds and
  assert a four-second return, so a loaded CI box cannot trip them.

## 0.06 (2026-09-10)

- A stop also settles writer waves: queued waves are cancelled, running
  ones get the same `stop_grace`, and any still running are reported as
  `writer_waves_abandoned`. Found by an end-to-end run of the 0.5 wheel
  against hosted Qwen, where four writer threads outlived `simulate()`.

## 0.05 (2026-09-10)

- `simulate(reproducible=True)`: round-synchronous scheduling. Same seed,
  same concurrency, same agent gives the same rows at any concurrency.
  Costs throughput under uneven latency and needs the clock off.
- README leads with bring-your-own-model; hosted Qwen is the fallback.
- This changelog.

## 0.04 (2026-09-10)

- `zeroproof` is trace ingest only. The pre-0.3 encrypted agent-to-agent
  messaging client (`ZeroProof`, `send_encrypted`, reputation, approval
  workflows) is gone. Pin `zeroproof<0.3` to keep it.
- `simulate()` is a `run/` package with named phases (inputs, build, loop,
  finish). Behavior unchanged; verified row-for-row on a 17-configuration
  serial harness.
- A seeded serial run reproduces bit-for-bit across processes. Scenario ids
  and the axis-gap hint no longer depend on Python's per-process hash.
- A stop (clock, cap, saturation) cancels queued rollouts, waits up to
  `advanced["stop_grace"]` (5 s) for running ones, keeps what finishes, and
  reports the rest as `rollouts_abandoned`. Nothing calls the agent after
  `simulate()` returns.
- Progress goes through the `zeroproof.simulations` logger instead of
  `print()`.
- `zeroproof.simulations` ships `py.typed`.
- Trace ingest defaults to `https://api.zeroproofai.com`.
- README parameter tables match the code (`concurrency` 32, `time_budget`
  `None`) and a test keeps them matching.
- A failed tool draft for a prompt-only agent is a `tool_draft_unavailable`
  degraded note instead of a silent tool-free run.
- Two timing-dependent tests made deterministic. `zeroproof.__version__`
  reads package metadata.

## 0.3 (2026-08-31)

- The `zeroproof` package absorbed `zeroproof-simulations`, which is
  deprecated on PyPI. Releases of `zeroproof` before 0.3 were an unrelated
  encrypted messaging client.
