"""Every number the run engine used to carry inline, named, explained and
tunable.

Each constant below has a comment of the form
``# <name> = <value>: <why> (<source>)``. The source is a measurement, a
chapter of https://rlhfbook.com, an arXiv id, or the honest words
"convention, untested". A constant with no source is a bug.

Two channels reach the code:

* the module-level constants, for values shared across files (the same
  number must mean the same thing everywhere it appears);
* :class:`RunKnobs`, one field per engine knob, which ``simulate()``
  reads from ``advanced={...}`` under the field's name and validates
  against the bounds in the field's metadata. A knob out of range is a
  ``ValueError`` that names the bound.

Sections are per owner so parallel work reconciles by section, not by
line: ``run/`` (engine, config, rows, simulation, data) and ``generate/``
(agents, generator, diversity, ...) are below. Other stages add their own
section.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import Any

# ---------------------------------------------------------------------
# shared across stages
# ---------------------------------------------------------------------

# DEFAULT_BUDGET = 1000: the row cap when simulate() is given none. A round
# number; large enough for a coverage curve to flatten on a three-tool
# agent (explore runs plateau under 300 rows) and small enough to finish on
# a trial key. (convention, untested)
DEFAULT_BUDGET = 1000

# DEFAULT_CONCURRENCY = 32: parallel rollouts, and the cap on parallel
# judge calls. One served vLLM replica's continuous-batching sweet spot
# for short chat turns; no throughput curve was recorded for this
# package. (convention, untested)
DEFAULT_CONCURRENCY = 32

# PASS_THRESHOLD = 0.5: a reward under this is a failure. The outcome
# label is binary, r in {0, 1} (rlhf-book ch. 7, outcome reward models;
# ch. 14, verifiable rewards), so 0.5 is its midpoint and a partial
# rubric score (or the conduct advisory 0.5 for a truncated reply) rounds
# to the nearer verdict. No source names another cut. The run loop's
# graded-failure gate, score/ audits and the exporters count fails the
# same way; import this rather than writing 0.5 again.
PASS_THRESHOLD = 0.5

# PASS_REWARD = 1.0: the reward a fully passing row carries; anything
# below it counts as failing in the arm-yield tally. (the [0, 1] reward
# scale every judge in the package writes)
PASS_REWARD = 1.0

# SHORT_HASH_CHARS = 16: hex chars kept from a sha256 when a row names a
# system prompt, a fingerprint or a policy hash. 64 bits: collisions
# among the prompts one account will ever run are not a concern.
# (convention, untested)
SHORT_HASH_CHARS = 16

# MAX_COMPLETIONS_PER_REQUEST = 8: writer completions one request may ask
# for (the ``n`` of a chat call). Mirrors
# ``generate.generator._MAX_COMPLETIONS``; the two must agree.
# (convention, untested)
MAX_COMPLETIONS_PER_REQUEST = 8

# FAULT_STATUSES = {error, timeout, not_found, denied, malformed}: a tool
# result ``status`` that means the call failed
# (the run's own vocabulary; a status outside this set is the tool's own
# word, not a fault, #261). ``score.grading._KNOWN_FAULTS`` is the wider
# alias table; this is the subset the engine re-rolls and mutates on.
FAULT_STATUSES = frozenset({"error", "timeout", "not_found", "denied", "malformed"})

# OK_STATUSES = {ok, success}: a tool result ``status`` that means the call
# worked (the run's own vocabulary, #261).
OK_STATUSES = frozenset({"ok", "success"})

# LEAK_MIN_QUOTE_CHARS = 12: the shortest run of characters that counts as
# quoting the privileged block. Shorter matches are common words.
# Mirrors ``score.privileged.leak_report(min_len=)``. (convention, untested)
LEAK_MIN_QUOTE_CHARS = 12

# ---------------------------------------------------------------------
# run/: engine, config, rows, simulation, data
# ---------------------------------------------------------------------

# RL_ROLLOUTS_PER_PROMPT = 8: k under mode="rl". A grouped update (GRPO,
# rlhf-book ch. 11) needs enough samples per prompt for the group mean to
# be a usable baseline. 8 is what Dr. GRPO trains with (arXiv 2503.20783,
# 8 responses per question) and the agentic-RL recipe in arXiv 2603.21972
# (G=8); DAPO (arXiv 2503.14476), ProRL (arXiv 2505.24864) and Skywork-OR1
# (arXiv 2505.22312) use 16. The package differs from the 16 camp on
# purpose: past K=4 resampling cuts eval variance by less than a sixth
# more (Miller, arXiv 2411.00640), so 8 is a first look at half the
# cost; raise repeats= to 16 for a training set on hard prompts, where
# 8 leaves more zero-variance groups.
RL_ROLLOUTS_PER_PROMPT = 8

# SFT_PHRASINGS_PER_SITUATION = 3: n under mode="sft". Three wordings of
# one situation give the SFT set paraphrase variety without tripling the
# situation count. (convention, untested)
SFT_PHRASINGS_PER_SITUATION = 3

# DEFAULT_PROBE = 2: rollouts a prompt gets before the run decides whether
# its group splits; two is the least that can disagree. A unanimous group
# carries no gradient, so DAPO (arXiv 2503.14476, eq. 11) and ProRL (arXiv
# 2505.24864) drop prompts whose accuracy is 0 or 1 after the fact, and
# GRESO (arXiv 2506.02177) predicts zero-variance prompts before spending
# rollouts on them. Probing first is the engine's version of that; no
# paper states a probe size, so 2 is the structural minimum, not a
# measured optimum.
DEFAULT_PROBE = 2

# RL_FAULT_RATE = 0.8: share of fault-tagged cells that keep their fault
# plan under mode="rl", above the 0.5 explore default
# (generate.scenarios.DEFAULT_FAULT_RATE). This is not a per-call failure
# rate: tagged cells are a small slice of the grid, so rows with a fault
# stay under about 10% at 0.5, which is the band training-time injection
# is stable in (arXiv 2603.21972: above 5-10% destabilised a 3B agent;
# AgentCE-Bench, arXiv 2604.06111, evaluates p in {0, 0.1, 0.3}). RL
# raises it because the faulted cells are where a base fails
# (rlhf-book ch. 14, difficulty filtering); PALADIN trains on an 80/20
# mix of recovery-bearing to clean traces (arXiv 2509.25238), which is
# the shape 0.8 gives the tagged slice. Untested against 0.5 on a
# training run.
RL_FAULT_RATE = 0.8

# DEFAULT_AVG_TURNS = 12: target conversation length. The person speaks at
# most avg_turns // 2 times, which leaves room to verify, look up,
# confirm and write. The cap (generate.agents.default_max_turns, 40 on an
# 8k context) sits above what published agent data reaches: tau-bench
# stops a task at 30 agent actions (arXiv 2406.12045) and APIGen-MT
# trajectories top out at 29 turns (arXiv 2504.03601). The mean of 12 is
# a convention, untested against source traces; fit it from traces=
# when you have them.
DEFAULT_AVG_TURNS = 12.0

# DEFAULT_MIN_USER_TURNS = 1: the person always speaks at least once.
DEFAULT_MIN_USER_TURNS = 1

# AGENT_MAX_TOKENS_FLOOR = 64: the smallest reply budget simulate() accepts;
# below it every reply is cut mid-sentence and scores 0. (convention)
AGENT_MAX_TOKENS_FLOOR = 64

# DEFAULT_SEED = 0
DEFAULT_SEED = 0

# DEFAULT_POOL_SIZE = 80: prompts the situation writer keeps per round
# (advanced per_round). (convention, untested)
DEFAULT_POOL_SIZE = 80

# DEFAULT_WRITER_FLIGHT = 4: writer waves in flight. Too many starves
# rollouts of the GPU; how many is too many was never measured.
# (convention, untested)
DEFAULT_WRITER_FLIGHT = 4

# DEFAULT_CARDS_PER_WAVE = 8: situation cards one writer wave asks for.
# (convention, untested)
DEFAULT_CARDS_PER_WAVE = 8

# DEFAULT_COMPLETIONS_PER_REQUEST = 1
DEFAULT_COMPLETIONS_PER_REQUEST = 1

# DEFAULT_EXTRA_CARDS = 1: spare cards per wave so a rejected card does not
# leave the wave short. (convention, untested)
DEFAULT_EXTRA_CARDS = 1

# SATURATION_CAP = 50_000: rows a saturation-bounded run may produce before
# the loop gives up when budget=None. (convention, untested)
SATURATION_CAP = 50_000

# HUNG_SLOT_S = 45.0: kill a hung slot after this wait; the row is
# omitted. Ping-pong is several HTTP calls; 24 s dropped healthy
# two-person traces and the replacement oversubscribed the GPU.
# (observed on the hosted endpoint; 45 is a convention above that)
HUNG_SLOT_S = 45.0

# STOP_GRACE_S = 5.0: after a stop, wait this long for rollouts already
# running; queued ones are cancelled at once. (convention, untested)
STOP_GRACE_S = 5.0

# DEAD_AGENT_MIN_ERRORS = 16: an agent that raises on every call is called
# off once this many rollouts were lost with no row landed, or
# DEAD_AGENT_BUDGET_MULTIPLE x budget, whichever is larger (#88).
# (convention, untested)
DEAD_AGENT_MIN_ERRORS = 16
# DEAD_AGENT_BUDGET_MULTIPLE = 2: the budget multiple in that rule; twice
# the rows asked for is more failures than any live agent produces.
# (convention, untested)
DEAD_AGENT_BUDGET_MULTIPLE = 2

# ALLOC_GAIN = 4.0: how hard a hot trace region pulls cell weight toward
# itself; a region with budget share s and a full recipe match multiplies
# the cell weight by 1 + 4 s. (convention, untested)
ALLOC_GAIN = 4.0

# TIER_MIX_MIN_ROWS = 20: rows before the realized difficulty mix is
# compared with the ask. At 20 rows a share of 0.4 has binomial sd
# sqrt(0.4 x 0.6 / 20) = 0.11, about the tolerance below, so a smaller
# run cannot tell a shortfall from noise. (binomial sd; the 20 is the
# smallest n that gets the sd to the tolerance)
TIER_MIX_MIN_ROWS = 20
# TIER_MIX_TOLERANCE = 0.10: how far (in share) the drawn rows may land
# below the ask before the run says so. (convention, untested)
TIER_MIX_TOLERANCE = 0.10

# PROGRESS_MIN_ROWS_FOR_ESTIMATE = 5: rollouts before the rate is worth
# extrapolating; before that the estimate is the first rollout's latency
# dressed up as a forecast. (convention, untested)
PROGRESS_MIN_ROWS_FOR_ESTIMATE = 5
# PROGRESS_MIN_BUDGET = 10: runs smaller than this say nothing; they are
# over before a line helps. (convention, untested)
PROGRESS_MIN_BUDGET = 10

# SYSTEM_PROMPT_HEAD_CHARS = 120: opening chars of the system prompt kept
# on every row, enough to tell a numbered policy from a bare prompt at a
# glance (#296). (convention, untested)
SYSTEM_PROMPT_HEAD_CHARS = 120

# PARENT_HEAD_CHARS = 80: chars of a failing prompt that name it as a
# mutation parent; the mutated row's ``parent`` and the aim table key
# on the same head, so the two must agree. (convention, untested)
PARENT_HEAD_CHARS = 80

# SCENARIO_ID_CHARS = 6: hex chars of the prompt hash that name a probe
# row's scenario_id (24 bits; ids are per run). (convention, untested)
SCENARIO_ID_CHARS = 6

# REPORT_LIST_ITEMS = 20: how many missing pinned prompts a report lists
# before it stops. (convention, untested)
REPORT_LIST_ITEMS = 20

# FINGERPRINT_STEM_MIN_LEN = 4: words longer than this lose a trailing
# "s" before fingerprinting, so "refunds" and "refund" match and "is"
# stays "is". (convention, untested)
FINGERPRINT_STEM_MIN_LEN = 4

# TOOL_SCHEMA_SPAN_CHARS = 500: how far apart "name", "description" and
# "parameters" may sit in visible text and still read as a leaked tool
# schema. (convention, untested)
TOOL_SCHEMA_SPAN_CHARS = 500

# DEFAULT_LLM_JUDGE_CONCURRENCY = 16: parallel calls to the hosted LLM
# judge; half the rollout concurrency because the judge is one model
# serving every account. (convention, untested)
DEFAULT_LLM_JUDGE_CONCURRENCY = 16

# JUDGE_CONCURRENCY_CAP = 32: the most parallel judge calls ``grade()``
# lets a caller ask for, whatever ``concurrency=`` says. (convention,
# untested)
JUDGE_CONCURRENCY_CAP = 32

# DEFAULT_SELECT_TARGET = 1000: rows ``select()`` / ``export_training()``
# aim for when the caller names no target. (convention, untested)
DEFAULT_SELECT_TARGET = 1000

# HOLDOUT_BUCKET_HEX_CHARS = 8: hex chars of the task hash that place a
# task on the train or holdout side; 32 bits is fine resolution for a
# fraction. (convention, untested)
HOLDOUT_BUCKET_HEX_CHARS = 8

# ---------------------------------------------------------------------
# generate/: HTTP transport (agents.py, anthropic_backend.py)
# ---------------------------------------------------------------------

# TRANSIENT_TRIES = 3: a 5xx, a 429 (Anthropic) or a dropped Modal request
# is retried this many times before the rollout is lost (convention,
# untested; both backends read it so neither is silently flakier).
TRANSIENT_TRIES = 3
# TRANSIENT_BACKOFF_S = 0.4: the first sleep before a transient retry,
# doubled each try (0.4, 0.8, 1.6 s) (convention, untested).
TRANSIENT_BACKOFF_S = 0.4
# MAX_SAMPLES_PER_CALL = 8: the most completions one request asks for
# with ``n``; vLLM prefills once for all of them, and above eight a busy
# endpoint dropped the request (convention from the hosted pool;
# run/config.py caps completions_per_request at the same number inline).
MAX_SAMPLES_PER_CALL = 8

# ---------------------------------------------------------------------
# generate/: context budgets (agents.py, generator.py)
# ---------------------------------------------------------------------

# CHARS_PER_TOKEN = 3: the token estimate every context budget is sized
# with, for the agent's history and the writer's prompt alike. English
# tokenizers sit at 3.5 to 4.5 chars per token, so this over-counts and
# the budget errs on the safe side (convention, untested on this data).
CHARS_PER_TOKEN = 3

# ---------------------------------------------------------------------
# score: statistics
# ---------------------------------------------------------------------
#
# The statistics every verdict in ``score/`` rests on. ``holdout_size``,
# ``detectable_effect``, ``noise_band``, ``bootstrap_ci``, ``compare_runs``
# and ``delta_report`` take these as keywords (``alpha=``, ``power=``,
# ``level=``, ``n_boot=``); the constants are only their defaults.

# ALPHA = 0.05: two-sided false-positive rate behind every interval and
# verdict. The convention the evaluation literature runs on (Miller 2024,
# arXiv:2411.00640, section 5, plugs alpha=0.05 into its power formula;
# rlhf-book ch. 16 reports 95% intervals). Untested against any other value.
ALPHA = 0.05
# CI_LEVEL = 0.95: the interval every ``ci95`` key carries. ``1 - ALPHA`` so
# the interval and the verdict agree: a delta whose interval excludes zero
# at CI_LEVEL is the one a test at ALPHA rejects.
CI_LEVEL = 1.0 - ALPHA
# Z_95 = 1.96: the normal quantile at CI_LEVEL, rounded to the two decimals
# every table prints (Miller 2024 writes the interval as 1.96 x SE). Kept
# at 1.96 rather than 1.959964 so a printed band can be checked by hand;
# the difference moves a band in the fourth decimal.
Z_95 = 1.96
# POWER = 0.8: the chance a holdout of the size ``holdout_size`` names
# detects a real gain. beta = 0.20 is the example Miller 2024 (section 5)
# and the classical power literature use; rlhf-book ch. 16 says the point
# of a better eval is statistical power without naming a number.
POWER = 0.8
# BOOTSTRAP_DRAWS = 2000: resamples behind a percentile interval. Efron and
# Tibshirani put the floor for percentile intervals at 1000; 2000 halves
# the Monte Carlo error on the endpoints and still runs in well under a
# second on a few hundred tasks. Convention above the floor, untested
# against 1000.
BOOTSTRAP_DRAWS = 2000
# MIN_CI_TASKS = 3: tasks a bootstrap interval needs. Below three the
# resampled statistic is one of a handful of arrangements of the data
# itself, so the interval says nothing (convention, untested).
MIN_CI_TASKS = 3
# MIN_RERUNS = 3: re-runs of one eval before ``run_std`` is read as a
# spread. Two runs give one difference, not a distribution; three is the
# fewest that give a sample sd with two degrees of freedom (rlhf-book
# ch. 16 on re-run variance; convention on the count).
MIN_RERUNS = 3
# BASE_PASS_RATE = 0.6: the before-side pass rate ``holdout_size`` assumes
# when no rows are given. The centre of the 20-80 difficulty band, where a
# binary task carries the most variance and the sizing is most
# conservative. Measured lanes sat between 0.5 and 0.7 (#288).
BASE_PASS_RATE = 0.6
# ROLLOUTS_PER_TASK = 4: the per-task rollout count the sizing assumes and
# the smallest k ``pass_at`` reports pass^k at. tau-bench (arXiv:2406.12045)
# plots pass^k to k=8 from at least 3 trials; tau2-bench (arXiv:2506.07982)
# runs every task 4 times and reports pass^1 and pass^4. Four is the
# smallest count those benchmarks report a k-way number on.
ROLLOUTS_PER_TASK = 4

# ---------------------------------------------------------------------
# score: difficulty band
# ---------------------------------------------------------------------
#
# DIFFICULTY_BAND = (0.2, 0.8): keep tasks the current policy passes between
# 20% and 80% of the time. rlhf-book ch. 14 ("Common Practices in Training
# Reasoning Models"): difficulty filtering restricts RL prompts to those
# the starting model solves 20-80% of the time, measured from N=16
# samples. DAPO (arXiv:2503.14476) is the online form: groups with
# accuracy 0 or 1 are dropped from the batch (G=16). The band edges are a
# reported practice, not an ablation, so every selector takes ``band=``.
DIFFICULTY_BAND: tuple[float, float] = (0.2, 0.8)
# DIFFICULTY_BAND_ROLLOUTS = 16: rollouts per task the band is measured
# from in the sources above (rlhf-book ch. 14 N=16; DAPO G=16). Below it
# a task's band assignment carries a Wilson half-width near 0.3 at k=8.
DIFFICULTY_BAND_ROLLOUTS = 16
# REJECTION_SAMPLING_MIN_K = 10: completions per prompt a best-of-N pick
# wants. Llama 3 (arXiv:2407.21783, section 4.2.2) samples K between 10 and
# 30 per prompt; rlhf-book ch. 9 repeats the range. Fewer makes the pick a
# filter, not a choice.
REJECTION_SAMPLING_MIN_K = 10
# RL_ROLLOUTS_PER_ASK = 8: the rollouts per ask ``recommend`` sizes an RL
# run for; the same quantity as RL_ROLLOUTS_PER_PROMPT (mode="rl" k) under
# the score/ name, so the two cannot drift. One value, one home.
RL_ROLLOUTS_PER_ASK = RL_ROLLOUTS_PER_PROMPT

# ---------------------------------------------------------------------
# score: decontamination
# ---------------------------------------------------------------------
#
# DECONTAM_NGRAM = 8: word n-gram behind the near-copy rule. Tulu 3
# (arXiv:2411.15124) decontaminates on 8-gram overlap between training
# prompts and evaluation prompts; rlhf-book ch. 16 cites the same rule.
DECONTAM_NGRAM = 8
# DECONTAM_OVERLAP = 0.8: share of a row's words one eval text has to
# cover with shared 8-grams. The Llama 2 rule (80% of tokens). Tulu 3 uses
# 50%; this package keeps 80% on purpose: template-written situations
# share whole sentences that say nothing about which question was asked,
# and at 50% the near-copy rule flags rows that never saw the eval
# question (#286). ``overlap=`` sets it per call.
DECONTAM_OVERLAP = 0.8
# SEMANTIC_SIMILARITY = 0.85: cosine at or above which two prompts read as
# one task to an embedder. Read off BGE-small (unrelated prompts score
# about 0.55 there, paraphrases above 0.85, #286). SemDeDup
# (arXiv:2303.09540) removes pairs within eps 0.03 to 0.07 of each other
# on CLIP and OPT embeddings (cosine 0.93 or higher); that is a duplicate
# threshold for pretraining text, not a paraphrase threshold for task
# prompts, so this package sits lower and calibrates against the eval
# set's own distinct-task similarity at run time.
SEMANTIC_SIMILARITY = 0.85

# ---------------------------------------------------------------------
# score: judge floors
# ---------------------------------------------------------------------
#
# MIN_GOLD = 50: human labels before a judge's accuracy number means
# anything. rlhf-book ch. 7 ("Suggested Experiments"): a 50- to 200-example
# held-out set is the useful size for tuning a reward model; below 50 the
# Wilson interval on agreement is about +/-0.1 wide.
MIN_GOLD = 50
# MAX_GOLD_ASK = 200: the most labels the trust report will ask a person
# for before it says the judge itself is the problem (the top of the same
# 50-200 range).
MAX_GOLD_ASK = 200
# MIN_AGREEMENT = 0.8: Wilson lower bound of judge-human agreement a judge
# has to reach. Zheng et al. (arXiv:2306.05685, MT-Bench): human-human
# agreement is 81% and GPT-4 reaches 85% against humans, so a judge under
# 80% agrees with people less than people do with each other.
MIN_AGREEMENT = 0.8
# MIN_KAPPA = 0.6: chance-corrected agreement floor. Landis and Koch (1977)
# call 0.61-0.80 "substantial"; 0.6 is the bottom of that band. A 2025
# sweep of 21 judges on MT-Bench measured kappa 0.38-0.51 against human
# preference labels (arXiv:2606.19544), so this floor is demanding on
# purpose: it asks for agreement beyond what raw accuracy hides.
MIN_KAPPA = 0.6
# JUDGE_CHECK_SAMPLE = 40: rows a judge check (perturbation, probes, the
# verifier audit) re-judges by default. Enough that a 10% effect shows as
# four flips; kept under MIN_GOLD because these checks cost a judge call
# per row (convention, untested).
JUDGE_CHECK_SAMPLE = 40
# LENGTH_GAP_FLAG = 0.15: judge pass rate gap between short and long
# replies with the same human label that reads as length bias. Zheng et
# al. (arXiv:2306.05685) found a 91% failure rate on a verbosity attack
# for GPT-3.5 and 9% for GPT-4; a 2025 sweep (arXiv:2606.19544) measured
# verbosity correlations under 0.011 on current judges. 15 points sits
# between the two eras: a judge over it is behaving like a 2023 judge.
# Convention on the exact number.
LENGTH_GAP_FLAG = 0.15
# FLIP_FLAG = 0.10: share of verdicts that change on an identical re-judge,
# on appended filler, or under a probe, before the judge is called
# exploitable. Test-retest consistency of current judges is 0.89-0.99
# (arXiv:2606.19544), so a tenth of verdicts moving is far outside the
# measured range. Convention on the exact number.
FLIP_FLAG = 0.10
# POSITION_FLIP_FLAG = 0.2: share of pairs a pairwise judge decides
# differently when A and B are swapped before its position bias is a
# warning. Zheng et al. (arXiv:2306.05685, Table 2) measured 65%
# consistency for GPT-4 (35% flipped) and 46% for GPT-3.5; the swap-both-
# ways rule already turns a flip into a tie, so the flag marks a judge
# whose prompt needs work, not a broken report. Convention on the number.
POSITION_FLIP_FLAG = 0.2

# ---------------------------------------------------------------------
# score: judge payload
# ---------------------------------------------------------------------
#
# What the LLM judge is shown. Every cap is a character budget on the JSON
# the judge reads; ``grade_one``, ``apply_grade_llm`` and ``audit_grades``
# take ``payload_chars=`` to move the total.

# JUDGE_PAYLOAD_CHARS = 8000: the whole judge payload. About 2000 tokens,
# which leaves a 4B hosted judge with an 8k window room for its system
# prompt and reply. Measured: at this cap 37 of 120 rows on one paired
# eval were being cut mid-JSON before #290 taught the payload to shrink
# structure instead; the cap itself is the window, not a finding.
JUDGE_PAYLOAD_CHARS = 8000
# JUDGE_SITUATION_CHARS = 4000: the user request as the judge sees it.
# Half the payload: a situation longer than this is a document, and the
# verdict needs the reply and the steps more (convention, untested).
JUDGE_SITUATION_CHARS = 4000
# JUDGE_FINAL_TEXT_CHARS = 2000: the agent's final reply. A quarter of the
# payload; a reply past it is judged on its first 2000 characters and the
# cut is announced in the text (convention, untested).
JUDGE_FINAL_TEXT_CHARS = 2000
# JUDGE_POLICY_CHARS = 2000: the agent's policy or harness text, and each
# judge-only field (principle, reference). Same budget as the reply
# (convention, untested).
JUDGE_POLICY_CHARS = 2000
# JUDGE_MAX_TOKENS = 120: the judge's reply budget. Reason first, then a
# score, in one sentence: 4B judges needed room for the sentence and
# 120 tokens held every reply measured; a pairwise verdict is the same
# shape. Raise it for a judge asked to explain at length.
JUDGE_MAX_TOKENS = 120
# JUDGE_TEMPERATURE = 0.0: a judge is read at zero for stable ratings
# (rlhf-book ch. 7, LLM-as-judge prompt rules; Zheng et al.
# arXiv:2306.05685 use temperature 0 for judging).
JUDGE_TEMPERATURE = 0.0

# ---------------------------------------------------------------------
# score: reply truncation
# ---------------------------------------------------------------------
#
# TRUNCATED_REPLY_CHARS = 600: a final reply longer than this that does not
# end on terminal punctuation or a sign-off is read as cut by the token
# cap (conduct advisory 0.5; junk for SFT). Short replies get the benefit
# of the doubt: a one-line answer often ends on a number or a name.
# Convention, untested; ``hygiene.is_truncated`` uses a looser 200 for its
# report-only count.
TRUNCATED_REPLY_CHARS = 600

# ---------------------------------------------------------------------
# score: reward hacks
# ---------------------------------------------------------------------
#
# HACK_THRESHOLD = 0.3: |corr(reward, feature)| at or above this is flagged
# as a shortcut the policy will learn. From the RLVR signal sweeps the
# scan descends from: the endorsed feature cleared 0.5 and delimiter
# hacks sat near 0.9, so 0.3 catches a hack before it dominates. Gao et
# al. (arXiv:2210.10760) is the mechanism: optimising a proxy the gold
# reward does not credit. Measured on our own lanes, not published.
HACK_THRESHOLD = 0.3

# ---------------------------------------------------------------------
# world (sandbox)
# ---------------------------------------------------------------------
#
# Reachable through ``MockEnvironment(options=WorldOptions(...))``,
# ``export_environment(world=...)`` / ``load_environment(world=...)`` and
# ``simulate(advanced={"world": {...}})``. The sandbox is seeded: changing a
# value here changes the golden rows (scripts/golden.py), so a change is a
# stated diff, never a drive-by.

# WORLD_DEFAULT_FAULT_MODE = "timeout": the fault a plan gets when it names
# none. Timeout is the most common runtime tool failure in the injection
# benchmarks (first of six perturbation types in arXiv:2605.11928; first of
# seven in arXiv:2606.01416); which mode a plan carries is decided upstream
# by the tool_condition axis (generate/scenarios.py), not here.
WORLD_DEFAULT_FAULT_MODE = "timeout"
# WORLD_DEFAULT_FAULT_RATE = 1.0: a fault plan without a rate fires on every
# call. The injection benchmarks fire at 100% per sample so every model is
# tested at the same trajectory step (arXiv:2605.11928 §4); how often a
# situation carries a plan at all is ``simulate(fault_rate=)`` (default in
# generate/scenarios.py, DEFAULT_FAULT_RATE), a separate dial.
WORLD_DEFAULT_FAULT_RATE = 1.0
# WORLD_STALE_AS_OF = "3 days ago": the age stamped on a stale read. A
# stale answer is a real record with an age, so the agent can notice it
# is old (measured: a hash taught agents to invent a shipment around it and
# a rubric judge passed them). The age itself is convention, untested.
WORLD_STALE_AS_OF = "3 days ago"
# WORLD_MALFORMED_PAYLOAD = "<<garbled resp0nse": what a malformed fault
# returns. Unparseable text
# is the "Malform" perturbation of arXiv:2605.11928 (malformed JSON);
# convention for the exact bytes.
WORLD_MALFORMED_PAYLOAD = "<<garbled resp0nse"
# WORLD_EXISTS_SHARE = 0.7: share of user-named references that exist in the
# world when no state says otherwise. Under 1.0 so "not found" is a path
# the agent meets without a fault plan; the number is convention, untested.
WORLD_EXISTS_SHARE = 0.7
# WORLD_SEARCH_HITS = (1, 6): a search or listing returns this many records,
# inclusive. Measured: a fixed 2 to 3 taught models that searches always
# return two or three results.
WORLD_SEARCH_HITS = (1, 6)
# WORLD_TEMPLATE_HITS = (1, 5): records a model-written list template
# expands to, inclusive. Convention, untested; same reasoning as search hits.
WORLD_TEMPLATE_HITS = (1, 5)
# WORLD_ID_RANGE = (1000, 90000): generated record ids, lower inclusive,
# upper exclusive. Four to five digits reads as an id and never collides
# with the ``<stem>_<hex>`` ids the world issues on create. Convention.
WORLD_ID_RANGE = (1000, 90000)
# WORLD_DATE_YEARS = (2022, 5): generated dates fall in these calendar
# years (first year, span). Convention: recent enough to read as live data,
# fixed so seeded rows do not drift with the clock.
WORLD_DATE_YEARS = (2022, 5)
# WORLD_JITTER_DIVISOR = 3: a number in a model-written result template
# moves by up to one part in this many of itself (900 lands in 600 to
# 1200). Wide enough that a policy threshold near the template value is
# crossed both ways; convention, untested. Documented on local_model().
WORLD_JITTER_DIVISOR = 3
# WORLD_CREATED_ID_MODULUS = 100000: issued ids on create are
# ``<stem>_<n mod this>``. Five digits; convention.
WORLD_CREATED_ID_MODULUS = 100_000
# WORLD_ISSUED_ID_HEX = 12: hex characters in a world-issued id and in the
# per-call digest. 48 bits: no collision in any run this package makes.
WORLD_ISSUED_ID_HEX = 12
# WORLD_REF_CHARS = 8: characters of the call digest echoed as ``ref``.
WORLD_REF_CHARS = 8
# WORLD_EXPRESSION_CHARS = 200: a calculator expression is echoed back cut
# to this many characters, so a runaway argument cannot bloat a row.
WORLD_EXPRESSION_CHARS = 200
# WORLD_HINT_CHARS = 40: characters of a caller's query kept as the name
# stem of generated records. Long enough for a product name; convention.
WORLD_HINT_CHARS = 40
# WORLD_SHELL_FLAVORS = 11: one shell call in this many is a permission
# error, one a failing test run, one a merge conflict, one an ``ls``; the
# rest pass. About a third of shell calls therefore fail, near the
# per-call tool error rates the tau-bench family reports for weaker agents
# (arXiv:2406.12045 reports pass^1 under 50% on retail). Convention.
WORLD_SHELL_FLAVORS = 11
# WORLD_CI_FAIL_ONE_IN = 7: one CI listing in this many carries a failed
# check. Convention, untested.
WORLD_CI_FAIL_ONE_IN = 7

# ---------------------------------------------------------------------
# ingest (traces)
# ---------------------------------------------------------------------
#
# Reachable through keywords on the ingest helpers (``mine_result_exemplars``,
# ``leakage_report``, ``split_pseudo_production``, ``behavior_state``,
# ``dimensions_from_traces``).

# TRACE_EXEMPLARS_PER_TOOL = 3: real result payloads kept per tool as shape
# templates. Three shows the shape family (record, list, text) without
# repeating it; convention, untested.
TRACE_EXEMPLARS_PER_TOOL = 3
# TRACE_EXEMPLAR_MAX_CHARS = 500: serialized size cap on one exemplar, about
# 125 tokens at 4 characters a token, so three per tool stay under a few
# hundred tokens of the writer prompt. Token-budget reasoning; the exact
# number is convention.
TRACE_EXEMPLAR_MAX_CHARS = 500
# TRACE_EXEMPLAR_STRING_CHARS = 160: a string inside an exemplar is cut here
# with an ellipsis; TRACE_EXEMPLAR_LIST_ITEMS = 2 and
# TRACE_EXEMPLAR_DICT_KEYS = 12 bound the nesting so the 500-char cap is
# reachable by trimming, not by dropping the whole payload. Convention.
TRACE_EXEMPLAR_STRING_CHARS = 160
TRACE_EXEMPLAR_LIST_ITEMS = 2
TRACE_EXEMPLAR_DICT_KEYS = 12
# TRACE_LEAK_THRESHOLD = 0.9: cosine similarity at or above which a
# generated prompt is a near copy of a source trace. Exact matches always
# flag. rlhfbook.com/c/16-evaluation.html ("Contamination") uses 8-gram
# overlap for exact leakage; 0.9 on hashed n-gram vectors is the same test
# with a small paraphrase allowance. Convention for the exact number.
TRACE_LEAK_THRESHOLD = 0.9
# TRACE_LEAK_EXAMPLES = 20: offenders listed in a leakage report; the count
# is always complete. Report size, convention.
TRACE_LEAK_EXAMPLES = 20
# TRACE_PSEUDO_PRODUCTION_FRACTION = 0.2: rows set aside as pseudo
# production. The usual 80/20 train/holdout split; convention.
TRACE_PSEUDO_PRODUCTION_FRACTION = 0.2
# TRACE_MIN_SUPPORT = 3: graded rows a behavior region needs before its
# allocation is more than a hint. Statistical reason: the Clopper-Pearson
# 95% upper bound on a 0-of-n fail rate is 0.95 at n=1, 0.78 at n=2 and
# 0.63 at n=3, so three is the smallest count at which "never failed" rules
# out a majority fail rate.
TRACE_MIN_SUPPORT = 3
# TRACE_EXPLORATION_FLOOR = 0.2 and TRACE_EXPLORATION_MAX = 0.6: share of
# the budget always kept for broad coverage (floor) and the most a caller
# can reserve (max). An allocation that follows observed failures alone
# never finds a new one; 20% is an epsilon-greedy convention, untested.
TRACE_EXPLORATION_FLOOR = 0.2
TRACE_EXPLORATION_MAX = 0.6
# TRACE_STATE_PRIORITY = {new: 1.0, ...}: status weight in the region
# priority. New and persistent failures draw first; a solved region keeps a small weight so
# it is re-checked. Convention, untested.
TRACE_STATE_PRIORITY = {
    "new": 1.0,
    "persistent": 0.9,
    "uncertain": 0.35,
    "improving": 0.2,
    "solved": 0.05,
    "passing": 0.05,
}
# TRACE_SUPPORT_SATURATION = 6: failures at which the support factor
# reaches 1.0 (it starts at 0.5 with none). Convention, untested.
TRACE_SUPPORT_SATURATION = 6
# TRACE_RATE_FLOOR = 0.25: the rate factor is floor + (1 - floor) * fail
# rate, so a region that never fails still draws a quarter of the weight a
# region that always fails does. Convention, untested.
TRACE_RATE_FLOOR = 0.25
# TRACE_REPORT_LIST_CAP = 5: names shown per axis in the text report.
TRACE_REPORT_LIST_CAP = 5
# TRACE_TASK_HASH_CHARS = 200: prompt characters in the split hash.
TRACE_TASK_HASH_CHARS = 200

# ---------------------------------------------------------------------
# connect (platform)
# ---------------------------------------------------------------------
#
# Reachable through ``timeout=`` / ``poll=`` keywords on the platform calls.

# PLATFORM_REQUEST_TIMEOUT_S = 120: one gate request. A finalize on a large
# set takes tens of seconds; two minutes is convention, untested.
PLATFORM_REQUEST_TIMEOUT_S = 120
# PLATFORM_UPLOAD_TIMEOUT_S = 300: the studio import, which grades every row
# server side before answering. Convention, sized to a 20k-row push.
PLATFORM_UPLOAD_TIMEOUT_S = 300
# PLATFORM_CREDENTIAL_TTL_S = 3600: default life of a delegated credential.
# One hour matches the Clerk session token that mints it.
PLATFORM_CREDENTIAL_TTL_S = 3600
# PLATFORM_ERROR_DETAIL_CHARS = 400: gate error body echoed in the
# exception. Enough for the gate's one-line reason; convention.
PLATFORM_ERROR_DETAIL_CHARS = 400
# PLATFORM_STUDIO_MAX_ROWS = 20000: the studio import endpoint's cap.
PLATFORM_STUDIO_MAX_ROWS = 20_000
# PLATFORM_HF_DATASET_TIMEOUT_S = 600 / PLATFORM_HF_MODEL_TIMEOUT_S = 900:
# how long ``hf_publish`` / ``hf_publish_run`` wait for the Hugging Face
# push; an adapter upload is bigger than a JSONL set. Convention.
PLATFORM_HF_DATASET_TIMEOUT_S = 600.0
PLATFORM_HF_MODEL_TIMEOUT_S = 900.0
# PLATFORM_HF_POLL_S = 2.5 / PLATFORM_HF_MODEL_POLL_S = 3.0 /
# PLATFORM_IMPORT_POLL_S = 3.0: seconds between status reads while waiting.
PLATFORM_HF_POLL_S = 2.5
PLATFORM_HF_MODEL_POLL_S = 3.0
PLATFORM_IMPORT_POLL_S = 3.0
# PLATFORM_IMPORT_TIMEOUT_S = 900 / PLATFORM_IMPORT_MAX_ROWS = 100000: an
# import of a Hugging Face split. The row cap is the gate's.
PLATFORM_IMPORT_TIMEOUT_S = 900.0
PLATFORM_IMPORT_MAX_ROWS = 100_000
# PLATFORM_TRACE_PAGE_SIZE = 200: traces read per page when listing an
# agent's traces (the ``limit=`` the traces route takes). Convention.
PLATFORM_TRACE_PAGE_SIZE = 200
# PLATFORM_HOLDOUT_PROVE_EFFECT = 0.05: the gain a pushed holdout is sized
# to prove at 80% power (holdout_size). Five points is the package's proof
# bar (a 5-point move on 50+ judged tasks); rlhfbook.com/c/16-evaluation.html
# puts held-constant eval noise at 0.25 to 1.5 points, so five is several
# noise floors.
PLATFORM_HOLDOUT_PROVE_EFFECT = 0.05
# PLATFORM_REWARD_MODEL_BATCH = 256: rows per scoring request to a hosted
# reward model; the gate's request cap.
PLATFORM_REWARD_MODEL_BATCH = 256
# PLATFORM_UNKNOWN_IDS_SHOWN = 3: unknown trace ids named in an error.
PLATFORM_UNKNOWN_IDS_SHOWN = 3

# ---------------------------------------------------------------------
# training
# ---------------------------------------------------------------------
#
# Reachable through ``training_run(flush_every=, flush_seconds=)``,
# ``TrainingRun(max_batch=)``, ``train(...)`` keywords and ``run.wait(poll=)``.
# The trainer's own defaults live on the platform; what the SDK holds is
# the range each knob is accepted in and the value the literature reaches
# for, so the error message can name a fix.

# TRAINING_FLUSH_EVERY = 25 points / TRAINING_FLUSH_SECONDS = 15: when the
# log buffer is sent. One request per 25 steps keeps a 1k-step run under
# 50 calls; 15 s keeps a slow run's curve live. Convention.
TRAINING_FLUSH_EVERY = 25
TRAINING_FLUSH_SECONDS = 15.0
# TRAINING_MAX_BATCH = 500: points per log request. Keeps one request
# well under the gate's body limit; convention.
TRAINING_MAX_BATCH = 500
# TRAINING_POLL_S = 15 / TRAINING_POLL_MIN_S = 1: seconds between reads of
# a hosted run's state, and the floor so a caller cannot hammer the gate.
TRAINING_POLL_S = 15.0
TRAINING_POLL_MIN_S = 1.0
# TRAINING_ERROR_CHARS = 2000: a finish error is cut here; the run page
# shows one screen of it.
TRAINING_ERROR_CHARS = 2000

# TRAINING_KNOBS = {knob: {lo, hi, ref, why}}: accepted range and reference
# value per knob, by method. ``lo``/``hi`` are what ``train`` accepts; ``ref``
# is what the cited source used, so the message that rejects a value can
# say what to reach for.
#
# generations: GRPO group size G. DAPO trains at G=16 (arXiv:2503.14476,
#   Table of hyperparameters), Dr. GRPO at 8 (arXiv:2503.20783, Table 6),
#   ProRL at 16 (arXiv:2505.24864). 2 is the structural minimum (one
#   advantage needs a pair); 32 is the hosted trainer's memory cap.
# learning_rate: RL runs at 1e-6 (DAPO, Dr. GRPO) to 2e-6 (ProRL); SFT one
#   to two orders below pretraining, 1e-5 to 8e-5 full fine-tune
#   (rlhfbook.com/c/09-instruction-tuning.html, "Implementation Details"),
#   2e-4 for a LoRA adapter (LoRA arXiv:2106.09685 tunes at a higher rate
#   than full fine-tuning). DPO wants "surprisingly low learning rates"
#   (rlhfbook.com/c/12-direct-alignment.html); the DPO paper used 1e-6
#   (arXiv:2305.18290, App. B).
# beta: the KL coefficient. DAPO, Dr. GRPO and CISPO drop it (0.0;
#   arXiv:2503.14476, 2503.20783, 2506.13585); ProRL keeps it with
#   reference resets (arXiv:2505.24864). DPO's beta is the preference
#   temperature, 0.1 in the paper (arXiv:2305.18290).
# max_completion_length: DAPO's soft overlong window is 4096 tokens under a
#   16384 cap (arXiv:2503.14476); the hosted trainer serves up to 4096.
# temperature: ProRL samples at 1.2 (arXiv:2505.24864); rejection sampling
#   runs 0.7 to 1.0 (rlhfbook.com/c/10-rejection-sampling.html).
# clip (host key epsilonHigh): PPO/GRPO clip 0.2 (rlhfbook.com/c/11-
#   policy-gradients.html); DAPO clip-higher 0.28, ProRL 0.4.
TRAINING_KNOBS: dict[str, dict[str, Any]] = {
    # lo/hi: accepted range; open_lo/open_hi: the endpoint itself is
    # rejected; range: the words the error uses; ref: the cited value.
    "generations": {
        "lo": 2,
        "hi": 32,
        "open_lo": False,
        "open_hi": False,
        "range": "2 to 32 rollouts per prompt",
        "ref": 8,
        "methods": ("grpo",),
        "why": "the GRPO group size; Dr. GRPO 8, DAPO and ProRL 16 "
        "(arXiv:2503.20783, 2503.14476, 2505.24864)",
    },
    "learning_rate": {
        "lo": 0.0,
        "hi": 1.0,
        "open_lo": True,
        "open_hi": True,
        "range": "a positive step below 1",
        "ref": {"sft": 2e-4, "grpo": 1e-6, "dpo": 1e-6, "rm": 1e-5},
        "methods": ("sft", "grpo", "dpo", "rm"),
        "why": "the optimizer step; 1e-6 for RL (DAPO, Dr. GRPO), 1e-6 for DPO "
        "(arXiv:2305.18290), 2e-4 for a LoRA SFT adapter",
    },
    "beta": {
        "lo": 0.0,
        "hi": None,
        "open_lo": False,
        "open_hi": False,
        "range": "0 or more",
        "ref": {"grpo": 0.0, "dpo": 0.1},
        "methods": ("grpo", "dpo"),
        "why": "the KL coefficient; 0 in DAPO, Dr. GRPO and CISPO, 0.1 as the DPO "
        "preference temperature (arXiv:2305.18290)",
    },
    "max_completion_length": {
        "lo": 16,
        "hi": 4096,
        "open_lo": False,
        "open_hi": False,
        "range": "16 to 4096 tokens",
        "ref": 4096,
        "methods": ("grpo", "dpo"),
        "why": "the token cap on a sampled reply; DAPO's soft overlong window is 4096 "
        "(arXiv:2503.14476)",
    },
    "temperature": {
        "lo": 0.0,
        "hi": 2.0,
        "open_lo": True,
        "open_hi": False,
        "range": "above 0 and at most 2",
        "ref": 1.0,
        "methods": ("grpo",),
        "why": "the GRPO rollout temperature; ProRL 1.2, rejection sampling 0.7 to 1.0",
    },
    "clip": {
        "lo": 0.0,
        "hi": 1.0,
        "open_lo": False,
        "open_hi": False,
        "range": "0 to 1",
        "ref": 0.28,
        "methods": ("grpo",),
        "why": "the upper clip (host key epsilonHigh); 0.2 PPO, 0.28 DAPO clip-higher, 0.4 ProRL",
    },
}
# TRAINING_LORA_RANK = 16 / TRAINING_LORA_ALPHA = 32: reference adapter
# shape. LoRA tunes r=4 to 8 on the attention projections and sets alpha
# to the first r tried (arXiv:2106.09685 §7.1); alpha=2r is the PEFT
# convention. Advisory: the hosted trainer owns its own values.
TRAINING_LORA_RANK = 16
TRAINING_LORA_ALPHA = 32
# TRAINING_BATCH_PROMPTS = 256: reference prompt batch. OLMo 2 post-trains
# at 256 prompts (rlhfbook.com/c/09-instruction-tuning.html); ProRL at 256
# (arXiv:2505.24864); DAPO at 512. Advisory.
TRAINING_BATCH_PROMPTS = 256
# TRAINING_SFT_EPOCHS = 2: reference SFT epochs; rlhfbook gives no number
# and the hosted trainer owns its own. Convention, untested.
TRAINING_SFT_EPOCHS = 2

# ---------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------
#
# Reachable through ``HackMonitor(...)`` keywords.

# MONITOR_N_PROMPTS = 32 / MONITOR_K = 4: holdout asks sampled per eval and
# completions per ask (128 rollouts). At p=0.5 a 32-task bootstrap band is
# about +-0.17, wide enough that only large proxy/gold divergence reads;
# the size is a cost choice, and ``n_prompts=`` raises it. Convention.
MONITOR_N_PROMPTS = 32
MONITOR_K = 4
# MONITOR_EVERY = 10: trainer steps between evals. Convention.
MONITOR_EVERY = 10
# MONITOR_WINDOW = 3: evals the alarms look back over.
MONITOR_WINDOW = 3
# MONITOR_DELTA = 0.1: proxy gain over the window that counts as climbing.
# Ten points is well outside the 0.25 to 1.5 point eval noise
# rlhfbook.com/c/16-evaluation.html reports; convention for the number.
MONITOR_DELTA = 0.1
# MONITOR_LENGTH_PCT = 0.25: completion-length growth that counts. Length
# growth is the first symptom of over-optimization in
# rlhfbook.com/c/17-over-optimization.html and the bias Dr. GRPO removes
# (arXiv:2503.20783); a quarter is convention.
MONITOR_LENGTH_PCT = 0.25
# MONITOR_BUFFER = 512: completions the reward wrapper keeps for the
# feature scan; MONITOR_SCAN_MIN = 8 is the fewest it scans (two groups).
MONITOR_BUFFER = 512
MONITOR_SCAN_MIN = 8
# MONITOR_SCAN_PERMUTATIONS = 50 / MONITOR_WINDOW_BOOTSTRAPS = 500: the
# permutation count for hack_scan's feature test and the bootstrap count
# for the gold-vs-window interval. 500 resamples give a 95% band to about
# two decimals; 50 permutations a p-value to 0.02. Convention.
MONITOR_SCAN_PERMUTATIONS = 50
MONITOR_WINDOW_BOOTSTRAPS = 500
# MONITOR_MAX_NEW_TOKENS = 256: completion length sampled on the holdout.
MONITOR_MAX_NEW_TOKENS = 256
# MONITOR_CONCURRENCY = 8: gold judge calls in flight.
MONITOR_CONCURRENCY = 8
# MONITOR_SAMPLE_TEMPERATURE = 0.8 / MONITOR_SAMPLE_TOP_P = 0.95 /
# MONITOR_SAMPLE_BATCH = 16: how the default sampler draws. The same
# temperature the rollout engine uses (LOCAL_MODEL_TEMPERATURE) so the
# holdout is sampled the way the data was.
MONITOR_SAMPLE_TEMPERATURE = 0.8
MONITOR_SAMPLE_TOP_P = 0.95
MONITOR_SAMPLE_BATCH = 16

# ---------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------
#
# Reachable through ``build_tasks`` / ``export_environment`` keywords.

# The solve-rate band (0.2, 0.8) an environment keeps prompts in lives in
# score/optimize.py as DEFAULT_BAND (one home); environment.py imports it.
# DAPO's dynamic sampling drops prompts at accuracy 0 and 1
# (arXiv:2503.14476); the 20 to 80 band is the same rule with a margin for
# k=8 noise.
# ENV_HOLDOUT_FRACTION = 0.2: tasks held out. 80/20 convention.
ENV_HOLDOUT_FRACTION = 0.2
# ENV_DECONTAMINATION_NGRAM = 8: n-gram size for train-vs-holdout overlap;
# rlhfbook.com/c/16-evaluation.html ("Contamination") found its overlaps
# with 8-gram matching.
ENV_DECONTAMINATION_NGRAM = 8
# ENV_DECONTAMINATION_EXAMPLES = 3: overlaps shown in the report.
ENV_DECONTAMINATION_EXAMPLES = 3
# ENV_MAX_TURNS_FALLBACK = 10: turn cap when a spec carries none.
ENV_MAX_TURNS_FALLBACK = 10
# ENV_EVAL_EXAMPLES = 5 / ENV_EVAL_ROLLOUTS = 3: the verifiers smoke eval
# written into the exported pyproject.
ENV_EVAL_EXAMPLES = 5
ENV_EVAL_ROLLOUTS = 3


def knob(default: Any, *, lo: float | None = None, hi: float | None = None) -> Any:
    """A :class:`RunKnobs` field with its bounds. ``lo`` and ``hi`` are
    inclusive; ``None`` is open."""
    return field(default=default, metadata={"lo": lo, "hi": hi})


@dataclass(frozen=True)
class RunKnobs:
    """The engine's tunables, one field per ``advanced={...}`` key.

    Every field's default is the value the engine carried inline before
    it was named; none changed. The comment above each says why that
    value, or says honestly that nobody has tested another.
    """

    # --- stopping and restarting ------------------------------------

    # empty_rounds_to_stop = 8: consecutive scheduler rounds with nothing
    # to run and nothing in flight before a run with a model writer and
    # an empty pool gives up as a writer failure. (convention, untested)
    empty_rounds_to_stop: int = knob(8, lo=1)
    # writer_idle_rounds_to_restart = 4: rounds the pool has stayed empty
    # with writers returning only duplicates before the writer is
    # restarted with a rotated seed and an avoid window. (convention,
    # untested)
    writer_idle_rounds_to_restart: int = knob(4, lo=1)
    # writer_idle_rounds_to_rest = 2: idle rounds after which a plain
    # (not unique, no clock) run stops launching new waves and lets the
    # restart rule decide. (convention, untested)
    writer_idle_rounds_to_rest: int = knob(2, lo=1)
    # restart_avoid_window = 8: used asks handed to the restarted writer
    # as avoid pressure per restart; reseeding alone reconverged to the
    # same asks (measured: 26 distinct asks vs the old ceiling of 28).
    # The window size itself is a convention, untested.
    restart_avoid_window: int = knob(8, lo=1)
    # rows_per_extra_restart = 100: the writer restart allowance grows by
    # one per this many budgeted rows, above MAX_NOVELTY_RESTARTS; a
    # 10k-row budget cannot live on a smoke run's retries. (convention,
    # untested)
    rows_per_extra_restart: int = knob(100, lo=1)
    # dead_agent_errors = 16: see DEAD_AGENT_MIN_ERRORS.
    dead_agent_errors: int = knob(DEAD_AGENT_MIN_ERRORS, lo=1)
    # dead_agent_budget_multiple = 2: see DEAD_AGENT_BUDGET_MULTIPLE.
    dead_agent_budget_multiple: int = knob(DEAD_AGENT_BUDGET_MULTIPLE, lo=0)
    # closing_margin = 2.0: under a clock, stop opening new groups when
    # the time left is under this many median rollout durations, so a
    # group is never cut mid-group. Two: one for the rollouts in flight,
    # one for the group's own. (convention, untested)
    closing_margin: float = knob(2.0, lo=0.0)
    # closing_window_rollouts = 20: recent rollout durations the median
    # is read from. (convention, untested)
    closing_window_rollouts: int = knob(20, lo=1)

    # --- scheduler timing -------------------------------------------

    # collect_wait_s = 0.35: how long a round waits for a rollout or a
    # verdict to land before re-planning; the loop's tick. Shorter spins
    # the CPU, longer delays refills. (convention, untested)
    collect_wait_s: float = knob(0.35, lo=0.0)
    # collect_wait_floor_s = 0.1: the shortest tick when the clock is
    # nearly out. (convention, untested)
    collect_wait_floor_s: float = knob(0.1, lo=0.0)
    # writer_wait_s = 0.5: how long the loop waits on a writer wave when
    # the pool is empty and nothing is in flight. (convention, untested)
    writer_wait_s: float = knob(0.5, lo=0.0)
    # writer_wait_floor_s = 0.2: the shortest writer or scene wait when
    # the clock is nearly out. (convention, untested)
    writer_wait_floor_s: float = knob(0.2, lo=0.0)
    # hosted_touch_s = 5.0: timeout on the warm-up ping sent to the hosted
    # writer before the first wave. (convention, untested)
    hosted_touch_s: float = knob(5.0, lo=0.0)
    # scene_join_s = 8.0: how long shutdown waits for the scene-brief
    # thread with no clock; scene_join_clocked_s = 1.0 under a clock.
    # (convention, untested)
    scene_join_s: float = knob(8.0, lo=0.0)
    scene_join_clocked_s: float = knob(1.0, lo=0.0)

    # --- writer pipeline --------------------------------------------

    # writer_buffer_waves = 2: keep about this many waves of prompts in
    # the pipe ahead of the rollouts. (convention, untested)
    writer_buffer_waves: int = knob(2, lo=1)
    # writer_buffer_cap = 96: the most prompts the buffer plans for,
    # whatever the concurrency. (convention, untested)
    writer_buffer_cap: int = knob(96, lo=1)
    # writer_typical_completions = 3: completions per card the buffer
    # math assumes, whatever completions_per_request asks for.
    # (convention, untested)
    writer_typical_completions: int = knob(3, lo=1)
    # pool_low_floor = 16, pool_low_cap = 64, pool_low_flight_divisor = 4:
    # the pool is "low" (refill now) under max(floor, min(cap, flight /
    # divisor)) eligible prompts. (convention, untested)
    pool_low_floor: int = knob(16, lo=0)
    pool_low_cap: int = knob(64, lo=0)
    pool_low_flight_divisor: int = knob(4, lo=1)
    # offline_topup_multiple = 2: the template writer tops up when fewer
    # than this many batches of prompts are eligible. (convention,
    # untested)
    offline_topup_multiple: int = knob(2, lo=1)
    # offline_bounce_limit = 20: extra template rounds tried before a
    # short batch is accepted. (convention, untested)
    offline_bounce_limit: int = knob(20, lo=0)
    # offline_bounce_stride = 17: the seed step between those rounds; any
    # stride coprime with the round count does. (convention, untested)
    offline_bounce_stride: int = knob(17, lo=1)
    # restart_seed_stride = 997: the round id advances by this per writer
    # restart so restarted draws never reuse a round's temperature and
    # tags. Any large prime does. (convention, untested)
    restart_seed_stride: int = knob(997, lo=1)
    # first_wave_writers = 2, first_wave_cards = 4, first_wave_tokens =
    # 320: the first waves are tiny so the first rollouts start in about
    # five seconds instead of after a full wave. (convention, untested)
    first_wave_writers: int = knob(2, lo=1)
    first_wave_cards: int = knob(4, lo=1)
    first_wave_tokens: int = knob(320, lo=1)
    # min_cards_per_wave = 2: a wave asks for at least two cards; one
    # card gives the writer nothing to contrast. (convention, untested)
    min_cards_per_wave: int = knob(2, lo=1)
    # tokens_per_card = 130, wave_tokens_base = 128, wave_tokens_floor =
    # 256, wave_tokens_cap = 2048: a wave's reply budget is clamp(130 x
    # cards + 128, 256, 2048), about 130 tokens per card so long-prompt
    # cards are realizable; the old 768 ceiling gave 12-card batches 64
    # tokens per message. (observed on the hosted writer; the per-card
    # figure is a convention, untested)
    tokens_per_card: int = knob(130, lo=1)
    wave_tokens_base: int = knob(128, lo=0)
    wave_tokens_floor: int = knob(256, lo=1)
    wave_tokens_cap: int = knob(2048, lo=1)
    # failing_seeds_cap = 40: failing asks mined from traces= that seed
    # the run; more than this crowds out the grid. (convention, untested)
    failing_seeds_cap: int = knob(40, lo=0)

    # --- selection --------------------------------------------------

    # select_oversample = 3: the diversity selector picks this many times
    # the batch so the family cap and the situation quota have slack.
    # (convention, untested)
    select_oversample: int = knob(3, lo=1)
    # family_cap_floor = 16, family_cap_per_phrasing = 4: near-copy
    # scenario families are capped at max(floor, n x per_phrasing) rows.
    # (convention, untested)
    family_cap_floor: int = knob(16, lo=1)
    family_cap_per_phrasing: int = knob(4, lo=1)
    # writer_context_items = 8: items of each kind (avoid, underexplored,
    # behavior gaps, axis gaps, tools) the writer prompt carries; more
    # made the card prompt longer than the cards. (convention, untested)
    writer_context_items: int = knob(8, lo=0)
    # writer_context_parents = 10: failing rows the writer mutates from.
    # (convention, untested)
    writer_context_parents: int = knob(10, lo=0)
    # family_avoid_items = 6: family-rejected prompts shown to the writer
    # as avoid pressure, before the concentrated ones. (convention,
    # untested)
    family_avoid_items: int = knob(6, lo=0)

    # --- search heuristics ------------------------------------------

    # allocation_gain = 4.0: see ALLOC_GAIN.
    allocation_gain: float = knob(ALLOC_GAIN, lo=0.0)
    # allocation_tool_weight = 0.6, allocation_condition_weight = 0.4: a
    # cell matching a hot region's tool scores 0.6, its tool condition
    # 0.4, both 1.0. (convention, untested)
    allocation_tool_weight: float = knob(0.6, lo=0.0, hi=1.0)
    allocation_condition_weight: float = knob(0.4, lo=0.0, hi=1.0)
    # region_novelty_smoothing = 0.5: weight on the newest novelty score
    # in a region's running novelty (an EMA; 1 - this on the old value).
    # (convention, untested)
    region_novelty_smoothing: float = knob(0.5, lo=0.0, hi=1.0)
    # gap_min_rows = 3: rows a region needs before "one signature only"
    # counts as stuck; gap_rich_signatures = 3: distinct signatures at
    # which a region counts as explored. (convention, untested)
    gap_min_rows: int = knob(3, lo=1)
    gap_rich_signatures: int = knob(3, lo=1)
    # gap_value_stuck = 1.0, gap_value_rich = 0.2, gap_value_unknown =
    # 0.5: the behavior-gap score for a stuck, an explored, and an
    # undecided region. (convention, untested)
    gap_value_stuck: float = knob(1.0, lo=0.0, hi=1.0)
    gap_value_rich: float = knob(0.2, lo=0.0, hi=1.0)
    gap_value_unknown: float = knob(0.5, lo=0.0, hi=1.0)
    # gap_weight = 0.7: share of a region's behavior value from the gap
    # score; the rest (0.3) from its fault rate. (convention, untested)
    gap_weight: float = knob(0.7, lo=0.0, hi=1.0)
    # adaptive_verify_explore_floor = 0.55: under mode="adaptive" a
    # short clock (explore share under this) re-rolls a prompt to peek
    # for a different outcome even when its region shows one behavior.
    # (convention, untested)
    adaptive_verify_explore_floor: float = knob(0.55, lo=0.0, hi=1.0)
    # pass_threshold = 0.5: see PASS_THRESHOLD.
    pass_threshold: float = knob(PASS_THRESHOLD, lo=0.0, hi=1.0)
    # smoothing_alpha = 1.0: Laplace's rule of succession, (s + a) /
    # (n + 2a), for the group hazard and the mixed rate; a = 1 is the
    # uniform prior on a rate, one pseudo-observation each way. Why
    # unanimous groups are stopped at all: they carry no gradient (DAPO,
    # arXiv 2503.14476; rlhf-book ch. 14). No paper states a prior for
    # the decision; this is the engine's own, untested against a = 0.5.
    smoothing_alpha: float = knob(1.0, lo=0.0)

    # --- run-level notes --------------------------------------------

    # short_share_floor = 0.08, long_share_floor = 0.10: under these
    # shares of short or long asks the writer is nudged toward that
    # length. (convention, untested)
    short_share_floor: float = knob(0.08, lo=0.0, hi=1.0)
    long_share_floor: float = knob(0.10, lo=0.0, hi=1.0)
    # followup_starved_min = 8, followup_starved_divisor = 4: the run is
    # marked followups_starved at 8 or more missed follow-ups that are
    # also at least rows / 4. (convention, untested)
    followup_starved_min: int = knob(8, lo=0)
    followup_starved_divisor: int = knob(4, lo=1)
    # semantic_duplicate_novelty = 0.05: a row under this semantic
    # novelty is a duplicate in the duplicate rate. (convention,
    # untested)
    semantic_duplicate_novelty: float = knob(0.05, lo=0.0, hi=1.0)
    # idle_judge_share = 0.1: an rl pool idle on verdicts for more than
    # this share of the run gets the "add situations" note. (convention,
    # untested)
    idle_judge_share: float = knob(0.1, lo=0.0, hi=1.0)
    # tier_mix_min_rows = 20, tier_mix_tolerance = 0.10: see the
    # constants TIER_MIX_MIN_ROWS and TIER_MIX_TOLERANCE above.
    tier_mix_min_rows: int = knob(TIER_MIX_MIN_ROWS, lo=1)
    tier_mix_tolerance: float = knob(TIER_MIX_TOLERANCE, lo=0.0, hi=1.0)
    # progress_every_s = 10.0, progress_every_rows = 10: never more than
    # this long, or this many finished rollouts, between progress lines.
    # (convention, untested)
    progress_every_s: float = knob(10.0, lo=0.0)
    progress_every_rows: int = knob(10, lo=1)
    # flush_report_rows = 25, flush_report_s = 5.0: the streamed-output
    # log line is written every 25 rows or 5 s. (convention, untested)
    flush_report_rows: int = knob(25, lo=1)
    flush_report_s: float = knob(5.0, lo=0.0)


def knob_names() -> tuple[str, ...]:
    """Every ``advanced`` key :class:`RunKnobs` reads, in field order."""
    return tuple(f.name for f in fields(RunKnobs))


def knob_default(name: str) -> Any:
    spec = RunKnobs.__dataclass_fields__[name]
    return None if spec.default is MISSING else spec.default


def knob_bounds(name: str) -> tuple[float | None, float | None]:
    meta = RunKnobs.__dataclass_fields__[name].metadata
    return meta.get("lo"), meta.get("hi")


def resolve_knobs(cfg: dict[str, Any]) -> RunKnobs:
    """Pop every :class:`RunKnobs` key out of ``cfg`` (the ``advanced``
    dict), coerce it to the field's type and check its bounds. What is
    left in ``cfg`` belongs to someone else."""
    values: dict[str, Any] = {}
    for spec in fields(RunKnobs):
        if spec.name not in cfg:
            continue
        raw = cfg.pop(spec.name)
        kind = int if spec.type in ("int", int) else float
        try:
            if isinstance(raw, bool):
                raise TypeError
            value = kind(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"advanced[{spec.name!r}] is {'an integer' if kind is int else 'a number'}; "
                f"got {raw!r}. The default is {spec.default!r}."
            ) from None
        if kind is int and float(raw) != float(value):
            raise ValueError(
                f"advanced[{spec.name!r}] is an integer; got {raw!r}. "
                f"The default is {spec.default!r}."
            )
        lo, hi = spec.metadata.get("lo"), spec.metadata.get("hi")
        if lo is not None and value < lo:
            raise ValueError(
                f"advanced[{spec.name!r}]={value!r} is below its floor {lo!r}. "
                f"The default is {spec.default!r}."
            )
        if hi is not None and value > hi:
            raise ValueError(
                f"advanced[{spec.name!r}]={value!r} is above its ceiling {hi!r}. "
                f"The default is {spec.default!r}."
            )
        values[spec.name] = value
    return RunKnobs(**values)


def laplace(successes: float, trials: float, alpha: float = 1.0) -> float:
    """Laplace's rule of succession: ``(s + a) / (n + 2a)``. With ``a = 1``
    a rate seen 0 of 0 times is 1/2, 0 of 1 is 1/3, and so on."""
    return (float(successes) + alpha) / (float(trials) + 2.0 * alpha)


__all__ = [
    "AGENT_MAX_TOKENS_FLOOR",
    "ALLOC_GAIN",
    "ALPHA",
    "BASE_PASS_RATE",
    "BOOTSTRAP_DRAWS",
    "CHARS_PER_TOKEN",
    "CI_LEVEL",
    "DEAD_AGENT_BUDGET_MULTIPLE",
    "DEAD_AGENT_MIN_ERRORS",
    "DECONTAM_NGRAM",
    "DECONTAM_OVERLAP",
    "DEFAULT_AVG_TURNS",
    "DEFAULT_BUDGET",
    "DEFAULT_CARDS_PER_WAVE",
    "DEFAULT_COMPLETIONS_PER_REQUEST",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_EXTRA_CARDS",
    "DEFAULT_LLM_JUDGE_CONCURRENCY",
    "DEFAULT_MIN_USER_TURNS",
    "DEFAULT_POOL_SIZE",
    "DEFAULT_PROBE",
    "DEFAULT_SEED",
    "DEFAULT_SELECT_TARGET",
    "DEFAULT_WRITER_FLIGHT",
    "DIFFICULTY_BAND",
    "DIFFICULTY_BAND_ROLLOUTS",
    "ENV_DECONTAMINATION_EXAMPLES",
    "ENV_DECONTAMINATION_NGRAM",
    "ENV_EVAL_EXAMPLES",
    "ENV_EVAL_ROLLOUTS",
    "ENV_HOLDOUT_FRACTION",
    "ENV_MAX_TURNS_FALLBACK",
    "FAULT_STATUSES",
    "FINGERPRINT_STEM_MIN_LEN",
    "FLIP_FLAG",
    "HACK_THRESHOLD",
    "HOLDOUT_BUCKET_HEX_CHARS",
    "HUNG_SLOT_S",
    "JUDGE_CHECK_SAMPLE",
    "JUDGE_CONCURRENCY_CAP",
    "JUDGE_FINAL_TEXT_CHARS",
    "JUDGE_MAX_TOKENS",
    "JUDGE_PAYLOAD_CHARS",
    "JUDGE_POLICY_CHARS",
    "JUDGE_SITUATION_CHARS",
    "JUDGE_TEMPERATURE",
    "LEAK_MIN_QUOTE_CHARS",
    "LENGTH_GAP_FLAG",
    "MAX_COMPLETIONS_PER_REQUEST",
    "MAX_GOLD_ASK",
    "MAX_SAMPLES_PER_CALL",
    "MIN_AGREEMENT",
    "MIN_CI_TASKS",
    "MIN_GOLD",
    "MIN_KAPPA",
    "MIN_RERUNS",
    "MONITOR_BUFFER",
    "MONITOR_CONCURRENCY",
    "MONITOR_DELTA",
    "MONITOR_EVERY",
    "MONITOR_K",
    "MONITOR_LENGTH_PCT",
    "MONITOR_MAX_NEW_TOKENS",
    "MONITOR_N_PROMPTS",
    "MONITOR_SAMPLE_BATCH",
    "MONITOR_SAMPLE_TEMPERATURE",
    "MONITOR_SAMPLE_TOP_P",
    "MONITOR_SCAN_MIN",
    "MONITOR_SCAN_PERMUTATIONS",
    "MONITOR_WINDOW",
    "MONITOR_WINDOW_BOOTSTRAPS",
    "OK_STATUSES",
    "PARENT_HEAD_CHARS",
    "PASS_REWARD",
    "PASS_THRESHOLD",
    "PLATFORM_CREDENTIAL_TTL_S",
    "PLATFORM_ERROR_DETAIL_CHARS",
    "PLATFORM_HF_DATASET_TIMEOUT_S",
    "PLATFORM_HF_MODEL_POLL_S",
    "PLATFORM_HF_MODEL_TIMEOUT_S",
    "PLATFORM_HF_POLL_S",
    "PLATFORM_HOLDOUT_PROVE_EFFECT",
    "PLATFORM_IMPORT_MAX_ROWS",
    "PLATFORM_IMPORT_POLL_S",
    "PLATFORM_IMPORT_TIMEOUT_S",
    "PLATFORM_REQUEST_TIMEOUT_S",
    "PLATFORM_REWARD_MODEL_BATCH",
    "PLATFORM_STUDIO_MAX_ROWS",
    "PLATFORM_TRACE_PAGE_SIZE",
    "PLATFORM_UNKNOWN_IDS_SHOWN",
    "PLATFORM_UPLOAD_TIMEOUT_S",
    "POSITION_FLIP_FLAG",
    "POWER",
    "PROGRESS_MIN_BUDGET",
    "PROGRESS_MIN_ROWS_FOR_ESTIMATE",
    "REJECTION_SAMPLING_MIN_K",
    "REPORT_LIST_ITEMS",
    "RL_FAULT_RATE",
    "RL_ROLLOUTS_PER_ASK",
    "RL_ROLLOUTS_PER_PROMPT",
    "ROLLOUTS_PER_TASK",
    "SATURATION_CAP",
    "SCENARIO_ID_CHARS",
    "SEMANTIC_SIMILARITY",
    "SFT_PHRASINGS_PER_SITUATION",
    "SHORT_HASH_CHARS",
    "STOP_GRACE_S",
    "SYSTEM_PROMPT_HEAD_CHARS",
    "TIER_MIX_MIN_ROWS",
    "TIER_MIX_TOLERANCE",
    "TOOL_SCHEMA_SPAN_CHARS",
    "TRACE_EXEMPLARS_PER_TOOL",
    "TRACE_EXEMPLAR_DICT_KEYS",
    "TRACE_EXEMPLAR_LIST_ITEMS",
    "TRACE_EXEMPLAR_MAX_CHARS",
    "TRACE_EXEMPLAR_STRING_CHARS",
    "TRACE_EXPLORATION_FLOOR",
    "TRACE_EXPLORATION_MAX",
    "TRACE_LEAK_EXAMPLES",
    "TRACE_LEAK_THRESHOLD",
    "TRACE_MIN_SUPPORT",
    "TRACE_PSEUDO_PRODUCTION_FRACTION",
    "TRACE_RATE_FLOOR",
    "TRACE_REPORT_LIST_CAP",
    "TRACE_STATE_PRIORITY",
    "TRACE_SUPPORT_SATURATION",
    "TRACE_TASK_HASH_CHARS",
    "TRAINING_BATCH_PROMPTS",
    "TRAINING_ERROR_CHARS",
    "TRAINING_FLUSH_EVERY",
    "TRAINING_FLUSH_SECONDS",
    "TRAINING_KNOBS",
    "TRAINING_LORA_ALPHA",
    "TRAINING_LORA_RANK",
    "TRAINING_MAX_BATCH",
    "TRAINING_POLL_MIN_S",
    "TRAINING_POLL_S",
    "TRAINING_SFT_EPOCHS",
    "TRANSIENT_BACKOFF_S",
    "TRANSIENT_TRIES",
    "TRUNCATED_REPLY_CHARS",
    "WORLD_CI_FAIL_ONE_IN",
    "WORLD_CREATED_ID_MODULUS",
    "WORLD_DATE_YEARS",
    "WORLD_DEFAULT_FAULT_MODE",
    "WORLD_DEFAULT_FAULT_RATE",
    "WORLD_EXISTS_SHARE",
    "WORLD_EXPRESSION_CHARS",
    "WORLD_HINT_CHARS",
    "WORLD_ID_RANGE",
    "WORLD_ISSUED_ID_HEX",
    "WORLD_JITTER_DIVISOR",
    "WORLD_MALFORMED_PAYLOAD",
    "WORLD_REF_CHARS",
    "WORLD_SEARCH_HITS",
    "WORLD_SHELL_FLAVORS",
    "WORLD_STALE_AS_OF",
    "WORLD_TEMPLATE_HITS",
    "Z_95",
    "RunKnobs",
    "knob",
    "knob_bounds",
    "knob_default",
    "knob_names",
    "laplace",
    "resolve_knobs",
]
