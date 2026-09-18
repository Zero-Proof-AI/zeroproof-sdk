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
line: ``run/`` (engine, config, rows, simulation, data) is below. Other
stages add their own section.
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
# rubric score rounds to the nearer verdict. No source names another
# cut.
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

# FAULT_STATUSES: a tool result ``status`` that means the call failed
# (the run's own vocabulary; a status outside this set is the tool's own
# word, not a fault, #261). ``score.grading._KNOWN_FAULTS`` is the wider
# alias table; this is the subset the engine re-rolls and mutates on.
FAULT_STATUSES = frozenset({"error", "timeout", "not_found", "denied", "malformed"})

# OK_STATUSES: a tool result ``status`` that means the call worked.
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
