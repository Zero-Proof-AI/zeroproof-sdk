# ZeroProof Simulations

The SDK makes post-training data for an agent you already have, or one you
can describe. You give it the agent's definition; it gives you graded
conversations you can train on. This document is about how it thinks and
why, not a tour of every option.

## The problem it solves

An agent's failures are specific. It hands off too early on one kind of
request, invents an order number under a particular kind of pressure,
loses its manners on the fourth turn. Fixing that with a system prompt
works until it does not. Fixing it in the weights needs examples of the
situation done right, and enough of them, with enough variety, that the
model learns the behavior rather than the example. Writing those by hand
is the expensive part of every post-training pipeline. The simulator
replaces the writing, not the judgment.

## Two ways in

**Describe the behavior.** One sentence is enough to start: "a personal
finance assistant that confirms before it moves money." The SDK drafts the
tools such an agent would have, builds a world around them, writes the
people who would talk to it, and runs the conversations. This is the
cold-start path for an agent that does not exist yet, or a behavior you
want to add to one that does.

**Point at the agent's traces.** If the agent is in production, its
telemetry already says where it is weak. The SDK reads graded traces,
plain OpenTelemetry spans included, into a picture of which situations
fail, which are new since the last model version, and which have stopped
failing. That picture sets the generation budget, so new rows land where
the deployed agent actually needs them and not where it is already fine.

Both paths use the same engine. The first one is what produced a training
set that took a base model from 5% to 30% on a public benchmark, from
nothing but the agent's tool list and policy.

## How the simulator thinks

**Situations are coordinates, not prompts.** Asking a model for a thousand
user requests gives you a thousand variations of the same polite,
well-specified ask. The SDK instead declares axes (which tool, which
policy clause, what the world looks like, what condition the tool is in,
what stance the person takes, what has already happened) and renders
points in that space. A covering design guarantees every pair of values
appears together at least once, which is the coverage strength the
testing literature settled on because most real failures come from two
things interacting. Coverage becomes a number you can read, not a hope.

**People are sampled, not described.** A coordinate says the customer is
in a hurry and their order was already cancelled. A second layer decides
how that person writes: lowercase, clipped, run-on, with a typo, polite,
sarcastic. The writer never sees those labels; it sees an aside in prose,
because a model told to be terse writes an essay about being terse. The
same person shows up on turn five that showed up on turn one.

**The world answers honestly.** Tool calls go to a simulated world that is
deterministic for a given seed, returns records shaped like the tool's own
schema, remembers what it created, and says no. An unknown identifier is
not found. An argument that echoes the schema instead of the person's
details ("first name", user@example.com) is refused with a hint. A world
that never says no teaches an agent that never expects it; we learned
that the expensive way and built it in.

**Grading is the customer's authority.** Rows come back ungraded on
purpose. The deterministic conduct checks catch structural failures (an
action claimed without a tool call, an identifier the person never gave,
success declared after a failed call), and then your grader, a function
you write, decides what good means for your agent. The SDK's job is to
make every row worth grading; it does not get a vote on what passes.

**Failure is loud.** If the writer, the world, or a judge cannot do its
job, the run says so. Template fallbacks are never quietly substituted for
model-written situations, because a dataset that looks real and is not is
worse than no dataset.

## What you get

A JSONL file of conversations in chat format, with tool schemas, each
row carrying its situation (which axes, which world state, which faults
were scheduled), its persona tags, and, once graded, its reward and the
reason. From there: `training_set()` for supervised fine-tuning,
preference pairs and repeated groups for reinforcement learning, and a
leakage check against any evaluation you care about, so what you train on
is provably not what you test on.

## What it is not

It is not ground truth. Every row is a simulation, kept by a grader, and
should be reviewed the way you would review a contractor's work. The world
is not your database. The people are drawn from a persona distribution,
not from your customers. The value is coverage, variety, and honesty
about all three.

## Where it goes next

The same simulator that produces a frozen dataset can serve as a live
environment for on-policy reinforcement learning: the trainer drives the
policy, and the SDK supplies the situations, the world, the person, and
the reward. That export is designed and follows once the supervised loop
is closed end to end.
