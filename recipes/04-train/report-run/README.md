# Report a run so a person can decide

**What you learn**: the four objects the platform tracks (agent, behavior,
run, live traffic), and why a version is scored on every behavior.

**Needs**: `WHILEAI_API_KEY` for the real thing; nothing for the smoke run.

**Takes**: 10 seconds.

```bash
python run.py                 # posts one full loop for agent refund-bot, prints the verdict
python run.py --agent my-bot  # your own name
python run.py --offline       # no key, no network: prints the calls it would make
```

The script registers an agent (model + harness + the frontier model you pay
for today), declares five behaviors with their own held-out tests, opens
one run per version (base, v1..v4) with a training curve and a score on
every behavior, marks v3 as served, posts two weeks of traffic, and reads
the verdict back. Open while.ai/platform/runs afterwards: that is the screen
the person decides on.

Two rules the platform holds you to, both from rlhf-book's evaluation
chapter: the held-out test for a behavior does not change under you (bump
`test_version` when it does), and a score without an interval is not a
result (`ci` is the half-width of the 95% interval).
