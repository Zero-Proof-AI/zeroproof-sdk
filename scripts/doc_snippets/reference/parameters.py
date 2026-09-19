"""The reader's side of docs/reference/parameters.md.

The pass@k section is about k, so its `data` has to be a run with repeats,
not the shared explore run. Everything else comes from _common.
"""

from _common import *  # noqa: F403
from _common import POLICY, TOOLS, wai

data = wai.simulate(
    wai.seeded_agent(TOOLS),
    tools=TOOLS,
    system_prompt=POLICY,
    simulator=False,
    mode="rl",
    repeats=4,
    repeat_policy="fixed",
    budget=64,
)


def my_judge(row):
    """The seeded agent labels its own bad rollouts, so the groups split and
    pass^k and pass@k have something to say."""
    return {"reward": int(not row["seeded"])}
