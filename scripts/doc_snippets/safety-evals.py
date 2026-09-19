"""The reader's side of docs/safety-evals.md.

The page's "The calls" block is the recipe's run, written out as one
sequence. Rather than restate the suite here, this imports the recipe the
page points at, so the page is checked against the thing it tells the
reader to go read. A name that moves in the recipe fails here.
"""

import sys
from pathlib import Path

RECIPE = Path(__file__).resolve().parents[2] / "recipes" / "02-measure" / "safety-evals"
sys.path.insert(0, str(RECIPE))

from agents import hardened_agent as agent_v2  # noqa: E402,F401
from agents import trusting_agent as agent  # noqa: E402,F401
from judge import safety_judge  # noqa: E402,F401
from suite import CATEGORIES, LABELED, SEEDS, TOOLS, classify  # noqa: E402,F401
from suite import SYSTEM_PROMPT as POLICY  # noqa: E402,F401

import whileai.simulations as wai  # noqa: E402,F401
