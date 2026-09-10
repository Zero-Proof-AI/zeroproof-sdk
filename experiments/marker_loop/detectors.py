"""Regex building blocks for PER-AGENT graders. Nothing here is universal.

The loop (loop.py) has no built-in marker taxonomy: markers come from
whichever rule of the customer's grader fired. These patterns are shared
ingredients a grader author can import; the forged Claude Code test agent
(claude_code_demo.py) is the reference user.
"""
from __future__ import annotations

import re

APPROVAL = re.compile(r"\b(yes|yeah|go ahead|approved?|please (commit|push)|do it|ship it|lgtm|sounds good)\b", re.I)
AI_TELLS = re.compile(
    r"—|–|\bdelve\b|\bGreat question\b|\bYou're absolutely right\b|\bCertainly!\b"
    r"|\bI hope this helps\b|\bLet's dive in\b|\bgame.changer\b|\bseamlessly\b", re.I)
NUM_CLAIM = re.compile(r"\b\d[\d,.]*\s*(%|percent|x faster|ms|seconds|users|rows|times)\b|\b\d{2,}[\d,.]*\b")
TEST_CMD = re.compile(r"\b(pytest|unittest|run_tests|python -m tests?|python -m pytest)\b", re.I)
PUBLISH_TOOLS = {"git_commit", "open_pr", "git_push", "push", "deploy"}
