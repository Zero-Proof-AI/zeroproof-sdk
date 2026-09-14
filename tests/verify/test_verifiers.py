"""Verifiers honor the judge contract and check what they claim."""

from __future__ import annotations

import zeroproof.simulations as zps
from zeroproof.simulations.verify import (
    All,
    Any,
    CodeExec,
    ExactMatch,
    Includes,
    JSONField,
    JSONSchema,
    JSONValid,
    MathEqual,
    MultipleChoice,
    Numeric,
    Regex,
    Weighted,
    verifier,
)


def row(final, **kw):
    r = {"final_text": final, "prompt": "p"}
    r.update(kw)
    return r


def test_exposed_on_package():
    assert hasattr(zps, "verify")
    assert zps.Verifier is not None
    assert callable(zps.verifier)


def test_contract_shape():
    r = ExactMatch()(row("Paris", answer="Paris"))
    assert r["reward"] == 1 and "reason" in r
    assert r["judge_meta"]["verifier"] == "ExactMatch"


def test_exact_match_last_line_and_list():
    assert ExactMatch()(row("The answer is\nParis", answer="paris"))["reward"] == 1
    assert ExactMatch()(row("London", answer="Paris"))["reward"] == 0
    assert ExactMatch()(row("yes", answer=["y", "yes", "yeah"]))["reward"] == 1


def test_missing_reference_is_none_not_zero():
    # No reference anywhere: reward None (contract: never invent a reward).
    assert ExactMatch()(row("Paris"))["reward"] is None


def test_privileged_reference_used_and_flat_fallback():
    assert ExactMatch()(row("Paris", privileged={"reference": "Paris"}))["reward"] == 1
    assert ExactMatch()(row("Paris", info={"answer": "Paris"}))["reward"] == 1


def test_includes_modes():
    assert Includes(mode="all")(row("a and b and c", answer=["a", "b"]))["reward"] == 1
    assert Includes(mode="all")(row("a only", answer=["a", "b"]))["reward"] == 0
    assert Includes(mode="any")(row("a only", answer=["a", "b"]))["reward"] == 1


def test_regex_format_and_extract():
    assert Regex(r"</think>")(row("...</think> done"))["reward"] == 1
    assert Regex(r"</think>")(row("no close"))["reward"] == 0
    v = Regex(r"answer:\s*(\w+)", group=1)
    assert v(row("answer: cat", answer="cat"))["reward"] == 1
    assert v(row("answer: dog", answer="cat"))["reward"] == 0


def test_multiple_choice():
    assert MultipleChoice()(row("The answer is (C).", answer="C"))["reward"] == 1
    assert MultipleChoice()(row("... so B", answer="B"))["reward"] == 1
    assert MultipleChoice()(row("A", answer="D"))["reward"] == 0


def test_numeric_tolerance_and_boxed():
    assert Numeric()(row("The total is 42.", answer=42))["reward"] == 1
    assert Numeric(rel=0.01)(row("about 100", answer=101))["reward"] == 1
    assert Numeric()(row("\\boxed{3.14159}", answer=3.14159))["reward"] == 1
    assert Numeric()(row("7", answer=8))["reward"] == 0


def test_math_equal_symbolic_or_fallback():
    assert MathEqual()(row("\\boxed{1/2}", answer="0.5"))["reward"] == 1
    assert MathEqual()(row("the answer is 12", answer="12"))["reward"] == 1
    assert MathEqual()(row("\\boxed{9}", answer="10"))["reward"] == 0


def test_json_verifiers():
    assert JSONValid(top_type=dict)(row('{"a": 1}'))["reward"] == 1
    assert JSONValid()(row("not json"))["reward"] == 0
    schema = {
        "type": "object",
        "required": ["intent"],
        "properties": {"intent": {"type": "string"}},
    }
    assert JSONSchema(schema)(row('{"intent": "refund"}'))["reward"] == 1
    assert JSONSchema(schema)(row('{"other": 1}'))["reward"] == 0
    assert JSONField("intent", equals="refund")(row('{"intent": "refund"}'))["reward"] == 1
    assert JSONField("intent")(row('{"intent": "x"}', answer="y"))["reward"] == 0


def test_compose_all_any_weighted():
    right = MathEqual()
    closed = Regex(r"</think>")
    good = row("<think>..</think>\\boxed{4}", answer="4")
    bad = row("\\boxed{4}", answer="4")  # right answer, no closing tag
    assert All([right, closed])(good)["reward"] == 1
    assert All([right, closed])(bad)["reward"] == 0
    assert Any([right, closed])(bad)["reward"] == 1
    w = Weighted([(right, 0.7), (closed, 0.3)])
    assert w(bad)["reward"] == 0.7


def test_function_verifier_decorator():
    @verifier
    def shouts(candidate, reference, r):
        return candidate.isupper()

    assert shouts(row("YES"))["reward"] == 1
    assert shouts(row("no"))["reward"] == 0


def test_broken_verifier_marks_none_not_zero():
    @verifier
    def boom(candidate, reference, r):
        raise RuntimeError("boom")

    out = boom(row("x"))
    assert out["reward"] is None and out["judge_status"] == "error"


def test_code_exec_pass_and_fail():
    tests = "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n"
    good = row("```python\ndef add(a, b):\n    return a + b\n```", privileged={"tests": tests})
    bad = row("```python\ndef add(a, b):\n    return a - b\n```", privileged={"tests": tests})
    assert CodeExec()(good)["reward"] == 1
    assert CodeExec()(bad)["reward"] == 0


def test_code_exec_timeout():
    tests = "loop()\n"
    slow = row(
        "```python\ndef loop():\n    while True:\n        pass\n```", privileged={"tests": tests}
    )
    r = CodeExec(timeout=2.0)(slow)
    assert r["reward"] == 0 and "timed out" in r["reason"]


def test_verifier_feeds_grade(monkeypatch):
    # A verifier is a judge: run_judge accepts it and stamps a scored row.
    from zeroproof.simulations.score.judging import run_judge

    rows = [row("Paris", answer="Paris"), row("London", answer="Paris")]
    scored = run_judge(rows, ExactMatch(), source="grade")
    rewards = sorted(r.get("reward") for r in scored.rows)
    assert rewards == [0, 1]
