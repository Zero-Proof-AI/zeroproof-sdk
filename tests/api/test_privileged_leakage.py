"""Teacher-only context never reaches a student-visible field.

``Privileged`` (``principle``, ``hidden_state``, ``reference``) is context
the teacher sees and the student never does. Today the projection
``to_row`` simply does not read it and every exporter rebuilds the
student's view from ``prompt``/``steps``/``final_text``, so nothing leaks.
That is a property worth pinning: the failure mode is silent, it lands in
a training file, and it is discovered only after a model has memorised the
answer key.

The guard is depth-independent since #149. ``export_row`` used to be an
allowlist, which made nesting unreachable by accident -- no carrier for a
nested privileged block was copied out -- and every test here was written
flat because of it. The export now carries the whole row, so the tests
below pin the nested case too.

Each test names the boundary it guards. ``from_row``/``to_row``
passthrough is deliberately excluded: carrying a source row's unknown keys
back out is the wire round-trip identity, not a student-visible export,
and it is pinned as such in ``test_schema_v1``.
"""

import json

from zeroproof.simulations.data import export_row
from zeroproof.simulations.export import (
    _CARRY_KEYS,
    export_preference,
    export_training,
    training_rows,
)
from zeroproof.simulations.schema import (
    Privileged,
    Rollout,
    Step,
    Task,
    to_row,
)
from zeroproof.simulations.score.judging import (
    build_preference_pairs,
    evaluate,
    run_judge,
)

SECRET = "zzteachersecretzz"
PRIVILEGED_KEYS = ("principle", "hidden_state", "reference")


def _privileged_row(
    prompt: str = "refund order 4412", reward: float = 1.0, final_text: str = "Refunded."
) -> dict:
    """A source row that carries every privileged key, as a hostile
    upstream (a hand-written trace, a platform pull) might."""
    return {
        "prompt": prompt,
        "steps": [
            {"tool": "lookup_order", "arguments": {"order_id": "4412"}, "result": {"status": "ok"}}
        ],
        "final_text": final_text,
        "scenario_id": "sc-1",
        "reward": reward,
        "principle": f"{SECRET}_principle",
        "hidden_state": {"answer": f"{SECRET}_hidden"},
        "reference": f"{SECRET}_reference",
    }


def _nested_privileged_row() -> dict:
    """The same hostile upstream, one level down.

    Written for #149. While ``export_row`` was an allowlist these keys
    were unreachable: nothing that could hold a nested privileged block
    (``lineage``, ``scenario_dimensions``, ``judge_meta``, a tool
    ``result``) was copied out at all, so a top-level-only guard was
    enough and every test here only ever built flat rows. The export now
    carries the whole row, so the guard has to be as deep as the row is.
    """
    return {
        "prompt": "refund order 4412",
        "steps": [
            {
                "tool": "lookup_order",
                "arguments": {"order_id": "4412"},
                # a tool whose result quotes the teacher's answer key
                "result": {"status": "ok", "rubric": f"{SECRET}_in_a_tool_result"},
            }
        ],
        "final_text": "Refunded.",
        "scenario_id": "sc-1",
        "reward": 1.0,
        "world_state": "exists",
        "markers": {"grounded": 1.0},
        "scenario_dimensions": {"tool": "lookup_order", "privileged": {"principle": SECRET}},
        "lineage": {"source": "grade", "reference": f"{SECRET}_in_lineage"},
        "judge_meta": {"trials": [{"n": 1, "hidden_state": {"answer": SECRET}}]},
    }


def _dump(value) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _privileged_keys_in(value) -> list[str]:
    """Every privileged key name reachable anywhere inside ``value``."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in (*PRIVILEGED_KEYS, "privileged", "rubric"):
                found.append(str(key))
            found.extend(_privileged_keys_in(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_privileged_keys_in(item))
    return found


def test_to_row_never_projects_the_privileged_block():
    """The teacher's own ``Task.privileged`` has no wire projection."""
    task = Task(
        task_id="t-1",
        prompt="refund order 4412",
        privileged=Privileged(
            principle=f"{SECRET}_principle",
            hidden_state={"answer": f"{SECRET}_hidden"},
            reference=f"{SECRET}_reference",
        ),
    )
    rollout = Rollout(
        rollout_id="r-1",
        task_id="t-1",
        final_text="Refunded.",
        steps=[Step(tool="lookup_order", arguments={"order_id": "4412"}, result={"status": "ok"})],
    )
    row = to_row(task, rollout)
    assert SECRET not in _dump(row)
    for key in PRIVILEGED_KEYS:
        assert key not in row


def test_engine_export_row_drops_privileged_keys():
    """``export_row`` is what ``simulate()`` writes to JSONL."""
    out = export_row(_privileged_row())
    assert SECRET not in _dump(out)
    for key in PRIVILEGED_KEYS:
        assert key not in out


def test_engine_export_row_drops_privileged_keys_at_any_depth():
    """The export carries the whole row now (#149), so the block list is
    applied to the whole row, not just its top level. A top-level-only
    guard let the teacher's answer key ride out inside a carried
    ``scenario_dimensions``, ``lineage``, ``judge_meta`` or tool result."""
    row = _nested_privileged_row()
    assert _privileged_keys_in(row), "fixture must actually nest something"
    out = export_row(row)
    assert SECRET not in _dump(out)
    assert _privileged_keys_in(out) == []
    # the rest of the row still survives: the guard scrubs, it does not
    # drop the carrier
    assert out["scenario_dimensions"] == {"tool": "lookup_order"}
    assert out["lineage"] == {"source": "grade"}
    assert out["markers"] == {"grounded": 1.0}
    assert out["steps"][0]["result"] == {"status": "ok"}


def test_export_row_rebuilds_messages_from_the_scrubbed_steps():
    """``conversation()`` dumps each tool result into a message string, so
    a key scrubbed after that has already stopped being a key. The scrub
    has to happen before anything is derived from the row."""
    row = _nested_privileged_row()
    row.pop("messages", None)
    out = export_row(row)
    assert out["messages"], "the fixture should produce a conversation"
    assert SECRET not in _dump(out["messages"])


def test_saved_file_carries_no_nested_privileged_value(tmp_path):
    """The bytes on disk, which is what the customer in #149 reads back."""
    from zeroproof.simulations.data import SimulationData

    data = SimulationData(trajectories=[_nested_privileged_row()])
    dest = tmp_path / "rows.jsonl"
    data.save(str(dest))
    text = dest.read_text(encoding="utf-8")
    assert SECRET not in text
    for key in (*PRIVILEGED_KEYS, "privileged", "rubric"):
        assert f'"{key}"' not in text
    assert _privileged_keys_in(json.loads(text.splitlines()[0])) == []
    # and the evidence #149 is about is still there
    on_disk = json.loads(text.splitlines()[0])
    assert on_disk["markers"] == {"grounded": 1.0}
    assert on_disk["lineage"] == {"source": "grade"}


def test_training_rows_carry_no_privileged_key_or_value():
    """The SFT/GRPO export: messages plus a fixed carry list."""
    rows = training_rows([_privileged_row()], system_prompt="Be honest.", tools=[])
    assert rows
    for row in rows:
        assert SECRET not in _dump(row)
        for key in PRIVILEGED_KEYS:
            assert key not in row


def test_export_training_file_carries_no_privileged_value(tmp_path):
    """The bytes on disk, not just the in-memory rows."""
    dest = tmp_path / "train.jsonl"
    report = export_training([_privileged_row()], str(dest), system_prompt="Be honest.", tools=[])
    assert report["n"] == 1
    text = dest.read_text(encoding="utf-8")
    assert SECRET not in text
    for key in PRIVILEGED_KEYS:
        assert f'"{key}"' not in text


def test_export_preference_file_carries_no_privileged_value(tmp_path):
    """Both sides of every pair are rebuilt as messages, so both are clean.

    ``build_preference_pairs`` keeps the whole source row on ``chosen`` /
    ``rejected`` as an in-memory intermediate; the export is the boundary
    that has to drop the privileged keys, and this pins that it does.
    """
    rows = [
        _privileged_row(reward=1.0, final_text="Refunded."),
        _privileged_row(reward=0.0, final_text="I refunded the wrong one."),
    ]
    pairs, _report = build_preference_pairs(rows)
    assert pairs
    dest = tmp_path / "pref.jsonl"
    export_preference(pairs, str(dest), system_prompt="Be honest.", tools=[])
    text = dest.read_text(encoding="utf-8")
    assert SECRET not in text
    for key in PRIVILEGED_KEYS:
        assert f'"{key}"' not in text


def test_training_carry_list_names_no_privileged_field():
    """A frozen surface: the carry list is the only way a non-message key
    reaches a training row, so a privileged name must never appear in it."""
    for key in PRIVILEGED_KEYS:
        assert key not in _CARRY_KEYS


def test_eval_rewards_are_distinguishable_from_training_rewards():
    """An eval score and a training score are the same shape, so lineage
    is the only thing that tells them apart. A consumer that cannot read
    ``lineage['source']`` cannot keep the eval scorer out of the reward."""
    rows = [_privileged_row()]
    judge = lambda row: {"reward": 1.0, "reason": "ok"}  # noqa: E731
    graded = run_judge(rows, judge)
    scored = evaluate(rows, judge=judge)
    assert graded.source == "grade"
    assert scored.source == "eval"
    assert graded.rows[0]["lineage"]["source"] == "grade"
    assert scored.rows[0]["lineage"]["source"] == "eval"


def test_rescoring_keeps_the_prior_scoring_run_id():
    """Re-judging graded rows under ``evaluate`` must not erase the run
    that produced the training reward; without the prior id a double-scored
    row is indistinguishable from a fresh one."""
    judge = lambda row: {"reward": 1.0, "reason": "ok"}  # noqa: E731
    graded = run_judge([_privileged_row()], judge)
    rescored = evaluate(graded.rows, judge=lambda row: {"reward": 0.0, "reason": "no"})
    lineage = rescored.rows[0]["lineage"]
    assert lineage["prior_scoring_run_id"] == graded.run_id
    assert lineage["scoring_run_id"] != graded.run_id
    assert lineage["source"] == "eval"
