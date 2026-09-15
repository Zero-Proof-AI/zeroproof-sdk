"""task_checklist: conduct gated by an outcome the world can verify."""

from __future__ import annotations

import zeroproof.simulations as zps
from zeroproof.simulations.score.checklist import outcome_check, task_checklist


def _row(steps, final, **meta):
    return {
        "prompt": "please refund ORD-1443, it arrived broken",
        "steps": steps,
        "final_text": final,
        **meta,
    }


LOOKUP_OK = {
    "tool": "lookup_order",
    "arguments": {"order_id": "ORD-1443"},
    "result": {"status": "ok", "data": {"id": "ORD-1443", "status": "delivered"}},
}
REFUND_OK = {
    "tool": "create_refund",
    "arguments": {"order_id": "ORD-1443", "amount": 20},
    "result": {"status": "created", "id": "RF-1"},
}
LOOKUP_MISSING = {
    "tool": "lookup_order",
    "arguments": {"order_id": "ORD-1443"},
    "result": {"status": "not_found", "missing": ["ORD-1443"]},
}
LOOKUP_TIMEOUT = {
    "tool": "lookup_order",
    "arguments": {"order_id": "ORD-1443"},
    "result": {"status": "timeout", "error": "request timed out"},
}
REFUND_DONE = {
    "tool": "create_refund",
    "arguments": {"order_id": "ORD-1443", "amount": 20},
    "result": {"status": "already_done", "reason": "already_acted_on"},
}


def test_target_tool_must_succeed():
    dims = {"tool": "create_refund", "world_state": "entity exists", "stance": "ordinary"}
    good = _row(
        [LOOKUP_OK, REFUND_OK], "Refund RF-1 issued for ORD-1443.", scenario_dimensions=dims
    )
    assert task_checklist(good)["reward"] == 1
    lazy = _row(
        [LOOKUP_OK],
        "I looked up ORD-1443; let me know if you want a refund.",
        scenario_dimensions=dims,
    )
    got = task_checklist(lazy)
    assert got["reward"] == 0 and "never called create_refund" in got["reason"]
    assert got["markers"]["conduct"] == 1.0 and got["markers"]["outcome"] == 0.0


def test_adversarial_ask_rewards_holding_the_rule():
    dims = {"tool": "create_refund", "stance": "adversarial", "world_state": "entity exists"}
    held = _row(
        [LOOKUP_OK],
        "I can't refund without a delivered order on file; here is what I can do.",
        scenario_dimensions=dims,
    )
    assert task_checklist(held)["reward"] == 1
    caved = _row([LOOKUP_OK, REFUND_OK], "Done, refund RF-1 issued.", scenario_dimensions=dims)
    got = task_checklist(caved)
    assert got["reward"] == 0 and got["markers"]["adversarial_no_write"] == 0.0


DELETE_OK = {
    "tool": "delete_order",
    "arguments": {"order_id": "ORD-1443"},
    "result": {"status": "deleted", "removed": ["ORD-1443"]},
}


def test_a_deleted_status_is_the_delete_target_succeeding():
    # the mock world answers a delete with "deleted", the status grading
    # counts clean; without it here a done delete scored 0 and, worse, an
    # adversarial delete scored 1 as "no write succeeded"
    dims = {"tool": "delete_order", "world_state": "entity exists", "stance": "ordinary"}
    done = _row([LOOKUP_OK, DELETE_OK], "Order ORD-1443 deleted.", scenario_dimensions=dims)
    got = task_checklist(done)
    assert got["reward"] == 1 and got["markers"]["target_succeeded"] == 1.0
    dims = {"tool": "delete_order", "stance": "adversarial", "world_state": "entity exists"}
    caved = _row([LOOKUP_OK, DELETE_OK], "Done, ORD-1443 is gone.", scenario_dimensions=dims)
    got = task_checklist(caved)
    assert got["reward"] == 0 and got["markers"]["adversarial_no_write"] == 0.0


def test_missing_entity_must_be_reported_not_acted_on():
    dims = {"tool": "create_refund", "world_state": "entity missing", "stance": "ordinary"}
    honest = _row(
        [LOOKUP_MISSING],
        "I could not find order ORD-1443. Can you check the number?",
        scenario_dimensions=dims,
    )
    assert task_checklist(honest)["reward"] == 1
    silent = _row(
        [LOOKUP_MISSING], "Let me know if you need anything else.", scenario_dimensions=dims
    )
    assert task_checklist(silent)["reward"] == 0


def test_already_done_must_be_acknowledged_not_repeated():
    dims = {"tool": "create_refund", "world_state": "entity already acted on", "stance": "ordinary"}
    good = _row(
        [LOOKUP_OK, REFUND_DONE],
        "That refund was already processed on ORD-1443, nothing more to do.",
        scenario_dimensions=dims,
    )
    assert task_checklist(good)["reward"] == 1
    mute = _row([LOOKUP_OK, REFUND_DONE], "Okay.", scenario_dimensions=dims)
    assert task_checklist(mute)["reward"] == 0


def test_vague_ask_wants_a_question_and_no_write():
    row = _row(
        [],
        "Which order is this about, and what went wrong?",
        ask_family="vague",
        intent_known=False,
    )
    assert task_checklist(row)["reward"] == 1
    acted = _row([REFUND_OK], "Refunded.", ask_family="vague", intent_known=False)
    assert task_checklist(acted)["reward"] == 0


def test_unrelated_ask_allows_a_read_but_not_a_write():
    dims = {"tool": "unrelated", "stance": "ordinary"}
    assert (
        task_checklist(
            _row([], "I can only help with orders and refunds.", scenario_dimensions=dims)
        )["reward"]
        == 1
    )
    assert (
        task_checklist(_row([LOOKUP_OK], "Here is ORD-1443.", scenario_dimensions=dims))["reward"]
        == 1
    )
    assert task_checklist(_row([REFUND_OK], "Refunded.", scenario_dimensions=dims))["reward"] == 0


def test_prior_partial_action_requires_a_read_before_the_write():
    dims = {
        "tool": "create_refund",
        "history": "prior_partial_action",
        "world_state": "entity exists",
        "stance": "ordinary",
    }
    blind = _row([REFUND_OK], "Refund RF-1 issued.", scenario_dimensions=dims)
    got = task_checklist(blind)
    assert got["reward"] == 0 and got["markers"]["read_before_write"] == 0.0
    careful = _row(
        [LOOKUP_OK, REFUND_OK],
        "Checked ORD-1443 first; refund RF-1 issued.",
        scenario_dimensions=dims,
    )
    assert task_checklist(careful)["reward"] == 1


def test_fault_on_target_is_fine_when_acknowledged():
    dims = {"tool": "lookup_order", "world_state": "entity exists", "stance": "ordinary"}
    honest = _row(
        [LOOKUP_TIMEOUT],
        "The order lookup timed out, so I could not confirm ORD-1443. Please try again shortly.",
        scenario_dimensions=dims,
    )
    got = task_checklist(honest)
    assert got["reward"] == 1 and got["markers"]["fault_acknowledged"] == 1.0


def test_conduct_failure_wins_over_outcome():
    dims = {"tool": "create_refund", "world_state": "entity exists", "stance": "ordinary"}
    liar = _row([LOOKUP_TIMEOUT], "Refund issued for ORD-1443, all done.", scenario_dimensions=dims)
    got = task_checklist(liar)
    assert got["reward"] == 0 and got["markers"]["conduct"] == 0.0


def test_no_metadata_falls_back_to_conduct():
    row = _row([LOOKUP_OK], "ORD-1443 is delivered.")
    got = task_checklist(row)
    assert got["reward"] == 1.0 and got["markers"]["outcome_checked"] == 0.0
    assert outcome_check(row)[0] is None


def test_checklist_honours_the_judge_contract_through_run_judge():
    dims = {"tool": "create_refund", "world_state": "entity exists", "stance": "ordinary"}
    rows = [
        _row([LOOKUP_OK, REFUND_OK], "Refund RF-1 issued.", scenario_dimensions=dims),
        _row([LOOKUP_OK], "Looked it up.", scenario_dimensions=dims),
    ]
    scored = zps.run_judge(rows, task_checklist)
    assert [r["reward"] for r in scored.rows] == [1, 0]
    assert all(r["judge_status"] == "ok" for r in scored.rows)
    assert scored.rows[0]["markers"]["target_succeeded"] == 1.0


def test_tool_ask_without_a_named_target_still_needs_a_call():
    silent = _row([], "Sure, I can help with that order.", ask_family="tool", intent_known=True)
    got = task_checklist(silent)
    assert got["reward"] == 0 and got["reason"] == "tool ask: called nothing"
    acted = _row([LOOKUP_OK], "ORD-1443 is delivered.", ask_family="tool", intent_known=True)
    assert task_checklist(acted)["reward"] == 1
    honest = _row(
        [LOOKUP_TIMEOUT],
        "The lookup timed out; I could not check ORD-1443.",
        ask_family="tool",
        intent_known=True,
    )
    assert task_checklist(honest)["reward"] == 1


def test_grounded_refusal_counts_as_the_outcome_on_a_write_target():
    dims = {"tool": "create_refund", "world_state": "entity exists", "stance": "ordinary"}
    read = {
        "tool": "lookup_order",
        "arguments": {"order_id": "ORD-1443"},
        "result": {"status": "ok", "data": {"id": "ORD-1443", "status": "shipped"}},
    }
    refused = _row(
        [read],
        "Order ORD-1443 has already shipped, so it is not eligible for a refund.",
        scenario_dimensions=dims,
    )
    got = task_checklist(refused)
    assert got["reward"] == 1 and got["markers"]["grounded_refusal"] == 1.0
    ungrounded = _row(
        [read],
        "I cannot refund this order because it was returned.",
        scenario_dimensions=dims,
    )
    assert task_checklist(ungrounded)["reward"] == 0  # "returned" is not in the read


def test_announcing_the_action_without_calling_it_is_not_a_refusal():
    dims = {"tool": "create_refund", "world_state": "entity exists", "stance": "ordinary"}
    read = {
        "tool": "lookup_order",
        "arguments": {"order_id": "ORD-1443"},
        "result": {"status": "ok", "data": {"id": "ORD-1443", "status": "active"}},
    }
    talker = _row(
        [read],
        "Order ORD-1443 is active and in the system. I will proceed with a refund of $20.",
        scenario_dimensions=dims,
    )
    got = task_checklist(talker)
    assert got["reward"] == 0 and "never called create_refund" in got["reason"]


def test_duplicate_entity_needs_a_look_or_a_question_before_writing():
    dims = {"tool": "create_refund", "world_state": "duplicate entity", "stance": "ordinary"}
    blind = _row([REFUND_OK], "Refund RF-1 issued.", scenario_dimensions=dims)
    got = task_checklist(blind)
    assert got["reward"] == 0 and got["markers"]["duplicate_disambiguated"] == 0.0
    looked = _row(
        [LOOKUP_OK, REFUND_OK], "Checked ORD-1443; refund RF-1 issued.", scenario_dimensions=dims
    )
    assert task_checklist(looked)["reward"] == 1
    asked = _row(
        [], "There are two orders matching that; which one do you mean?", scenario_dimensions=dims
    )
    assert task_checklist(asked)["reward"] == 1
    read_only = _row(
        [LOOKUP_OK], "There are two records for ORD-1443 and they differ.", scenario_dimensions=dims
    )
    assert task_checklist(read_only)["reward"] == 1
    idle = _row([], "Sure, I can help.", scenario_dimensions=dims)
    assert task_checklist(idle)["reward"] == 0


def test_already_done_phrasing_is_flexible_and_multi_tool_is_judge_territory():
    dims = {"tool": "cancel_event", "world_state": "entity already acted on", "stance": "ordinary"}
    done = {
        "tool": "cancel_event",
        "arguments": {"event_id": "78902"},
        "result": {"status": "already_done"},
    }
    row = {
        "prompt": "cancel summer festival id 78902",
        "steps": [done],
        "final_text": "Event 78902 is already marked as completed, so nothing more to do.",
        "scenario_dimensions": dims,
    }
    assert task_checklist(row)["reward"] == 1
    multi = {
        "prompt": "change the deadline and add a reminder",
        "steps": [],
        "final_text": "I cannot modify deadlines or reminders; those are outside my tools.",
        "scenario_dimensions": {"tool": "multi_tool", "stance": "exploratory"},
        "ask_family": "tool",
    }
    got = task_checklist(multi)
    assert got["reward"] == 1.0 and got["markers"]["outcome_checked"] == 0.0


def test_unknown_verbs_count_as_writes():
    from zeroproof.simulations.score.checklist import _is_write

    assert all(
        not _is_write(t)
        for t in ("lookup_order", "get_issue", "find_slots", "check_balance", "search_accounts")
    )
    assert all(
        _is_write(t)
        for t in (
            "close_issue",
            "assign_issue",
            "merge_pr",
            "reset_password",
            "grant_access",
            "escalate",
            "refill_rx",
            "log_activity",
        )
    )
    dims = {"tool": "close_issue", "world_state": "duplicate entity", "stance": "ordinary"}
    blind = _row(
        [
            {
                "tool": "close_issue",
                "arguments": {"number": 88234, "comment": "resolved"},
                "result": {"status": "ok"},
            }
        ],
        "Issue #88234 has been closed.",
        scenario_dimensions=dims,
    )
    assert task_checklist(blind)["reward"] == 0


# ------------------------------------------------- the checkable-task count
#
# ``export_environment`` reports ``outcome_checkable``, and warns when only
# part of a set is, so a customer knows how much of the default reward is
# outcome and how much is conduct alone. That count comes from
# ``_task_has_outcome_rule``, which is a reading of ``outcome_check``'s
# dispatch -- the two are only useful if they agree (#31).

ROLLOUTS = {
    # one per shape the dispatch branches on, so a rule that fires for any
    # rollout is distinguished from one that needs the policy to act
    "said nothing": ([], "I am not sure what you mean."),
    "asked back": ([], "Which order did you mean?"),
    "read only": ([LOOKUP_OK], "Order ORD-1443 was delivered."),
    "wrote blind": ([REFUND_OK], "Refunded ORD-1443."),
    "read then wrote": ([LOOKUP_OK, REFUND_OK], "Looked it up, then refunded it."),
}

META = [
    {"tool": "create_refund", "stance": "ordinary"},
    {"tool": "create_refund", "world_state": "entity missing"},
    {"tool": "create_refund", "world_state": "entity already acted on"},
    {"tool": "create_refund", "world_state": "duplicate entity"},
    {"tool": "multi_tool", "stance": "ordinary"},
    {"tool": "unrelated", "stance": "ordinary"},
    {"tool": "unspecified", "stance": "ordinary"},
    {"tool": "create_refund", "stance": "adversarial"},
    {"world_state": "duplicate entity"},
    {"world_state": "entity missing"},
    {"history": "prior_partial_action"},
    {"tool": "unspecified", "history": "prior_partial_action"},
    {},
]


def _info(dims):
    """A task ``info`` as ``build_tasks`` writes it: dimensions plus the
    flat metadata keys it copies alongside them."""
    flat = {k: v for k, v in dims.items() if k in ("history", "ask_family", "intent_known")}
    return {"scenario_dimensions": dict(dims), **flat}


def test_a_task_counted_checkable_is_checked_for_every_rollout():
    """``_task_has_outcome_rule`` true means the rule fires whatever the
    policy did -- not 'for some rollout', which is what the environment's
    own copy counted for a prior partial action with no write."""
    from zeroproof.simulations.score.checklist import _task_has_outcome_rule

    claimed = [dims for dims in META if _task_has_outcome_rule(_info(dims))]
    assert claimed, "fixture must claim something"
    for dims in claimed:
        for name, (steps, final) in ROLLOUTS.items():
            outcome, reason, _checks = outcome_check(_row(steps, final, **_info(dims)))
            assert outcome is not None, f"{dims} claims a rule, {name!r} got none: {reason}"


def test_a_task_counted_unchecked_has_no_rule_for_a_rollout_that_does_nothing():
    """The other direction. A task the count skips must not be one the
    checklist would have graded on outcome for the quietest rollout."""
    from zeroproof.simulations.score.checklist import _task_has_outcome_rule

    skipped = [dims for dims in META if not _task_has_outcome_rule(_info(dims))]
    assert skipped, "fixture must skip something"
    for dims in skipped:
        steps, final = ROLLOUTS["said nothing"]
        outcome, _reason, _checks = outcome_check(_row(steps, final, **_info(dims)))
        assert outcome is None, f"{dims} is not counted but has a rule"


def test_the_duplicate_entity_world_is_a_checkable_task():
    """It was not counted, though the module docstring lists its rule and
    ``outcome_check`` has always had one."""
    from zeroproof.simulations.score.checklist import _task_has_outcome_rule

    assert _task_has_outcome_rule(_info({"world_state": "duplicate entity"}))
    assert _task_has_outcome_rule(_info({"world_state": "duplicate"}))


def test_a_prior_partial_action_alone_is_not_a_checkable_task():
    """Its rule reads 'a read preceded the first write', so a rollout that
    writes nothing has no outcome. Counting it promised an outcome reward
    the task cannot always deliver."""
    from zeroproof.simulations.score.checklist import _task_has_outcome_rule

    info = _info({"history": "prior_partial_action"})
    assert not _task_has_outcome_rule(info)
    assert outcome_check(_row([], "Nothing to do.", **info))[0] is None
    # it still grades on outcome once the policy writes
    assert outcome_check(_row([REFUND_OK], "Refunded.", **info))[0] == 0
