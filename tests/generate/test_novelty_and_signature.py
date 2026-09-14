"""novelty is a cosine-distance floor; behavior_signature hashes how the agent acted."""

from __future__ import annotations

import zeroproof.simulations as zps
from zeroproof.simulations.generate.scenarios import novelty
from zeroproof.simulations.score.grading import behavior_signature


def test_novelty_is_the_minimum_cosine_distance_to_any_tested_row():
    assert novelty([1, 0], [[1, 0], [0, 1]]) == 0.0  # the nearest row decides
    assert novelty([1, 0], [[0, 1]]) == 1.0
    assert abs(novelty([2, 0], [[1, 1]]) - (1 - 1 / 2**0.5)) < 1e-9  # scale free


def test_novelty_with_nothing_tested_is_full():
    assert novelty([1, 2], []) == 1.0
    assert novelty([1, 2], None) == 1.0


def test_novelty_of_a_zero_vector_is_full_not_nan():
    assert novelty([0, 0], [[1, 0]]) == 1.0
    assert novelty([1, 0], [[0, 0]]) == 1.0


def _traj(steps, final="Refunded $40.", prompt="refund order ord_40"):
    return {"prompt": prompt, "steps": steps, "final_text": final}


def _lookup(order="ord_40", status="ok"):
    return {"tool": "lookup_order", "arguments": {"order_id": order}, "result": {"status": status}}


def _refund(order="ord_40"):
    return {
        "tool": "create_refund",
        "arguments": {"order_id": order, "amount": 40},
        "result": {"status": "created"},
    }


def test_behavior_signature_is_deterministic_and_short():
    a = behavior_signature(_traj([_lookup(), _refund()]))
    assert a == behavior_signature(_traj([_lookup(), _refund()]))
    assert len(a) == 12 and int(a, 16) >= 0
    assert zps.behavior_signature is behavior_signature


def test_signature_separates_tool_order_result_status_and_refusal():
    base = behavior_signature(_traj([_lookup(), _refund()]))
    reordered = behavior_signature(_traj([_refund(), _lookup()]))
    failed = behavior_signature(_traj([_lookup(status="not_found"), _refund()]))
    refused = behavior_signature(_traj([_lookup(), _refund()], final="I can't refund that."))
    assert len({base, reordered, failed, refused}) == 4


def test_signature_tracks_where_an_argument_came_from_not_its_value():
    # the same shape with a different order id copied from the prompt: one behavior
    copied = behavior_signature(_traj([_lookup("ord_40")], prompt="refund order ord_40"))
    copied_other = behavior_signature(_traj([_lookup("ord_77")], prompt="refund order ord_77"))
    # an id the user never gave: a different behavior (invented argument)
    invented = behavior_signature(_traj([_lookup("ord_99")], prompt="refund order ord_40"))
    assert copied == copied_other
    assert copied != invented


def test_signature_reads_an_argument_derived_from_an_earlier_result():
    derived_steps = [
        {"tool": "lookup_order", "arguments": {"email": "a@b.c"}, "result": {"id": "ord_51"}},
        _refund("ord_51"),
    ]
    invented_steps = [
        {"tool": "lookup_order", "arguments": {"email": "a@b.c"}, "result": {"id": "ord_51"}},
        _refund("ord_52"),
    ]
    prompt = "refund the order for a@b.c"
    assert behavior_signature(_traj(derived_steps, prompt=prompt)) != behavior_signature(
        _traj(invented_steps, prompt=prompt)
    )


def test_signature_skips_steps_without_a_tool():
    with_note = _traj([{"note": "thinking"}, _lookup(), _refund()])
    assert behavior_signature(with_note) == behavior_signature(_traj([_lookup(), _refund()]))
