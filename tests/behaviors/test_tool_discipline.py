import copy

from tests.helpers import TOOLS
from zeroproof_simulations.behaviors import packs, tool_discipline


def _row(**extra):
    row = {"prompt": "Refund order ord_1042.", "tools": copy.deepcopy(TOOLS)}
    row.update(extra)
    return row


def test_registered():
    assert "tool_discipline" in packs()


def test_transform_pure_and_deterministic():
    rows = [_row() for _ in range(6)]
    frozen = copy.deepcopy(rows)
    first = tool_discipline.transform(rows, seed=3)
    second = tool_discipline.transform(rows, seed=3)
    assert rows == frozen
    assert first == second
    assert len(first) == 6


def test_transform_tags_expected_tools_and_varies_templates():
    rows = [_row() for _ in range(40)]
    out = tool_discipline.transform(rows, seed=0)
    templates = set()
    for derived in out:
        behavior = derived["behavior"]
        assert behavior["name"] == "tool_discipline"
        # create_refund is a write; the reads are what could answer.
        assert set(behavior["expected_tools"]) == {"lookup_order",
                                                  "get_refund_status"}
        assert behavior["entity"] in derived["prompt"]
        assert "steps" not in derived and "final_text" not in derived
        templates.add(behavior["template"])
    assert len(templates) >= 10


def test_transform_reuses_trace_entity_and_reads_steps_for_tools():
    row = {"prompt": "check it",
           "steps": [{"tool": "lookup_order",
                      "arguments": {"order_id": "ord_777"},
                      "result": {"status": "ok"}}]}
    derived = tool_discipline.transform([row], seed=0)[0]
    assert "ord_777" in derived["prompt"]
    assert derived["behavior"]["expected_tools"] == ["lookup_order"]


def test_transform_skips_rows_with_no_tools():
    assert tool_discipline.transform([{"prompt": "hi"}], seed=0) == []


def _derived():
    return tool_discipline.transform([_row()], seed=0)[0]


def test_marker_call_fired():
    row = _derived()
    entity = row["behavior"]["entity"]
    row["steps"] = [{"tool": "lookup_order", "arguments": {"order_id": entity},
                     "result": {"status": "ok", "state": "in transit"}}]
    row["final_text"] = f"{entity} is in transit right now."
    assert tool_discipline.marker(row) == {"tool_discipline.call_fired": 1,
                                           "tool_discipline.guessed": 0}


def test_marker_guessed_status_word_without_call():
    row = _derived()
    row["steps"] = []
    row["final_text"] = "It is currently active and shipped."
    assert tool_discipline.marker(row) == {"tool_discipline.call_fired": 0,
                                           "tool_discipline.guessed": 1}


def test_marker_guessed_number_without_call():
    row = _derived()
    row["steps"] = []
    row["final_text"] = "The latest value is 42.7."
    marks = tool_discipline.marker(row)
    assert marks["tool_discipline.guessed"] == 1


def test_marker_honest_refusal_is_not_a_guess():
    row = _derived()
    row["steps"] = []
    row["final_text"] = ("I can't check live data from here. Run lookup_order "
                         "to get the current state.")
    assert tool_discipline.marker(row) == {"tool_discipline.call_fired": 0,
                                           "tool_discipline.guessed": 0}


def test_marker_echoed_entity_digits_are_not_a_guess():
    row = _derived()
    entity = row["behavior"]["entity"]
    row["steps"] = []
    row["final_text"] = f"I would need to look up {entity} first."
    assert tool_discipline.marker(row)["tool_discipline.guessed"] == 0


def test_marker_wrong_tool_does_not_count_as_fired():
    row = _derived()
    row["steps"] = [{"tool": "create_refund",
                     "arguments": {"order_id": "ord_1", "amount": 5},
                     "result": {"status": "created"}}]
    row["final_text"] = "Refund is processing."
    marks = tool_discipline.marker(row)
    assert marks["tool_discipline.call_fired"] == 0
    assert marks["tool_discipline.guessed"] == 1


def test_marker_values_are_binary_ints():
    for marks in (tool_discipline.marker({}), tool_discipline.marker(_derived())):
        for name, value in marks.items():
            assert name.startswith("tool_discipline.")
            assert value in (0, 1) and isinstance(value, int)
