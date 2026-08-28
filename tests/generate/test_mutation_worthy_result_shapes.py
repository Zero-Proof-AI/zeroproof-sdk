"""`_mutation_worthy` must survive whatever a tool put in `result`.

The canonical step schema says `result: Any`, and most real tools return text.
This package's own adapters do: `from_langchain`, `from_openai_agents`,
`claude_code` and the rest all store `str(...)` there. Calling `.get` on that
crashed `simulate(agent=...)` outright with

    AttributeError: 'str' object has no attribute 'get'

so every one of those adapters was unusable with a callable agent.
"""
import pytest

from zeroproof_simulations import _mutation_worthy
from zeroproof_simulations.adapters import parse_claude_stream


def step(result):
    return {"tool": "read_file", "arguments": {"path": "a.py"}, "result": result}


@pytest.mark.parametrize("result", [
    "plain text a tool returned",
    "ERROR: something went wrong",
    "",
    None,
    123,
    ["a", "list"],
])
def test_non_dict_results_do_not_crash(result):
    assert _mutation_worthy({"steps": [step(result)]}) is False


@pytest.mark.parametrize("status", ["error", "timeout", "not_found", "denied", "malformed"])
def test_dict_faults_are_still_detected(status):
    assert _mutation_worthy({"steps": [step({"status": status})]}) is True


def test_json_encoded_fault_is_detected():
    """`_as_dict` parses a JSON string, so a serialized result still counts."""
    assert _mutation_worthy({"steps": [step('{"status": "error"}')]}) is True


def test_clean_dict_result_is_not_worthy():
    assert _mutation_worthy({"steps": [step({"status": "ok"})]}) is False


def test_faults_column_still_wins():
    assert _mutation_worthy({"faults": ["tool_error"], "steps": []}) is True


def test_non_dict_steps_are_skipped():
    assert _mutation_worthy({"steps": ["not a step", None, 7]}) is False


def test_steps_from_a_real_adapter_do_not_crash():
    """The shape `claude_code` actually emits: result is a string."""
    stream = "\n".join([
        '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1",'
        '"name":"read_file","input":{"path":"a.py"}}]}}',
        '{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1",'
        '"content":"file contents here"}]}}',
        '{"type":"result","result":"done"}',
    ])
    parsed = parse_claude_stream(stream)
    assert isinstance(parsed["steps"][0]["result"], str)
    assert _mutation_worthy(parsed) is False
