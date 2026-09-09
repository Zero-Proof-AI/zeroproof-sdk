"""The JSONL row keeps what a grader needs to group and reproduce a rollout."""
from zeroproof_simulations.data import _export_row


def test_export_row_keeps_group_identity_and_reproduction_fields():
    row = {
        "prompt": "fix the failing test", "steps": [], "final_text": "done",
        "scenario_id": "sc-1", "rollout_index": 3, "seed": 7,
        "scenario_dimensions": {"tool": "run_tests", "stance": "hurried"},
        "model_version": "Qwen/Qwen3-4B-Instruct-2507",
        "vector": [0.1, 0.2], "semantic_novelty": 0.9,
    }
    out = _export_row(row)
    assert out["rollout_index"] == 3
    assert out["model_version"] == "Qwen/Qwen3-4B-Instruct-2507"
    # search bookkeeping stays in memory; situation labels are flattened elsewhere
    for key in ("vector", "semantic_novelty", "seed", "scenario_dimensions"):
        assert key not in out


def test_export_row_omits_missing_group_fields():
    out = _export_row({"prompt": "p", "steps": [], "final_text": "t"})
    for key in ("rollout_index", "model_version"):
        assert key not in out



def test_faults_export_carries_fault_modes_only():
    from zeroproof_simulations.data import _export_row
    row = {"prompt": "p", "steps": [], "final_text": "f", "scenario_id": "s",
           "stance": "hurried",
           "faults": {"*": {"mode": "timeout", "rate": 1.0}, "stance": "hurried",
                      "texture": "lowercase", "world_state": "entity missing"}}
    out = _export_row(row)
    assert out["faults"] == {"*": {"mode": "timeout", "rate": 1.0}}
    assert out["stance"] == "hurried"
    assert "faults" not in _export_row({**row, "faults": {"stance": "hurried"}})
