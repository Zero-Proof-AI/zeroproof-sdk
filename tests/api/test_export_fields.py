"""The JSONL row keeps what a grader needs to group and reproduce a rollout."""

import json

import zeroproof.simulations as zps
from tests.helpers import POLICY, TOOLS, scripted_agent
from zeroproof.simulations.data import _export_row


def test_export_row_keeps_group_identity_and_reproduction_fields():
    row = {
        "prompt": "fix the failing test",
        "steps": [],
        "final_text": "done",
        "scenario_id": "sc-1",
        "rollout_index": 3,
        "seed": 7,
        "scenario_dimensions": {"tool": "run_tests", "stance": "hurried"},
        "model_version": "Qwen/Qwen3-4B-Instruct-2507",
        "vector": [0.1, 0.2],
        "semantic_novelty": 0.9,
    }
    out = _export_row(row)
    assert out["rollout_index"] == 3
    assert out["model_version"] == "Qwen/Qwen3-4B-Instruct-2507"
    # #149: reproducibility and coverage evidence ride to disk with the row
    assert out["seed"] == 7
    assert out["scenario_dimensions"] == {"tool": "run_tests", "stance": "hurried"}
    assert out["semantic_novelty"] == 0.9
    # the raw embedding is in-memory search bookkeeping and never exported
    assert "vector" not in out


def test_export_row_omits_missing_group_fields():
    out = _export_row({"prompt": "p", "steps": [], "final_text": "t"})
    for key in ("rollout_index", "model_version"):
        assert key not in out


def test_faults_export_carries_fault_modes_only():
    from zeroproof.simulations.data import _export_row

    row = {
        "prompt": "p",
        "steps": [],
        "final_text": "f",
        "scenario_id": "s",
        "stance": "hurried",
        "faults": {
            "*": {"mode": "timeout", "rate": 1.0},
            "stance": "hurried",
            "texture": "lowercase",
            "world_state": "entity missing",
        },
    }
    out = _export_row(row)
    assert out["faults"] == {"*": {"mode": "timeout", "rate": 1.0}}
    assert out["stance"] == "hurried"
    assert "faults" not in _export_row({**row, "faults": {"stance": "hurried"}})


# --------------------------------------------------------------------- #149

#: What a graded row has to still say once it has left memory: who judged
#: it and how that went, which coverage cell it filled, and the seed that
#: reproduces it. Dropping any of these made a saved run unable to prove
#: its own provenance (#149), and losing ``markers`` re-broke #56 for
#: everyone reading ``rows()`` instead of ``trajectories``.
_EVIDENCE = ("markers", "judge_status", "judge_name", "lineage", "seed", "scenario_dimensions")


def _graded_run(output=None):
    def judge(row):
        return {"reward": 1, "reason": "ok", "markers": {"m1": 1.0}}

    return zps.simulate(
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        budget=6,
        seed=0,
        simulator=False,
        concurrency=4,
        grader=judge,
        output=output,
        advanced={"per_round": 6, "mutate_failures": False},
    )


def test_graded_row_survives_rows_and_a_round_trip_through_output(tmp_path):
    dest = tmp_path / "check.jsonl"
    data = _graded_run(output=str(dest))
    trajectory = data.trajectories[0]
    for key in _EVIDENCE:
        assert trajectory.get(key) is not None, f"fixture lost {key} before the export"

    exported = data.rows()[0]
    on_disk = [json.loads(line) for line in dest.read_text().splitlines() if line.strip()]
    assert len(on_disk) == len(data.trajectories)

    for name, row in (("rows()", exported), ("output=", on_disk[0])):
        for key in _EVIDENCE:
            assert row.get(key) == trajectory[key], f"{name} dropped {key}"
        assert row["world_state"] == trajectory["world_state"]

    # nothing a trajectory carries goes missing on either path
    for row in (exported, on_disk[0]):
        missing = {k for k, v in trajectory.items() if v is not None} - set(row)
        assert not missing, missing

    # #56 on the rows() path: marker_summary needs the markers to be there
    summary = zps.marker_summary(data.rows())
    assert summary and "m1" in summary


def test_rows_reads_as_a_list_and_as_a_call():
    """``ScoredData.rows`` is a list, so ``SimulationData.rows`` answers to
    both spellings rather than raising ``TypeError: 'method' object is not
    iterable`` on the one that is not a call."""
    data = _graded_run()
    as_attribute = list(data.rows)
    as_call = data.rows()
    assert as_attribute == list(as_call) == [_export_row(t) for t in data.trajectories]
    assert len(data.rows) == len(data.trajectories)
    assert data.rows[0]["prompt"] == data.trajectories[0]["prompt"]
