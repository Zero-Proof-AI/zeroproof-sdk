from tests.helpers import TOOLS, POLICY, scripted_agent
import zeroproof_simulations as zps
from zeroproof_simulations.coverage import space_saturated


def test_space_saturated_needs_five_copies():
    assert space_saturated({}, {}) is False
    assert space_saturated({"a": 4, "b": 5}, {}) is False
    assert space_saturated({"a": 5, "b": 5}, {}) is True
    assert space_saturated(
        {"a": 5}, {"s": 1}, uncovered_shapes=2, walked_shapes=True) is False
    assert space_saturated(
        {"a": 5}, {"s": 5}, uncovered_shapes=0, walked_shapes=True) is True


def test_space_saturated_requires_the_planned_cell_universe():
    assert space_saturated(
        {"a": 5}, {}, expected_cells={"a", "b"}) is False
    assert space_saturated(
        {"a": 5, "b": 5}, {}, expected_cells={"a", "b"}) is True


def test_coverage_curve_grows_each_batch(tmp_path):
    data = zps.simulate(
        scripted_agent, tools=TOOLS, policy=POLICY, budget=40, seed=0,
        grade=False, concurrency=8, simulator=False,
        advanced={"per_round": 16, "mutate_failures": False})
    assert data.coverage_curve
    rows = [p["n_rows"] for p in data.coverage_curve]
    assert rows == sorted(rows)
    assert rows[-1] == len(data.trajectories)
    assert data.coverage["rows"] == len(data.trajectories)
    assert data.coverage["unique_signatures"] > 0
    path = data.save(str(tmp_path / "r.jsonl"), meta=True)
    meta = tmp_path / "r.meta.json"
    assert meta.exists()
    import json
    blob = json.loads(meta.read_text())
    assert blob["coverage_curve"]
    assert blob["search"]["cell_counts"]
    assert blob["search"]["copies_needed"] == 5
    assert "arm_weights" in blob["search"] or blob.get("arm_weights")


def test_coverage_tracks_cells_without_halting():
    dims = {
        "tool": ["lookup_order"], "rule": ["unspecified"], "stance": ["ordinary"],
        "world_state": ["unspecified"], "tool_condition": ["success"],
        "history": ["fresh"],
    }
    data = zps.simulate(
        lambda m: {"steps": [], "final_text": "ok"},
        tools=TOOLS, policy=POLICY, budget=80, seed=0, grade=False,
        concurrency=8, until="compute", dimensions=dims, simulator=False,
        time_budget=None, mode="adaptive", rollouts_per_request=12,
        advanced={"per_round": 4, "mutate_failures": False})
    assert data.stopped_because == "budget"
    assert data.coverage["saturation"] is False
    assert data.coverage["rows"] == 80
    assert data.coverage.get("predicted_to_saturation") is not None
    assert data.search.get("copies_needed") == 5
    assert data.search.get("cell_counts")


def test_until_saturation_halts_on_tiny_grid():
    dims = {
        "tool": ["lookup_order"], "rule": ["unspecified"], "stance": ["ordinary"],
        "world_state": ["unspecified"], "tool_condition": ["success"],
        "history": ["fresh"],
    }
    data = zps.simulate(
        lambda m: {"steps": [], "final_text": "ok"},
        tools=TOOLS, policy=POLICY, budget=80, seed=0, grade=False,
        concurrency=4, until="saturation", dimensions=dims, simulator=False,
        time_budget=None, rollouts_per_request=5, mode="adaptive",
        advanced={"per_round": 4, "mutate_failures": False})
    assert data.stopped_because == "saturation"
    assert len(data.trajectories) < 80
    assert data.coverage.get("until") == "saturation"
    assert space_saturated(data.search.get("cell_counts") or {},
                           data.search.get("shape_counts") or {})


def test_budget_mode_predicts_toward_budget():
    data = zps.simulate(
        scripted_agent, tools=TOOLS, policy=POLICY, budget=80, seed=0,
        grade=False, until="budget_only", concurrency=8, simulator=False,
        rollouts_per_request=2,
        advanced={"per_round": 16, "mutate_failures": False})
    assert data.stopped_because == "budget"
    assert data.coverage["rows"] == 80
    assert data.coverage["predicted_to_saturation"] == 80
