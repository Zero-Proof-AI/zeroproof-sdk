"""behavior_state: trace history in, budget allocation out. Each of the
lifecycle rules gets a synthetic timeline; exploration is always
reserved; loud repeated failures saturate instead of monopolizing."""
from zeroproof_simulations.traces import behavior_state


def _row(tool, ok, ts):
    result = {"status": "ok"} if ok else {"status": "timeout"}
    return {"prompt": f"p{ts}", "ts": ts, "reward": 1 if ok else 0,
            "steps": [{"tool": tool, "arguments": {}, "result": result}],
            "final_text": "done"}


def test_lifecycle_classification():
    rows = []
    rows += [_row("refund", False, t) for t in (1, 2)]      # old fail
    rows += [_row("refund", False, t) for t in (11, 12)]    # recent fail -> persistent
    rows += [_row("lookup", False, 3)] + [_row("lookup", True, 13)]   # improving
    rows += [_row("cancel", False, 4)]                      # old fail, no recent -> uncertain
    rows += [_row("book", False, 14)]                       # new
    rows += [_row("status", True, 5), _row("status", True, 15)]  # passing
    state = behavior_state(rows, recent_fraction=0.5)
    by = {r["region"]: r["status"] for r in state["regions"]}
    assert by["refund"] == "persistent"
    assert by["lookup"] == "improving"
    assert by["cancel"] == "uncertain"
    assert by["book"] == "new"
    assert by["status"] == "passing"
    shares = {r["region"]: r["budget_share"]
              for r in state["regions"]}
    assert shares["book"] > shares["cancel"] > shares["lookup"]
    assert shares["refund"] > shares["improving"] if "improving" in shares \
        else shares["refund"] > shares["lookup"]


def test_exploration_always_reserved_and_shares_sum():
    rows = [_row("a", False, t) for t in range(10)]
    state = behavior_state(rows)
    assert state["exploration_share"] >= 0.2
    total = sum(r["budget_share"] for r in state["regions"])
    assert abs(total + state["exploration_share"] - 1.0) < 0.01


def test_repetition_saturates():
    loud = [_row("loud", False, t) for t in range(1, 40)]
    quiet = [_row("quiet", False, 39)]
    state = behavior_state(loud + quiet)
    shares = {r["region"]: r["budget_share"]
              for r in state["regions"]}
    assert shares["loud"] < 3 * shares["quiet"]


def test_empty_history():
    state = behavior_state([])
    assert state["regions"] == [] and state["exploration_share"] == 1.0
