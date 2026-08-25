"""RL filter keeps gold ``reward`` rows. Offline: no writes, no GPU."""
from zeroproof_simulations.optimize import (
    INCOMPLETE_JUNK, KEPT_VERIFIED_ZERO, UNUSABLE_LABEL, drop_reason,
    filter_rl_rows, is_unusable_label, is_verified_zero,
)


def _kept(**extra):
    row = {
        "prompt": "look up issue 4412",
        "final_text": "Issue 4412 is open.",
        "steps": [{"tool": "get_issue", "arguments": {"number": 4412},
                   "result": {"status": "ok"}}],
        "messages": [
            {"role": "user", "content": "look up issue 4412"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "get_issue",
                             "arguments": {"number": 4412}}]},
            {"role": "tool", "name": "get_issue", "content": '{"status": "ok"}'},
            {"role": "assistant", "content": "Issue 4412 is open."},
        ],
    }
    row.update(extra)
    return row


def test_grade_style_reward_only_is_kept():
    row = _kept(reward=1)
    assert "qwen_reward" not in row
    assert is_unusable_label(row) is False
    assert drop_reason(row) is None


def test_legacy_qwen_reward_only_is_kept():
    row = _kept(qwen_reward=0)
    assert "reward" not in row
    assert is_unusable_label(row) is False
    assert drop_reason(row) is None


def test_missing_label_is_unusable():
    row = _kept()
    assert is_unusable_label(row) is True
    assert drop_reason(row) == UNUSABLE_LABEL


def test_filter_keeps_reward_and_qwen_rows():
    rows = [_kept(reward=1), _kept(qwen_reward=0), _kept()]
    kept, report = filter_rl_rows(rows)
    assert report["n_kept"] == 2
    assert report["dropped"][UNUSABLE_LABEL] == 1
    assert len(kept) == 2


def _junk_zero(**extra):
    # Degenerate final text on an actionable ask: junk by every gate.
    row = _kept(reward=0, final_text="a" * 40)
    row["steps"] = []
    row["messages"] = [
        {"role": "user", "content": "look up issue 4412"},
        {"role": "assistant", "content": "a" * 40},
    ]
    row.update(extra)
    return row


def test_verified_zero_bypasses_junk_gates():
    row = _junk_zero(label_source="claude_fleet_blind_20260821")
    assert is_verified_zero(row) is True
    assert drop_reason(row) is None
    kept, report = filter_rl_rows([row])
    assert len(kept) == 1
    assert report[KEPT_VERIFIED_ZERO] == 1


def test_unverified_junk_zero_still_drops():
    assert drop_reason(_junk_zero()) == INCOMPLETE_JUNK
    assert drop_reason(_junk_zero(label_source="judge")) == INCOMPLETE_JUNK


def test_verified_one_does_not_bypass():
    row = _junk_zero(reward=1, label_source="claude_fleet_blind_20260821")
    assert is_verified_zero(row) is False
    assert drop_reason(row) == INCOMPLETE_JUNK


def test_verified_zero_needs_trainable_content():
    row = _junk_zero(label_source="gold", final_text="")
    row["messages"] = []
    assert drop_reason(row) == INCOMPLETE_JUNK


def _graded(prompt, reward, suffix=""):
    row = _kept(reward=reward)
    row["prompt"] = prompt
    row["final_text"] = f"Issue 4412 is open.{suffix}"
    return row


def _grouped_rows():
    rows = []
    rows += [_graded("mixed ask", r) for r in (1, 0, 1, 0)]
    rows += [_graded("all zero ask", 0) for _ in range(4)]
    rows += [_graded("all one ask", 1) for _ in range(4)]
    rows += [_graded("solo ask", 1)]
    return rows


def test_group_signal_counts_mix_and_band():
    from zeroproof_simulations.optimize import group_signal
    signal = group_signal(_grouped_rows())
    assert signal["n_groups"] == 4
    assert signal["n_mixed"] == 1
    assert signal["n_all_zero"] == 1
    assert signal["n_all_one"] == 1
    assert signal["n_single"] == 1
    assert signal["n_in_band"] == 1  # p = 0.5
    assert signal["mixed_rate"] == 1 / 3


def test_trim_unanimous_drops_dead_groups_keeps_singles():
    from zeroproof_simulations.optimize import trim_unanimous_groups
    kept, report = trim_unanimous_groups(_grouped_rows())
    prompts = {row["prompt"] for row in kept}
    assert prompts == {"mixed ask", "solo ask"}
    assert report["n_groups_dropped"] == 2
    assert report["signal"]["n_mixed"] == 1


def test_select_for_rl_keeps_whole_groups():
    from zeroproof_simulations.optimize import select_for_rl
    picked, report = select_for_rl(_grouped_rows(), target=4)
    prompts = [row["prompt"] for row in picked]
    assert prompts.count("mixed ask") == 4  # the group came whole
    assert report["unanimous_groups_dropped"] == 2
    assert report["signal"]["n_mixed"] == 1


def test_select_for_sft_takes_only_passes_and_spreads_behaviors():
    from zeroproof_simulations.optimize import select_for_sft
    rows = _grouped_rows()
    picked, report = select_for_sft(rows, target=3)
    assert picked
    assert all(row["reward"] == 1 for row in picked)
    # Duplicate prompts never ship twice.
    prompts = [row["prompt"] for row in picked]
    assert len(prompts) == len(set(prompts))
    assert report["n_selected"] == len(picked)


def test_optimize_dispatches_on_mode_and_never_overwrites(tmp_path):
    import json
    from zeroproof_simulations.optimize import optimize
    src = tmp_path / "batch.jsonl"
    rows = _grouped_rows()
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    picked, report = optimize(str(src), mode="rl", target=4)
    assert report["mode"] == "rl"
    assert report["path"].endswith("batch.rl.jsonl")
    assert src.read_text().count("\n") == len(rows)  # source untouched
    sft_rows, sft_report = optimize(rows, mode="sft", target=2)
    assert sft_report["mode"] == "sft"
    assert all(r["reward"] == 1 for r in sft_rows)


def test_no_tool_agent_keeps_refusal_demonstrations():
    from zeroproof_simulations.optimize import is_do_nothing, select_for_sft
    row = {
        "prompt": "check the status of order 98765 for me",
        "ask_family": "tool",
        "final_text": "I cannot look up orders. For billing, see the front desk.",
        "reward": 1,
        "steps": [],
        "messages": [
            {"role": "user", "content": "check the status of order 98765 for me"},
            {"role": "assistant",
             "content": "I cannot look up orders. For billing, see the front desk."},
        ],
    }
    assert is_do_nothing(row) is True            # tool agent: a real drop
    assert is_do_nothing(row, has_tools=False) is False
    # Judge-labeled rows bypass the heuristic entirely: a 1 is the
    # judge's call, whatever the grid expected.
    picked, report = select_for_sft([row], target=5)
    assert len(picked) == 1
    from zeroproof_simulations.optimize import drop_reason
    unlabeled = dict(row)
    unlabeled.pop("reward")
    assert drop_reason(unlabeled) == "do_nothing"
    assert drop_reason(unlabeled, has_tools=False) != "do_nothing"
