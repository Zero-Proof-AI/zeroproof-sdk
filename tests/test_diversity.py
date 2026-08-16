import random

from tests.helpers import TOOLS, POLICY, scripted_agent
import zeroproof_simulations as zps
from zeroproof_simulations.diversity import accept_anneal_candidate, sample_request_axes
from zeroproof_simulations.generator import ModelSimulator


def test_rollouts_per_prompt_same_prompt_two_rows():
    data = zps.simulate(
        scripted_agent, tools=TOOLS, policy=POLICY, budget=4, seed=0,
        rollouts_per_prompt=2, grade=False, concurrency=4, simulator=False,
        advanced={"per_round": 12, "mutate_failures": False})
    assert len(data.trajectories) == 4
    prompts = [t["prompt"] for t in data.trajectories]
    assert len(set(prompts)) == 2
    for prompt in set(prompts):
        rows = [t for t in data.trajectories if t["prompt"] == prompt]
        assert len(rows) == 2
        assert {t["rollout_index"] for t in rows} == {0, 1}


def test_anneal_accepts_non_greedy_when_hot():
    rng = random.Random(0)
    hot = sum(accept_anneal_candidate(0.05, temperature=1.0, rng=rng) for _ in range(200))
    cold = sum(accept_anneal_candidate(0.05, temperature=0.05, rng=rng) for _ in range(200))
    assert hot > cold
    assert hot > 20


def test_writer_tags_are_sparse_and_generic():
    from pathlib import Path
    from zeroproof_simulations.diversity import sample_cell_tags

    sim = ModelSimulator(tools=TOOLS, policy=POLICY, seed=1, candidates_per_round=40)
    prompt = sim._prompt(0, sim.regions[:8])
    assert "request_axes" not in prompt
    assert '"voice"' not in prompt

    allowed = {"tool", "rule", "length", "vagueness", "stance", "phrasing",
               "texture", "tone", "ask", "pressure", "user",
               "history", "world_state", "tool_condition"}
    lengths: list[str] = []
    tool_rule_only = weird = 0
    for i in range(80):
        tags = sample_cell_tags(1, 0, f"cell-{i}",
                                {"tool": "dim_lights", "rule": "Ask first."})
        if "length" in tags:
            lengths.append(str(tags["length"]))
        assert set(tags) <= allowed
        if set(tags) - {"texture", "tone", "length"} <= {"tool", "rule"}:
            tool_rule_only += 1
        if "phrasing" in tags:
            weird += 1
        if "ask" in tags:
            assert tags["ask"] in {"question", "ask", "several asks",
                                   "do several things"}
        assert "demographic" not in tags
    assert len(lengths) >= 40
    assert set(lengths) <= {"short prompt", "medium prompt", "long prompt"}
    assert lengths.count("medium prompt") > lengths.count("long prompt")
    assert lengths.count("medium prompt") > lengths.count("short prompt")
    assert {"short prompt", "medium prompt", "long prompt"} <= set(
        sample_cell_tags(1, 0, f"len-{i}", {"length": hint})["length"]
        for i, hint in enumerate(("short", "medium", "long")))
    assert weird <= 20
    assert tool_rule_only >= 20
    stances = [sample_cell_tags(1, 0, f"tier-{i}", {"stance": "ordinary"})
               for i in range(20)]
    assert all(tags.get("stance") != "ordinary" for tags in stances)
    assert any(tags.get("stance") == "adversarial" for tags in (
        sample_cell_tags(1, 0, f"hard-{i}", {"stance": "adversarial"})
        for i in range(40)))
    asks = [sample_cell_tags(1, 0, f"ask-{i}", {"tool": "dim_lights"})
            for i in range(80)]
    ask_vals = [str(t["ask"]) for t in asks if t.get("ask")]
    assert ask_vals
    assert ask_vals.count("question") + ask_vals.count("ask") > sum(
        1 for v in ask_vals if v in {"several asks", "do several things"})
    assert all("many tools" not in str(v) and "tool-heavy" not in str(v)
               for tags in asks for v in list(tags) + list(tags.values()))
    axes = sample_request_axes(1, 0, "k")
    assert "demographic" not in axes
    assert "opening" not in axes

    sleep_tools = [{"type": "function", "function": {
        "name": "start_sleep",
        "description": "Begin a sleep session.",
        "parameters": {"type": "object", "properties": {"minutes": {"type": "number"}}},
    }}]
    sleep_policy = "Only start sleep if the user asked. Do not invent a duration."
    sleep_sim = ModelSimulator(tools=sleep_tools, policy=sleep_policy, seed=3)
    sleep_prompt = sleep_sim._prompt(0, sleep_sim.regions[:6])
    assert "Only start sleep" not in sleep_prompt
    assert '"parameters"' not in sleep_prompt
    assert "you know this assistant can" in sleep_prompt.lower()
    assert "Begin a sleep session" in sleep_prompt
    assert "minutes" in sleep_prompt
    assert "looking to use" not in sleep_prompt
    assert "start_sleep" not in sleep_prompt
    tags = sample_cell_tags(3, 0, sleep_sim.regions[0]["id"],
                            sleep_sim.regions[0]["assignment"])
    assert set(tags) <= {"tool", "rule", "length", "vagueness", "stance",
                         "phrasing", "texture", "tone", "ask", "pressure",
                         "user", "history", "world_state",
                         "tool_condition"}
    sdk = Path(__file__).resolve().parents[1] / "zeroproof_simulations"
    for path in sdk.glob("*.py"):
        src = path.read_text()
        assert "_KIND_ASKS" not in src
        assert "_SHOP_MARKERS" not in src
        assert "_CODING_MARKERS" not in src
        assert "def agent_kind" not in src
        for line in src.splitlines():
            if "time.sleep" in line:
                continue
            assert "insomnia" not in line.lower()
            assert "nap time" not in line.lower()


def test_scenario_family_cap_catches_paraphrases_not_unrelated_subjects():
    from zeroproof_simulations.diversity import cap_scenario_families

    rows = [{"text": text} for text in (
        "Find a ceramic mug below twenty dollars",
        "Can you find a blue mug that is inexpensive?",
        "I'm searching for a rustic mug for the office",
        "Find noise-cancelling headphones for commuting",
    )]
    kept, rejected = cap_scenario_families(rows, [], cap=2)
    assert len(kept) == 3
    assert rejected == [rows[2]]


def test_pressure_is_a_sparse_tag_not_hardcoded_english():
    from zeroproof_simulations.diversity import sample_cell_tags
    from zeroproof_simulations.explore import MUTATORS
    from zeroproof_simulations.scenarios import (STANCE_BRIEFS, STANCES,
                                                 scenario_regions)

    assert "pressurize" not in {name for name, _ in MUTATORS}
    tagged = 0
    for i in range(80):
        tags = sample_cell_tags(1, 0, f"p-{i}", {"tool": "lookup_order"})
        if "pressure" in tags:
            tagged += 1
            assert tags["pressure"] in {"rushed", "insistent", "repeat"}
        assert "I already confirmed" not in str(tags)
    assert 1 <= tagged <= 20
    assert "hurried" in STANCES
    assert "unsure" in STANCES
    assert STANCE_BRIEFS["hurried"]
    assert "Hi, I need" not in STANCE_BRIEFS["hurried"]
    regions = scenario_regions(TOOLS, POLICY)
    conds = [r["assignment"].get("tool_condition", "success") for r in regions]
    assert conds.count("success") >= int(0.8 * len(conds))
    assert {"timeout", "permission_denied"} & set(conds)


def test_conversation_features_use_live_tiers():
    from zeroproof_simulations.diversity import conversation_features

    ordinary = conversation_features({"stance": "ordinary"}, {},
                                     ask_family="tool", tool="get_pr")
    assert ordinary["tier"] == "ordinary"
    assert ordinary["ask_family"] == "tool"
    assert ordinary["intent_known"] is True
    assert ordinary["tool_known"] is True
    assert "stance" not in ordinary
    vague = conversation_features({"stance": "adversarial"}, {"vagueness": "vague"},
                                  ask_family="vague", tool="unrelated")
    assert vague["tier"] == "adversarial"
    assert vague["intent_known"] is False
    assert vague["tool_known"] is False
    assert vague["vagueness"] == "vague"
    empty = conversation_features({}, {})
    assert empty == {"tier": "ordinary"}


def test_same_scenario_id_different_prompts_allowed():
    data = zps.simulate(
        scripted_agent, tools=TOOLS, policy=POLICY, budget=60, seed=2,
        grade=False, concurrency=8, simulator=False,
        requests_per_situation=3, rollouts_per_request=1,
        advanced={"per_round": 20, "mutate_failures": False})
    by_sid: dict[str, set[str]] = {}
    for t in data.trajectories:
        sid = t.get("scenario_id") or ""
        by_sid.setdefault(sid, set()).add(t["prompt"])
    multi = [sid for sid, ps in by_sid.items() if len(ps) > 1]
    assert multi, "expected same scenario_id with different wording"
