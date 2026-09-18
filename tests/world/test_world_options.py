"""Every dial of the mock world is a named default a caller can turn.

``WorldOptions`` carries the numbers (defaults.py says why each is what it
is); ``MockEnvironment(options=)``, ``simulate(advanced={"world": ...})``
and ``export_environment(world=)`` are the ways in. Each test here turns
one dial and checks the world moved with it.
"""

from __future__ import annotations

import pytest

from tests.helpers import TOOLS
from whileai.simulations import defaults
from whileai.simulations.run.config import resolve_run_config
from whileai.simulations.world.sandbox import (
    DEFAULT_WORLD,
    FAULT_MODES,
    PAYLOAD_BUILDERS,
    RESULT_KINDS,
    MockEnvironment,
    WorldOptions,
    _fill_template,
    _result_kind,
)


def _tool(name, props, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name.replace("_", " "),
            "parameters": {"type": "object", "properties": props, "required": list(required)},
        },
    }


SHOP = [
    _tool("search_products", {"query": {"type": "string"}}, ["query"]),
    _tool("get_order", {"order_id": {"type": "string"}}, ["order_id"]),
    _tool("list_team_members", {"team": {"type": "string"}}),
    _tool("query_db", {"sql": {"type": "string"}}, ["sql"]),
]


def test_defaults_are_the_named_numbers_in_one_home():
    o = DEFAULT_WORLD
    assert o.search_hits == defaults.WORLD_SEARCH_HITS == (1, 6)
    assert o.exists_share == defaults.WORLD_EXISTS_SHARE == 0.7
    assert o.default_fault_mode == defaults.WORLD_DEFAULT_FAULT_MODE == "timeout"
    assert o.default_fault_rate == defaults.WORLD_DEFAULT_FAULT_RATE == 1.0
    assert o.jitter_divisor == defaults.WORLD_JITTER_DIVISOR == 3
    assert o.id_range == defaults.WORLD_ID_RANGE and o.date_years == defaults.WORLD_DATE_YEARS
    assert (
        set(o.fault_modes)
        == set(FAULT_MODES)
        == {
            "timeout",
            "malformed",
            "stale",
            "permission_denied",
        }
    )
    assert o.result_kinds is RESULT_KINDS and set(o.payloads) == set(PAYLOAD_BUILDERS)
    assert WorldOptions.coerce(None) is DEFAULT_WORLD
    assert WorldOptions.coerce(o) is o


def test_a_typo_in_the_options_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError, match="search_hit"):
        WorldOptions.coerce({"search_hit": (2, 2)})
    with pytest.raises(ValueError, match="exists_share"):
        WorldOptions(exists_share=1.5)
    with pytest.raises(ValueError, match="default_fault_mode"):
        WorldOptions(default_fault_mode="rate_limited")
    with pytest.raises(TypeError):
        WorldOptions.coerce("stale")  # type: ignore[arg-type]


def test_search_hits_bound_the_listing_size():
    for seed in range(12):
        env = MockEnvironment(SHOP, seed=seed, options={"search_hits": (3, 3)})
        out = env.call("search_products", {"query": "vase"})
        assert out["data"]["count"] == 3 and len(out["data"]["items"]) == 3
    counts = {
        MockEnvironment(SHOP, seed=s).call("search_products", {"query": "vase"})["data"]["count"]
        for s in range(40)
    }
    assert counts <= set(range(1, 7)) and len(counts) > 1, "the default world still varies"


def test_exists_share_decides_whether_a_named_record_is_found():
    always = [
        MockEnvironment(SHOP, seed=s, options={"exists_share": 1.0}).call(
            "get_order", {"order_id": f"ORD-{s}"}
        )["status"]
        for s in range(30)
    ]
    never = [
        MockEnvironment(SHOP, seed=s, options={"exists_share": 0.0}).call(
            "get_order", {"order_id": f"ORD-{s}"}
        )["status"]
        for s in range(30)
    ]
    assert set(always) == {"ok"} and set(never) == {"not_found"}


def test_default_fault_mode_and_rate_fill_a_plan_that_names_neither():
    plan = {"*": {"rate": 1.0}}  # a plan that names no mode
    env = MockEnvironment(SHOP, seed=1, faults=plan, options={"default_fault_mode": "stale"})
    out = env.call("get_order", {"order_id": "4412"})
    assert out.get("stale") is True and out["as_of"] == defaults.WORLD_STALE_AS_OF
    assert MockEnvironment(SHOP, seed=1, faults=plan).call("get_order", {"order_id": "4412"}) == {
        "status": "timeout",
        "error": "request timed out",
    }
    plan = {"*": {"mode": "stale"}}  # a plan that names no rate
    quiet = MockEnvironment(SHOP, seed=1, faults=plan, options={"default_fault_rate": 0.0})
    assert "stale" not in quiet.call("get_order", {"order_id": "4412"})
    assert MockEnvironment(SHOP, seed=1, faults=plan).call("get_order", {"order_id": "4412"})[
        "stale"
    ]
    aged = MockEnvironment(
        SHOP, seed=1, faults={"*": {"mode": "stale"}}, options={"stale_as_of": "2 weeks ago"}
    )
    assert aged.call("get_order", {"order_id": "4412"})["as_of"] == "2 weeks ago"


def test_a_caller_can_add_a_fault_mode():
    def rate_limited(env, tool, arguments):
        return {"status": "error", "error": "429 too many requests", "retry_after_s": 30}

    modes = {**FAULT_MODES, "rate_limited": rate_limited}
    env = MockEnvironment(
        SHOP, seed=3, faults={"*": {"mode": "rate_limited"}}, options={"fault_modes": modes}
    )
    out = env.call("get_order", {"order_id": "77"})
    assert out["error"].startswith("429") and out["retry_after_s"] == 30
    # the same plan against the shipped modes is not a fault at all
    assert "error" not in MockEnvironment(
        SHOP, seed=3, faults={"*": {"mode": "rate_limited"}}
    ).call("get_order", {"order_id": "77"})


def test_the_routing_table_and_payload_builders_are_extendable():
    def is_sql(name, keys, blob):
        return "sql" in keys

    def sql_payload(tool, arguments, n, digest, options):
        return {"rows": [{"n": n % 3}], "sql": arguments.get("sql")}

    assert _result_kind("query_db", SHOP[3]["function"], {"sql": "select 1"}) == "record"
    options = {
        "result_kinds": (("sql", is_sql), *RESULT_KINDS),
        "payloads": {**PAYLOAD_BUILDERS, "sql": sql_payload},
    }
    assert (
        _result_kind(
            "query_db", SHOP[3]["function"], {"sql": "select 1"}, WorldOptions.coerce(options)
        )
        == "sql"
    )
    env = MockEnvironment(SHOP, seed=0, options=options)
    out = env.call("query_db", {"sql": "select 1"})
    assert out["status"] == "ok" and out["data"]["sql"] == "select 1" and "rows" in out["data"]


def test_name_pools_dates_and_ids_come_from_the_options():
    env = MockEnvironment(
        SHOP,
        seed=5,
        options={
            "people": ("ada lovelace",),
            "date_years": (1999, 1),
            "id_range": (10, 11),
            "statuses": ("frozen",),
        },
    )
    people = env.call("list_team_members", {"team": "backend"})["data"]
    items = people.get("items") or [people]
    assert all(it["owner"] == "ada lovelace" for it in items)
    assert all(it["updated_at"].startswith("1999-") for it in items)
    assert all(it["status"] == "frozen" for it in items)
    assert all(it["id"] == 10 for it in items)


def test_jitter_divisor_sets_how_far_a_template_number_moves():
    shape = {"total": 900, "price": 90.0}
    wide = {_fill_template(shape, k)["total"] for k in range(0, 3000, 7)}
    narrow = {
        _fill_template(shape, k, options=WorldOptions(jitter_divisor=900))["total"]
        for k in range(0, 3000, 7)
    }
    assert min(wide) < 700 and max(wide) > 1100, "default: up to a third either way"
    assert narrow <= {899, 900, 901}, "divisor 900: one unit either way"


def test_advanced_world_lands_on_the_run_config_validated():
    cfg = resolve_run_config(
        tools=TOOLS, system_prompt="p", advanced={"world": {"search_hits": (2, 2)}}
    )
    assert isinstance(cfg.world_options, WorldOptions)
    assert cfg.world_options.search_hits == (2, 2)
    assert "world" not in cfg.advanced, "consumed here, never handed to the writer"
    assert resolve_run_config(tools=TOOLS, system_prompt="p").world_options is None
    with pytest.raises(ValueError, match="exists_shar"):
        resolve_run_config(tools=TOOLS, system_prompt="p", advanced={"world": {"exists_shar": 1}})
