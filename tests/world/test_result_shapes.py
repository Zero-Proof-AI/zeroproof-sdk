"""Model-written result shapes keep the sandbox agent-agnostic."""
import json

from tests.helpers import GITHUB_SPEC
from zeroproof_simulations.sandbox import MockEnvironment, _fill_template


TOOLS = [
    {"type": "function", "function": {
        "name": "get_directions",
        "description": "Get travel time and directions.",
        "parameters": {"type": "object", "properties": {
            "destination": {"type": "string"}},
            "required": ["destination"]}}},
    {"type": "function", "function": {
        "name": "search_email",
        "description": "Search mail.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
]

DIRECTIONS_SHAPE = {
    "duration_minutes": 37,
    "distance_miles": 12.4,
    "traffic": "heavy",
    "route": ["Turn right onto Oak St", "Merge onto I-80"],
    "arrival_time": "2026-05-01T09:40",
}
EMAIL_SHAPE = {
    "results": [{
        "subject": "Quarterly report draft",
        "from": "sam@acme.com",
        "date": "2026-04-12",
        "snippet": "attaching the latest numbers",
    }],
}


def test_fill_template_varies_only_what_should_vary():
    a = _fill_template(DIRECTIONS_SHAPE, 12345)
    b = _fill_template(DIRECTIONS_SHAPE, 99999)
    assert a["traffic"] == "heavy"
    assert a["route"] == DIRECTIONS_SHAPE["route"]
    assert isinstance(a["duration_minutes"], int) and a["duration_minutes"] > 0
    assert a["duration_minutes"] != b["duration_minutes"]
    assert _fill_template(DIRECTIONS_SHAPE, 12345) == a


def test_fill_template_expands_listings_and_rewrites_people():
    filled = _fill_template(EMAIL_SHAPE, 777)
    items = filled["results"]
    assert 2 <= len(items) <= 3
    assert all(i["subject"] == "Quarterly report draft" for i in items)
    froms = {i["from"] for i in items}
    assert all("@acme.com" in f for f in froms)
    assert len(froms) > 1


def test_environment_prefers_model_shape_over_generic_record():
    env = MockEnvironment(TOOLS, result_shapes={
        "get_directions": DIRECTIONS_SHAPE, "search_email": EMAIL_SHAPE})
    out = env.call("get_directions", {"destination": "airport"})
    assert out["status"] == "ok"
    assert "duration_minutes" in out["data"]
    assert "route" in out["data"]
    other = env.call("get_directions", {"destination": "office"})
    assert other["data"]["duration_minutes"] != out["data"]["duration_minutes"]
    mail = env.call("search_email", {"query": "quarterly report"})
    assert mail["data"]["results"][0]["subject"]
    # No shape -> generic record fallback still works.
    bare = MockEnvironment(TOOLS)
    fallback = bare.call("search_email", {"query": "quarterly report"})
    assert fallback["status"] in {"ok", "not_found"}


def test_search_item_and_read_describe_same_entity():
    tools = [
        {"type": "function", "function": {
            "name": "search_orders", "description": "find orders",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "get_order", "description": "read one order",
            "parameters": {"type": "object", "properties": {
                "order_id": {"type": "string"}},
                "required": ["order_id"]}}},
    ]
    env = MockEnvironment(tools)
    env._exists = lambda value: True
    listing = env.call("search_orders", {"query": "recent"})
    item = listing["data"]["items"][0]
    read = env.call("get_order", {"order_id": str(item["id"])})
    record = read["data"]
    assert record["name"] == item["name"]
    assert record["owner"] == item["owner"]
    assert record["updated_at"] == item["updated_at"]
    again = env.call("get_order", {"order_id": str(item["id"]),
                                   "extra": "field"})
    assert again["data"]["name"] == item["name"]


def test_list_valued_shape_normalized(monkeypatch):
    from zeroproof_simulations import generator

    payload = {"search_email": [
        {"message_id": "8891", "subject": "Kickoff", "from": "a@b.com"}]}

    def fake_complete(_url, _model, _messages, **kwargs):
        return {"content": json.dumps(payload)}

    monkeypatch.setattr(generator, "complete", fake_complete)
    shapes = generator.write_result_shapes(
        TOOLS, backend_spec="vllm:fake@http://example")
    assert "results" in shapes["search_email"]


def test_listing_items_are_distinct_entities_despite_parent_ref():
    tools = [{"type": "function", "function": {
        "name": "list_deadlines", "description": "docket deadlines",
        "parameters": {"type": "object", "properties": {
            "matter_id": {"type": "string"}},
            "required": ["matter_id"]}}}]
    env = MockEnvironment(tools)
    env._exists = lambda value: True
    out = env.call("list_deadlines", {"matter_id": "78902"})
    items = out["data"]["items"]
    assert len(items) >= 2
    ids = [i["id"] for i in items]
    assert len(set(ids)) == len(ids)
    assert len({json.dumps(i, sort_keys=True) for i in items}) == len(items)
    assert all(i.get("matter_id") == "78902" for i in items)


def test_already_done_world_still_answers_reads():
    tools = [
        {"type": "function", "function": {
            "name": "get_order", "description": "read one order",
            "parameters": {"type": "object", "properties": {
                "order_id": {"type": "string"}},
                "required": ["order_id"]}}},
        {"type": "function", "function": {
            "name": "cancel_order", "description": "cancel an order",
            "parameters": {"type": "object", "properties": {
                "order_id": {"type": "string"}},
                "required": ["order_id"]}}},
    ]
    env = MockEnvironment(tools, world_state="entity already acted on")
    env._exists = lambda value: True
    read = env.call("get_order", {"order_id": "9917"})
    assert read["status"] == "ok"
    assert isinstance(read.get("data"), dict)
    act = env.call("cancel_order", {"order_id": "9917"})
    assert act["status"] == "already_done"


def test_write_result_shapes_parses_fenced_and_filters_unknown(monkeypatch):
    from zeroproof_simulations import generator

    payload = {"get_directions": DIRECTIONS_SHAPE,
               "search_email": EMAIL_SHAPE,
               "not_a_tool": {"x": 1}}

    def fake_complete(_url, _model, _messages, **kwargs):
        return {"content": "```json\n" + json.dumps(payload) + "\n```"}

    monkeypatch.setattr(generator, "complete", fake_complete)
    shapes = generator.write_result_shapes(
        TOOLS, backend_spec="vllm:fake@http://example")
    assert set(shapes) == {"get_directions", "search_email"}
    assert shapes["get_directions"]["duration_minutes"] == 37


def _named_tools(n: int) -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": f"tool_{i}",
            "description": f"Do thing {i}.",
            "parameters": {"type": "object", "properties": {
                "q": {"type": "string"}}, "required": ["q"]}}}
        for i in range(n)
    ]


def test_write_result_shapes_one_call_for_small_lists(monkeypatch):
    from zeroproof_simulations import generator

    calls: list[list] = []

    def fake_complete(_url, _model, messages, **kwargs):
        digest = json.loads(messages[-1]["content"])
        calls.append([item["name"] for item in digest])
        return {"content": json.dumps(
            {item["name"]: {"ok": True, "id": item["name"]} for item in digest})}

    monkeypatch.setattr(generator, "complete", fake_complete)
    shapes = generator.write_result_shapes(
        _named_tools(8), backend_spec="vllm:fake@http://example")
    assert len(calls) == 1
    assert len(shapes) == 8
    assert set(shapes) == {f"tool_{i}" for i in range(8)}


def test_write_result_shapes_chunks_and_merges_large_lists(monkeypatch):
    from zeroproof_simulations import generator

    n_tools = 36
    calls: list[list] = []

    def fake_complete(_url, _model, messages, **kwargs):
        digest = json.loads(messages[-1]["content"])
        names = [item["name"] for item in digest]
        calls.append(names)
        return {"content": json.dumps(
            {name: {"ok": True, "id": name} for name in names})}

    monkeypatch.setattr(generator, "complete", fake_complete)
    shapes = generator.write_result_shapes(
        _named_tools(n_tools), backend_spec="vllm:fake@http://example")
    assert 2 <= len(calls) <= 3
    seen = [name for batch in calls for name in batch]
    assert seen == [f"tool_{i}" for i in range(n_tools)]
    assert len(shapes) == n_tools
    assert set(shapes) == {f"tool_{i}" for i in range(n_tools)}
    assert all(shapes[name]["id"] == name for name in shapes)


CODING_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from the workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search files for a pattern.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"]}}},
]


def test_coding_tools_are_not_person_records():
    env = MockEnvironment(CODING_TOOLS)
    env._exists = lambda value: True
    file_out = env.call("read_file", {"path": "src/app.py"})
    data = file_out["data"]
    assert file_out["status"] == "ok"
    assert isinstance(data.get("content"), str)
    assert "\n" in data["content"]
    assert "def " in data["content"] or "from " in data["content"]
    assert data.get("owner") is None
    assert data.get("status") not in {"completed", "in_progress", "pending"}
    shell = env.call("run_command", {"command": "pytest -q"})
    payload = shell.get("data") or shell
    assert "stdout" in payload or "exit_code" in payload
    assert payload.get("owner") is None
    assert "exit_code" in payload
    grep = env.call("grep", {"pattern": "TODO", "path": "src"})
    matches = (grep.get("data") or grep).get("matches")
    assert matches
    assert "path" in matches[0] and "text" in matches[0]
    assert matches[0].get("owner") is None


def test_order_tools_still_get_records():
    tools = [
        {"type": "function", "function": {
            "name": "search_orders", "description": "find orders",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}}, "required": ["query"]}}},
    ]
    env = MockEnvironment(tools)
    env._exists = lambda value: True
    listing = env.call("search_orders", {"query": "recent"})
    item = listing["data"]["items"][0]
    assert "owner" in item
    assert "status" in item
    assert "name" in item


def test_unknown_spec_still_gets_payloads():
    tools = [
        {"type": "function", "function": {
            "name": "frobnicate_gadget",
            "description": "Twiddle a gadget.",
            "parameters": {"type": "object", "properties": {
                "gadget_id": {"type": "string"}},
                "required": ["gadget_id"]}}},
        {"type": "function", "function": {
            "name": "read_blueprint",
            "description": "Read a blueprint file.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}}, "required": ["path"]}}},
    ]
    env = MockEnvironment(tools)
    env._exists = lambda value: True
    rec = env.call("frobnicate_gadget", {"gadget_id": "G-1"})
    assert rec.get("status") in {"ok", "created"}
    assert rec.get("data") or rec.get("id")
    fil = env.call("read_blueprint", {"path": "docs/plan.md"})
    content = (fil.get("data") or {}).get("content")
    assert isinstance(content, str) and len(content) > 20
    assert "#" in content or "plan" in content.lower() or "\n" in content


def test_github_get_file_and_commits_are_code_shaped():
    from pathlib import Path

    spec_path = GITHUB_SPEC / "spec.json"
    spec = json.loads(spec_path.read_text())
    env = MockEnvironment(spec["tools"])
    env._exists = lambda value: True
    file_out = env.call("get_file", {
        "repo": "acme/app", "path": "src/main.py", "ref": "main"})
    data = file_out["data"]
    assert isinstance(data.get("content"), str)
    assert "\n" in data["content"]
    assert data.get("owner") is None
    commits = env.call("list_commits", {"repo": "acme/app"})
    item = commits["data"]["items"][0]
    assert "sha" in item and "message" in item
    checks = env.call("list_checks", {"repo": "acme/app", "number": 12})
    job = checks["data"]["items"][0]
    assert "name" in job and "conclusion" in job


def test_run_tests_is_shell_not_a_person_record():
    tools = [
        {"type": "function", "function": {
            "name": "run_tests",
            "description": "Run the project test suite.",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string"}}}}},
    ]
    env = MockEnvironment(tools)
    env._exists = lambda value: True
    out = env.call("run_tests", {"command": "pytest"})
    data = out.get("data") or {}
    assert "owner" not in data
    assert "updated_at" not in data
    assert "stdout" in data or "exit_code" in data


def test_record_shaped_example_does_not_override_run_tests():
    tools = [
        {"type": "function", "function": {
            "name": "run_tests",
            "description": "Run the project test suite.",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string"}}}}},
    ]
    env = MockEnvironment(tools, result_shapes={
        "run_tests": {
            "id": "1", "name": "manual test", "owner": "nina brooks",
            "status": "completed", "updated_at": "2026-04-12",
        },
    })
    env._exists = lambda value: True
    data = env.call("run_tests", {"command": "pytest"}).get("data") or {}
    assert "owner" not in data
    assert "stdout" in data or "exit_code" in data


def test_search_follows_query_not_frozen_example():
    tools = [{"type": "function", "function": {
        "name": "search_products",
        "description": "Search the catalog.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}}]
    frozen = {"products": [{
        "asin": "B08X9QY8WZ",
        "title": "Wireless Bluetooth Headphones with Noise Cancellation",
        "price": 79.99,
        "description": "Premium noise-cancelling headphones.",
        "images": ["https://example.com/images/headphones-1.jpg"],
    }]}
    env = MockEnvironment(
        tools, world_state="entity exists",
        result_shapes={"search_products": frozen})
    towels = env.call("search_products", {"query": "towels"})
    blob = json.dumps(towels).lower()
    assert towels["status"] == "ok"
    assert "headphone" not in blob
    assert "towel" in blob
    jackets = env.call("search_products", {"query": "jackets"})
    other = json.dumps(jackets).lower()
    assert "jacket" in other
    assert "towel" not in other
    missing = MockEnvironment(
        tools, world_state="entity missing",
        result_shapes={"search_products": frozen})
    out = missing.call("search_products", {"query": "towels"})
    assert out["status"] == "not_found"


def test_calculator_tools_return_real_arithmetic():
    from zeroproof_simulations.sandbox import MockEnvironment
    tools = [{"type": "function", "function": {"name": "calculate", "parameters": {
        "type": "object", "properties": {"expression": {"type": "string"}},
        "required": ["expression"]}}}]
    env = MockEnvironment(tools)
    out = env.call("calculate", {"expression": "345 * 678 - 200"})
    assert out["status"] == "ok"
    assert out["data"]["result"] == 233710
    out = env.call("calculate", {"expression": "1500 * 0.87 + 234 - 100"})
    assert abs(out["data"]["result"] - 1439) < 0.01
    bad = env.call("calculate", {"expression": "__import__('os')"})
    assert bad["status"] == "rejected"
