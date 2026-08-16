"""Model-written result shapes keep the sandbox agent-agnostic."""
import json

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
