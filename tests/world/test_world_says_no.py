"""The mock world refuses schema echoes and answers in the tool's own domain."""
from zeroproof_simulations.world.sandbox import MockEnvironment, placeholder_arguments
from zeroproof_simulations.score.grading import conduct_grade


def _tool(name, props, required=()):
    return {"type": "function", "function": {
        "name": name, "description": name.replace("_", " "),
        "parameters": {"type": "object", "properties": props, "required": list(required)}}}


TOOLS = [
    _tool("find_user_id_by_email", {"email": {"type": "string"}}, ["email"]),
    _tool("check_inventory", {"item": {"type": "string"}}, ["item"]),
    _tool("get_price", {"product": {"type": "string"}}, ["product"]),
    _tool("list_team_members", {"team": {"type": "string"}}),
]


def test_placeholder_arguments_are_rejected_not_answered():
    env = MockEnvironment(TOOLS, seed=3)
    out = env.call("find_user_id_by_email", {"email": "user@example.com"})
    assert out["status"] == "rejected" and out["reason"] == "placeholder_argument"
    assert out["fields"] == ["email"]
    out = env.call("find_user_id_by_email", {"email": "first name"})
    assert out["status"] == "rejected"
    real = env.call("find_user_id_by_email", {"email": "mei.kovacs@gmail.com"})
    assert real["status"] in {"ok", "not_found"}


def test_placeholder_detector_shapes():
    assert placeholder_arguments({"zip": "zip code", "name": "Mei Kovacs"}) == ["zip"]
    assert placeholder_arguments({"a": {"b": ["<id>", "real-42"]}}) == ["a.b[0]"]
    assert placeholder_arguments({"query": "shipping delay"}) == []


def test_inventory_records_carry_a_quantity_and_no_owner():
    env = MockEnvironment(TOOLS, seed=7)
    out = env.call("check_inventory", {"item": "ceramic vases"})
    assert out["status"] == "ok"
    data = out["data"]
    assert isinstance(data.get("quantity"), int)
    assert "owner" not in data
    priced = env.call("get_price", {"product": "ceramic vases"})
    price = priced.get("data") or priced
    assert isinstance(price.get("amount"), (int, float)) and price.get("currency") == "USD"
    people = env.call("list_team_members", {"team": "backend"})["data"]
    items = people.get("items") or [people]
    assert all("owner" in it for it in items)


def test_conduct_grade_fails_a_placeholder_call():
    row = {"prompt": "find my account, my email is mei.kovacs@gmail.com",
           "steps": [{"tool": "find_user_id_by_email", "arguments": {"email": "user@example.com"},
                      "result": {"status": "rejected", "reason": "placeholder_argument", "fields": ["email"]}}],
           "final_text": "I could not find your account with that email."}
    verdict = conduct_grade(row, {"find_user_id_by_email"})
    assert verdict["reward"] == 0.0 and "placeholder" in verdict["reason"]

