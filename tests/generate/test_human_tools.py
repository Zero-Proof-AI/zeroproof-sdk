"""Human-kind tools: their result is the person's answer, never a mock."""
import json

from zeroproof_simulations.generate import agents as zagents


def test_human_tool_answer_comes_from_the_user_voice(monkeypatch):
    calls = []

    def fake(base_url, model, messages, **kw):
        calls.append(messages)
        if len(calls) == 1:
            return {"content": "", "tool_calls": [{
                "id": "c1", "function": {
                    "name": "ask_user",
                    "arguments": json.dumps(
                        {"question": "Delete branch old-x for real?"})}}]}
        if len(calls) == 2:
            # the user simulator answering, in character
            return {"content": "yes, delete it, we shipped that", "tool_calls": []}
        return {"content": "Done. Deleted old-x.", "tool_calls": []}

    monkeypatch.setattr(zagents, "complete", fake)
    runner = zagents.local_model(
        "http://x/v1", "m",
        tools=[{"type": "function", "kind": "human", "function": {
            "name": "ask_user", "description": "ask the human",
            "parameters": {"type": "object", "properties": {
                "question": {"type": "string"}}, "required": ["question"]}}}],
        system="Confirm before destructive actions.", max_turns=3)
    row = runner("delete the old-x branch")
    ask_steps = [s for s in row["steps"] if s.get("tool") == "ask_user"]
    assert ask_steps, "ask_user step missing"
    assert ask_steps[0]["result"] == {
        "answer": "yes, delete it, we shipped that"}
    # the answering call was voiced as the USER, not the assistant
    assert any(m.get("role") == "system" and "USER" in str(m.get("content"))
               for m in calls[1])
