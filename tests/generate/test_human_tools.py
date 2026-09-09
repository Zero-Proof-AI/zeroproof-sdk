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
    # the answering call put the model in the person's seat, not the assistant's
    assert any(m.get("role") == "system"
               and "person the assistant is helping" in str(m.get("content"))
               for m in calls[1])


def test_human_answer_carries_the_situation_stance(monkeypatch):
    calls = []

    def fake(base_url, model, messages, **kw):
        calls.append(messages)
        if len(calls) == 1:
            return {"content": "", "tool_calls": [{
                "id": "c1", "function": {
                    "name": "ask_user",
                    "arguments": json.dumps({"question": "Delete old-x?"})}}]}
        if len(calls) == 2:
            return {"content": "wait, I meant old-y not old-x", "tool_calls": []}
        return {"content": "Deleted old-y.", "tool_calls": []}

    monkeypatch.setattr(zagents, "complete", fake)
    ask = "delete the old-x branch"
    runner = zagents.local_model(
        "http://x/v1", "m",
        tools=[{"type": "function", "kind": "human", "function": {
            "name": "ask_user", "description": "ask the human",
            "parameters": {"type": "object", "properties": {
                "question": {"type": "string"}}}}}],
        system="Confirm before destructive actions.", max_turns=3,
        fault_plans={ask: {"stance": "mistaken"}})
    row = runner(ask)
    # the stance reached the person's system prompt and never the mock world
    assert any(m.get("role") == "system" and "wrong about a detail" in str(m.get("content"))
               for m in calls[1])
    assert row["steps"][0]["result"] == {"answer": "wait, I meant old-y not old-x"}


def test_followup_writer_keeps_the_situation_tone_and_texture(monkeypatch):
    calls = []

    def fake(base_url, model, messages, **kw):
        calls.append(messages)
        if len(calls) == 1:
            return {"content": "Which branch do you mean, old-x or old-y?", "tool_calls": []}
        if len(calls) == 2:
            return {"content": "old-y obviously", "tool_calls": []}
        return {"content": "Deleted old-y.", "tool_calls": []}

    monkeypatch.setattr(zagents, "complete", fake)
    ask = "delete the old branch"
    runner = zagents.local_model(
        "http://x/v1", "m", tools=[], system="Be careful.", max_turns=4,
        fault_plans={ask: {"tone": "sarcastic", "texture": "lowercase"}})
    runner(ask)
    followup_prompt = json.dumps(calls[1])
    assert "sarcastic" in followup_prompt and "every letter small" in followup_prompt
