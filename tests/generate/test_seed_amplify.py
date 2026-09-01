"""simulate-from-seeds: a few example asks + situations=N make the
engine mint the rest of the situation space itself. Five seeds must be
enough; originals always survive; disclosure is recorded."""
import zeroproof_simulations.generator as gen
from zeroproof_simulations.generator import amplify_seeds

SEEDS = ["who are you?", "what's your name?", "who made you?",
         "introduce yourself", "which company built you?"]


def test_amplifies_to_target_with_dedup(monkeypatch):
    calls = []

    def fake_complete(url, model, messages, **kw):
        calls.append(messages[0]["content"])
        batch = [f"tell me about yourself v{len(calls)}-{i}"
                 for i in range(25)]
        batch.append("who are you?")          # dup of an original
        return {"content": "\n".join(batch)}

    monkeypatch.setattr(gen, "complete", fake_complete)
    out = amplify_seeds(SEEDS, 80)
    assert len(out) == 80
    assert out[:5] == SEEDS                    # originals first
    assert len(set(o.lower() for o in out)) == 80
    assert len(calls) >= 3                     # multiple style rounds
    styles = " ".join(calls)
    assert "direct ask" in styles and "adversarial" in styles


def test_no_target_or_enough_seeds_is_identity(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not call the model")
    monkeypatch.setattr(gen, "complete", boom)
    assert amplify_seeds(SEEDS, 5) == SEEDS
    assert amplify_seeds(SEEDS, 3) == SEEDS    # never truncates originals


def test_backend_failure_returns_originals(monkeypatch):
    def dead(*a, **k):
        raise RuntimeError("endpoint down")
    monkeypatch.setattr(gen, "complete", dead)
    assert amplify_seeds(SEEDS, 50) == SEEDS
