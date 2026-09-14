"""Reference scoring: prefix alignment, summing the generated span, mean_kl."""

from __future__ import annotations

import pytest

from zeroproof.simulations.score.logprobs import mean_kl
from zeroproof.simulations.score.reference import reference_logprobs, score_turns

END = "<|im_end|>"


class FakeServer:
    """Renders messages as one token per word, scores each token -1.0, and
    answers like vLLM: usage.prompt_tokens on a prefix call, prompt_logprobs
    plus prompt_token_ids on a scoring call."""

    def __init__(self, per_token: float = -1.0, with_ids: bool = True):
        self.per_token = per_token
        self.with_ids = with_ids
        self.calls: list[dict] = []

    def render(self, messages, add_generation_prompt):
        toks = []
        for m in messages:
            toks += [f"<{m['role']}>", *str(m.get("content") or "").split(), END, "\n"]
        if add_generation_prompt:
            toks += ["<assistant>"]
        return toks

    def __call__(self, body):
        self.calls.append(body)
        toks = self.render(body["messages"], body.get("add_generation_prompt", True))
        out = {"usage": {"prompt_tokens": len(toks)}}
        if body.get("prompt_logprobs"):
            ids = list(range(1, len(toks) + 1))
            entries = [None] + [
                {
                    str(i): {"logprob": self.per_token, "rank": 1, "decoded_token": t},
                    "999": {"logprob": -0.1},
                }
                for i, t in zip(ids[1:], toks[1:])
            ]
            out["prompt_logprobs"] = entries
            # vLLM returns the ids only with return_token_ids; the top
            # candidate ("999", rank 1) must not be mistaken for the token.
            out["prompt_token_ids"] = ids if self.with_ids else None
            for e in entries[1:]:
                e["999"]["rank"] = 1
                next(v for k, v in e.items() if k != "999")["rank"] = 7
        return out


def test_score_turns_sums_only_the_generated_span():
    server = FakeServer()
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "refund order 4412"},
        {"role": "assistant", "content": "done for 4412"},
    ]
    total, count, turns = score_turns(messages, post=server, model="ref")
    # "done for 4412" = 3 tokens + the end token; the trailing newline the
    # template adds is not a generated token and is dropped.
    assert (count, turns) == (4, 1)
    assert total == pytest.approx(-4.0)
    prefix, full = server.calls
    assert prefix["add_generation_prompt"] is True and "prompt_logprobs" not in prefix
    assert full["add_generation_prompt"] is False and full["prompt_logprobs"] == 1
    assert full["max_tokens"] == 1
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    server.calls.clear()
    score_turns(messages, post=server, model="ref", tools=tools)
    assert server.calls[0]["tools"] == tools and server.calls[0]["tool_choice"] == "none"


def test_without_token_ids_the_worse_ranked_candidate_is_the_token():
    server = FakeServer(with_ids=False)
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a b"}]
    total, count, turns = score_turns(messages, post=server, model="ref")
    assert (count, turns) == (3, 1) and total == pytest.approx(-3.0)
    assert server.calls[-1]["return_token_ids"] is True


def test_every_assistant_turn_is_scored_with_its_own_prefix():
    server = FakeServer()
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
        {"role": "user", "content": "refund 4412"},
        {"role": "assistant", "content": "done"},
    ]
    total, count, turns = score_turns(messages, post=server, model="ref")
    assert turns == 2 and count == (2 + 1) + (1 + 1)
    assert total == pytest.approx(-count)
    assert len(server.calls) == 4


def test_reference_logprobs_stamps_rows_and_feeds_mean_kl():
    server = FakeServer(per_token=-1.5)
    rows = [
        {
            "prompt": "refund order 4412",
            "final_text": "done for 4412",
            "steps": [],
            "logprob": -2.0,
            "n_tokens": 4,
            "messages": [
                {"role": "user", "content": "refund order 4412"},
                {"role": "assistant", "content": "done for 4412"},
            ],
        },
        {
            "prompt": "no reply",
            "final_text": "",
            "steps": [],
            "messages": [{"role": "user", "content": "x"}],
        },
        "not a row",
    ]
    report = reference_logprobs(
        rows, "vllm:Qwen/Qwen3-4B@https://serve.example/v1", transport=server
    )
    assert report["n_rows"] == 1 and report["n_skipped"] == 1
    assert report["model"] == "Qwen/Qwen3-4B"
    assert report["token_count_gap"] == 0.0
    assert rows[0]["ref_logprob"] == pytest.approx(-6.0)
    assert rows[0]["ref_n_tokens"] == 4 and rows[0]["ref_model"] == "Qwen/Qwen3-4B"
    assert "ref_logprob" not in rows[1]
    kl = mean_kl(rows[:2])
    # (policy -2.0) - (reference -6.0) over 4 tokens
    assert kl["mean_kl"] == pytest.approx(1.0)
    assert kl["n_rows"] == 1


def test_a_failing_row_is_counted_not_raised():
    def broken(body):
        raise RuntimeError("boom")

    rows = [
        {
            "prompt": "p",
            "final_text": "r",
            "steps": [],
            "messages": [{"role": "user", "content": "p"}, {"role": "assistant", "content": "r"}],
        }
    ]
    report = reference_logprobs(rows, "vllm:m@https://x", transport=broken)
    assert report["n_rows"] == 0 and report["n_skipped"] == 1
    assert report["errors"] and "boom" in report["errors"][0]


def test_endpoint_without_prompt_logprobs_is_a_clear_error():
    def plain(body):
        return {"usage": {"prompt_tokens": 3}}

    with pytest.raises(RuntimeError, match="prompt_logprobs"):
        score_turns(
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
            post=plain,
            model="m",
        )
