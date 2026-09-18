"""Reference scoring: prefix alignment, summing the generated span, mean_kl."""

from __future__ import annotations

import sys
from unittest import mock

import pytest

from whileai.simulations.score.logprobs import mean_kl
from whileai.simulations.score.reference import (
    REF_KEYS,
    _chat_url,
    _post,
    reference_logprobs,
    score_turns,
)

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


# --------------------------------------------------- reaching the endpoint


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("serve.example", "https://serve.example/v1/chat/completions"),
        ("https://serve.example", "https://serve.example/v1/chat/completions"),
        ("https://serve.example/v1", "https://serve.example/v1/chat/completions"),
        ("https://serve.example/v1/", "https://serve.example/v1/chat/completions"),
        (
            "https://serve.example/v1/chat/completions",
            "https://serve.example/v1/chat/completions",
        ),
    ],
)
def test_every_spelling_of_a_base_url_reaches_one_endpoint(spec, expected):
    # A caller writes the endpoint four different ways; all four must hit
    # the same URL, or the reference is silently scored somewhere else.
    assert _chat_url(spec) == expected


def test_a_failing_endpoint_names_the_url_and_the_status():
    class Res:
        status_code = 503
        text = "upstream is warming up" * 40

    sent: dict = {}

    class FakeRequests:
        @staticmethod
        def post(url, headers, json, timeout):
            sent.update(url=url, headers=headers, body=json, timeout=timeout)
            return Res()

    with (
        mock.patch.dict(sys.modules, {"requests": FakeRequests}),
        pytest.raises(RuntimeError) as err,
    ):
        _post("https://serve.example/v1/chat/completions", "zp_key", {"model": "m"}, 30.0)
    message = str(err.value)
    assert "https://serve.example/v1/chat/completions" in message
    assert "503" in message and "upstream is warming up" in message
    # The body is truncated, so a megabyte of HTML cannot become the error.
    assert len(message) < 400
    assert sent["headers"]["Authorization"] == "Bearer zp_key"
    assert sent["timeout"] == 30.0


def test_no_key_sends_no_authorization_header():
    class Res:
        status_code = 200

        @staticmethod
        def json():
            return {"ok": True}

    seen: dict = {}

    class FakeRequests:
        @staticmethod
        def post(url, headers, json, timeout):
            seen.update(headers)
            return Res()

    with mock.patch.dict(sys.modules, {"requests": FakeRequests}):
        assert _post("https://serve.example/v1/chat/completions", "", {}, 5.0) == {"ok": True}
    assert "Authorization" not in seen
    assert seen["Content-Type"] == "application/json"


def test_without_a_transport_the_endpoint_comes_from_the_backend_spec(monkeypatch):
    # The default path: no transport=, so the URL and key are built from
    # ``ref`` and the environment. Only the HTTP call itself is replaced.
    server = FakeServer()
    calls: list[tuple[str, str, float]] = []

    def fake_post(url, key, body, timeout):
        calls.append((url, key, timeout))
        return server(body)

    monkeypatch.setattr("whileai.simulations.score.reference._post", fake_post)
    monkeypatch.setenv("WHILEAI_API_KEY", "zp_from_env")
    rows = [
        {
            "prompt": "p",
            "final_text": "r",
            "steps": [],
            "messages": [{"role": "user", "content": "p"}, {"role": "assistant", "content": "r"}],
        }
    ]
    report = reference_logprobs(
        rows, "vllm:Qwen/Qwen3-4B@https://serve.zeroproofai.com/v1", timeout=42.0
    )
    assert report["n_rows"] == 1
    url, key, timeout = calls[0]
    assert url == "https://serve.zeroproofai.com/v1/chat/completions"
    assert key == "zp_from_env"
    assert timeout == 42.0


def test_an_explicit_api_key_wins_over_the_environment(monkeypatch):
    server = FakeServer()
    keys: list[str] = []

    monkeypatch.setattr(
        "whileai.simulations.score.reference._post",
        lambda url, key, body, timeout: (keys.append(key), server(body))[1],
    )
    monkeypatch.setenv("WHILEAI_API_KEY", "zp_from_env")
    rows = [
        {
            "prompt": "p",
            "final_text": "r",
            "steps": [],
            "messages": [{"role": "user", "content": "p"}, {"role": "assistant", "content": "r"}],
        }
    ]
    reference_logprobs(rows, "vllm:m@https://serve.zeroproofai.com/v1", api_key="zp_explicit")
    assert keys and set(keys) == {"zp_explicit"}


# ------------------------------------------- what the reference is shown


def test_system_prompt_and_tools_override_what_the_source_carried():
    # A row list carries neither, so these are the only way the reference
    # sees the prompt the policy saw. Scoring under a different prompt is
    # not a KL, and nothing else in the report would say so.
    server = FakeServer()
    tools = [{"type": "function", "function": {"name": "lookup_invoice", "parameters": {}}}]
    rows = [
        {
            "prompt": "p",
            "final_text": "r",
            "steps": [],
            "messages": [{"role": "user", "content": "p"}, {"role": "assistant", "content": "r"}],
        }
    ]
    reference_logprobs(
        rows,
        "vllm:m@https://x/v1",
        system_prompt="you are a billing agent",
        tools=tools,
        transport=server,
    )
    assert server.calls, "nothing was sent"
    for body in server.calls:
        assert body["messages"][0] == {"role": "system", "content": "you are a billing agent"}
        assert body["tools"] == tools and body["tool_choice"] == "none"


def test_chat_template_kwargs_reach_every_request():
    # Qwen3 samples with enable_thinking=False; if the reference renders
    # with thinking on, the token spans do not line up and mean_kl is a
    # template artifact. Both calls per turn must carry it.
    server = FakeServer()
    rows = [
        {
            "prompt": "p",
            "final_text": "r",
            "steps": [],
            "messages": [{"role": "user", "content": "p"}, {"role": "assistant", "content": "r"}],
        }
    ]
    reference_logprobs(
        rows,
        "vllm:m@https://x/v1",
        chat_template_kwargs={"enable_thinking": False},
        transport=server,
    )
    assert len(server.calls) == 2
    for body in server.calls:
        assert body["chat_template_kwargs"] == {"enable_thinking": False}


# ------------------------------------------------- degenerate server replies


def test_positions_the_server_cannot_explain_are_skipped_not_guessed():
    # A position with no usable record contributes nothing: no token is
    # counted and no logprob is invented. Under-counting is recoverable
    # (token_count_gap shows it); a guessed logprob is not.
    class Sparse(FakeServer):
        def __call__(self, body):
            out = super().__call__(body)
            entries = out.get("prompt_logprobs")
            if entries:
                out["prompt_token_ids"] = None
                # Inside the scored span: not a mapping; a mapping holding
                # no record; a record whose logprob is not a number.
                entries[-5] = "junk"
                entries[-4] = {"a": 1}
                entries[-3] = {"7": {"logprob": None, "rank": 1, "decoded_token": "x"}}
            return out

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "a b c d e"},
    ]
    # The reply is 5 tokens plus the end token; 3 of those 6 positions
    # come back unusable.
    total, count, turns = score_turns(messages, post=Sparse(), model="m")
    assert (turns, count) == (1, 3)
    assert total == pytest.approx(-3.0)


def test_an_entry_keyed_by_the_token_id_itself_is_still_found():
    # vLLM keys prompt_logprobs by token id; JSON makes those strings, but
    # a transport that hands back parsed objects keeps them as ints.
    class IntKeyed(FakeServer):
        def __call__(self, body):
            out = super().__call__(body)
            entries = out.get("prompt_logprobs")
            if entries:
                out["prompt_logprobs"] = [
                    e if e is None else {int(k) if k.isdigit() else k: v for k, v in e.items()}
                    for e in entries
                ]
            return out

    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a b"}]
    total, count, turns = score_turns(messages, post=IntKeyed(), model="m")
    assert (count, turns) == (3, 1) and total == pytest.approx(-3.0)


def test_a_row_with_no_assistant_turn_is_skipped_without_an_error():
    server = FakeServer()
    rows = [{"prompt": "p", "final_text": "", "steps": [], "messages": [{"role": "user", "x": 1}]}]
    report = reference_logprobs(rows, "vllm:m@https://x/v1", transport=server)
    assert report == {
        "n_rows": 0,
        "n_skipped": 1,
        "n_tokens": 0,
        "model": "m",
        "token_count_gap": None,
        "errors": [],
    }
    assert not any(key in rows[0] for key in REF_KEYS)


def test_the_error_list_is_capped_but_every_skip_is_counted():
    # A batch where the endpoint is down must not return a 200-line report;
    # n_skipped is the number to read, errors is the sample.
    def broken(body):
        raise RuntimeError("endpoint refused the connection")

    rows = [
        {
            "prompt": f"p{i}",
            "final_text": "r",
            "steps": [],
            "messages": [
                {"role": "user", "content": f"p{i}"},
                {"role": "assistant", "content": "r"},
            ],
        }
        for i in range(12)
    ]
    report = reference_logprobs(rows, "vllm:m@https://x/v1", transport=broken, concurrency=2)
    assert report["n_rows"] == 0 and report["n_skipped"] == 12
    assert len(report["errors"]) == 5
    assert all("endpoint refused the connection" in e for e in report["errors"])


def test_token_count_gap_reports_the_mean_distance_from_the_policy():
    # n_tokens on the row is the policy's count; the gap is the check that
    # the two sides share a tokenizer, and so that mean_kl is a KL.
    server = FakeServer()
    rows = [
        {
            "prompt": "p",
            "final_text": "a b",
            "steps": [],
            "n_tokens": n,
            "messages": [
                {"role": "user", "content": "p"},
                {"role": "assistant", "content": "a b"},
            ],
        }
        for n in (3, 5)
    ]
    report = reference_logprobs(rows, "vllm:m@https://x/v1", transport=server)
    # The reference counts 3 tokens ("a", "b", end): gaps of 0 and 2.
    assert report["n_rows"] == 2 and report["token_count_gap"] == 1.0
