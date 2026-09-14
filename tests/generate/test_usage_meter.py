"""Hosted-model tokens reach the platform; bring-your-own calls never do."""

import json

import pytest

from zeroproof.simulations.generate import usage_meter
from zeroproof.simulations.generate.usage_meter import UsageMeter, report_usage


class _Posts:
    def __init__(self, ok=True):
        self.bodies: list[dict] = []
        self.ok = ok

    def __call__(self, tokens_in, tokens_out):
        # Set on the class as a plain callable, so it is not bound: no meter arg.
        self.bodies.append({"input_tokens": tokens_in, "output_tokens": tokens_out})
        return self.ok


@pytest.fixture
def meter(monkeypatch):
    monkeypatch.delenv("ZEROPROOF_NO_USAGE_REPORT", raising=False)
    m = UsageMeter()
    posts = _Posts()
    monkeypatch.setattr(UsageMeter, "_post", posts)
    monkeypatch.setattr(usage_meter, "METER", m)
    return m, posts


def test_batches_and_flushes_totals(meter):
    m, posts = meter
    m.add(100, 10)
    m.add(50, 5)
    assert posts.bodies == []
    assert m.flush() is True
    assert posts.bodies == [{"input_tokens": 150, "output_tokens": 15}]
    assert m.flush() is True and len(posts.bodies) == 1  # nothing owed, no request
    assert m.sent == [{"input_tokens": 150, "output_tokens": 15}]


def test_flushes_itself_after_enough_calls(meter):
    m, posts = meter
    for _ in range(usage_meter.FLUSH_EVERY_CALLS):
        m.add(1, 1)
    assert posts.bodies == [{"input_tokens": 50, "output_tokens": 50}]


def test_keeps_owed_tokens_when_the_post_fails_then_gives_up(meter):
    m, posts = meter
    posts.ok = False
    m.add(10, 1)
    assert m.flush() is False
    assert m.flush() is False
    assert m.flush() is False
    assert m.dropped == 3
    # Three strikes and the owed tokens are let go: a meter that cannot reach
    # the platform must not grow without bound inside a long run.
    posts.ok = True
    assert m.flush() is True
    assert len(posts.bodies) == 3 and all(b["input_tokens"] == 10 for b in posts.bodies)
    m.add(2, 2)
    assert m.flush() is True
    assert posts.bodies[-1] == {"input_tokens": 2, "output_tokens": 2}
    assert m.dropped == 0


def test_report_usage_only_counts_hosted_calls(meter):
    m, posts = meter
    report_usage({"_usage": {"input_tokens": 7, "output_tokens": 3}}, hosted=False)
    report_usage({"content": "no usage block"}, hosted=True)
    report_usage("not a dict", hosted=True)
    assert m.flush() is True and posts.bodies == []
    report_usage({"_usage": {"input_tokens": 7, "output_tokens": 3}}, hosted=True)
    m.flush()
    assert posts.bodies == [{"input_tokens": 7, "output_tokens": 3}]


def test_opt_out_env(meter, monkeypatch):
    m, posts = meter
    monkeypatch.setenv("ZEROPROOF_NO_USAGE_REPORT", "1")
    m.add(100, 100)
    m.flush()
    assert posts.bodies == []


def test_post_shape(monkeypatch):
    seen = {}

    class _Res:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url
        seen["key"] = req.get_header("X-api-key")
        seen["body"] = json.loads(req.data)
        return _Res()

    monkeypatch.setenv("ZEROPROOF_API_KEY", "zp_test")
    monkeypatch.setenv("ZEROPROOF_API_URL", "https://api.example.test/")
    monkeypatch.setattr(usage_meter.urllib.request, "urlopen", fake_urlopen)
    assert UsageMeter()._post(12, 3) is True
    assert seen == {
        "url": "https://api.example.test/usage",
        "key": "zp_test",
        "body": {"input_tokens": 12, "output_tokens": 3},
    }


def test_no_key_means_nothing_is_sent(monkeypatch):
    monkeypatch.delenv("ZEROPROOF_API_KEY", raising=False)
    monkeypatch.setattr("zeroproof.auth.stored_api_key", lambda: None)
    called = []
    monkeypatch.setattr(usage_meter.urllib.request, "urlopen", lambda *a, **k: called.append(1))
    assert UsageMeter()._post(1, 1) is False
    assert called == []
