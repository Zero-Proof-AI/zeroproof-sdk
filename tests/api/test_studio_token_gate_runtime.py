"""Runtime quota tests for the studio token gate."""
from __future__ import annotations

import importlib
import sys

import pytest

from tests.helpers import REPO_ROOT


class FakeDynamo:
    def __init__(self, *, key_items=None, usage_items=None):
        self.key_items = key_items or {}
        self.usage_items = usage_items or {}
        self.updates = []

    def get_item(self, TableName, Key):
        api_key = Key["apiKey"]["S"]
        if TableName == "zeroproof-api-keys":
            return {"Item": self.key_items.get(api_key)} if api_key in self.key_items else {}
        usage_key = (api_key, Key["date"]["S"])
        return {"Item": self.usage_items.get(usage_key)} if usage_key in self.usage_items else {}

    def update_item(self, **kwargs):
        self.updates.append(kwargs)


@pytest.fixture
def token_gate(monkeypatch):
    studio_path = str(REPO_ROOT / "studio")
    if studio_path not in sys.path:
        sys.path.insert(0, studio_path)
    sys.modules.pop("token_gate", None)
    module = importlib.import_module("token_gate")
    monkeypatch.setattr(module, "_dynamo", None)
    module._key_cache.clear()
    return module


def test_check_key_allows_usage_below_daily_limits(token_gate, monkeypatch):
    fake_db = FakeDynamo(
        key_items={
            "zp_valid": {"active": {"BOOL": True}, "userId": {"S": "user-1"}},
        },
        usage_items={
            ("user#user-1", token_gate._today()): {
                "inputTokens": {"N": str(token_gate.DAILY_INPUT_LIMIT - 1)},
                "outputTokens": {"N": str(token_gate.DAILY_OUTPUT_LIMIT - 1)},
            },
        },
    )
    monkeypatch.setattr(token_gate, "_dynamo", fake_db)

    token_gate.check_key("zp_valid")


def test_check_key_blocks_once_account_limit_is_reached(token_gate, monkeypatch):
    fake_db = FakeDynamo(
        key_items={
            "zp_at_limit": {"active": {"BOOL": True}, "userId": {"S": "user-2"}},
        },
        usage_items={
            ("user#user-2", token_gate._today()): {
                "inputTokens": {"N": str(token_gate.DAILY_INPUT_LIMIT)},
                "outputTokens": {"N": "0"},
            },
        },
    )
    monkeypatch.setattr(token_gate, "_dynamo", fake_db)

    with pytest.raises(token_gate.QuotaExceeded, match="daily account quota exceeded"):
        token_gate.check_key("zp_at_limit")


def test_record_usage_updates_account_daily_counter(token_gate, monkeypatch):
    fake_db = FakeDynamo(
        key_items={
            "zp_shared": {"active": {"BOOL": True}, "userId": {"S": "shared-user"}},
        },
    )
    monkeypatch.setattr(token_gate, "_dynamo", fake_db)

    token_gate.record_usage("zp_shared", input_tokens=500, output_tokens=200)

    assert len(fake_db.updates) == 1
    update = fake_db.updates[0]
    assert update["TableName"] == token_gate.USAGE_TABLE
    assert update["Key"] == {
        "apiKey": {"S": "user#shared-user"},
        "date": {"S": token_gate._today()},
    }
    assert update["ExpressionAttributeValues"][":i"] == {"N": "500"}
    assert update["ExpressionAttributeValues"][":o"] == {"N": "200"}


def test_record_usage_ignores_invalid_keys(token_gate, monkeypatch):
    fake_db = FakeDynamo()
    monkeypatch.setattr(token_gate, "_dynamo", fake_db)

    token_gate.record_usage("bad_key", input_tokens=1, output_tokens=1)

    assert fake_db.updates == []