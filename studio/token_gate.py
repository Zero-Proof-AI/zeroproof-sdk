"""
Token-gate enforcement for the studio engine (Modal).

Drop-in for studio/serve.py. Requires the Modal secret `zeroproof-token-gate`
(already created in the zeroproofai workspace), which carries scoped AWS
credentials for the `zeroproof-modal-gate` IAM user plus table names and
daily limits.

Usage in serve.py:

    from token_gate import check_key, record_usage, QuotaExceeded, InvalidKey

    # attach the secret to the serving function:
    #   @app.function(secrets=[modal.Secret.from_name("zeroproof-token-gate")], ...)

    # at the top of a request handler:
    api_key = request.headers.get("x-api-key", "")
    try:
        check_key(api_key)          # raises InvalidKey / QuotaExceeded
    except InvalidKey:
        return JSONResponse({"error": "invalid or missing API key"}, status_code=401)
    except QuotaExceeded as e:
        return JSONResponse({"error": str(e)}, status_code=429)

    # ... run the model ...

    record_usage(api_key, input_tokens=n_in, output_tokens=n_out)

Design: dumb-simple, and the quota is PER ACCOUNT. A key resolves to its
owning Clerk user, and counters live under user#<clerkId> per UTC day, so
a user with five keys still gets one 500k/1M daily allowance. Checks read
the counters as they were at request start; a burst of parallel requests
can overshoot by one request's worth, which is acceptable for a free tier.
Usage items carry a TTL so the table stays small.
"""

import os
import time
import datetime
import threading

import boto3

_dynamo = None

KEYS_TABLE = os.environ.get("TABLE_API_KEYS", "zeroproof-api-keys")
USAGE_TABLE = os.environ.get("TABLE_DAILY_USAGE", "zeroproof-daily-usage")
DAILY_INPUT_LIMIT = int(os.environ.get("DAILY_INPUT_LIMIT", "500000"))
DAILY_OUTPUT_LIMIT = int(os.environ.get("DAILY_OUTPUT_LIMIT", "1000000"))
USAGE_TTL_DAYS = 14

# tiny in-process cache so hot keys don't hit Dynamo on every request:
# api_key -> (owner_user_id or None if invalid, fetched_at)
_key_cache: dict[str, tuple[str | None, float]] = {}
_KEY_CACHE_TTL = 60.0
_KEY_CACHE_MAX = 10_000
_key_cache_lock = threading.Lock()


class InvalidKey(Exception):
    pass


class QuotaExceeded(Exception):
    pass


def _db():
    global _dynamo
    if _dynamo is None:
        _dynamo = boto3.client("dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    return _dynamo


def _today() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _owner(api_key: str) -> str:
    """Resolve a key to its owning user id, or raise InvalidKey."""
    if not api_key or not api_key.startswith("zp_"):
        raise InvalidKey("missing or malformed key")
    now = time.time()
    with _key_cache_lock:
        cached = _key_cache.get(api_key)
    if cached and now - cached[1] < _KEY_CACHE_TTL:
        owner = cached[0]
    else:
        item = _db().get_item(TableName=KEYS_TABLE, Key={"apiKey": {"S": api_key}}).get("Item")
        active = bool(item and item.get("active", {}).get("BOOL"))
        owner = item["userId"]["S"] if (active and item and "userId" in item) else None
        with _key_cache_lock:
            if len(_key_cache) >= _KEY_CACHE_MAX:
                # Evict stale entries first; if still too large, drop cache.
                stale_before = now - _KEY_CACHE_TTL
                stale_keys = [k for k, (_, fetched_at) in _key_cache.items() if fetched_at < stale_before]
                for k in stale_keys:
                    _key_cache.pop(k, None)
                if len(_key_cache) >= _KEY_CACHE_MAX:
                    _key_cache.clear()
            _key_cache[api_key] = (owner, now)
    if not owner:
        raise InvalidKey("unknown or deactivated key")
    return owner


def check_key(api_key: str) -> None:
    """Raise InvalidKey or QuotaExceeded; return None when the request may proceed."""
    owner = _owner(api_key)
    usage = _db().get_item(
        TableName=USAGE_TABLE,
        Key={"apiKey": {"S": f"user#{owner}"}, "date": {"S": _today()}},
    ).get("Item") or {}
    used_in = int(usage.get("inputTokens", {}).get("N", "0"))
    used_out = int(usage.get("outputTokens", {}).get("N", "0"))
    if used_in >= DAILY_INPUT_LIMIT or used_out >= DAILY_OUTPUT_LIMIT:
        raise QuotaExceeded(
            f"daily account quota exceeded ({used_in}/{DAILY_INPUT_LIMIT} input, "
            f"{used_out}/{DAILY_OUTPUT_LIMIT} output tokens); resets at midnight UTC"
        )


def record_usage(api_key: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
    """Atomically add this request's token counts to the ACCOUNT's daily counters."""
    try:
        owner = _owner(api_key)
    except InvalidKey:
        return
    expires = int(time.time()) + USAGE_TTL_DAYS * 86400
    _db().update_item(
        TableName=USAGE_TABLE,
        Key={"apiKey": {"S": f"user#{owner}"}, "date": {"S": _today()}},
        UpdateExpression="ADD inputTokens :i, outputTokens :o SET expiresAt = :e",
        ExpressionAttributeValues={
            ":i": {"N": str(int(input_tokens))},
            ":o": {"N": str(int(output_tokens))},
            ":e": {"N": str(expires)},
        },
    )
