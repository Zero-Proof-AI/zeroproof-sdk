"""Token-gate integration guards for studio deployment."""
from __future__ import annotations

from tests.helpers import REPO_ROOT

MODAL_SRC = (REPO_ROOT / "infra" / "modal_studio.py").read_text()
SERVE_SRC = (REPO_ROOT / "studio" / "serve.py").read_text()
SIM_SRC = (REPO_ROOT / "studio" / "api" / "simulate.py").read_text()
GATE_SRC = (REPO_ROOT / "studio" / "token_gate.py").read_text()


def test_modal_studio_attaches_token_gate_secret():
    assert 'modal.Secret.from_name("zeroproof-token-gate")' in MODAL_SRC


def test_serve_requires_token_gate_import_and_gate_check():
    assert "from token_gate import check_key" in SERVE_SRC
    assert "def _check_api_key(handler)" in SERVE_SRC
    assert "check_key(api_key)" in SERVE_SRC


def test_simulate_records_usage_for_api_key_calls():
    assert '"/api/simulate": lambda: simulate.start({**body, "_api_key": api_key})' in SERVE_SRC
    assert 'api_key = str(spec.get("_api_key") or "").strip()' in SIM_SRC
    assert "from token_gate import record_usage" in SIM_SRC
    assert "record_usage(api_key" in SIM_SRC


def test_token_gate_bounds_key_cache_growth():
    assert "_KEY_CACHE_MAX" in GATE_SRC
    assert "if len(_key_cache) >= _KEY_CACHE_MAX:" in GATE_SRC


def test_token_gate_uses_lock_for_cache_access():
    assert "_key_cache_lock = threading.Lock()" in GATE_SRC
    assert "with _key_cache_lock:" in GATE_SRC
