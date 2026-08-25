"""CPU studio Modal app: one warm CPU, existing secret, no GPU edits."""
from __future__ import annotations

from tests.helpers import REPO_ROOT

SRC = (REPO_ROOT / "infra" / "modal_studio.py").read_text()


def test_studio_modal_is_cpu_kept_warm():
    assert 'modal.App("zeroproof-studio-api"' in SRC
    assert "min_containers=1" in SRC
    assert "min_containers=0" not in SRC
    assert "gpu=" not in SRC
    assert 'modal.App("stressd-vllm"' not in SRC
    assert "min_containers=2" not in SRC


def test_studio_modal_reuses_existing_vllm_secret_and_serve_py():
    assert 'Secret.from_name("stressd-vllm-key")' in SRC
    assert 'pip_install("boto3")' in SRC
    assert "studio/serve.py" in SRC
    assert "huggingface-secret" not in SRC
