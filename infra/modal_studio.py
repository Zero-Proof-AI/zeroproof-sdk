"""Public CPU API for the Vercel studio desk.

Serves existing studio/serve.py. Hosted Qwen reads VLLM_API_KEY from the
already-deployed Modal secret `stressd-vllm-key`. The browser never sees it.

CPU only. Keep one container warm so in-memory simulate jobs survive
the desk demo. Do not edit stressd-vllm (GPU). Do not attach a GPU.

Deploy from the SDK repo root:

    modal deploy infra/modal_studio.py
"""
from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PORT = 8765
IGNORE = ["**/__pycache__/**", "**/.DS_Store", "**/.env", "**/.env.*"]

image = modal.Image.debian_slim(python_version="3.12").env({
    "PYTHONUNBUFFERED": "1",
    "STUDIO_HOST": "0.0.0.0",
    "PORT": str(PORT),
    "PYTHONPATH": "/root",
})
image = image.add_local_dir(
    str(ROOT / "zeroproof_simulations"), "/root/zeroproof_simulations",
    ignore=IGNORE)
image = image.add_local_dir(
    str(ROOT / "studio"), "/root/studio", ignore=IGNORE)
if (ROOT / "tests" / "fixtures").is_dir():
    image = image.add_local_dir(
        str(ROOT / "tests" / "fixtures"), "/root/tests/fixtures", ignore=IGNORE)
if (ROOT / "specs").is_dir():
    image = image.add_local_dir(
        str(ROOT / "specs"), "/root/specs", ignore=IGNORE)

app = modal.App("zeroproof-studio-api", image=image)


@app.function(
    cpu=1,
    memory=2048,
    min_containers=1,
    max_containers=1,
    scaledown_window=300,
    timeout=900,
    secrets=[modal.Secret.from_name("stressd-vllm-key")],
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=60)
def serve():
    import subprocess

    subprocess.Popen(["python", "-u", "/root/studio/serve.py"])
