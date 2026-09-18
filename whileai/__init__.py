"""
While platform client: OTLP trace ingest for the token gate.

    >>> import whileai
    >>> whileai.send_traces(otlp_batch, api_key="zp_...")
    >>> whileai.list_traces("zp_...")["traces"]

Sign in once from a terminal with ``whileai login``; ``resolve_api_key()``
then finds the saved key. Agent simulations live next door in
``whileai.simulations``.
"""

from .auth import LoginError, account, login, logout, resolve_api_key, signup
from .ingest import (
    WhileIngestError,
    ZeroProofIngestError,
    ingest_traces,
    list_traces,
    otel_env,
    send_runs,
    send_traces,
)

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("whileai")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"

__all__ = [
    "LoginError",
    "WhileIngestError",
    "ZeroProofIngestError",
    "__version__",
    "account",
    "ingest_traces",
    "list_traces",
    "login",
    "logout",
    "otel_env",
    "resolve_api_key",
    "send_runs",
    "send_traces",
    "signup",
]
