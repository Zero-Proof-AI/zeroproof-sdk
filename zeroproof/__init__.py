"""
ZeroProof platform client: OTLP trace ingest for the token gate.

    >>> import zeroproof
    >>> zeroproof.send_traces(otlp_batch, api_key="zp_...")
    >>> zeroproof.list_traces("zp_...")["traces"]

Sign in once from a terminal with ``zeroproof login``; ``resolve_api_key()``
then finds the saved key. Agent simulations live next door in
``zeroproof.simulations``.
"""

from .auth import LoginError, login, logout, resolve_api_key
from .ingest import (
    ZeroProofIngestError,
    ingest_traces,
    list_traces,
    otel_env,
    send_traces,
)

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("zeroproof")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"

__all__ = [
    "LoginError",
    "ZeroProofIngestError",
    "__version__",
    "ingest_traces",
    "list_traces",
    "login",
    "logout",
    "otel_env",
    "resolve_api_key",
    "send_traces",
]
