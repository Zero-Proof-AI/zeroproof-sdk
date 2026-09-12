"""
ZeroProof platform client: OTLP trace ingest for the token gate.

    >>> import zeroproof
    >>> zeroproof.send_traces(otlp_batch, api_key="zp_...")
    >>> zeroproof.list_traces("zp_...")["traces"]

Agent simulations live next door in ``zeroproof.simulations``.
"""

from .ingest import (
    ZeroProofIngestError,
    ingest_traces,
    list_traces,
    otel_env,
    send_traces,
)

try:
    from importlib.metadata import PackageNotFoundError, version as _dist_version
    __version__ = _dist_version("zeroproof")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"

__all__ = [
    "ingest_traces",
    "list_traces",
    "otel_env",
    "send_traces",
    "ZeroProofIngestError",
    "__version__",
]
