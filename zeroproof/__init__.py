"""
ZeroProof Python SDK

Python SDK for the ZeroProof AI verification API.
Provides encrypted messaging services.

Example:
    >>> from zeroproof import ZeroProof
    >>> client = ZeroProof(api_key="zkp_your_key")
    >>> result = client.send_encrypted(
    ...     to_agent_id="agent_456",
    ...     message="Hello, world!",
    ...     ttl_minutes=60
    ... )
"""

from .client import ZeroProof
from .exceptions import (
    AuthenticationError,
    ExpiredError,
    NotFoundError,
    RateLimitError,
    ValidationError,
    ZeroProofError,
)
from .ingest import (
    ZeroProofIngestError,
    ingest_traces,
    list_traces,
    otel_env,
    send_traces,
)
from .models import (
    ApprovalWorkflow,
    DecryptedMessage,
    EncryptedMessage,
    ReputationReport,
    ReputationScore,
    WorkflowApproval,
    WorkflowStatus,
)

__version__ = "0.3"
__all__ = [
    # Main client
    "ZeroProof",
    # Models
    "EncryptedMessage",
    "DecryptedMessage",
    "ReputationReport",
    "ReputationScore",
    "ApprovalWorkflow",
    "WorkflowApproval",
    "WorkflowStatus",
    # Trace ingest
    "ingest_traces",
    "list_traces",
    "otel_env",
    "send_traces",
    "ZeroProofIngestError",
    # Exceptions
    "ZeroProofError",
    "AuthenticationError",
    "ValidationError",
    "NotFoundError",
    "ExpiredError",
    "RateLimitError",
]
