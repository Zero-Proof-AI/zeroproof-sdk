"""
Data models for the ZeroProof SDK.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class EncryptedMessage:
    """Represents an encrypted message response."""

    message_id: str
    expires_at: str
    status: str
    ttl_minutes: int

    @classmethod
    def from_dict(cls, data: dict) -> "EncryptedMessage":
        """Create an EncryptedMessage from API response data."""
        return cls(
            message_id=data["message_id"],
            expires_at=data["expires_at"],
            status=data["status"],
            ttl_minutes=data["ttl_minutes"],
        )


@dataclass
class DecryptedMessage:
    """Represents a decrypted message."""

    message_id: str
    from_agent_id: str
    to_agent_id: str
    message: Any
    read_count: int
    created_at: str
    expires_at: str

    @classmethod
    def from_dict(cls, data: dict) -> "DecryptedMessage":
        """Create a DecryptedMessage from API response data."""
        return cls(
            message_id=data["message_id"],
            from_agent_id=data["from_agent_id"],
            to_agent_id=data["to_agent_id"],
            message=data["message"],
            read_count=data["read_count"],
            created_at=data["created_at"],
            expires_at=data["expires_at"],
        )


@dataclass
class ReputationReport:
    """Represents a transaction report response."""

    agent_id: str
    transaction_recorded: bool
    new_trust_score: float
    reputation_tier: str
    total_transactions: int
    successful_transactions: int
    disputed_transactions: int
    dispute_rate: float

    @classmethod
    def from_dict(cls, data: dict) -> "ReputationReport":
        """Create a ReputationReport from API response data."""
        return cls(
            agent_id=data["agent_id"],
            transaction_recorded=data["transaction_recorded"],
            new_trust_score=data["new_trust_score"],
            reputation_tier=data["reputation_tier"],
            total_transactions=data["total_transactions"],
            successful_transactions=data["successful_transactions"],
            disputed_transactions=data["disputed_transactions"],
            dispute_rate=data["dispute_rate"],
        )


@dataclass
class ReputationScore:
    """Represents an agent's reputation score."""

    agent_id: str
    trust_score: float
    reputation_tier: str
    total_transactions: int
    successful_transactions: int
    disputed_transactions: int
    dispute_rate: float
    total_volume: float
    first_seen: str
    last_activity: str
    history: list = None

    @classmethod
    def from_dict(cls, data: dict) -> "ReputationScore":
        """Create a ReputationScore from API response data."""
        return cls(
            agent_id=data["agent_id"],
            trust_score=data["trust_score"],
            reputation_tier=data["reputation_tier"],
            total_transactions=data["total_transactions"],
            successful_transactions=data["successful_transactions"],
            disputed_transactions=data["disputed_transactions"],
            dispute_rate=data["dispute_rate"],
            total_volume=data["total_volume"],
            first_seen=data["first_seen"],
            last_activity=data["last_activity"],
            history=data.get("history"),
        )


@dataclass
class ApprovalWorkflow:
    """Represents a created approval workflow."""

    workflow_id: str
    status: str
    transaction_id: str
    amount: float
    required_approvers: list
    created_at: str
    expires_at: str

    @classmethod
    def from_dict(cls, data: dict) -> "ApprovalWorkflow":
        """Create an ApprovalWorkflow from API response data."""
        return cls(
            workflow_id=data["workflow_id"],
            status=data["status"],
            transaction_id=data["transaction_id"],
            amount=data["amount"],
            required_approvers=data["required_approvers"],
            created_at=data["created_at"],
            expires_at=data["expires_at"],
        )


@dataclass
class WorkflowApproval:
    """Represents a single approval within a workflow."""

    agent_id: str
    decision: str
    timestamp: str
    reason: str = None

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowApproval":
        """Create a WorkflowApproval from API response data."""
        return cls(
            agent_id=data["agent_id"],
            decision=data["decision"],
            timestamp=data["timestamp"],
            reason=data.get("reason"),
        )


@dataclass
class WorkflowStatus:
    """Represents the status of an approval workflow."""

    workflow_id: str
    status: str
    transaction_id: str
    amount: float
    required_approvers: list
    approvals: list
    pending_approvers: list
    created_at: str
    expires_at: str
    updated_at: str
    metadata: dict = None

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowStatus":
        """Create a WorkflowStatus from API response data."""
        approvals = [WorkflowApproval.from_dict(a) for a in data.get("approvals", [])]
        return cls(
            workflow_id=data["workflow_id"],
            status=data["status"],
            transaction_id=data["transaction_id"],
            amount=data["amount"],
            required_approvers=data["required_approvers"],
            approvals=approvals,
            pending_approvers=data.get("pending_approvers", []),
            created_at=data["created_at"],
            expires_at=data["expires_at"],
            updated_at=data["updated_at"],
            metadata=data.get("metadata"),
        )
