"""
Main client for the ZeroProof SDK.
"""

import json
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests

from .exceptions import (
    AuthenticationError,
    ExpiredError,
    NotFoundError,
    RateLimitError,
    ValidationError,
    ZeroProofError,
)
from .models import (
    ApprovalWorkflow,
    DecryptedMessage,
    EncryptedMessage,
    ReputationReport,
    ReputationScore,
    WorkflowStatus,
)


class ZeroProof:
    """
    ZeroProof API client for encrypted messaging.

    Example:
        >>> client = ZeroProof(api_key="zkp_your_key")
        >>> result = client.send_encrypted(
        ...     to_agent_id="agent_456",
        ...     message="Hello, world!",
        ...     ttl_minutes=60
        ... )
        >>> print(result.message_id)
    """

    DEFAULT_BASE_URL = "https://api.zeroproofai.com/v1"

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        Initialize the ZeroProof client.

        Args:
            api_key: Your ZeroProof API key (starts with 'zkp_')
            base_url: Custom API base URL (optional)

        Raises:
            ValueError: If API key is invalid or missing
        """
        if not api_key:
            raise ValueError("API key is required")

        if not api_key.startswith("zkp_"):
            raise ValueError("Invalid API key format. API key must start with 'zkp_'")

        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.session = requests.Session()
        
        # Get version safely to avoid circular import
        try:
            from . import __version__
            version = __version__
        except ImportError:
            version = "unknown"
        
        self.session.headers.update(
            {
                "X-Api-Key": self.api_key,
                "Content-Type": "application/json",
                "User-Agent": f"zeroproof-python/{version}",
            }
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Close the HTTP session."""
        self.session.close()

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters

        Returns:
            API response as dictionary

        Raises:
            ZeroProofError: If the request fails
        """
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=30,
            )

            # Try to parse response as JSON
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = {"error": response.text}

            # Handle error responses
            if response.status_code == 401:
                raise AuthenticationError(
                    message=response_data.get("message", "Authentication failed"),
                    status_code=response.status_code,
                    response=response_data,
                )
            elif response.status_code == 400:
                raise ValidationError(
                    message=response_data.get("message", "Validation failed"),
                    status_code=response.status_code,
                    response=response_data,
                )
            elif response.status_code == 404:
                raise NotFoundError(
                    message=response_data.get("message", "Resource not found"),
                    status_code=response.status_code,
                    response=response_data,
                )
            elif response.status_code == 410:
                raise ExpiredError(
                    message=response_data.get("message", "Resource expired"),
                    status_code=response.status_code,
                    response=response_data,
                )
            elif response.status_code == 429:
                raise RateLimitError(
                    message=response_data.get("message", "Rate limit exceeded"),
                    status_code=response.status_code,
                    response=response_data,
                )
            elif response.status_code >= 400:
                raise ZeroProofError(
                    message=response_data.get(
                        "message", f"Request failed with status {response.status_code}"
                    ),
                    status_code=response.status_code,
                    response=response_data,
                )

            return response_data

        except requests.exceptions.Timeout:
            raise ZeroProofError("Request timed out")
        except requests.exceptions.ConnectionError:
            raise ZeroProofError("Failed to connect to API")
        except requests.exceptions.RequestException as e:
            raise ZeroProofError(f"Request failed: {str(e)}")

    def send_encrypted(
        self,
        to_agent_id: str,
        message: Any,
        ttl_minutes: int = 60,
    ) -> EncryptedMessage:
        """
        Send an encrypted message to another agent.

        Args:
            to_agent_id: Target agent identifier
            message: Message content (string, dict, or JSON-serializable data)
            ttl_minutes: Time-to-live in minutes (default: 60, max: 1440)

        Returns:
            EncryptedMessage: Object containing message_id, expires_at, etc.

        Raises:
            ZeroProofError: If the request fails

        Example:
            >>> result = client.send_encrypted(
            ...     to_agent_id="agent_456",
            ...     message={"order_id": "12345", "status": "shipped"},
            ...     ttl_minutes=30
            ... )
            >>> print(result.message_id)
        """
        data = {
            "to_agent_id": to_agent_id,
            "message": message,
            "ttl_minutes": ttl_minutes,
        }

        response = self._make_request("POST", "/encryption/send", data=data)
        return EncryptedMessage.from_dict(response)

    def receive_encrypted(self, message_id: str) -> DecryptedMessage:
        """
        Receive and decrypt an encrypted message.

        Args:
            message_id: The message ID from send_encrypted()

        Returns:
            DecryptedMessage: Object with decrypted message and metadata

        Raises:
            NotFoundError: If message not found
            ExpiredError: If message has expired
            ZeroProofError: If decryption fails

        Example:
            >>> message = client.receive_encrypted(message_id="msg_abc123...")
            >>> print(message.message)
            >>> print(f"Read {message.read_count} times")
        """
        data = {"message_id": message_id}

        response = self._make_request("POST", "/encryption/receive", data=data)
        return DecryptedMessage.from_dict(response)

    def report_transaction(
        self,
        agent_id: str,
        transaction_id: str,
        outcome: str,
        amount: float = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReputationReport:
        """
        Report a transaction outcome for reputation tracking.

        Args:
            agent_id: Unique identifier for the agent
            transaction_id: Unique transaction identifier
            outcome: Transaction outcome ("success", "disputed", or "failed")
            amount: Transaction amount in USD (default: 0)
            metadata: Additional transaction context (optional)

        Returns:
            ReputationReport: Updated reputation information

        Raises:
            ZeroProofError: If the request fails

        Example:
            >>> report = client.report_transaction(
            ...     agent_id="vendor_123",
            ...     transaction_id="txn_001",
            ...     outcome="success",
            ...     amount=299.99,
            ...     metadata={"order_id": "ORD-12345"}
            ... )
            >>> print(f"New trust score: {report.new_trust_score}")
        """
        data = {
            "agent_id": agent_id,
            "transaction_id": transaction_id,
            "outcome": outcome,
            "amount": amount,
        }
        if metadata:
            data["metadata"] = metadata

        response = self._make_request("POST", "/reputation/report-transaction", data=data)
        return ReputationReport.from_dict(response)

    def check_reputation(
        self,
        agent_id: str,
        include_history: bool = False,
    ) -> ReputationScore:
        """
        Check an agent's reputation and trust score.

        Args:
            agent_id: Unique identifier for the agent to check
            include_history: Include last 50 transactions (default: False)

        Returns:
            ReputationScore: Agent's reputation information

        Raises:
            ZeroProofError: If the request fails

        Example:
            >>> reputation = client.check_reputation(agent_id="vendor_123")
            >>> print(f"Trust score: {reputation.trust_score}")
            >>> print(f"Tier: {reputation.reputation_tier}")
            >>> if reputation.trust_score < 0.7:
            ...     print("Low trust - require escrow")
        """
        data = {
            "agent_id": agent_id,
            "include_history": include_history,
        }

        response = self._make_request("POST", "/reputation/check", data=data)
        return ReputationScore.from_dict(response)

    def create_approval_workflow(
        self,
        transaction_id: str,
        amount: float,
        required_approvers: list,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApprovalWorkflow:
        """
        Create a multi-agent approval workflow.

        Args:
            transaction_id: Unique transaction identifier
            amount: Transaction amount in USD
            required_approvers: List of approvers with agent_id and optional timeout_minutes
            metadata: Additional workflow context (optional)

        Returns:
            ApprovalWorkflow: Created workflow information

        Raises:
            ZeroProofError: If the request fails

        Example:
            >>> workflow = client.create_approval_workflow(
            ...     transaction_id="txn_12345",
            ...     amount=5000.00,
            ...     required_approvers=[
            ...         {"agent_id": "budget_agent", "timeout_minutes": 30},
            ...         {"agent_id": "compliance_agent", "timeout_minutes": 60}
            ...     ],
            ...     metadata={"vendor": "TechCorp", "category": "software"}
            ... )
            >>> print(f"Workflow created: {workflow.workflow_id}")
        """
        data = {
            "transaction_id": transaction_id,
            "amount": amount,
            "required_approvers": required_approvers,
        }
        if metadata:
            data["metadata"] = metadata

        response = self._make_request("POST", "/workflows/approval/create", data=data)
        return ApprovalWorkflow.from_dict(response)

    def approve_workflow(
        self,
        workflow_id: str,
        decision: str,
        reason: Optional[str] = None,
    ) -> WorkflowStatus:
        """
        Approve or reject an approval workflow.

        Args:
            workflow_id: The workflow ID to approve/reject
            decision: Either "approved" or "rejected"
            reason: Optional reason for the decision

        Returns:
            WorkflowStatus: Updated workflow status

        Raises:
            ZeroProofError: If the request fails

        Example:
            >>> result = client.approve_workflow(
            ...     workflow_id="wf_abc123",
            ...     decision="approved",
            ...     reason="Within budget limits"
            ... )
            >>> print(f"Status: {result.status}")
            >>> print(f"Pending: {len(result.pending_approvers)} approvers")
        """
        data = {
            "workflow_id": workflow_id,
            "decision": decision,
        }
        if reason:
            data["reason"] = reason

        response = self._make_request("POST", "/workflows/approval/approve", data=data)
        return WorkflowStatus.from_dict(response)

    def get_workflow_status(self, workflow_id: str) -> WorkflowStatus:
        """
        Get the status of an approval workflow.

        Args:
            workflow_id: The workflow ID to check

        Returns:
            WorkflowStatus: Current workflow status

        Raises:
            ZeroProofError: If the request fails

        Example:
            >>> status = client.get_workflow_status(workflow_id="wf_abc123")
            >>> print(f"Status: {status.status}")
            >>> print(f"Approvals: {len(status.approvals)}/{len(status.required_approvers)}")
            >>> for approval in status.approvals:
            ...     print(f"  {approval.agent_id}: {approval.decision}")
        """
        response = self._make_request("GET", f"/workflows/approval/{workflow_id}")
        return WorkflowStatus.from_dict(response)
