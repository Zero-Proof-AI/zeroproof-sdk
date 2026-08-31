"""
Exceptions for the ZeroProof SDK.
"""

from typing import Optional, Dict, Any


class ZeroProofError(Exception):
    """Base exception for all ZeroProof SDK errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a ZeroProofError.

        Args:
            message: Error message
            status_code: HTTP status code if applicable
            response: Full API response if available
        """
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class AuthenticationError(ZeroProofError):
    """Raised when API authentication fails."""

    pass


class ValidationError(ZeroProofError):
    """Raised when request validation fails."""

    pass


class NotFoundError(ZeroProofError):
    """Raised when a resource is not found."""

    pass


class ExpiredError(ZeroProofError):
    """Raised when a message or session has expired."""

    pass


class RateLimitError(ZeroProofError):
    """Raised when rate limit is exceeded."""

    pass
