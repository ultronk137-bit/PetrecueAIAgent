"""
Custom exception hierarchy for the application.
Provides specific exceptions for different failure scenarios.
"""
from typing import Any, Optional


class PetRescueException(Exception):
    """Base exception for all application-specific errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationException(PetRescueException):
    """Exception raised for validation errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)


class ImageUploadException(PetRescueException):
    """Exception raised when image upload fails."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class AIException(PetRescueException):
    """Base exception for all AI provider errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class GeminiException(AIException):
    """Exception raised when Gemini API fails."""


class OpenAIException(AIException):
    """Exception raised when OpenAI API fails."""


class FirestoreException(PetRescueException):
    """Exception raised when Firestore operations fail."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class LocationServiceException(PetRescueException):
    """Exception raised when location service fails."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class AgentExecutionException(PetRescueException):
    """Exception raised when agent execution fails."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)
