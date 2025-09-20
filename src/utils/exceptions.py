# utils/exceptions.py
"""
Custom exception classes for LegalEase AI
"""

class DocumentProcessingError(Exception):
    """Raised when document processing fails (e.g., invalid PDF, parsing issues)."""
    def __init__(self, message="Error occurred while processing the document"):
        self.message = message
        super().__init__(self.message)


class SecurityError(Exception):
    """Raised when a security validation fails (e.g., unsafe file, malicious input)."""
    def __init__(self, message="Security validation failed"):
        self.message = message
        super().__init__(self.message)
