"""
Enhanced Legal Document Demystifier - Validators and Exception Handlers
Production-ready validation and error handling utilities
"""

import re
import mimetypes
from pathlib import Path
from typing import List, Optional, Union

class LegalDocumentError(Exception):
    """Base exception for legal document processing"""
    pass

class DocumentProcessingError(LegalDocumentError):
    """Exception raised during document processing"""
    pass

class SecurityError(LegalDocumentError):
    """Exception raised for security violations"""
    pass

class ValidationError(LegalDocumentError):
    """Exception raised for validation failures"""
    pass

class FileValidator:
    """File validation utilities"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.max_file_size = self.config.get('max_file_size', 25 * 1024 * 1024)  # 25MB
        self.allowed_extensions = self.config.get('allowed_extensions', ['.pdf'])
        self.allowed_mime_types = self.config.get('allowed_mime_types', ['application/pdf'])
    
    def validate_file_size(self, file_path: Union[str, Path], max_size: int = None) -> bool:
        """Validate file size"""
        if max_size is None:
            max_size = self.max_file_size
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return False
            
            file_size = file_path.stat().st_size
            return file_size <= max_size
            
        except Exception as e:
            return False
    
    def validate_file_type(self, filename: str, allowed_extensions: List[str] = None) -> bool:
        """Validate file type by extension"""
        if allowed_extensions is None:
            allowed_extensions = self.allowed_extensions
        
        try:
            file_path = Path(filename)
            file_extension = file_path.suffix.lower()
            
            return file_extension in allowed_extensions
            
        except Exception as e:
            return False
    
    def validate_mime_type(self, file_path: Union[str, Path]) -> bool:
        """Validate MIME type"""
        try:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return mime_type in self.allowed_mime_types
        except Exception:
            return False
    
    def validate_filename(self, filename: str) -> bool:
        """Validate filename for security"""
        try:
            # Check for path traversal attempts
            if '..' in filename or '/' in filename or '\\' in filename:
                return False
            
            # Check for suspicious patterns
            suspicious_patterns = [
                r'[<>:"|?*]',  # Windows invalid characters
                r'^\.',        # Hidden files
                r'\.exe$|\.bat$|\.cmd$|\.scr$',  # Executable files
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, filename, re.I):
                    return False
            
            # Check length
            if len(filename) > 255:
                return False
            
            return True
        except Exception:
            return False
    
    def comprehensive_validation(self, file_path: Union[str, Path]) -> bool:
        """Run all validations"""
        try:
            file_path = Path(file_path)
            
            # Validate filename
            if not self.validate_filename(file_path.name):
                return False
            
            # Validate file exists and size
            if not self.validate_file_size(file_path):
                return False
            
            # Validate file type
            if not self.validate_file_type(file_path.name):
                return False
            
            # Validate MIME type
            if not self.validate_mime_type(file_path):
                return False
            
            return True
        except Exception:
            return False

class TextValidator:
    """Text content validation utilities"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.max_text_length = self.config.get('max_text_length', 1000000)  # 1MB
        self.min_text_length = self.config.get('min_text_length', 10)
    
    def validate_text_length(self, text: str) -> bool:
        """Validate text length"""
        try:
            return self.min_text_length <= len(text) <= self.max_text_length
        except Exception:
            return False
    
    def validate_text_content(self, text: str) -> bool:
        """Validate text content for suspicious patterns"""
        try:
            # Check for potential code injection
            suspicious_patterns = [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'data:text/html',
                r'eval\s*\(',
                r'exec\s*\(',
                r'__import__',
                r'import\s+os',
                r'subprocess\.',
                r'system\(',
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, text, re.I | re.S):
                    return False
            
            return True
        except Exception:
            return False
    
    def validate_encoding(self, text: str) -> bool:
        """Validate text encoding"""
        try:
            # Try to encode/decode to check for encoding issues
            text.encode('utf-8').decode('utf-8')
            return True
        except UnicodeError:
            return False
        except Exception:
            return False
    
    def comprehensive_text_validation(self, text: str) -> bool:
        """Run all text validations"""
        try:
            return (self.validate_text_length(text) and 
                    self.validate_text_content(text) and 
                    self.validate_encoding(text))
        except Exception:
            return False

class InputValidator:
    """General input validation utilities"""
    
    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        """Validate session ID format"""
        try:
            if not session_id:
                return False
            
            # UUID format validation
            uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            return bool(re.match(uuid_pattern, session_id, re.I))
        except Exception:
            return False
    
    @staticmethod
    def validate_language_code(language: str) -> bool:
        """Validate language code"""
        try:
            supported_languages = [
                'english', 'hindi', 'tamil', 'telugu', 'bengali', 
                'gujarati', 'marathi', 'kannada'
            ]
            return language in supported_languages
        except Exception:
            return False
    
    @staticmethod
    def validate_privacy_mode(privacy_mode: str) -> bool:
        """Validate privacy mode"""
        try:
            allowed_modes = ['standard', 'high']
            return privacy_mode in allowed_modes
        except Exception:
            return False
    
    @staticmethod
    def validate_clause_id(clause_id: str) -> bool:
        """Validate clause ID format"""
        try:
            # Format: p{page}-c{clause_number}
            clause_pattern = r'^p\d+-c\d+$'
            return bool(re.match(clause_pattern, clause_id))
        except Exception:
            return False
    
    @staticmethod
    def validate_question(question: str) -> bool:
        """Validate user question"""
        try:
            if not question or not question.strip():
                return False
            
            if len(question) > 1000:
                return False
            
            # Check for suspicious content
            if re.search(r'[<>{}]', question):
                return False
            
            return True
        except Exception:
            return False

# Convenience functions for backward compatibility
def validate_file_size(file_path: Union[str, Path], max_size: int) -> bool:
    """Validate file size - convenience function"""
    try:
        validator = FileValidator({'max_file_size': max_size})
        return validator.validate_file_size(file_path, max_size)
    except Exception:
        return False

def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
    """Validate file type - convenience function"""
    try:
        validator = FileValidator({'allowed_extensions': allowed_types})
        return validator.validate_file_type(filename, allowed_types)
    except Exception:
        return False

# Error formatting utilities
class ErrorFormatter:
    """Format errors for API responses"""
    
    @staticmethod
    def format_validation_error(error: ValidationError) -> dict:
        """Format validation error for API response"""
        return {
            "error_type": "validation_error",
            "message": str(error),
            "code": "VALIDATION_FAILED"
        }
    
    @staticmethod
    def format_security_error(error: SecurityError) -> dict:
        """Format security error for API response"""
        return {
            "error_type": "security_error",
            "message": "Security validation failed",
            "code": "SECURITY_VIOLATION"
        }
    
    @staticmethod
    def format_processing_error(error: DocumentProcessingError) -> dict:
        """Format processing error for API response"""
        return {
            "error_type": "processing_error",
            "message": str(error),
            "code": "PROCESSING_FAILED"
        }
    
    @staticmethod
    def format_generic_error(error: Exception) -> dict:
        """Format generic error for API response"""
        return {
            "error_type": "internal_error",
            "message": "An internal error occurred",
            "code": "INTERNAL_ERROR"
        }

# Context manager for validation
class ValidationContext:
    """Context manager for comprehensive validation"""
    
    def __init__(self, config: dict = None):
        self.file_validator = FileValidator(config)
        self.text_validator = TextValidator(config)
        self.errors = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.errors:
            error_messages = '; '.join(self.errors)
            raise ValidationError(f"Validation failed: {error_messages}")
    
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """Validate file within context"""
        try:
            return self.file_validator.comprehensive_validation(file_path)
        except ValidationError as e:
            self.errors.append(str(e))
            return False
    
    def validate_text(self, text: str) -> bool:
        """Validate text within context"""
        try:
            return self.text_validator.comprehensive_text_validation(text)
        except ValidationError as e:
            self.errors.append(str(e))
            return False