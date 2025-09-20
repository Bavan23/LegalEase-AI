"""
Enhanced Legal Document Demystifier - Security and Privacy Utils
Production-ready security, privacy, and data protection utilities
"""

import re
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)

class SecurityManager:
    """Comprehensive security and privacy management"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.encryption_key = self._generate_or_load_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # PII patterns for different privacy levels
        self.pii_patterns = {
            'standard': self._get_standard_pii_patterns(),
            'high': self._get_high_privacy_patterns()
        }
    
    def _generate_or_load_key(self) -> bytes:
        """Generate or load encryption key"""
        key_file = Path(self.config.get('key_file', '.encryption_key'))
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            # Set secure permissions
            key_file.chmod(0o600)
            return key
    
    def encrypt_text(self, text: str) -> str:
        """Encrypt text data"""
        try:
            encrypted = self.cipher_suite.encrypt(text.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise
    
    def decrypt_text(self, encrypted_text: str) -> str:
        """Decrypt text data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_text.encode())
            decrypted = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise
    
    def hash_document(self, file_path: Path) -> str:
        """Generate secure hash of document for privacy tracking"""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Document hashing failed: {str(e)}")
            raise
    
    def redact_sensitive_info(self, text: str, privacy_level: str = 'standard') -> str:
        """
        Redact sensitive information based on privacy level
        
        Args:
            text: Text to redact
            privacy_level: 'standard' or 'high'
            
        Returns:
            Redacted text
        """
        patterns = self.pii_patterns.get(privacy_level, self.pii_patterns['standard'])
        
        redacted_text = text
        for pattern_name, pattern_data in patterns.items():
            pattern = pattern_data['pattern']
            replacement = pattern_data['replacement']
            
            try:
                redacted_text = re.sub(pattern, replacement, redacted_text, flags=re.I)
            except re.error as e:
                logger.warning(f"Regex error in pattern {pattern_name}: {str(e)}")
                continue
        
        return redacted_text
    
    def _get_standard_pii_patterns(self) -> Dict:
        """Standard privacy patterns"""
        return {
            'credit_card': {
                'pattern': r'\b(\d{4})\s*(\d{4})\s*(\d{4})\s*(\d{4})\b',
                'replacement': r'\1 XXXX XXXX XXXX'
            },
            'debit_card': {
                'pattern': r'\b(\d{4})\s*(\d{4})\s*(\d{4})\b',
                'replacement': r'\1 XXXX XXXX'
            },
            'phone_number': {
                'pattern': r'\b(\d{3})\s*(\d{3})\s*(\d{4})\b',
                'replacement': r'\1 XXX XXXX'
            },
            'email': {
                'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'replacement': '[REDACTED_EMAIL]'
            },
            'indian_phone': {
                'pattern': r'\b(\+91\s*)?(\d{2})\s*(\d{4})\s*(\d{4})\b',
                'replacement': r'\1\2 XXXX XXXX'
            },
            'pan_card': {
                'pattern': r'\b[A-Z]{5}\d{4}[A-Z]\b',
                'replacement': '[REDACTED_PAN]'
            },
            'aadhaar': {
                'pattern': r'\b(\d{4})\s*(\d{4})\s*(\d{4})\b',
                'replacement': r'\1 XXXX XXXX'
            }
        }
    
    def _get_high_privacy_patterns(self) -> Dict:
        """High privacy level patterns (more aggressive)"""
        standard_patterns = self._get_standard_pii_patterns()
        
        high_privacy_patterns = {
            'any_number_sequence': {
                'pattern': r'\b\d{6,}\b',
                'replacement': '[REDACTED_NUMBER]'
            },
            'currency_amounts': {
                'pattern': r'[₹$€£]\s*[\d,]+(?:\.\d{2})?',
                'replacement': '[REDACTED_AMOUNT]'
            },
            'id_patterns': {
                'pattern': r'\b[A-Z]{2,8}\d{4,8}\b',
                'replacement': '[REDACTED_ID]'
            },
            'dates': {
                'pattern': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                'replacement': '[REDACTED_DATE]'
            },
            'names_pattern': {
                'pattern': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
                'replacement': '[REDACTED_NAME]'
            }
        }
        
        # Combine standard and high privacy patterns
        combined_patterns = {**standard_patterns, **high_privacy_patterns}
        return combined_patterns
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for security"""
        # Remove path traversal attempts
        filename = filename.replace('..', '').replace('/', '').replace('\\', '')
        
        # Keep only alphanumeric, dots, hyphens, and underscores
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
            max_name_length = 255 - len(ext) - 1 if ext else 255
            sanitized = name[:max_name_length] + ('.' + ext if ext else '')
        
        return sanitized
    
    def generate_session_token(self) -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
    
    def validate_input(self, input_text: str, max_length: int = 10000) -> bool:
        """Basic input validation"""
        if not isinstance(input_text, str):
            return False
        
        if len(input_text) > max_length:
            return False
        
        # Check for potential injection patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'eval\s*\(',
            r'exec\s*\(',
            r'import\s+os',
            r'__import__'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, input_text, re.I | re.S):
                logger.warning(f"Potentially dangerous input detected: {pattern}")
                return False
        
        return True

class PrivacyManager:
    """Privacy compliance and data protection manager"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.data_retention_days = self.config.get('data_retention_days', 30)
        self.audit_log = []
    
    def log_data_access(self, action: str, data_type: str, user_id: str = None):
        """Log data access for audit purposes"""
        import datetime
        
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'action': action,
            'data_type': data_type,
            'user_id': user_id or 'anonymous',
            'session_id': getattr(self, 'current_session_id', 'unknown')
        }
        
        self.audit_log.append(log_entry)
        logger.info(f"Data access logged: {action} on {data_type}")
    
    def check_consent(self, consent_type: str) -> bool:
        """Check if user has given consent for data processing"""
        # In production, this would check against a consent database
        consents = self.config.get('user_consents', {})
        return consents.get(consent_type, False)
    
    def anonymize_results(self, results: Dict) -> Dict:
        """Anonymize results for privacy compliance"""
        anonymized = results.copy()
        
        # Remove or hash identifying information
        if 'file_name' in anonymized:
            anonymized['file_name'] = self._hash_identifier(anonymized['file_name'])
        
        if 'metadata' in anonymized:
            metadata = anonymized['metadata'].copy()
            if 'file_hash' in metadata:
                metadata['file_hash'] = metadata['file_hash'][:8] + '...'
            anonymized['metadata'] = metadata
        
        return anonymized
    
    def _hash_identifier(self, identifier: str) -> str:
        """Hash identifier for anonymization"""
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    def get_privacy_notice(self, language: str = 'english') -> str:
        """Get privacy notice in specified language"""
        notices = {
            'english': """
            Privacy Notice: Your document is processed locally and temporarily. 
            We do not store your documents permanently. Personal information is 
            automatically redacted based on your privacy settings. This analysis 
            is for informational purposes only and does not constitute legal advice.
            """,
            'hindi': """
            गोपनीयता सूचना: आपका दस्तावेज़ स्थानीय रूप से और अस्थायी रूप से संसाधित किया जाता है। 
            हम आपके दस्तावेज़ों को स्थायी रूप से संग्रहीत नहीं करते। व्यक्तिगत जानकारी आपकी 
            गोपनीयता सेटिंग्स के आधार पर स्वचालित रूप से हटा दी जाती है।
            """
        }
        
        return notices.get(language, notices['english']).strip()

# Utility functions for backward compatibility
def sanitize_text(text: str) -> str:
    """Sanitize text for security"""
    security_manager = SecurityManager()
    return security_manager.validate_input(text)

def encrypt_file(file_path: Path, password: str = None) -> Path:
    """Encrypt file (simplified implementation)"""
    security_manager = SecurityManager()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    encrypted_content = security_manager.encrypt_text(content)
    
    encrypted_path = file_path.with_suffix(file_path.suffix + '.encrypted')
    with open(encrypted_path, 'w', encoding='utf-8') as f:
        f.write(encrypted_content)
    
    return encrypted_path

def redact_sensitive_info(text: str, privacy_level: str = 'standard') -> str:
    """Redact sensitive information"""
    security_manager = SecurityManager()
    return security_manager.redact_sensitive_info(text, privacy_level)