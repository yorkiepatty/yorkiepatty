"""
Utilities for Derek's Learning Engine
"""
from .logging import setup_logging, get_audit_logger, get_api_logger, get_learning_logger, log_user_action, sanitize_log_message, create_sanitized_logger

try:
    from .encryption import Encryptor, get_encryptor, encrypt_file, decrypt_file
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

__all__ = [
    'setup_logging',
    'get_audit_logger', 
    'get_api_logger',
    'get_learning_logger',
    'log_user_action',
    'sanitize_log_message',
    'create_sanitized_logger'
]

if ENCRYPTION_AVAILABLE:
    __all__.extend([
        'Encryptor',
        'get_encryptor',
        'encrypt_file',
        'decrypt_file'
    ])