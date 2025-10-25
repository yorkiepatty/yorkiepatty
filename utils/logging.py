"""
HIPAA-compliant logging setup for Derek's Learning Engine
"""
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """
    Setup HIPAA-compliant logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Get log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter with HIPAA-compliant format (no PII)
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler (for development)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Rotating file handler for general logs (10MB max, 5 backups)
    general_log_file = log_path / "derek_learning_engine.log"
    file_handler = RotatingFileHandler(
        general_log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)
    
    # Separate handler for audit logs (HIPAA compliance)
    audit_log_file = log_path / f"derek_audit_{datetime.now().strftime('%Y%m%d')}.log"
    audit_handler = logging.FileHandler(audit_log_file, mode='a')
    audit_handler.setFormatter(logging.Formatter(
        '%(asctime)s [AUDIT] %(name)s - %(message)s'
    ))
    audit_handler.setLevel(logging.INFO)
    
    # Create audit logger
    audit_logger = logging.getLogger('derek.audit')
    audit_logger.addHandler(audit_handler)
    audit_logger.setLevel(logging.INFO)
    
    # Setup specific loggers
    setup_api_logger(log_path, formatter)
    setup_learning_logger(log_path, formatter)
    
    # Log the initialization
    logger = logging.getLogger(__name__)
    logger.info("HIPAA-compliant logging initialized")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log directory: {log_path.absolute()}")
    
    return root_logger


def setup_api_logger(log_path: Path, formatter):
    """Setup API-specific logger"""
    api_log_file = log_path / "derek_api.log"
    api_handler = logging.FileHandler(api_log_file, mode='a')
    api_handler.setFormatter(formatter)
    
    api_logger = logging.getLogger('derek.api')
    api_logger.addHandler(api_handler)
    api_logger.setLevel(logging.INFO)


def setup_learning_logger(log_path: Path, formatter):
    """Setup learning-specific logger"""
    learning_log_file = log_path / "derek_learning.log"
    learning_handler = logging.FileHandler(learning_log_file, mode='a')
    learning_handler.setFormatter(formatter)
    
    learning_logger = logging.getLogger('derek.learning')
    learning_logger.addHandler(learning_handler)
    learning_logger.setLevel(logging.INFO)


def get_audit_logger():
    """Get the audit logger for HIPAA compliance"""
    return logging.getLogger('derek.audit')


def get_api_logger():
    """Get the API logger"""
    return logging.getLogger('derek.api')


def get_learning_logger():
    """Get the learning logger"""
    return logging.getLogger('derek.learning')


# Utility functions for secure logging
def sanitize_log_message(message: str) -> str:
    """
    Sanitize log messages to remove potential PII for HIPAA compliance
    
    Args:
        message: Original log message
        
    Returns:
        Sanitized message safe for logging
    """
    # Remove email addresses
    message = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[REDACTED_EMAIL]', message)
    
    # Remove phone numbers (multiple patterns)
    message = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]', message)
    message = re.sub(r'\(\d{3}\)\s?\d{3}[-.]?\d{4}', '[REDACTED_PHONE]', message)
    
    # Remove SSN patterns
    message = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', message)
    message = re.sub(r'\b\d{9}\b', '[REDACTED_SSN]', message)
    
    # Remove credit card patterns (basic)
    message = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[REDACTED_CC]', message)
    
    # Remove IP addresses
    message = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[REDACTED_IP]', message)
    
    # Remove potential API keys (long alphanumeric strings)
    message = re.sub(r'\b[A-Za-z0-9_-]{32,}\b', '[REDACTED_KEY]', message)
    
    return message


def create_sanitized_logger(logger_name: str):
    """
    Create a logger with automatic PII sanitization
    
    Args:
        logger_name: Name of the logger
        
    Returns:
        Logger with sanitized output
    """
    logger = logging.getLogger(logger_name)
    
    # Wrap logging methods to sanitize messages
    original_info = logger.info
    original_warning = logger.warning
    original_error = logger.error
    original_debug = logger.debug
    
    logger.info = lambda msg, *args, **kwargs: original_info(sanitize_log_message(str(msg)), *args, **kwargs)
    logger.warning = lambda msg, *args, **kwargs: original_warning(sanitize_log_message(str(msg)), *args, **kwargs)
    logger.error = lambda msg, *args, **kwargs: original_error(sanitize_log_message(str(msg)), *args, **kwargs)
    logger.debug = lambda msg, *args, **kwargs: original_debug(sanitize_log_message(str(msg)), *args, **kwargs)
    
    return logger


def log_user_action(action: str, user_id: str = None, details: dict = None):
    """
    Log user actions for audit purposes (HIPAA compliant)
    
    Args:
        action: Description of the action
        user_id: Anonymized user identifier (no PII)
        details: Additional non-PII details
    """
    audit_logger = get_audit_logger()
    
    log_entry = {
        'action': sanitize_log_message(action),
        'user_id': user_id or 'anonymous',
        'timestamp': datetime.now().isoformat()
    }
    
    if details:
        # Ensure details don't contain PII
        sanitized_details = {k: sanitize_log_message(str(v)) for k, v in details.items()}
        log_entry['details'] = sanitized_details
    
    audit_logger.info(f"USER_ACTION: {log_entry}")


if __name__ == "__main__":
    # Test the logging setup
    setup_logging("DEBUG")
    
    logger = logging.getLogger("test")
    logger.info("Testing logging setup")
    
    # Test audit logging
    log_user_action("test_action", "user123", {"test": "data"})