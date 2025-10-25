"""
Encryption utilities for HIPAA-compliant data storage
"""
from cryptography.fernet import Fernet
import os
import logging

logger = logging.getLogger(__name__)

class Encryptor:
    """HIPAA-compliant encryption utility using Fernet symmetric encryption"""
    
    def __init__(self):
        """Initialize encryptor with key from environment or generate new one"""
        key = os.getenv("ENCRYPTION_KEY")
        
        if not key:
            # Generate new key if none exists
            key = Fernet.generate_key()
            logger.warning("⚠️ No ENCRYPTION_KEY found in environment. Generated new key.")
            logger.warning("Please add this to your .env file:")
            logger.warning(f"ENCRYPTION_KEY={key.decode()}")
        else:
            # Convert string key to bytes if needed
            if isinstance(key, str):
                key = key.encode()
        
        try:
            self.cipher = Fernet(key)
            logger.info("✅ Encryptor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize encryptor: {e}")
            raise

    def encrypt(self, data: str) -> bytes:
        """
        Encrypt string data to bytes
        
        Args:
            data: String data to encrypt
            
        Returns:
            Encrypted data as bytes
        """
        try:
            return self.cipher.encrypt(data.encode('utf-8'))
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt(self, data: bytes) -> str:
        """
        Decrypt bytes data to string
        
        Args:
            data: Encrypted bytes data
            
        Returns:
            Decrypted string data
        """
        try:
            return self.cipher.decrypt(data).decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_json(self, data: dict) -> bytes:
        """
        Encrypt JSON-serializable data
        
        Args:
            data: Dictionary or JSON-serializable object
            
        Returns:
            Encrypted data as bytes
        """
        import json
        json_string = json.dumps(data, indent=2)
        return self.encrypt(json_string)
    
    def decrypt_json(self, data: bytes) -> dict:
        """
        Decrypt and parse JSON data
        
        Args:
            data: Encrypted bytes data
            
        Returns:
            Decrypted dictionary/object
        """
        import json
        json_string = self.decrypt(data)
        return json.loads(json_string)


# Global encryptor instance (lazy initialization)
_encryptor = None

def get_encryptor() -> Encryptor:
    """Get global encryptor instance"""
    global _encryptor
    if _encryptor is None:
        _encryptor = Encryptor()
    return _encryptor


def encrypt_file(file_path: str, data: dict):
    """
    Encrypt and save data to file
    
    Args:
        file_path: Path to save encrypted file
        data: Data to encrypt and save
    """
    encryptor = get_encryptor()
    encrypted_data = encryptor.encrypt_json(data)
    
    with open(file_path, 'wb') as f:
        f.write(encrypted_data)
    
    logger.info(f"💾 Encrypted data saved to {file_path}")


def decrypt_file(file_path: str) -> dict:
    """
    Load and decrypt data from file
    
    Args:
        file_path: Path to encrypted file
        
    Returns:
        Decrypted data
    """
    if not os.path.exists(file_path):
        logger.warning(f"Encrypted file not found: {file_path}")
        return {}
    
    encryptor = get_encryptor()
    
    with open(file_path, 'rb') as f:
        encrypted_data = f.read()
    
    try:
        decrypted_data = encryptor.decrypt_json(encrypted_data)
        logger.info(f"🔓 Decrypted data loaded from {file_path}")
        return decrypted_data
    except Exception as e:
        logger.error(f"Failed to decrypt {file_path}: {e}")
        return {}