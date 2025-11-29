"""
AWS Polly TTS Service for Sonny
Enhanced text-to-speech using AWS Polly neural voices
"""

import boto3
import os
import tempfile
import uuid
import subprocess
import platform
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# AWS Polly Neural Voices optimized for Sonny
SONNY_POLLY_VOICES = {
    "matthew": {
        "VoiceId": "Matthew",
        "Engine": "neural",
        "LanguageCode": "en-US",
        "Gender": "Male",
        "Description": "Friendly, warm male voice - Sonny's default"
    },
    "joanna": {
        "VoiceId": "Joanna", 
        "Engine": "neural",
        "LanguageCode": "en-US",
        "Gender": "Female",
        "Description": "Professional, clear female voice"
    },
    "stephen": {
        "VoiceId": "Stephen",
        "Engine": "neural", 
        "LanguageCode": "en-US",
        "Gender": "Male",
        "Description": "Calm, measured male voice"
    },
    "ruth": {
        "VoiceId": "Ruth",
        "Engine": "neural",
        "LanguageCode": "en-US", 
        "Gender": "Female",
        "Description": "Warm, encouraging female voice"
    },
    "kevin": {
        "VoiceId": "Kevin",
        "Engine": "neural",
        "LanguageCode": "en-US",
        "Gender": "Male", 
        "Description": "Conversational, approachable male voice"
    },
    "gregory": {
        "VoiceId": "Gregory",
        "Engine": "neural",
        "LanguageCode": "en-US",
        "Gender": "Male",
        "Description": "Authoritative, confident male voice"
    }
}

class SonnyPollyTTS:
    """AWS Polly TTS service specifically configured for Sonny"""
    
    def __init__(self, default_voice="matthew", region_name="us-east-1"):
        """
        Initialize Sonny's Polly TTS service
        
        Args:
            default_voice: Default voice ID from SONNY_POLLY_VOICES
            region_name: AWS region for Polly service
        """
        self.default_voice = default_voice
        self.region_name = region_name
        self.polly_client = None
        self.is_available = False
        
        self._initialize_polly()
    
    def _initialize_polly(self):
        """Initialize AWS Polly client with error handling"""
        try:
            self.polly_client = boto3.client('polly', region_name=self.region_name)
            
            # Test connection with a simple call
            self.polly_client.describe_voices(MaxItems=1)
            
            self.is_available = True
            logger.info(f"✅ AWS Polly initialized successfully in {self.region_name}")
            logger.info(f"🎙️ Default voice: {self.default_voice} ({SONNY_POLLY_VOICES[self.default_voice]['Description']})")
            
        except Exception as e:
            self.is_available = False
            logger.error(f"❌ Failed to initialize AWS Polly: {e}")
            
            # Provide helpful error messages
            error_str = str(e)
            if "NoCredentialsError" in error_str:
                logger.error("💡 AWS credentials not found. Please configure AWS CLI or set environment variables.")
            elif "UnauthorizedOperation" in error_str:
                logger.error("💡 AWS credentials lack Polly permissions. Add 'polly:*' permissions to your IAM user/role.")
            elif "InvalidRegion" in error_str:
                logger.error(f"💡 Invalid AWS region: {self.region_name}. Try 'us-east-1' or 'us-west-2'.")
    
    def speak(self, text: str, voice_id: Optional[str] = None, play_audio: bool = True) -> Optional[str]:
        """
        Convert text to speech using AWS Polly
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID (defaults to default_voice)
            play_audio: Whether to play the audio immediately
            
        Returns:
            Path to generated audio file, or None if failed
        """
        if not self.is_available:
            logger.error("❌ AWS Polly not available")
            return None
        
        voice_id = voice_id or self.default_voice
        
        if voice_id not in SONNY_POLLY_VOICES:
            logger.warning(f"⚠️ Unknown voice '{voice_id}', using default '{self.default_voice}'")
            voice_id = self.default_voice
        
        try:
            voice_config = SONNY_POLLY_VOICES[voice_id]
            
            # Synthesize speech
            response = self.polly_client.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId=voice_config['VoiceId'],
                Engine=voice_config['Engine'],
                LanguageCode=voice_config.get('LanguageCode', 'en-US')
            )
            
            # Save to temporary file
            temp_dir = tempfile.gettempdir()
            audio_file = os.path.join(temp_dir, f"sonny_polly_{uuid.uuid4()}.mp3")
            
            with open(audio_file, 'wb') as f:
                f.write(response['AudioStream'].read())
            
            logger.info(f"🎵 Audio generated: {Path(audio_file).name}")
            
            # Play audio if requested
            if play_audio:
                self._play_audio(audio_file)
            
            return audio_file
            
        except Exception as e:
            logger.error(f"❌ Failed to synthesize speech: {e}")
            return None
    
    def _play_audio(self, audio_file: str):
        """Play audio file using system-appropriate method"""
        try:
            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.run(["afplay", audio_file], check=True)
            elif system == "Linux":
                # Try multiple Linux audio players
                players = ["aplay", "paplay", "mpg123", "mpv"]
                for player in players:
                    try:
                        subprocess.run([player, audio_file], check=True, timeout=30)
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                else:
                    logger.warning("⚠️ No suitable audio player found on Linux")
            elif system == "Windows":
                import winsound
                winsound.PlaySound(audio_file, winsound.SND_FILENAME)
            else:
                logger.warning(f"⚠️ Audio playback not supported on {system}")
        except Exception as e:
            logger.error(f"⚠️ Audio playback failed: {e}")
        finally:
            # Clean up temporary file after a delay
            try:
                import threading
                import time
                def cleanup():
                    time.sleep(2)  # Wait for playback to finish
                    try:
                        os.remove(audio_file)
                    except:
                        pass
                
                threading.Thread(target=cleanup, daemon=True).start()
            except:
                pass
    
    def get_available_voices(self) -> Dict[str, Dict[str, Any]]:
        """Get available Sonny voices"""
        return SONNY_POLLY_VOICES.copy()
    
    def test_voice(self, voice_id: str = None) -> bool:
        """Test a specific voice"""
        voice_id = voice_id or self.default_voice
        test_text = f"Hello! I'm Sonny, testing the {voice_id} voice. How do I sound?"
        
        result = self.speak(test_text, voice_id=voice_id, play_audio=True)
        return result is not None
    
    def set_default_voice(self, voice_id: str):
        """Change the default voice"""
        if voice_id in SONNY_POLLY_VOICES:
            old_voice = self.default_voice
            self.default_voice = voice_id
            logger.info(f"🔄 Default voice changed from '{old_voice}' to '{voice_id}'")
        else:
            logger.error(f"❌ Unknown voice '{voice_id}'. Available: {list(SONNY_POLLY_VOICES.keys())}")

# Singleton instance for easy import
sonny_polly_tts = SonnyPollyTTS()

def speak_with_polly(text: str, voice_id: str = "matthew", play_audio: bool = True) -> Optional[str]:
    """Convenience function for text-to-speech"""
    return sonny_polly_tts.speak(text, voice_id=voice_id, play_audio=play_audio)

def test_polly_connection() -> bool:
    """Test if AWS Polly is working"""
    return sonny_polly_tts.test_voice()

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================