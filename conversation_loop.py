"""
Conversation Loop for Sunny
The Christman AI Project - Speech-to-Action Core
------------------------------------------------
Listens for speech, processes it via Sunny’s brain, and replies with speech.
"""

import os
import sys
import time
import logging

# -------------------------------------------------------------
# Ensure project root is in path for imports
# -------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------------------
# Imports from the Sunny system
# -------------------------------------------------------------
from brain import sunny
from tts_bridge import speak_response as speak
from executor import execute_task
from memory_engine import MemoryEngine
from speech_recognition_engine import SpeechRecognitionEngine

# -------------------------------------------------------------
# Initialize components
# -------------------------------------------------------------
logger = logging.getLogger("conversation_loop")
memory = MemoryEngine()
engine = SpeechRecognitionEngine()  # main speech engine

# -------------------------------------------------------------
# Handle recognized speech
# -------------------------------------------------------------
def handle_recognition(text, confidence, meta=None):
    """Callback executed when speech is recognized."""
    if not text or confidence < 0.2:
        print("❌ No clear speech recognized.")
        speak("I didn’t catch that clearly. Could you please repeat it?")
        return

    print(f"\n👤 You: {text}  (confidence: {confidence:.2f})")

    try:
        # Load Sunny's context memory
        memory.load()
        context = memory.get_context()

        # Get Sunny’s thought process
        response = sunny.think(text)
        intent = response.get("intent", "general")

        # Execute the intent / generate reply
        reply = execute_task(text, intent, context)

        print(f"🤖 Sunny: {reply}")
        speak(reply)

    except Exception as e:
        logger.error(f"Error in handle_recognition: {e}", exc_info=True)
        speak("I encountered an internal issue while processing that.")

# -------------------------------------------------------------
# Main conversation loop
# -------------------------------------------------------------
def run_conversation():
    """Continuously listen for speech and process commands."""
    print("🎙️ Sunny is now live and listening...")
    try:
        # Start listening via the engine
        engine.start_listening(callback=handle_recognition)

        # Keep alive until interrupted
        while True:
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n🛑 Conversation manually stopped.")
        engine.stop_listening()
        speak("Conversation ended. Goodbye.")

    except Exception as e:
        logger.error(f"Conversation loop crashed: {e}", exc_info=True)
        speak("There was an error in the conversation system.")

# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    run_conversation()


# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
