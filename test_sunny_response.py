#!/usr/bin/env python3
"""
Quick test script to verify Sunny's AI response system
Tests the thinking/reasoning without speech recognition
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Test if we can import and initialize the core thinking system
def test_derek_thinking():
    """Test Sunny's AI response system"""
    
    print("=" * 60)
    print("🧠 Testing Sunny's AI Response System")
    print("=" * 60)
    print()
    
    # Check API keys
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    print("API Status:")
    print(f"  Anthropic Claude: {'✅' if has_anthropic else '❌'}")
    print(f"  OpenAI GPT: {'✅' if has_openai else '❌'}")
    print()
    
    if not (has_anthropic or has_openai):
        print("❌ No AI providers available. Please set API keys in .env")
        return False
    
    # Try to initialize Sunny's thinking system
    try:
        from sunny_ultimate_voice import DerekUltimateVoice
        
        print("🚀 Initializing Sunny (without speech/audio)...")
        sunny = DerekUltimateVoice(
            ai_provider="auto",
            voice_id="gtts",  # Use gtts to avoid AWS requirement
            use_web_search=False,
            enable_speech=False  # Skip speech recognition (no PyAudio/hardware)
        )
        print("✅ Sunny initialized!\n")
        
        # Test queries
        test_queries = [
            "How are you doing today?",
            "What is your purpose?",
            "Tell me about The Christman AI Project"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"Test {i}/{len(test_queries)}")
            print(f"{'='*60}")
            print(f"👤 User: {query}")
            print("🧠 Sunny thinking...")
            
            try:
                response = sunny.think(query)
                print(f"🤖 Sunny: {response}")
                
                if "Processing your input" in response and "cross-referencing" in response:
                    print("⚠️  WARNING: Sunny is still using template response!")
                    print("   This means the AI provider isn't being called properly.")
                else:
                    print("✅ Real AI response detected!")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print(f"\n{'='*60}")
        print("✅ All tests completed!")
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize Sunny: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_derek_thinking()
    exit(0 if success else 1)


# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
