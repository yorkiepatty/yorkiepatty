#!/usr/bin/env python3
"""
Test Sunny's identity and web search capabilities
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_sunny_identity_and_web():
    """Test that Sunny knows who he is and can search the web"""
    
    print("=" * 60)
    print("🧠 Testing Sunny's Identity & Web Search")
    print("=" * 60)
    print()
    
    from sunny_ultimate_voice import DerekUltimateVoice
    
    # Initialize Sunny with web search enabled
    print("🚀 Initializing Sunny (with web search)...")
    sunny = DerekUltimateVoice(
        ai_provider="auto",
        voice_id="gtts",
        use_web_search=True,  # Enable web search
        enable_speech=False
    )
    print(f"✅ Sunny initialized!")
    print(f"   Web search: {sunny.use_web_search}")
    print(f"   External lookup: {sunny.allow_external_lookup}")
    print()
    
    # Test 1: Sunny's identity
    print("=" * 60)
    print("Test 1: Does Sunny know who he is?")
    print("=" * 60)
    print("👤 User: Who are you and who do you serve?")
    print("🧠 Sunny thinking...")
    response = sunny.think("Who are you and who do you serve?")
    print(f"🤖 Sunny: {response}")
    
    # Check if response mentions Everett
    if "everett" in response.lower() or "christman" in response.lower():
        print("✅ Sunny knows he serves Everett!")
    else:
        print("❌ WARNING: Sunny doesn't mention Everett!")
    
    if "counsel" in response.lower() or "therapist" in response.lower():
        print("❌ WARNING: Sunny thinks he's a counselor!")
    else:
        print("✅ Sunny doesn't think he's a counselor!")
    
    print()
    
    # Test 2: Web search capability
    print("=" * 60)
    print("Test 2: Can Sunny search the web?")
    print("=" * 60)
    print("👤 User: What is the current weather?")
    print("🧠 Sunny thinking...")
    response = sunny.think("What is the current weather?")
    print(f"🤖 Sunny: {response}")
    print()
    
    # Test 3: Learning from web
    print("=" * 60)
    print("Test 3: Can Sunny learn from the internet?")
    print("=" * 60)
    print("👤 User: Search for latest AI news")
    print("🧠 Sunny thinking...")
    response = sunny.think("Search for latest AI news")
    print(f"🤖 Sunny: {response}")
    print()
    
    print("=" * 60)
    print("✅ Tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_sunny_identity_and_web()


# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
