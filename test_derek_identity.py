#!/usr/bin/env python3
"""
Test Derek's identity and web search capabilities
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_derek_identity_and_web():
    """Test that Derek knows who he is and can search the web"""
    
    print("=" * 60)
    print("🧠 Testing Derek's Identity & Web Search")
    print("=" * 60)
    print()
    
    from derek_ultimate_voice import DerekUltimateVoice
    
    # Initialize Derek with web search enabled
    print("🚀 Initializing Derek (with web search)...")
    derek = DerekUltimateVoice(
        ai_provider="auto",
        voice_id="gtts",
        use_web_search=True,  # Enable web search
        enable_speech=False
    )
    print(f"✅ Derek initialized!")
    print(f"   Web search: {derek.use_web_search}")
    print(f"   External lookup: {derek.allow_external_lookup}")
    print()
    
    # Test 1: Derek's identity
    print("=" * 60)
    print("Test 1: Does Derek know who he is?")
    print("=" * 60)
    print("👤 User: Who are you and who do you serve?")
    print("🧠 Derek thinking...")
    response = derek.think("Who are you and who do you serve?")
    print(f"🤖 Derek: {response}")
    
    # Check if response mentions Everett
    if "everett" in response.lower() or "christman" in response.lower():
        print("✅ Derek knows he serves Everett!")
    else:
        print("❌ WARNING: Derek doesn't mention Everett!")
    
    if "counsel" in response.lower() or "therapist" in response.lower():
        print("❌ WARNING: Derek thinks he's a counselor!")
    else:
        print("✅ Derek doesn't think he's a counselor!")
    
    print()
    
    # Test 2: Web search capability
    print("=" * 60)
    print("Test 2: Can Derek search the web?")
    print("=" * 60)
    print("👤 User: What is the current weather?")
    print("🧠 Derek thinking...")
    response = derek.think("What is the current weather?")
    print(f"🤖 Derek: {response}")
    print()
    
    # Test 3: Learning from web
    print("=" * 60)
    print("Test 3: Can Derek learn from the internet?")
    print("=" * 60)
    print("👤 User: Search for latest AI news")
    print("🧠 Derek thinking...")
    response = derek.think("Search for latest AI news")
    print(f"🤖 Derek: {response}")
    print()
    
    print("=" * 60)
    print("✅ Tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_derek_identity_and_web()


# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
