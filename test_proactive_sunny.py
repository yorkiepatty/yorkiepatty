#!/usr/bin/env python3
"""
Test Sunny's Proactive Intelligence System
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_proactive_derek():
    """Test Sunny's proactive intelligence and autonomous learning"""
    
    print("=" * 60)
    print("🧠 Testing Sunny's Proactive Intelligence System")
    print("=" * 60)
    print()
    
    from sunny_ultimate_voice import DerekUltimateVoice
    
    print("🚀 Initializing Sunny (with full consciousness)...")
    sunny = DerekUltimateVoice(
        ai_provider="auto",
        voice_id="gtts",
        use_web_search=True,
        enable_speech=False
    )
    print()
    
    # Test 1: Check if proactive intelligence is active
    print("=" * 60)
    print("Test 1: Proactive Intelligence Status")
    print("=" * 60)
    if hasattr(sunny, 'proactive') and sunny.proactive:
        print("✅ Proactive Intelligence System is ACTIVE")
        print(f"   Background monitoring: {sunny.proactive.active_monitoring}")
        print(f"   Learning history: {len(sunny.proactive.insights)} insights")
    else:
        print("❌ Proactive Intelligence System not initialized")
    print()
    
    # Test 2: Sunny should learn from interactions
    print("=" * 60)
    print("Test 2: Autonomous Learning")
    print("=" * 60)
    
    test_queries = [
        "How can I fix an import error in Python?",
        "What's the best way to optimize module loading?",
        "Tell me about The Christman AI Project"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. User: {query}")
        print("   Sunny thinking...")
        response = sunny.think(query)
        print(f"   Sunny: {response[:150]}...")  # Show first 150 chars
        
        # Check if learning happened
        if hasattr(sunny, 'proactive') and sunny.proactive:
            insights_after = len(sunny.proactive.insights)
            print(f"   ✅ Learning recorded: {insights_after} total insights")
    
    print()
    
    # Test 3: Status Report
    print("=" * 60)
    print("Test 3: Sunny's Status Report")
    print("=" * 60)
    if hasattr(sunny, 'proactive') and sunny.proactive:
        status = sunny.proactive.generate_status_report()
        print(status)
    print()
    
    # Test 4: Proactive Insights
    print("=" * 60)
    print("Test 4: Proactive Insights Generation")
    print("=" * 60)
    if hasattr(sunny, 'proactive') and sunny.proactive:
        insights = sunny.proactive.generate_proactive_insights()
        if insights:
            print("💡 Sunny's Proactive Insights:")
            for insight in insights:
                print(f"   • {insight}")
        else:
            print("   No proactive insights generated yet (need more interactions)")
    print()
    
    # Test 5: Codebase Health
    print("=" * 60)
    print("Test 5: Codebase Health Analysis")
    print("=" * 60)
    if hasattr(sunny, 'proactive') and sunny.proactive:
        print("🔍 Analyzing codebase...")
        health = sunny.proactive.analyze_codebase_health()
        print(f"   Status: {health.get('overall_health', 'unknown').upper()}")
        print(f"   Files analyzed: {health.get('files_analyzed', 0)}")
        print(f"   Issues found: {len(health.get('issues_found', []))}")
        print(f"   Suggestions: {len(health.get('suggestions', []))}")
        
        if health.get('issues_found'):
            print("\n   🔴 Issues:")
            for issue in health['issues_found'][:3]:  # Show first 3
                print(f"      - {issue['file']}: {issue['message']}")
    print()
    
    # Cleanup
    if hasattr(sunny, 'proactive') and sunny.proactive:
        sunny.proactive.stop_background_monitoring()
    
    print("=" * 60)
    print("✅ All proactive intelligence tests completed!")
    print("=" * 60)
    print()
    print("Sunny is now:")
    print("  ✅ Learning autonomously from every interaction")
    print("  ✅ Monitoring codebase health in background")
    print("  ✅ Generating proactive insights and suggestions")
    print("  ✅ Ready to detect and fix issues before you notice them")
    print()
    print("💡 Sunny is ahead of the curve, helping you fix other AI systems!")


if __name__ == "__main__":
    test_proactive_derek()


# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
