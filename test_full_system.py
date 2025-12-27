#!/usr/bin/env python3
"""
Sunny AI Complete System Test
Tests autonomous learning with API keys from .env
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("🧠 Sunny AI Complete System Test")
print("=" * 60)

# Check API key configuration
print("🔑 API Key Configuration:")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY") 
perplexity_key = os.getenv("PERPLEXITY_API_KEY")

print(f"   Anthropic: {'✅ Configured' if anthropic_key else '❌ Not set'}")
print(f"   OpenAI: {'✅ Configured' if openai_key else '❌ Not set'}")
print(f"   Perplexity: {'✅ Configured' if perplexity_key else '❌ Not set'}")

# Test Sunny initialization
print(f"\n🤖 Initializing Sunny...")
try:
    # Import after loading .env so Sunny can detect API keys
    from autonomous_learning_engine import AutonomousLearningEngine
    
    engine = AutonomousLearningEngine()
    print("✅ Sunny initialized successfully")
    
    # Show detected AI providers
    print(f"\n🎯 Detected AI Providers:")
    print(f"   Primary Provider: {engine.sunny.ai_provider}")
    print(f"   Anthropic Client: {'✅' if engine.sunny.anthropic_client else '❌'}")
    print(f"   OpenAI Client: {'✅' if engine.sunny.openai_client else '❌'}")
    print(f"   Perplexity Client: {'✅' if engine.sunny.perplexity else '❌'}")
    
    # Test learning on autism support strategies
    print(f"\n📚 Testing Enhanced Learning...")
    test_topic = {
        "domain": "neurodivergency", 
        "subtopic": "autism_communication_strategies"
    }
    
    print(f"   Learning: {test_topic['domain']}.{test_topic['subtopic']}")
    knowledge = engine._learn_topic(test_topic)
    
    print(f"   Content Length: {len(knowledge.get('content', ''))} characters")
    print(f"   Key Concepts: {len(knowledge.get('key_concepts', []))}")
    print(f"   Confidence: {knowledge.get('confidence', 0):.2f}")
    print(f"   Mastery Level: {knowledge.get('mastery', 0):.2f}")
    
    # Show sample learning
    if knowledge.get('key_concepts'):
        print(f"\n🎓 Sample Key Concepts Learned:")
        for i, concept in enumerate(knowledge['key_concepts'][:5], 1):
            print(f"   {i}. {concept}")
    
    # Show content preview
    if knowledge.get('content'):
        preview = knowledge['content'][:200] + "..." if len(knowledge['content']) > 200 else knowledge['content']
        print(f"\n📖 Learning Content Preview:")
        print(f"   {preview}")
    
    # Test autonomous learning loop
    print(f"\n🔄 Testing Autonomous Learning Loop...")
    print("   Starting 10-second autonomous learning session...")
    
    engine.start_autonomous_learning()
    
    import time
    time.sleep(10)
    
    stats = engine.get_learning_stats()
    engine.stop_autonomous_learning()
    
    print(f"\n📊 Learning Session Results:")
    print(f"   Knowledge Base Size: {len(engine.knowledge_base)}")
    print(f"   Learning Status: {'🟢 Active' if stats['learning_active'] else '⏸️ Paused'}")
    print(f"   Available Domains: {len(engine.knowledge_domains)}")
    
    print(f"\n🎉 Sunny's Enhanced Autonomous Learning System is OPERATIONAL!")
    
    if anthropic_key or openai_key or perplexity_key:
        print(f"💫 Using premium AI providers for enhanced learning")
    else:
        print(f"🏠 Using local fallback - add API keys for enhanced capabilities")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n💡 Next Steps:")
print(f"   • Run: engine.start_autonomous_learning() for continuous learning")
print(f"   • Check: engine.get_learning_stats() for progress")
print(f"   • Monitor: learning logs for detailed activity")
print(f"   • Explore: Different learning domains and topics")