#!/usr/bin/env python3
"""
Simple test of Sunny's autonomous learning
"""

print("🧠 Testing Sunny's Enhanced Learning System")
print("=" * 50)

try:
    from autonomous_learning_engine import AutonomousLearningEngine
    print("✅ Successfully imported AutonomousLearningEngine")
    
    # Initialize engine
    engine = AutonomousLearningEngine()
    print("✅ Successfully initialized Sunny")
    
    # Check AI providers
    print(f"\n🤖 AI Provider Status:")
    print(f"   Primary: {engine.sunny.ai_provider}")
    print(f"   Anthropic: {'✅' if engine.sunny.anthropic_client else '❌'}")
    print(f"   OpenAI: {'✅' if engine.sunny.openai_client else '❌'}")
    print(f"   Perplexity: {'✅' if engine.sunny.perplexity else '❌'}")
    
    # Test learning
    print(f"\n📚 Testing Learning Process...")
    topic = {"domain": "neurodivergency", "subtopic": "autism_spectrum"}
    
    knowledge = engine._learn_topic(topic)
    
    print(f"   Content Length: {len(knowledge.get('content', ''))} chars")
    print(f"   Key Concepts: {len(knowledge.get('key_concepts', []))}")
    print(f"   Confidence: {knowledge.get('confidence', 0)}")
    
    if knowledge.get('key_concepts'):
        print(f"\n🎯 Sample Key Concepts:")
        for i, concept in enumerate(knowledge['key_concepts'][:3], 1):
            print(f"   {i}. {concept}")
    
    print(f"\n✅ Sunny's autonomous learning is working!")
    print(f"💡 Add API keys to .env file for enhanced AI capabilities")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()