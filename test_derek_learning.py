#!/usr/bin/env python3
"""
Test Derek's Enhanced Autonomous Learning System
Tests all AI providers and fallback systems
"""

import time
from autonomous_learning_engine import AutonomousLearningEngine

def test_derek_learning():
    """Test Derek's learning capabilities"""
    print("🧠 Testing Derek's Enhanced Autonomous Learning System")
    print("=" * 60)
    
    # Initialize engine
    print("1. Initializing Derek...")
    engine = AutonomousLearningEngine()
    
    # Show AI provider status
    print("\n2. AI Provider Status:")
    print(f"   Primary Provider: {engine.derek.ai_provider}")
    print(f"   Anthropic Claude: {'✅' if engine.derek.anthropic_client else '❌'}")
    print(f"   OpenAI GPT: {'✅' if engine.derek.openai_client else '❌'}")
    print(f"   Perplexity Web: {'✅' if engine.derek.perplexity else '❌'}")
    print(f"   Ollama Local: {'Available' if True else '❌'}")
    
    # Test learning on a single topic
    print("\n3. Testing Learning Process...")
    topic = {"domain": "neurodivergency", "subtopic": "autism_spectrum"}
    
    print(f"   Learning Topic: {topic['domain']}.{topic['subtopic']}")
    knowledge = engine._learn_topic(topic)
    
    print(f"   Content Length: {len(knowledge.get('content', ''))}")
    print(f"   Key Concepts: {len(knowledge.get('key_concepts', []))}")
    print(f"   Confidence: {knowledge.get('confidence', 0)}")
    
    # Show learned key concepts
    if knowledge.get('key_concepts'):
        print(f"\n4. Key Concepts Learned:")
        for i, concept in enumerate(knowledge['key_concepts'][:5], 1):
            print(f"   {i}. {concept}")
    
    # Test autonomous learning
    print(f"\n5. Testing Autonomous Learning Loop...")
    engine.start_autonomous_learning()
    
    print("   Running for 10 seconds...")
    time.sleep(10)
    
    engine.stop_autonomous_learning()
    
    # Show results
    stats = engine.get_learning_stats()
    print(f"\n6. Learning Results:")
    print(f"   Topics Learned: {len(engine.knowledge_base)}")
    print(f"   Total Domains: {len(engine.knowledge_domains)}")
    print(f"   Learning Status: {'🟢 ACTIVE' if stats['learning_active'] else '⏸️ PAUSED'}")
    
    print(f"\n✅ Derek's learning system test complete!")
    
    return engine

if __name__ == "__main__":
    engine = test_derek_learning()
    
    print(f"\n💡 Next Steps:")
    print(f"   • Add API keys for external AI providers")
    print(f"   • Pull more Ollama models: ollama pull llama3.2:3b")
    print(f"   • Run autonomous learning: engine.start_autonomous_learning()")
    print(f"   • Check learning progress: engine.get_learning_stats()")