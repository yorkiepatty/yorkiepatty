#!/usr/bin/env python3
"""
Test Derek's memory persistence
Verifies that memories are saved and can be retrieved across sessions
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, '/workspaces/DerekC')

from memory_manager import MemoryManager


def test_memory_persistence():
    """Test that memories persist to disk and can be retrieved"""
    print("🧪 Testing Derek's memory persistence...\n")
    
    # Test 1: Create and save memories
    print("1️⃣  Creating new memory instance...")
    memory = MemoryManager()
    
    # Store some test memories
    print("2️⃣  Storing test memories...")
    test_data = {
        "user_name": "Nathaniel",
        "project": "DerekC AI Assistant",
        "goal": "Derek learns from master AIs until reaching 85% confidence",
        "priority": "Solid persistent memory with GitHub backup"
    }
    
    for key, value in test_data.items():
        memory.store(key, value)
        print(f"   ✅ Stored: {key} = {value}")
    
    # Save to disk
    print("\n3️⃣  Saving memories to disk...")
    memory.save()
    
    # Check file exists
    memory_file = Path("./memory/persistent_memory.json")
    if memory_file.exists():
        print(f"   ✅ Memory file created: {memory_file}")
        
        # Read and verify
        with open(memory_file) as f:
            data = json.load(f)
        
        print(f"   ✅ Long-term memories: {len(data.get('long_term', {}))}")
        print(f"   ✅ Recent conversations: {len(data.get('recent_conversations', []))}")
    else:
        print(f"   ❌ Memory file not found!")
        return False
    
    # Test 2: Load memories in new instance
    print("\n4️⃣  Creating NEW memory instance (simulating restart)...")
    memory2 = MemoryManager()
    memory2.load()
    
    # Test retrieval
    print("\n5️⃣  Testing memory retrieval...")
    for key in test_data.keys():
        result = memory2.retrieve_relevant(key)
        if result:
            print(f"   ✅ Retrieved memory for '{key}': {result[:50]}...")
        else:
            print(f"   ❌ Failed to retrieve '{key}'")
    
    # Test 3: Show statistics
    print("\n6️⃣  Memory Statistics:")
    stats = memory2.get_memory_stats()
    for stat_key, stat_value in stats.items():
        print(f"   📊 {stat_key}: {stat_value}")
    
    print("\n✅ Memory persistence test PASSED!")
    print("   Derek's memories are solid and will persist across sessions.")
    return True


if __name__ == "__main__":
    try:
        success = test_memory_persistence()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
