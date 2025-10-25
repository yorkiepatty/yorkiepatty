# memory_manager.py
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from json_guardian import JSONGuardian

guardian = JSONGuardian()


class MemoryManager:
    """Manager for Derek's long-term memory storage with persistent GitHub backup"""
    
    def __init__(self, memory_dir="./memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memories = {}
        self.memory_file = self.memory_dir / "persistent_memory.json"
        self.conversation_memory = []  # Current session
        self.long_term_memory = {}  # Persistent across sessions
        self.save_counter = 0
        self.auto_save_interval = 5  # Auto-save every 5 interactions
    
    def load(self):
        """Load ALL memories from disk - both persistent and legacy files"""
        print("🧠 Loading Derek's persistent memory...")
        
        # Load main persistent memory file
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.long_term_memory = data.get('long_term', {})
                    self.conversation_memory = data.get('recent_conversations', [])
                    print(f"✅ Loaded {len(self.long_term_memory)} long-term memories")
                    print(f"✅ Loaded {len(self.conversation_memory)} recent conversations")
            except Exception as e:
                print(f"⚠️  Error loading persistent memory: {e}")
                self.long_term_memory = {}
                self.conversation_memory = []
        
        # Load legacy memory files
        for file_path in self.memory_dir.glob("*.json"):
            if file_path.name == "persistent_memory.json":
                continue  # Already loaded above
            try:
                with open(file_path) as f:
                    self.memories[file_path.stem] = json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading {file_path}: {e}")
        
        if self.memories:
            print(f"✅ Loaded {len(self.memories)} legacy memory files")
    
    def save(self):
        """Save ALL memories to disk with automatic GitHub backup"""
        try:
            # Prepare data structure
            memory_data = {
                'last_updated': datetime.now().isoformat(),
                'long_term': self.long_term_memory,
                'recent_conversations': self.conversation_memory[-50:],  # Keep last 50
                'statistics': {
                    'total_memories': len(self.long_term_memory),
                    'total_conversations': len(self.conversation_memory),
                    'memory_keys': list(self.long_term_memory.keys())
                }
            }
            
            # Save to persistent file
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
            
            print(f"💾 Saved {len(self.long_term_memory)} memories to persistent storage")
            
            # Auto-commit to GitHub every 10 saves
            self.save_counter += 1
            if self.save_counter % 10 == 0:
                self._backup_to_github()
            
        except Exception as e:
            print(f"⚠️  Error saving memory: {e}")
    
    def _backup_to_github(self):
        """Automatically backup memory to GitHub"""
        try:
            # Check if we're in a git repo
            result = subprocess.run(
                ['git', 'rev-parse', '--is-inside-work-tree'],
                cwd=self.memory_dir.parent,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return  # Not in a git repo, skip
            
            # Add memory files
            subprocess.run(
                ['git', 'add', 'memory/'],
                cwd=self.memory_dir.parent,
                capture_output=True
            )
            
            # Commit with timestamp
            commit_msg = f"💾 Auto-backup: Derek's memory update ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
            subprocess.run(
                ['git', 'commit', '-m', commit_msg],
                cwd=self.memory_dir.parent,
                capture_output=True
            )
            
            print("☁️  Memory backed up to GitHub")
            
        except Exception as e:
            # Silently fail - GitHub backup is nice-to-have, not critical
            pass
    
    def store(self, key: str, value: Any):
        """Store a memory - both in session and long-term"""
        # Store in session
        self.memories[key] = value
        
        # Store in long-term with timestamp
        timestamp = datetime.now().isoformat()
        self.long_term_memory[key] = {
            'value': value,
            'timestamp': timestamp,
            'access_count': self.long_term_memory.get(key, {}).get('access_count', 0) + 1
        }
        
        # Add to conversation memory
        self.conversation_memory.append({
            'key': key,
            'value': str(value)[:200],  # First 200 chars
            'timestamp': timestamp
        })
        
        # Auto-save periodically
        if len(self.conversation_memory) % self.auto_save_interval == 0:
            self.save()
    
    def retrieve_relevant(self, query: str) -> str:
        """Retrieve memories relevant to query - ALWAYS checks memory"""
        relevant = []
        query_lower = query.lower()
        
        # Check long-term memory first
        for key, data in self.long_term_memory.items():
            value = str(data.get('value', ''))
            if query_lower in value.lower() or query_lower in key.lower():
                # Update access count
                data['access_count'] = data.get('access_count', 0) + 1
                data['last_accessed'] = datetime.now().isoformat()
                relevant.append(value)
        
        # Check session memories
        for key, value in self.memories.items():
            if query_lower in str(value).lower():
                relevant.append(str(value))
        
        # Check recent conversations
        for conv in self.conversation_memory[-20:]:  # Last 20 conversations
            if query_lower in conv.get('value', '').lower():
                relevant.append(conv.get('value', ''))
        
        # Return most relevant (unique)
        unique_relevant = list(dict.fromkeys(relevant))  # Remove duplicates
        return " | ".join(unique_relevant[:5]) if unique_relevant else ""
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about Derek's memory"""
        return {
            'long_term_memories': len(self.long_term_memory),
            'session_memories': len(self.memories),
            'recent_conversations': len(self.conversation_memory),
            'most_accessed': self._get_most_accessed(5),
            'memory_file_exists': self.memory_file.exists()
        }
    
    def _get_most_accessed(self, limit=5) -> List[str]:
        """Get most frequently accessed memories"""
        sorted_memories = sorted(
            self.long_term_memory.items(),
            key=lambda x: x[1].get('access_count', 0),
            reverse=True
        )
        return [key for key, _ in sorted_memories[:limit]]


def load_memory_block(name):
    file_path = guardian.memory_dir / f"{name}.json"
    guardian.validate_file(file_path)  # could be a lighter per-file check
    with open(file_path) as f:
        return json.load(f)

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
