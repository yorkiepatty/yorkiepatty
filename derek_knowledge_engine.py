"""
Derek's Knowledge-First Reasoning Engine
The Christman AI Project

Enables Derek to:
- Use his learned knowledge BEFORE external APIs
- Build confidence in his own knowledge
- Reduce API costs by 90%+
- Become truly self-sufficient

"The best knowledge is that which you've learned yourself."
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import re
import logging
from memory_mesh_bridge import MemoryMeshBridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("knowledge_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Derek:
    def __init__(self, memory_dir: str = "./derek_memory"):
        try:
            self.memory = MemoryMeshBridge(memory_dir=memory_dir)
            logger.info("Derek initialized with MemoryMeshBridge")
        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")
            raise RuntimeError(f"Memory initialization failed: {e}")

class KnowledgeEngine:
    def __init__(self, knowledge_dir: str = "./knowledge", memory_mesh=None, local_reasoning=None):
        """
        Initialize the Knowledge Engine
        
        Args:
            knowledge_dir: Directory containing learned knowledge
            memory_mesh: Reference to Derek memory system
            local_reasoning: Reference to local AI reasoning engine
        """
        self.derek = Derek()
        self.knowledge_dir = Path(knowledge_dir)
        self.memory_mesh = memory_mesh
        self.local_reasoning = local_reasoning
        
        # Knowledge confidence thresholds
        self.high_confidence = 0.8
        self.medium_confidence = 0.6
        self.low_confidence = 0.3
        
        # Statistics
        self.stats = {
            "queries_answered_locally": 0,
            "queries_needed_external": 0,
            "total_queries": 0,
            "knowledge_hits": 0,
            "api_calls_saved": 0
        }
    
    def _store_knowledge(self, topic, knowledge):
        """Store knowledge in Derek's memory"""
        if self.derek.memory is None:
            logger.error("Memory system not initialized")
            raise RuntimeError("Memory system not initialized")
        try:
            self.derek.memory.store(topic, knowledge)
            logger.info(f"Stored knowledge: {topic}")
        except Exception as e:
            logger.error(f"Failed to store knowledge: {e}")
            raise
        
    def reason(
        self,
        question: str,
        context: Optional[str] = None,
        require_external: bool = False
    ) -> Dict[str, Any]:
        """
        Main reasoning method - uses knowledge first, APIs as fallback
        
        Args:
            question: User's question
            context: Additional context
            require_external: Force external API usage
        
        Returns:
            dict: Response with metadata
        """
        self.stats["total_queries"] += 1
        
        # Step 1: Check if we should skip local reasoning
        if require_external:
            return self._use_external_api(question, context, reason="forced")
        
        # Step 2: Search Derek's learned knowledge
        knowledge_result = self._search_learned_knowledge(question)
        
        # Step 3: Check Derek's memory for relevant experiences
        memory_result = self._search_memory(question) if self.memory_mesh else None
        
        # Step 4: Combine knowledge and memory
        combined_confidence = self._calculate_confidence(knowledge_result, memory_result)
        
        # Step 5: Decide on reasoning strategy
        if combined_confidence >= self.high_confidence:
            # We have strong knowledge - use local reasoning
            return self._reason_from_knowledge(
                question, knowledge_result, memory_result, context
            )
        
        elif combined_confidence >= self.medium_confidence:
            # We have some knowledge - try local first, enhance with API if needed
            local_result = self._reason_from_knowledge(
                question, knowledge_result, memory_result, context
            )
            
            # If local result seems incomplete, enhance with external
            if self._is_response_incomplete(local_result):
                return self._enhance_with_external(question, local_result, context)
            
            return local_result
        
        else:
            # Low confidence - use external API
            return self._use_external_api(
                question, context,
                reason="insufficient_knowledge",
                partial_knowledge=knowledge_result
            )
    
    def _search_learned_knowledge(self, question: str) -> Dict[str, Any]:
        """
        Search Derek's learned knowledge base
        
        Args:
            question: User's question
        
        Returns:
            dict: Relevant knowledge with confidence score
        """
        if not self.knowledge_dir.exists():
            return {"items": [], "confidence": 0.0}
        
        relevant_knowledge = []
        question_lower = question.lower()
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        
        # Search all knowledge domains
        for domain_dir in self.knowledge_dir.iterdir():
            if not domain_dir.is_dir() or domain_dir.name == "generated_code":
                continue
            
            for knowledge_file in domain_dir.glob("*.json"):
                try:
                    with open(knowledge_file, 'r') as f:
                        data = json.load(f)
                        
                        # Extract searchable content
                        content = " ".join([
                            str(data.get('topic', '')),
                            str(data.get('summary', '')),
                            str(data.get('key_concepts', [])),
                            str(data.get('practical_applications', []))
                        ]).lower()
                        
                        content_words = set(re.findall(r'\b\w+\b', content))
                        
                        # Calculate relevance
                        word_overlap = len(question_words & content_words)
                        if word_overlap > 0:
                            relevance = word_overlap / len(question_words)
                            
                            # Boost if exact phrase match
                            if any(phrase in content for phrase in question_lower.split()):
                                relevance *= 1.5
                            
                            if relevance > 0.15:  # Minimum threshold
                                relevant_knowledge.append({
                                    "topic": data.get('topic', ''),
                                    "domain": data.get('domain', ''),
                                    "summary": data.get('summary', ''),
                                    "key_concepts": data.get('key_concepts', []),
                                    "applications": data.get('practical_applications', []),
                                    "learned_at": data.get('learned_at', ''),
                                    "relevance": min(relevance, 1.0)
                                })
                
                except Exception as e:
                    continue
        
        # Sort by relevance
        relevant_knowledge.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Calculate overall confidence
        if relevant_knowledge:
            top_relevance = relevant_knowledge[0]['relevance']
            num_sources = min(len(relevant_knowledge), 5)
            confidence = (top_relevance * 0.7) + (num_sources / 10)  # Max ~1.2
            confidence = min(confidence, 1.0)
        else:
            confidence = 0.0
        
        self.stats["knowledge_hits"] += len(relevant_knowledge)
        
        return {
            "items": relevant_knowledge[:10],  # Top 10
            "confidence": confidence,
            "total_found": len(relevant_knowledge)
        }
    
    def _search_memory(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Search Derek's memory mesh for relevant experiences
        
        Args:
            question: User's question
        
        Returns:
            dict: Relevant memories with confidence
        """
        if not self.memory_mesh:
            return None
        
        try:
            # Search working and episodic memory
            relevant_memories = []
            
            # Simple keyword search (could be enhanced)
            question_words = set(question.lower().split())
            
            # Search memory (simplified - actual implementation depends on memory_mesh API)
            # This is a placeholder for memory mesh integration
            
            return {
                "items": relevant_memories,
                "confidence": 0.3 if relevant_memories else 0.0
            }
        except Exception as e:
            return None
    
    def _calculate_confidence(
        self,
        knowledge_result: Dict[str, Any],
        memory_result: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall confidence from knowledge and memory
        
        Args:
            knowledge_result: Result from knowledge search
            memory_result: Result from memory search
        
        Returns:
            float: Combined confidence (0.0-1.0)
        """
        knowledge_conf = knowledge_result.get("confidence", 0.0)
        memory_conf = memory_result.get("confidence", 0.0) if memory_result else 0.0
        
        # Weighted combination (knowledge more important)
        combined = (knowledge_conf * 0.8) + (memory_conf * 0.2)
        
        return min(combined, 1.0)
    
    def _reason_from_knowledge(
        self,
        question: str,
        knowledge_result: Dict[str, Any],
        memory_result: Optional[Dict[str, Any]],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response using Derek's knowledge and local AI
        
        Args:
            question: User's question
            knowledge_result: Relevant knowledge
            memory_result: Relevant memories
            context: Additional context
        
        Returns:
            dict: Response with metadata
        """
        self.stats["queries_answered_locally"] += 1
        self.stats["api_calls_saved"] += 1
        
        # Build knowledge context
        knowledge_items = knowledge_result.get("items", [])
        
        if not knowledge_items:
            return {
                "response": None,
                "confidence": 0.0,
                "source": "no_knowledge",
                "needs_external": True
            }
        
        # Format knowledge for reasoning
        knowledge_context = self._format_knowledge_context(knowledge_items)
        
        # Use local reasoning if available
        if self.local_reasoning and self.local_reasoning.ollama_available:
            system_prompt = f"""You are Derek, an AI with specialized knowledge.

Your learned knowledge:
{knowledge_context}

Use this knowledge to answer questions accurately. If the knowledge doesn't fully answer the question, say so honestly.

Your mission: "How can we help you love yourself more?"
Focus on being helpful, compassionate, and evidence-based."""
            
            response = self.local_reasoning.query_local_model(
                prompt=question,
                system_prompt=system_prompt,
                temperature=0.6
            )
            
            if response:
                return {
                    "response": response,
                    "confidence": knowledge_result["confidence"],
                    "source": "local_knowledge_reasoning",
                    "model": self.local_reasoning.current_model,
                    "knowledge_used": [k["topic"] for k in knowledge_items[:5]],
                    "domains": list(set(k["domain"] for k in knowledge_items)),
                    "needs_external": False
                }
        
        # Fallback: Format knowledge directly (no local model)
        formatted_response = self._format_knowledge_response(question, knowledge_items)
        
        return {
            "response": formatted_response,
            "confidence": knowledge_result["confidence"] * 0.8,  # Slightly lower confidence
            "source": "knowledge_direct",
            "knowledge_used": [k["topic"] for k in knowledge_items[:5]],
            "domains": list(set(k["domain"] for k in knowledge_items)),
            "needs_external": False
        }
    
    def _format_knowledge_context(self, knowledge_items: List[Dict[str, Any]]) -> str:
        """Format knowledge items into context for reasoning"""
        context_parts = []
        
        for i, item in enumerate(knowledge_items[:5], 1):
            part = f"{i}. {item['topic']} ({item['domain']})\n"
            part += f"   {item['summary'][:300]}"
            
            if item.get('key_concepts'):
                concepts = item['key_concepts'][:3]
                part += f"\n   Key concepts: {', '.join(concepts)}"
            
            context_parts.append(part)
        
        return "\n\n".join(context_parts)
    
    def _format_knowledge_response(
        self,
        question: str,
        knowledge_items: List[Dict[str, Any]]
    ) -> str:
        """Format knowledge items into a direct response"""
        if not knowledge_items:
            return "I don't have learned knowledge about this topic yet."
        
        top_item = knowledge_items[0]
        
        response = f"Based on what I've learned about {top_item['topic']}:\n\n"
        response += top_item['summary']
        
        if len(knowledge_items) > 1:
            response += f"\n\nI also have knowledge about:\n"
            for item in knowledge_items[1:4]:
                response += f"• {item['topic']} ({item['domain']})\n"
        
        return response
    
    def _is_response_incomplete(self, result: Dict[str, Any]) -> bool:
        """Check if a response seems incomplete and needs enhancement"""
        response = result.get("response", "")
        confidence = result.get("confidence", 0.0)
        
        if not response:
            return True
        
        if confidence < 0.7:
            return True
        
        # Check for phrases indicating uncertainty
        uncertain_phrases = [
            "i don't know", "i'm not sure", "unclear", "insufficient",
            "don't have enough", "need more information"
        ]
        
        if any(phrase in response.lower() for phrase in uncertain_phrases):
            return True
        
        return False
    
    def _enhance_with_external(
        self,
        question: str,
        local_result: Dict[str, Any],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Enhance local response with external API"""
        self.stats["queries_needed_external"] += 1
        
        # Return signal to use external API with local knowledge as context
        return {
            "response": local_result.get("response"),
            "confidence": local_result.get("confidence", 0.0),
            "source": "hybrid_local_external",
            "needs_external": True,
            "enhancement_needed": True,
            "local_knowledge": local_result.get("knowledge_used", []),
            "partial_answer": local_result.get("response")
        }
    
    def _use_external_api(
        self,
        question: str,
        context: Optional[str] = None,
        reason: str = "unknown",
        partial_knowledge: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Signal that external API should be used"""
        self.stats["queries_needed_external"] += 1
        
        return {
            "response": None,
            "confidence": 0.0,
            "source": "external_api_needed",
            "needs_external": True,
            "reason": reason,
            "partial_knowledge": partial_knowledge
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reasoning statistics"""
        total = self.stats["total_queries"]
        if total == 0:
            local_pct = 0
            api_saved_pct = 0
        else:
            local_pct = (self.stats["queries_answered_locally"] / total) * 100
            api_saved_pct = (self.stats["api_calls_saved"] / total) * 100
        
        return {
            **self.stats,
            "local_answer_rate": f"{local_pct:.1f}%",
            "api_savings_rate": f"{api_saved_pct:.1f}%"
        }
    
    def print_statistics(self):
        """Print reasoning statistics"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("📊 DEREK'S KNOWLEDGE REASONING STATISTICS")
        print("=" * 60)
        print(f"\nTotal Queries: {stats['total_queries']}")
        print(f"Answered Locally: {stats['queries_answered_locally']} ({stats['local_answer_rate']})")
        print(f"Needed External: {stats['queries_needed_external']}")
        print(f"Knowledge Hits: {stats['knowledge_hits']}")
        print(f"\n💰 API Calls Saved: {stats['api_calls_saved']} ({stats['api_savings_rate']})")
        print("=" * 60 + "\n")

    def _store_knowledge(self, topic, knowledge):
        if self.derek.memory is None:
            logging.error("Memory system not initialized")
            raise RuntimeError("Memory system not initialized")
        try:
            self.derek.memory.store(topic, knowledge)
        except Exception as e:
            logging.error(f"Failed to store knowledge: {e}")
            raise

# Test function
if __name__ == "__main__":
    print("Testing Derek's Knowledge Engine...\n")
    
    engine = KnowledgeEngine()
    
    # Test query
    result = engine.reason("What is autism?")
    
    print(f"Confidence: {result['confidence']}")
    print(f"Source: {result['source']}")
    print(f"Needs External: {result.get('needs_external', False)}")
    
    if result.get('response'):
        print(f"\nResponse: {result['response'][:300]}...")
    
    engine.print_statistics()

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
