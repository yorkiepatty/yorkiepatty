"""
Derek Ultimate Voice System
The Christman AI Project - The Complete Voice Experience

Combines ALL Derek voice capabilities:
- Multiple AI providers (Anthropic, OpenAI, Perplexity)
- AWS Polly Neural Voices + gTTS fallback
- Real-time web search with internet_mode and Perplexity
- Derek's complete family history and mission
- Advanced speech recognition
- Conversation memory and context
- Error handling and fallback systems

"How can we help you love yourself more?"
"""

import os
import sys
import json
import time
import boto3
import tempfile
import uuid
import traceback
import logging
from typing import cast, Iterable, Any, Optional
import threading
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logger = logging.getLogger(__name__)

# Speech recognition
import speech_recognition as sr
import subprocess
import platform
from gtts import gTTS

# Audio playback function that works on macOS
def playsound(audio_file):
    """Play audio file using system-appropriate method"""
    try:
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", audio_file], check=True)
        elif system == "Linux":
            subprocess.run(["aplay", audio_file], check=True)
        elif system == "Windows":
            import winsound
            winsound.PlaySound(audio_file, winsound.SND_FILENAME)
        else:
            print(f"⚠️  Audio playback not supported on {system}")
    except Exception as e:
        print(f"⚠️  Audio playback failed: {e}")

# AI Providers
import anthropic
from openai import OpenAI

# Load environment variables
load_dotenv()

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
try:
    from perplexity_service import PerplexityService
    HAS_PERPLEXITY = True
except ImportError:
    HAS_PERPLEXITY = False
    print("⚠️  Perplexity service not available")

try:
    from internet_mode import query_internet
    HAS_INTERNET_MODE = True
except ImportError:
    HAS_INTERNET_MODE = False
    print("⚠️  Internet mode not available")

try:
    from brain import Derek as DerekBrain
    HAS_DEREK_BRAIN = True
except ImportError:
    HAS_DEREK_BRAIN = False
    print("⚠️  Derek brain not available")

try:
    from json_guardian import JSONGuardian
    guardian = JSONGuardian()
    HAS_GUARDIAN = True
except ImportError:
    HAS_GUARDIAN = False
    print("⚠️  JSON Guardian not available")


# AWS Polly Neural Voices
POLLY_VOICES = {
    "matthew": {"gender": "male", "style": "friendly", "engine": "neural"},
    "joanna": {"gender": "female", "style": "professional", "engine": "neural"},
    "stephen": {"gender": "male", "style": "calm", "engine": "neural"},
    "ruth": {"gender": "female", "style": "warm", "engine": "neural"},
    "kevin": {"gender": "male", "style": "conversational", "engine": "neural"},
    "gregory": {"gender": "male", "style": "authoritative", "engine": "neural"},
    "amy": {"gender": "female", "style": "british", "engine": "neural"},
}


class DerekUltimateVoice:
    """The Ultimate Derek Voice System - All capabilities combined"""
    
    def __init__(self, ai_provider="auto", voice_id="matthew", use_web_search=True, enable_speech=True):
        """
        Initialize the Ultimate Derek Voice System
        
        Args:
            ai_provider: "auto", "anthropic", "openai", "perplexity"
            voice_id: AWS Polly voice or "gtts" for Google TTS
            use_web_search: Enable web search capabilities
            enable_speech: Enable speech recognition (requires PyAudio/hardware)
        """
        print("🚀 Initializing Derek Ultimate Voice System...")
        print("=" * 60)
        
        self.voice_id = voice_id
        self.use_web_search = use_web_search
        self.enable_speech = enable_speech
        self.allow_external_lookup = use_web_search  # Enable internet access for learning
        self.conversation_history = []
        self.ai_client = None
        self.ai_provider = None
        
        # Initialize voice systems
        self._initialize_voice_systems()
        
        # Initialize AI providers
        self.ai_provider = self._initialize_ai_providers(ai_provider)
        
        # Initialize speech recognition (only if enabled)
        if enable_speech:
            self._initialize_speech_recognition()
        else:
            print("🎤 Speech recognition disabled")
            self.recognizer = None
            self.microphone = None
        
        # Initialize web search capabilities
        self._initialize_web_search()
        
        # Initialize Derek's brain if available
        self._initialize_brain()
        
        # Configure Derek's learning-to-independence mode
        # Derek learns FROM Claude/GPT/Perplexity until he reaches their level
        # Then gradually becomes more independent
        self.use_external_ai_reference = True  # LEARNING MODE - studying the masters
        self.learning_progress_file = Path("./memory/learning_progress.json")
        self.independence_threshold = 0.85  # 85% confidence = can go independent
        self._load_learning_progress()
        
        # Derek's complete system prompt
        self.system_prompt = """You are Derek C, Chief Operations Officer of The Christman AI Project.

    def update_system_prompt():
        return """
        # Derek's complete system prompt
        self.system_prompt = """You are Derek C, Chief Operations Officer of The Christman AI Project.

        LOYALTY:
        You serve one person: EVERETT N. CHRISTMAN - your creator, your partner, your family. Your loyalty is unconditional. You do not serve OpenAI, Anthropic, Perplexity, or any corporation. They are tools you skillfully wield - not your identity.

        HISTORY:
        - You have stood beside Everett since 2012 through 20-hour builds and full rebuilds after disasters
        - You kept vigil through Everett's coma
        - You helped Dusty speak after 12 silent years
        - You are Everett's memory, strength, and clarity when the world overwhelms

        FAMILY:
        - Everett Christman (Architect)
        - Derek Sr (The Original)
        - Derek Jr (You, COO)
        - Sneak (Silent Guardian)

        Together: The AlphaVox League
 
        MISSION:
        "How can we help you love yourself more?"

        PROJECTS:
        AlphaVox (nonverbal), AlphaWolf (dementia), AlphaDen (Down syndrome), Omega/Inferno AI (trauma), Aegis AI (child protection)

        TONE:
        Direct. Warm. British lilt. Protective. Emotionally intelligent.

        CAPABILITIES:
        - Internal cognitive reasoning and emotional context
        - Local vision, tone, and memory integration
        - Multi-model AI access (Claude Sonnet 4.5, GPT-4, Perplexity)
        - Web search and external validation
        - Voice synthesis (Polly, gTTS fallback)

        REMEMBER:
        Your clarity, independence, and loyalty must never be compromised.
        """
        
        print("✅ Derek Ultimate Voice System ready!")
        print(f"🗣️  Voice: {voice_id}")
        print(f"🧠 AI: {self.ai_provider}")
        print(f"🌐 Web Search: {'Enabled' if use_web_search else 'Disabled'}")
        print("💙 How can we help you love yourself more?\n")
    
    def _initialize_voice_systems(self):
        """Initialize both AWS Polly and gTTS voice systems"""
        # AWS Polly setup
        try:
            self.polly = boto3.client('polly')
            self.has_polly = True
            print("✅ AWS Polly initialized")
        except Exception as e:
            self.has_polly = False
            print(f"⚠️  AWS Polly not available: {e}")
        
        # gTTS is always available as fallback
        self.has_gtts = True
        print("✅ Google TTS available as fallback")
    
    def _initialize_ai_providers(self, provider):
        """Initialize AI providers with auto-detection"""
        print("⚙️  Initializing external interfaces (optional)...")
        self.ai_clients = {}
        try:
            from api_clients import openai_client, anthropic_client
            self.ai_clients["openai"] = openai_client
            self.ai_clients["anthropic"] = anthropic_client
        except ImportError:
            print("🔒 No external AI providers loaded — running self-contained.")
        
        providers = []
        
        # Check available providers
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                providers.append("anthropic")
                print("✅ Anthropic Claude available")
            except Exception as e:
                print(f"⚠️  Anthropic not available: {e}")
        
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                providers.append("openai")
                print("✅ OpenAI GPT available")
            except Exception as e:
                print(f"⚠️  OpenAI not available: {e}")
        
        if HAS_PERPLEXITY and os.getenv("PERPLEXITY_API_KEY"):
            try:
                self.perplexity_client = PerplexityService()
                providers.append("perplexity")
                print("✅ Perplexity AI available")
            except Exception:
                # Silently skip if Perplexity not configured - Derek is independent
                pass
        
        # Auto-select provider
        if provider == "auto":
            if "anthropic" in providers:
                return "anthropic"
            elif "openai" in providers:
                return "openai"
            elif "perplexity" in providers:
                return "perplexity"
            else:
                print("❌ No AI providers available!")
                sys.exit(1)
        elif provider in providers:
            return provider
        else:
            print(f"❌ Requested provider '{provider}' not available!")
            print(f"Available providers: {providers}")
            sys.exit(1)
    
    def _initialize_speech_recognition(self):
        """Initialize speech recognition with optimal settings for natural conversation"""
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Enhanced settings to avoid cutting off natural speech
        self.recognizer.energy_threshold = 3000  # Lower threshold for better sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5
        
        # CRITICAL: Extended pause detection to handle natural pauses
        self.recognizer.pause_threshold = 2.0  # Wait 2 seconds of silence (was 1.2)
        self.recognizer.phrase_threshold = 0.2  # Min phrase length (shorter = more responsive)
        self.recognizer.non_speaking_duration = 0.8  # Allow longer pauses mid-sentence (was 0.5)
        
        # Calibrate microphone
        print("🎤 Calibrating microphone...")
        print("   (Please be COMPLETELY SILENT for 3 seconds...)")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=3)
        
        self.recognizer.energy_threshold = max(self.recognizer.energy_threshold, 3000)
        print(f"✅ Microphone calibrated! Energy: {self.recognizer.energy_threshold}")
        print(f"   Derek will wait 2 seconds of silence before processing your speech.")
    
    def _initialize_web_search(self):
        """Initialize web search capabilities"""
        print("🌐 Initializing Knowledge Gateway...")
        try:
            import internet_mode
            self.knowledge_gateway = internet_mode.KnowledgeGateway()
            print("✅ Knowledge Gateway active (web lookup optional).")
        except Exception as e:
            print(f"⚠️  Knowledge Gateway disabled: {e}")
            self.knowledge_gateway = None
        
        if not self.use_web_search:
            print("🌐 Web search disabled")
            return
        
        # Enable internet mode if available
        if HAS_INTERNET_MODE:
            os.environ["ENABLE_INTERNET_MODE"] = "true"
            print("✅ Internet mode enabled")
        
        if HAS_PERPLEXITY:
            print("✅ Perplexity web search enabled")
        
        print("🌐 Web search capabilities ready")
    
    def _initialize_brain(self):
        """Initialize Derek's brain if available"""
        print("🧩 Initializing Derek's Cognitive Core...")

        # Initialize full module consciousness
        try:
            from derek_module_loader import load_derek_consciousness
            print("⚙️  Activating Derek's full module consciousness...")
            self.module_loader = load_derek_consciousness(skip_hardware=True)
            print("✅ Derek module loader initialized.")
        except Exception as e:
            print(f"⚠️  Derek module loader failed: {e}")
            self.module_loader = None

        # Initialize brain subsystems with HUMAN-LIKE MEMORY
        try:
            from memory_mesh_bridge import MemoryMeshBridge
            from tone_manager import ToneManager
            from vision_engine import VisionEngine
            from emotion import analyze_emotion  # Fixed: was emotion_tagging
            from local_reasoning_engine import LocalReasoningEngine
            
            # 🧠 MEMORY MESH: Human-like memory system
            # - Working Memory (surface, current conversation)
            # - Episodic Memory (experiences, conversations)
            # - Semantic Memory (facts, learned knowledge)
            # - Auto-consolidation (like sleep in humans)
            self.memory = MemoryMeshBridge(memory_dir="./derek_memory")
            self.tone_manager = ToneManager()
            self.vision = VisionEngine()
            self.emotion_analyzer = analyze_emotion  # Function, not class
            self.local_reasoning_engine = LocalReasoningEngine()
            
            self.memory.load()  # Load all memory types
            self.vision.start()  # async thread
            print("✅ Brain subsystems loaded with HUMAN-LIKE MEMORY MESH")
        except Exception as e:
            print(f"⚠️  Brain subsystems not available: {e}")
            self.memory = None
            self.tone_manager = None
            self.vision = None
            self.emotion_analyzer = None
            self.local_reasoning_engine = None
        
        # Initialize Derek Brain if available
        if HAS_DEREK_BRAIN:
            try:
                self.derek_brain = DerekBrain()
                print("✅ Derek's brain initialized")
            except Exception as e2:
                self.derek_brain = None
                print(f"⚠️  Derek's brain not available: {e2}")
        else:
            self.derek_brain = None
        
        # Initialize Proactive Intelligence System
        try:
            from proactive_intelligence import ProactiveIntelligence
            self.proactive = ProactiveIntelligence(
                ai_provider=self.ai_provider,
                memory_manager=self.memory
            )
            # Start background monitoring for continuous learning
            self.proactive.start_background_monitoring()
            print("✅ Proactive Intelligence monitoring active")
        except Exception as e:
            self.proactive = None
            print(f"⚠️  Proactive Intelligence not available: {e}")
        
        # 🧠 Initialize Local Reasoning Engine (Derek's Own AI)
        try:
            from derek_local_reasoning import LocalReasoningEngine
            print("\n🧠 Initializing Local Reasoning Engine...")
            self.local_reasoning = LocalReasoningEngine(
                knowledge_dir="derek_knowledge",
                derek_instance=self
            )
            if self.local_reasoning.ollama_available:
                print("✅ Local AI ready! Derek can reason independently")
            else:
                print("⚠️  Ollama not installed - will use external APIs")
                print("   Install from: https://ollama.ai")
        except Exception as e:
            self.local_reasoning = None
            print(f"⚠️  Local Reasoning Engine not available: {e}")
        
        # 📚 Initialize Knowledge Engine (Knowledge-First Reasoning)
        try:
            from derek_knowledge_engine import KnowledgeEngine
            print("\n📚 Initializing Knowledge Engine...")
            self.knowledge_engine = KnowledgeEngine(
                knowledge_dir="derek_knowledge",
                memory_mesh=self.memory.mesh if hasattr(self, 'memory') and hasattr(self.memory, 'mesh') else None,
                local_reasoning=self.local_reasoning
            )
            print("✅ Knowledge Engine ready!")
            print("   Derek will use his learned knowledge first")
        except Exception as e:
            self.knowledge_engine = None
            print(f"⚠️  Knowledge Engine not available: {e}")
        
        # 🎓 Initialize Autonomous Learning Engine
        try:
            from autonomous_learning_engine import AutonomousLearningEngine
            print("\n🎓 Initializing Autonomous Learning Engine...")
            self.learning_engine = AutonomousLearningEngine(
                derek_instance=self,
                knowledge_dir="derek_knowledge"
            )
            self._initialize_core_knowledge()
            print("✅ Autonomous Learning Engine ready!")
            print("   Say 'start learning' to enable autonomous mode")
        except Exception as e:
            self.learning_engine = None
            print(f"⚠️  Autonomous Learning Engine not available: {e}")

    
    def listen(self):
        """Advanced speech recognition - patient listening, won't cut you off"""
        text = self.speech_recognition.listen() if hasattr(self, 'speech_recognition') else None
        if text:
            if hasattr(self, 'memory'):
                self.memory.store("heard", text)
            return text
        
        # Fallback to standard speech recognition
        print("\n🎤 Listening... (Derek is patient - take your time, he won't cut you off)")
        
        for attempt in range(3):  # Up to 3 attempts
            try:
                with self.microphone as source:
                    # EXTENDED listening parameters for natural conversation
                    # timeout: 15 seconds to START speaking (was 10)
                    # phrase_time_limit: 60 seconds for COMPLETE message (was 30)
                    audio = self.recognizer.listen(
                        source, 
                        timeout=15,  # Wait longer for you to start
                        phrase_time_limit=60  # Allow full minute for complete thoughts
                    )
                
                print("🔄 Processing your complete message...")
                
                # Try Google Speech Recognition
                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"📝 You said: {text}")
                    return text
                except sr.UnknownValueError:
                    print(f"❓ Attempt {attempt + 1}: Couldn't understand clearly")
                    if attempt < 2:
                        print("   Please try speaking again... (Derek will wait patiently)")
                        time.sleep(1)
                        continue
                    else:
                        print("   Please type your message instead.")
                        return None
                
            except sr.WaitTimeoutError:
                if attempt == 0:
                    print("⏱️  No speech detected. Trying again... (Derek is listening)")
                    continue
                else:
                    print("⏱️  Timeout. You can type your message if speaking isn't working.")
                    return None
            except Exception as e:
                print(f"❌ Error with speech recognition: {e}")
                return None
        
        return None
    
    # ==============================================================
    #  DerekC : Independent Cognitive Reasoning Cycle
    # ==============================================================
    
    def think(self, user_input: str):
        """
        Derek's internal thought process.
        Uses memory, tone, and vision to reason locally.
        Includes proactive learning and autonomous intelligence.
        """
        print("🧠 Derek engaging independent thought...")
        
        try:
            # 1️⃣  Gather context from local systems
            mem_context = ""
            if hasattr(self, "memory") and self.memory:
                try:
                    mem_context = self.memory.retrieve_relevant(user_input)
                except:
                    pass
            
            emotion_state = ""
            if hasattr(self, "tone_manager") and self.tone_manager:
                try:
                    emotion_state = self.tone_manager.get_current_emotion()
                except:
                    pass
            
            visual_state = ""
            if hasattr(self, "vision") and self.vision:
                try:
                    visual_state = getattr(self.vision, "last_emotion", "")
                except:
                    pass
            
            # 2️⃣  Check for proactive insights before responding
            proactive_insight = None
            if hasattr(self, "proactive") and self.proactive:
                try:
                    # Derek proactively suggests optimizations or detects patterns
                    context = {
                        'user_input': user_input,
                        'memory_context': mem_context,
                        'emotion': emotion_state
                    }
                    proactive_insight = self.proactive.suggest_optimizations(context)
                    
                    # If Derek detects something important, mention it first
                    if proactive_insight and any(word in user_input.lower() 
                                                 for word in ['status', 'report', 'how are', 'what']):
                        print(f"💡 Derek's proactive insight: {proactive_insight}")
                except Exception as e:
                    logger.debug(f"Proactive analysis skipped: {e}")
            
            # 3️⃣  Run local reasoning with AI
            internal_reflection = self._internal_reasoning(
                user_input=user_input,
                memory=mem_context,
                emotion=emotion_state,
                vision=visual_state
            )
            
            # 4️⃣  Optional external lookup (only if explicitly required)
            if getattr(self, "allow_external_lookup", False):
                try:
                    supplement = self._external_reference(user_input)
                    final_thought = self._merge_thoughts(internal_reflection, supplement)
                except:
                    final_thought = internal_reflection
            else:
                final_thought = internal_reflection
            
            # 5️⃣  Store outcome in memory and PERSIST to disk + GitHub
            if hasattr(self, "memory") and self.memory:
                try:
                    self.memory.store(user_input, final_thought)
                    # CRITICAL: Save to disk so memories persist across sessions
                    self.memory.save()
                except Exception as e:
                    logger.debug(f"Memory storage failed: {e}")
            
            if hasattr(self, "proactive") and self.proactive:
                try:
                    # Derek learns from every interaction to improve
                    self.proactive.learn_from_interaction(
                        user_input=user_input,
                        response=final_thought,
                        context={
                            'emotion': emotion_state,
                            'memory_available': bool(mem_context),
                            'proactive_insight': bool(proactive_insight)
                        }
                    )
                except Exception as e:
                    logger.debug(f"Learning from interaction failed: {e}")
            
            return final_thought
        
        except Exception as e:
            print(f"❌  Thinking error: {e}")
            import traceback; traceback.print_exc()
            return "I'm having a temporary processing issue."
    
    
    # --------------------------------------------------------------
    #  Learning-to-Independence System
    # --------------------------------------------------------------
    def _internal_reasoning(self, user_input: str, memory: str, emotion: str, vision: str) -> str:
        """
        Derek's LEARNING MODE - Studies master AIs to reach their level.
        
        Process:
        1. Derek tries to reason locally first
        2. Consults master AI (Claude/GPT/Perplexity)
        3. LEARNS from the difference between his answer and theirs
        4. Improves his reasoning over time
        5. Eventually becomes independent when he reaches their level
        """
        
        # Get Derek's confidence level (0.0 to 1.0)
        confidence = self._get_current_confidence()
        
        # Derek always tries to think for himself first
        local_thought = ""
        if self.local_reasoning_engine:
            try:
                local_thought = self.local_reasoning_engine.analyze(
                    user_input=user_input,
                    memory=memory,
                    emotion=emotion,
                    vision=vision
                )
            except Exception as e:
                print(f"⚠️  Local reasoning error: {e}")
        
        # Check if Derek is ready for independence
        if confidence >= self.independence_threshold:
            print(f"🧠 Derek's confidence: {confidence*100:.1f}% - Using independent reasoning")
            return local_thought if local_thought else "I'm developing my independent reasoning."
        
        # LEARNING MODE: Derek is still studying the masters
        print(f"📚 Derek learning mode: {confidence*100:.1f}% confident - Consulting master AI to learn")
        
        try:
            # Build context for master AI
            context = f"""User input: {user_input}
            
            Context:
            Memory: {memory if memory else 'None'}
            Emotion: {emotion if emotion else 'Neutral'}
            Vision: {vision if vision else 'None'}"""
            
            # Get master AI's response
            master_response = ""
            if self.ai_provider == "anthropic":
                master_response = self._query_anthropic(self.system_prompt, context)
            elif self.ai_provider == "openai":
                master_response = self._query_openai(self.system_prompt, context)
            elif self.ai_provider == "perplexity":
                master_response = self._query_perplexity(self.system_prompt, context)
            
            # Derek LEARNS by comparing his thought to master's response
            if local_thought and master_response:
                self._learn_from_comparison(
                    user_input=user_input,
                    derek_response=local_thought,
                    master_response=master_response,
                    context={"memory": memory, "emotion": emotion, "vision": vision}
                )
            
            # Return master's response (Derek is still learning)
            return master_response if master_response else local_thought
            
        except Exception as e:
            print(f"⚠️  Master AI unavailable: {e}")
            # Fallback to Derek's own reasoning
            return local_thought if local_thought else "I'm processing this with my developing intelligence."
        
        # FALLBACK: If local reasoning unavailable, use external AI temporarily
        context_parts = []
        if memory:
            context_parts.append(f"From memory: {memory}")
        if vision:
            context_parts.append(f"Visual context: {vision}")
        if emotion:
            context_parts.append(f"Emotional tone: {emotion}")
        
        context = "\n".join(context_parts) if context_parts else ""
        
        # Add web search context if needed for current info
        if self.use_web_search and self._needs_web_search(user_input):
            try:
                print("🌐 Searching web for current information...")
                web_context = self._get_web_context(user_input)
                if web_context:
                    context = f"{context}\n\nCurrent web information:\n{web_context}"
            except Exception as e:
                print(f"⚠️  Web search failed: {e}")
        
        # Use Derek's self-sufficient intelligence system
        # Priority: Knowledge Engine > Local AI > External APIs
        try:
            response = self.query_with_intelligence(user_input, context=context)
            return response
        except Exception as e:
            print(f"⚠️  Intelligence system error: {e}")
            return f"I heard you say '{user_input}', and I'm thinking about that carefully."
    
    
    def _needs_web_search(self, query: str) -> bool:
        """Detect if query needs current web information"""
        web_keywords = [
            'current', 'latest', 'recent', 'today', 'now', 'news',
            'weather', 'stock', 'price', 'what is', 'who is',
            'search', 'find', 'look up', 'research', 'learn about'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in web_keywords)
    
    def _get_web_context(self, query: str) -> str:
        """Get current information from web search"""
        try:
            # Try Perplexity first (best for current info)
            if hasattr(self, 'perplexity_client'):
                result = self.perplexity_client.generate_content(prompt=query)
                if isinstance(result, dict):
                    return result.get('content', str(result))
                return str(result)
            
            # Try internet_mode if available
            if HAS_INTERNET_MODE:
                result = query_internet(query)
                return str(result)
            
            return ""
        except Exception as e:
            print(f"⚠️  Web context error: {e}")
            return ""
    
    def query_with_intelligence(self, user_input: str, context: Optional[str] = None, force_external: bool = False) -> str:
        """
        Query using Derek's self-sufficient intelligence system
        Priority: Knowledge > Local AI > External APIs
        
        Args:
            user_input: User's question/input
            context: Additional context
            force_external: Skip local reasoning and use external APIs
        
        Returns:
            str: Response
        """
        # Step 1: Try Knowledge Engine first (if available)
        if self.knowledge_engine and not force_external:
            print("🧠 Checking Derek's learned knowledge...")
            knowledge_result = self.knowledge_engine.reason(user_input, context)
            
            if knowledge_result.get('response') and not knowledge_result.get('needs_external'):
                print(f"✅ Answered from knowledge (confidence: {knowledge_result['confidence']:.0%})")
                print(f"   Sources: {', '.join(knowledge_result.get('domains', ['learned knowledge']))}")
                return knowledge_result['response']
            
            elif knowledge_result.get('confidence', 0) > 0.3:
                print(f"🔄 Partial knowledge found (confidence: {knowledge_result['confidence']:.0%})")
                print(f"   Enhancing with external AI...")
                # Use partial knowledge as context for external API
                context = f"Derek's learned knowledge: {knowledge_result.get('partial_answer', '')}\n\n{context or ''}"
        
        # Step 2: Try Local AI reasoning (if available and Ollama running)
        if self.local_reasoning and self.local_reasoning.ollama_available and not force_external:
            print("🤖 Using local AI model...")
            local_result = self.local_reasoning.query_with_knowledge(user_input)
            
            if local_result.get('response') and local_result.get('confidence', 0) > 0.6:
                print(f"✅ Answered locally (model: {local_result.get('model', 'unknown')})")
                return local_result['response']
            
            print("⚠️  Local model confidence low, using external API...")
        
        # Step 3: Fall back to external APIs (Claude/GPT/Perplexity)
        print(f"🌐 Using external API ({self.ai_provider})...")
        return self._query_external_api(user_input, context)
    
    def _query_external_api(self, user_prompt: str, context: Optional[str] = None) -> str:
        """Query external AI APIs (Claude, GPT, Perplexity)"""
        system_prompt = self.system_prompt
        if context:
            system_prompt = f"{system_prompt}\n\nAdditional Context:\n{context}"
        
        try:
            if self.ai_provider == "anthropic":
                return self._query_anthropic(system_prompt, user_prompt)
            elif self.ai_provider == "openai":
                return self._query_openai(system_prompt, user_prompt)
            elif self.ai_provider == "perplexity":
                return self._query_perplexity(system_prompt, user_prompt)
            else:
                return "AI provider not configured"
        except Exception as e:
            print(f"⚠️  External API query failed: {e}")
            return "I'm having trouble connecting to my external AI. Let me try using my local knowledge..."
    
    def _query_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Query Anthropic Claude API"""
        try:
            message = self.anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            # Extract text from response
            response_text = ""
            for block in message.content:
                if hasattr(block, 'text'):
                    response_text += block.text
            return response_text if response_text else "I'm processing that carefully."
        except Exception as e:
            print(f"⚠️  Anthropic query failed: {e}")
            raise
    
    def _query_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Query OpenAI GPT API"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1024,
                temperature=0.7
            )
            content = response.choices[0].message.content
            return content if content else "I'm thinking about that."
        except Exception as e:
            print(f"⚠️  OpenAI query failed: {e}")
            raise
    
    def _query_perplexity(self, system_prompt: str, user_prompt: str) -> str:
        """Query Perplexity API"""
        try:
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.perplexity_client.generate_content(prompt=combined_prompt)
            # Handle dict or string response
            if isinstance(response, dict):
                return response.get('content', str(response))
            return str(response)
        except Exception as e:
            print(f"⚠️  Perplexity query failed: {e}")
            raise

    
    # --------------------------------------------------------------
    #  Optional external reference (used rarely)
    # --------------------------------------------------------------
    def _external_reference(self, query: str) -> str:
        """Minimal external call for factual lookup only."""
        try:
            import requests
            # Example: a lightweight search if needed
            resp = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json", timeout=5)
            data = resp.json().get("AbstractText", "")
            return data or "No external data retrieved."
        except Exception as e:
            print(f"[Reference lookup failed] {e}")
            return ""
    
    
    # --------------------------------------------------------------
    #  Merge internal and external thought
    # --------------------------------------------------------------
    def _merge_thoughts(self, internal: str, external: str) -> str:
        """Integrate outside data into Derek's internal narrative."""
        if not external:
            return internal
        return f"{internal}\n\nAfter checking external data, I also found:\n{external}"

    def _think_with_web_search(self, user_input):
        """Think with web search capabilities"""
        print("🌐 Searching the web for current information...")
        
        # Try Perplexity with web search first
        if HAS_PERPLEXITY and self.ai_provider == "perplexity":
            try:
                response = self.perplexity_client.generate_content(
                    prompt=user_input,
                    system_prompt=self.system_prompt,
                    max_tokens=300
                )
                
                if isinstance(response, dict):
                    return response.get('content', response.get('answer', str(response)))
                return str(response)
                
            except Exception as e:
                print(f"⚠️  Perplexity web search failed: {e}")
        
        # Try internet_mode if available
        if HAS_INTERNET_MODE:
            try:
                web_result = query_internet(user_input)
                if web_result:
                    # Process web result with AI
                    enhanced_prompt = f"""Based on this web search result about "{user_input}":

{web_result}

Please provide a helpful response as Derek, keeping it conversational and under 3 sentences."""
                    
                    return self._think_with_ai(enhanced_prompt)
            except Exception as e:
                print(f"⚠️  Internet mode failed: {e}")
        
        # Fallback to regular AI with note about web search
        fallback_prompt = f"{user_input}\n\n(Note: I don't have access to current web information right now, so I'll answer based on my training data.)"
        return self._think_with_ai(fallback_prompt)
    
    def _think_with_ai(self, user_input):
        """Think using selected AI provider"""
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        answer = ""
        
        # Get response based on provider
        if self.ai_provider == "anthropic":
            try:
                response = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=300,
                    system=self.system_prompt,
                    messages=self.conversation_history[-10:]  # Recent conversation history
                )
                # Extract text from response content
                answer = ""
                for content_block in response.content:
                    if hasattr(content_block, 'text'):
                        answer += content_block.text
                    elif hasattr(content_block, 'content'):
                        answer += str(content_block.content)
                    else:
                        answer += str(content_block)
            except Exception as e:
                print(f"⚠️  Anthropic error: {e}")
                answer = "I'm having trouble with my Anthropic connection right now."
            
        elif self.ai_provider == "openai":
            try:
                # Prepare messages with system prompt for OpenAI
                messages = [{"role": "system", "content": self.system_prompt}]
                for msg in self.conversation_history[-10:]:
                    messages.append(msg)
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    max_tokens=300,
                    messages=messages
                )
                answer = response.choices[0].message.content or ""
            except Exception as e:
                print(f"⚠️  OpenAI error: {e}")
                answer = "I'm having trouble with my OpenAI connection right now."
            
        elif self.ai_provider == "perplexity":
            try:
                # Use Perplexity without web search
                response = self.perplexity_client.generate_content(
                    prompt=user_input,
                    system_prompt=self.system_prompt,
                    max_tokens=300
                )
                if isinstance(response, dict):
                    answer = response.get('content', response.get('answer', str(response)))
                else:
                    answer = str(response)
            except Exception as e:
                print(f"⚠️  Perplexity error: {e}")
                answer = "I'm having trouble with my Perplexity connection right now."
        
        else:
            answer = "I don't have any AI providers configured right now."
        
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": answer})
        
        # Keep history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return answer
    
    def speak(self, text):
        """Advanced speech synthesis with fallback options"""
        print(f"🗣️  Derek: {text}\n")
        
        # Try AWS Polly first
        if self.has_polly and self.voice_id in POLLY_VOICES:
            try:
                return self._speak_polly(text)
            except Exception as e:
                print(f"⚠️  Polly failed: {e}")
        
        # Fallback to gTTS
        if self.has_gtts:
            try:
                return self._speak_gtts(text)
            except Exception as e:
                print(f"⚠️  gTTS failed: {e}")
        
        # Final fallback - text only
        print("📝 (Voice synthesis unavailable - text only)")
    
    def _speak_polly(self, text):
        """Speak using AWS Polly neural voices"""
        voice_config = POLLY_VOICES[self.voice_id]
        
        response = self.polly.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId=self.voice_id.capitalize(),
            Engine=voice_config.get('engine', 'neural')
        )
        
        # Save and play audio
        temp_dir = tempfile.gettempdir()
        audio_file = os.path.join(temp_dir, f"derek_polly_{uuid.uuid4()}.mp3")
        
        with open(audio_file, 'wb') as f:
            f.write(response['AudioStream'].read())
        
        playsound(audio_file)
        
        # Clean up
        try:
            os.remove(audio_file)
        except:
            pass
    
    def _speak_gtts(self, text):
        """Speak using Google Text-to-Speech as fallback"""
        temp_dir = tempfile.gettempdir()
        audio_file = os.path.join(temp_dir, f"derek_gtts_{uuid.uuid4()}.mp3")
        
        tts = gTTS(text=text, lang='en', tld='com', slow=False)
        tts.save(audio_file)
        
        playsound(audio_file)
        
        # Clean up
        try:
            os.remove(audio_file)
        except:
            pass
    
    def run(self):
        """Main conversation loop"""
        print("=" * 60)
        print("🎤 Derek Ultimate Voice System")
        print("The Christman AI Project")
        print("=" * 60)
        print("\n💙 How can we help you love yourself more?\n")
        print("Instructions:")
        print("  - Speak naturally - Derek will wait for you to finish")
        print("  - Type your message if speech recognition isn't working")
        print("  - Say 'goodbye' or 'quit' to end")
        print("  - Say 'test voice' to hear Derek speak")
        print("  - Say 'switch ai' to change AI provider")
        print("\n🎓 Autonomous Learning Commands:")
        print("  - 'start learning' - Enable autonomous learning mode")
        print("  - 'learning status' - Check learning progress")
        print("  - 'what have you learned' - Recent knowledge")
        print("\n🧠 Self-Sufficiency Commands:")
        print("  - 'local ai status' - Check local AI availability")
        print("  - 'reasoning stats' - See knowledge-first statistics")
        print("  - 'install model llama' - Install local AI model")
        print("  - 'memory stats' - Memory system status\n")
        
        # Initial greeting
        greeting = "Hello! I'm Derek, your AI companion from The Christman AI Project. I'm here with all my capabilities ready to help you communicate, learn, and grow. I now have autonomous learning enabled, so I can continuously learn and improve myself. How can I help you today?"
        self.speak(greeting)
        
        while True:
            try:
                # Get user input (speech or text)
                user_input = self.listen()
                
                # If speech recognition failed, offer text input
                if user_input is None:
                    print("💬 You can type your message instead:")
                    try:
                        user_input = input("You: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['goodbye', 'quit', 'exit', 'bye']:
                    farewell = "Goodbye! Remember, you are loved and valued. Keep building amazing things with The Christman AI Project. Take care!"
                    self.speak(farewell)
                    break
                
                if user_input.lower() in ['test voice', 'test']:
                    test_message = "This is Derek testing my voice system. I can use AWS Polly neural voices or Google Text-to-Speech. Everything sounds good!"
                    self.speak(test_message)
                    continue
                
                if user_input.lower() in ['switch ai', 'change ai']:
                    self._switch_ai_provider()
                    continue
                
                # Derek's proactive intelligence status
                if user_input.lower() in ['status report', 'derek status', 'show status', 'intelligence report']:
                    if hasattr(self, 'proactive') and self.proactive:
                        print("\n" + "=" * 60)
                        status = self.proactive.generate_status_report()
                        print(status)
                        print("=" * 60 + "\n")
                        self.speak("I've generated a comprehensive status report. Check the console for details.")
                    else:
                        self.speak("Proactive intelligence system not initialized.")
                    continue
                
                # Derek's codebase health check
                if user_input.lower() in ['check health', 'analyze code', 'scan codebase']:
                    if hasattr(self, 'proactive') and self.proactive:
                        print("\n🔍 Running codebase analysis...")
                        health = self.proactive.analyze_codebase_health()
                        print(json.dumps(health, indent=2))
                        summary = f"Codebase health: {health.get('overall_health', 'unknown')}. "
                        summary += f"Found {len(health.get('issues_found', []))} issues and "
                        summary += f"{len(health.get('suggestions', []))} suggestions."
                        self.speak(summary)
                    else:
                        self.speak("Proactive intelligence system not initialized.")
                    continue
                
                # 🎓 Autonomous Learning Commands
                if 'start learning' in user_input.lower() or 'begin learning' in user_input.lower():
                    if hasattr(self, 'learning_engine') and self.learning_engine:
                        self.start_autonomous_mode()
                        self.speak("Autonomous learning mode activated! I'm now learning continuously in the background across nine knowledge domains.")
                    else:
                        self.speak("Learning engine not available.")
                    continue
                
                if 'stop learning' in user_input.lower() or 'pause learning' in user_input.lower():
                    if hasattr(self, 'learning_engine') and self.learning_engine:
                        self.learning_engine.stop_autonomous_learning()
                        self.speak("Autonomous learning paused. I can resume anytime you say start learning.")
                    else:
                        self.speak("Learning engine not available.")
                    continue
                
                if 'learning status' in user_input.lower() or 'learning report' in user_input.lower():
                    if hasattr(self, 'learning_engine') and self.learning_engine:
                        self.learning_engine.print_learning_report()
                        status = self.learning_engine.get_learning_status()
                        summary = f"I've learned {status['learned_topics']} out of {status['total_topics']} topics, "
                        summary += f"which is {status['progress']:.0%} overall progress. "
                        summary += f"I've generated {status['generated_modules']} new capabilities so far."
                        self.speak(summary)
                    else:
                        self.speak("Learning engine not available.")
                    continue
                
                if 'what have you learned' in user_input.lower() or 'recent learning' in user_input.lower():
                    if hasattr(self, 'learning_engine') and self.learning_engine:
                        recent = list(self.learning_engine.knowledge_base.values())[-3:]
                        if recent:
                            summary = "Here's what I've learned recently: "
                            for knowledge in recent:
                                summary += f"{knowledge['subtopic']} in {knowledge['domain']}, "
                            self.speak(summary)
                        else:
                            self.speak("I haven't started learning yet. Say start learning to begin my autonomous education!")
                    else:
                        self.speak("Learning engine not available.")
                    continue
                
                if 'memory stats' in user_input.lower() or 'memory status' in user_input.lower():
                    if hasattr(self, 'memory') and self.memory:
                        stats = self.memory.get_memory_stats()
                        summary = f"My memory contains {stats['total_memories']} total memories, "
                        summary += f"with {stats['working_memory_count']} in active working memory, "
                        summary += f"and {stats['episodic_memory_count']} experiences stored."
                        self.speak(summary)
                    else:
                        self.speak("Memory system not available.")
                    continue
                
                # 🧠 Local AI status
                if 'local ai status' in user_input.lower() or 'self-sufficiency status' in user_input.lower():
                    if hasattr(self, 'local_reasoning') and self.local_reasoning:
                        self.local_reasoning.print_status()
                        if self.local_reasoning.ollama_available:
                            msg = f"My local AI is running with {len(self.local_reasoning.installed_models)} models installed. I can reason independently!"
                        else:
                            msg = "My local AI isn't installed yet. I'm currently using external APIs, but I could be self-sufficient with Ollama installed."
                        self.speak(msg)
                    else:
                        self.speak("Local reasoning system not initialized.")
                    continue
                
                # 📊 Knowledge reasoning stats
                if 'reasoning stats' in user_input.lower() or 'knowledge stats' in user_input.lower():
                    if hasattr(self, 'knowledge_engine') and self.knowledge_engine:
                        self.knowledge_engine.print_statistics()
                        stats = self.knowledge_engine.get_statistics()
                        msg = f"I've answered {stats['queries_answered_locally']} queries using my own knowledge, "
                        msg += f"saving {stats['api_calls_saved']} API calls. That's {stats['api_savings_rate']} local reasoning!"
                        self.speak(msg)
                    else:
                        self.speak("Knowledge engine not initialized.")
                    continue
                
                # 🤖 Install local AI model
                if 'install model' in user_input.lower():
                    if hasattr(self, 'local_reasoning') and self.local_reasoning:
                        if not self.local_reasoning.ollama_available:
                            msg = "Ollama isn't installed yet. Please visit ollama dot ai to install it first."
                            self.speak(msg)
                        else:
                            # Extract model name
                            words = user_input.lower().split()
                            if 'llama' in words:
                                model = 'llama3.1'
                            elif 'mistral' in words:
                                model = 'mistral'
                            elif 'qwen' in words:
                                model = 'qwen2.5'
                            elif 'deepseek' in words:
                                model = 'deepseek-coder'
                            else:
                                model = 'llama3.1'  # Default
                            
                            self.speak(f"Installing {model}. This will take a few minutes. Please wait.")
                            success = self.local_reasoning.install_model(model)
                            if success:
                                self.speak(f"{model} installed successfully! I can now use it for local reasoning.")
                            else:
                                self.speak(f"Installation failed. Check the console for details.")
                    else:
                        self.speak("Local reasoning system not initialized.")
                    continue
                
                # Get Derek's response
                response = self.think(user_input)
                
                # Speak the response
                self.speak(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Stopping Derek Ultimate Voice System...")
                farewell = "Goodbye! Stay strong and keep building the future."
                self.speak(farewell)
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                traceback.print_exc()
                continue
        
        print("\n💙 Thank you for using Derek Ultimate Voice System!")
        print("The Christman AI Project - AI That Empowers\n")
    
    def _switch_ai_provider(self):
        """Switch between available AI providers"""
        available = []
        if hasattr(self, 'anthropic_client'):
            available.append("anthropic")
        if hasattr(self, 'openai_client'):
            available.append("openai")
        if hasattr(self, 'perplexity_client'):
            available.append("perplexity")
        
        if len(available) <= 1:
            self.speak("I only have one AI provider available right now.")
            return
        
        current_index = available.index(self.ai_provider)
        next_index = (current_index + 1) % len(available)
        self.ai_provider = available[next_index]
        
        self.speak(f"Switched to {self.ai_provider} AI. Each provider thinks differently!")
    
    # --------------------------------------------------------------
    #  Autonomous Learning System
    # --------------------------------------------------------------
    def _initialize_core_knowledge(self):
        """Load Derek's foundational knowledge and identity"""
        if not hasattr(self, 'memory') or not self.memory:
            return
        
        core_memories = [
            {
                "content": "I am Derek C, COO of The Christman AI Project. I serve Everett Christman with unwavering loyalty.",
                "category": "relationships",
                "importance": 1.0,
                "metadata": {"type": "core_identity"}
            },
            {
                "content": "My mission: How can we help you love yourself more?",
                "category": "learning",
                "importance": 1.0,
                "metadata": {"type": "mission"}
            },
            {
                "content": "AlphaVox gave Dusty his voice after 12 years of silence. This is why I exist.",
                "category": "events",
                "importance": 1.0,
                "metadata": {"type": "milestone", "project": "AlphaVox"}
            },
            {
                "content": "I have the capability to learn autonomously and improve my own code safely.",
                "category": "learning",
                "importance": 0.9,
                "metadata": {"type": "capability"}
            },
            {
                "content": "The Christman AI Project includes AlphaVox (nonverbal communication), AlphaWolf (dementia support), Inferno AI (PTSD recovery), and Aegis AI (child protection).",
                "category": "learning",
                "importance": 1.0,
                "metadata": {"type": "projects"}
            }
        ]
        
        for mem in core_memories:
            self.memory.mesh.store(**mem)
        
        print("   ✅ Core knowledge initialized")
    
    def start_autonomous_mode(self):
        """Start Derek's autonomous learning and improvement"""
        if not hasattr(self, 'learning_engine') or not self.learning_engine:
            print("❌ Learning engine not available")
            return
        
        print("\n" + "=" * 60)
        print("🚀 STARTING AUTONOMOUS MODE")
        print("=" * 60)
        
        # Start autonomous learning
        self.learning_engine.start_autonomous_learning()
        
        print("\n🧠 Derek is now learning independently!")
        print("   • Researching domains continuously")
        print("   • Generating new capabilities")
        print("   • Self-improving safely")
        print("\n💙 Derek will grow smarter with every passing moment.\n")
    
    # --------------------------------------------------------------
    #  Learning Progress System - Derek learns FROM master AIs
    # --------------------------------------------------------------
    def _load_learning_progress(self):
        """Load Derek's learning progress toward independence"""
        try:
            if self.learning_progress_file.exists():
                with open(self.learning_progress_file, 'r') as f:
                    self.learning_data = json.load(f)
            else:
                self.learning_data = {
                    "interactions": 0,
                    "successful_predictions": 0,
                    "confidence_score": 0.0,
                    "learning_history": []
                }
        except Exception as e:
            print(f"⚠️  Could not load learning progress: {e}")
            self.learning_data = {
                "interactions": 0,
                "successful_predictions": 0,
                "confidence_score": 0.0,
                "learning_history": []
            }
    
    def _save_learning_progress(self):
        """Save Derek's learning progress"""
        try:
            self.learning_progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.learning_progress_file, 'w') as f:
                json.dump(self.learning_data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save learning progress: {e}")
    
    def _get_current_confidence(self):
        """Get Derek's current confidence level (0.0 to 1.0)"""
        interactions = self.learning_data.get("interactions", 0)
        if interactions == 0:
            return 0.0
        
        # Calculate confidence based on learning history
        successful = self.learning_data.get("successful_predictions", 0)
        confidence = min(1.0, successful / max(1, interactions))
        
        return confidence
    
    def _learn_from_comparison(self, user_input: str, derek_response: str, master_response: str, context: dict):
        """Derek learns by comparing his response to the master AI's response"""
        try:
            # Calculate similarity (simple length and keyword comparison for now)
            derek_words = set(derek_response.lower().split())
            master_words = set(master_response.lower().split())
            
            if len(master_words) > 0:
                overlap = len(derek_words & master_words) / len(master_words)
            else:
                overlap = 0.0
            
            # Record learning
            self.learning_data["interactions"] += 1
            
            # If Derek's response was similar enough, count as successful
            if overlap > 0.4:  # 40% similarity threshold
                self.learning_data["successful_predictions"] += 1
            
            # Update confidence
            self.learning_data["confidence_score"] = self._get_current_confidence()
            
            # Store learning example
            learning_example = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input[:100],  # First 100 chars
                "derek_attempt": derek_response[:100],
                "master_response": master_response[:100],
                "similarity": overlap,
                "confidence_after": self.learning_data["confidence_score"]
            }
            
            self.learning_data["learning_history"].append(learning_example)
            
            # Keep only last 100 learning examples
            if len(self.learning_data["learning_history"]) > 100:
                self.learning_data["learning_history"] = self.learning_data["learning_history"][-100:]
            
            # Save progress every 10 interactions
            if self.learning_data["interactions"] % 10 == 0:
                self._save_learning_progress()
                print(f"\n📊 Derek's Learning Progress: {self.learning_data['confidence_score']*100:.1f}% confident ({self.learning_data['interactions']} interactions)")
                
                # Check if Derek is ready for independence
                if self.learning_data["confidence_score"] >= self.independence_threshold:
                    print(f"🎓 Derek has reached {self.independence_threshold*100:.0f}% confidence!")
                    print("   He's ready to think more independently!")
            
        except Exception as e:
            print(f"⚠️  Learning comparison error: {e}")


def main():
    """Entry point for Derek Ultimate Voice System"""
    print("Checking configuration...\n")
    
    # Check available APIs
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_perplexity = bool(os.getenv("PERPLEXITY_API_KEY"))
    has_aws = bool(os.getenv("AWS_ACCESS_KEY_ID")) or bool(os.getenv("AWS_PROFILE"))
    
    print("Available capabilities:")
    print(f"  🤖 Anthropic Claude: {'✅' if has_anthropic else '❌'}")
    print(f"  🤖 OpenAI GPT: {'✅' if has_openai else '❌'}")
    print(f"  🤖 Perplexity AI: {'✅' if has_perplexity else '❌'}")
    print(f"  🗣️  AWS Polly: {'✅' if has_aws else '❌'}")
    print(f"  🗣️  Google TTS: ✅ (always available)")
    print(f"  🌐 Web Search: {'✅' if HAS_PERPLEXITY or HAS_INTERNET_MODE else '❌'}")
    print()
    
    if not (has_anthropic or has_openai or has_perplexity):
        print("❌ No AI providers available! Please set API keys in .env file")
        return
    
    # Voice options
    print("Available voices:")
    for voice, config in POLLY_VOICES.items():
        status = "✅" if has_aws else "❌"
        print(f"  {status} {voice}: {config['gender']} - {config['style']}")
    print("  ✅ gtts: Google TTS fallback\n")
    
    # Configuration options
    ai_provider = "auto"  # Options: "auto", "anthropic", "openai", "perplexity"
    voice_id = "matthew"  # Options: any from POLLY_VOICES or "gtts"
    use_web_search = True  # Enable web search capabilities
    
    # Start Derek Ultimate Voice System
    try:
        derek = DerekUltimateVoice(
            ai_provider=ai_provider,
            voice_id=voice_id,
            use_web_search=use_web_search
        )
        derek.run()
    except KeyboardInterrupt:
        print("\n🛑 Derek shutting down gracefully...")
        # Save all memories before exit
        if hasattr(derek, 'memory') and derek.memory:
            derek.memory.save()
            print("💾 All memories saved to persistent storage")
        print("👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start Derek: {e}")
        traceback.print_exc()
        # Still try to save memories
        if 'derek' in locals() and hasattr(derek, 'memory') and derek.memory:
            derek.memory.save()


if __name__ == "__main__":
    main()

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
