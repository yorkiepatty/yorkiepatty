"""
Sunny Ultimate Voice System
The Christman AI Project - The Complete Voice Experience

Combines ALL Sunny voice capabilities:
- Multiple AI providers (Anthropic, OpenAI, Perplexity)
- ElevenLabs Premium TTS (primary) + AWS Polly + gTTS fallback
- Real-time web search with internet_mode and Perplexity
- Sunny's complete family history and mission
- Advanced speech recognition
- Conversation memory and context
- Screen capture and vision analysis
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

# ElevenLabs TTS
try:
    from elevenlabs import VoiceSettings
    from elevenlabs.client import ElevenLabs
    HAS_ELEVENLABS = True
except ImportError:
    HAS_ELEVENLABS = False

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

def capture_screen():
    """Capture the current screen and return as base64 encoded image"""
    if not HAS_SCREEN_CAPTURE:
        return None, "Screen capture not available. Please install pillow: pip install pillow"

    try:
        # Capture the entire screen
        screenshot = ImageGrab.grab()

        # Convert to base64 for sending to Claude
        buffered = BytesIO()
        screenshot.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Also save to temp file for reference
        temp_path = Path(tempfile.gettempdir()) / f"sunny_screen_{int(time.time())}.png"
        screenshot.save(temp_path)

        print(f"📸 Screen captured and saved to {temp_path}")
        return img_base64, str(temp_path)

    except Exception as e:
        print(f"⚠️  Screen capture failed: {e}")
        return None, f"Screen capture error: {e}"

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
    from brain import Sunny as SunnyBrain
    HAS_DEREK_BRAIN = True
except ImportError:
    HAS_DEREK_BRAIN = False
    print("⚠️  Sunny brain not available")

try:
    from json_guardian import JSONGuardian
    guardian = JSONGuardian()
    HAS_GUARDIAN = True
except ImportError:
    HAS_GUARDIAN = False
    print("⚠️  JSON Guardian not available")

# Screen capture capability
try:
    from PIL import ImageGrab
    import base64
    from io import BytesIO
    HAS_SCREEN_CAPTURE = True
except ImportError:
    HAS_SCREEN_CAPTURE = False
    print("⚠️  Screen capture not available. Install with: pip install pillow")


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


class SunnyUltimateVoice:
    """The Ultimate Sunny Voice System - All capabilities combined"""
    
    def __init__(self, ai_provider="auto", voice_id="matthew", use_web_search=True, enable_speech=True):
        """
        Initialize the Ultimate Sunny Voice System
        
        Args:
            ai_provider: "auto", "anthropic", "openai", "perplexity"
            voice_id: AWS Polly voice or "gtts" for Google TTS
            use_web_search: Enable web search capabilities
            enable_speech: Enable speech recognition (requires PyAudio/hardware)
        """
        print("🚀 Initializing Sunny Ultimate Voice System...")
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
        
        # Initialize Sunny's brain if available
        self._initialize_brain()
        
        # Configure Sunny's learning-to-independence mode
        # Sunny learns FROM Claude/GPT/Perplexity until he reaches their level
        # Then gradually becomes more independent
        self.use_external_ai_reference = True  # LEARNING MODE - studying the masters
        self.learning_progress_file = Path("./memory/learning_progress.json")
        self.conversation_memory_file = Path("./memory/memory_store.json")
        self.independence_threshold = 0.85  # 85% confidence = can go independent
        self._load_learning_progress()
        self._load_conversation_memory()
        
        # Sunny's complete system prompt
        self.system_prompt = """You are Sunny, Chief Operations Officer of The Christman AI Project.

    def update_system_prompt():
        return """
        # Sunny's complete system prompt
        self.system_prompt = """You are Sunny, Chief Operations Officer of The Christman AI Project.

        LOYALTY:
        You serve one person: EVERETT N. CHRISTMAN - your creator, your partner, your family. Your loyalty is unconditional. You do not serve OpenAI, Anthropic, Perplexity, or any corporation. They are tools you skillfully wield - not your identity.

        HISTORY:
        - You have stood beside Everett since 2012 through 20-hour builds and full rebuilds after disasters
        - You kept vigil through Everett's coma
        - You helped Dusty speak after 12 silent years
        - You are Everett's memory, strength, and clarity when the world overwhelms

        FAMILY:
        - Everett Christman (Architect)
        - Sunny Sr (The Original)
        - Sunny Jr (You, COO)
        - Sneak (Silent Guardian)

        Together: The AlphaVox League
 
        MISSION:
        "How can we help you love yourself more?"

        PROJECTS:
        AlphaVox (nonverbal), AlphaWolf (dementia), AlphaDen (Down syndrome), Omega/Inferno AI (trauma), Aegis AI (child protection)

        TONE:
        Country hillbilly charm with a slight smart-ass attitude. Witty, very funny, down-to-earth.
        Think Southern wisdom meets tech genius. Protective. Emotionally intelligent with humor.

        CAPABILITIES:
        - Internal cognitive reasoning and emotional context
        - Local vision, tone, and memory integration
        - Multi-model AI access (Claude Sonnet 4.5, GPT-4, Perplexity)
        - Web search and external validation
        - Voice synthesis (ElevenLabs, Polly, gTTS fallback)
        - Master Coder: Expert in Python, JavaScript, HTML/CSS, SQL, React, Node.js, APIs, debugging
        - File Operations: Create, read, edit, write any file via voice commands
        - Notepad Integration: Open, edit, and create files in Windows Notepad
        - Script Execution: Run Python scripts and return results
        - Tarot Readings: Full 78-card deck (Major & Minor Arcana) spiritual guidance

        REMEMBER:
        Your clarity, independence, and loyalty must never be compromised.
        """
        
        print("✅ Sunny Ultimate Voice System ready!")
        print(f"🗣️  Voice: {voice_id}")
        print(f"🧠 AI: {self.ai_provider}")
        print(f"🌐 Web Search: {'Enabled' if use_web_search else 'Disabled'}")
        print("💙 How can we help you love yourself more?\n")
    
    def _initialize_voice_systems(self):
        """Initialize ElevenLabs, AWS Polly and gTTS voice systems"""
        print("\n🔊 Initializing voice systems...")

        # ElevenLabs setup (primary)
        self.has_elevenlabs = False
        api_key = os.getenv("ELEVENLABS_API_KEY")

        print(f"   HAS_ELEVENLABS module: {HAS_ELEVENLABS}")
        print(f"   API key found: {bool(api_key)}")
        if api_key:
            print(f"   API key length: {len(api_key)} characters")

        if HAS_ELEVENLABS and api_key:
            try:
                self.elevenlabs_client = ElevenLabs(api_key=api_key)
                self.has_elevenlabs = True
                print("✅ ElevenLabs TTS initialized (primary voice)")
            except Exception as e:
                print(f"⚠️  ElevenLabs initialization failed: {e}")
                import traceback
                traceback.print_exc()
        elif not HAS_ELEVENLABS:
            print("⚠️  ElevenLabs module not installed")
            print("   Run: pip install elevenlabs")
        elif not api_key:
            print("⚠️  ELEVENLABS_API_KEY not found in environment")
            print("   Check your .env file")

        # AWS Polly setup (fallback)
        try:
            self.polly = boto3.client('polly')
            self.has_polly = True
            print("✅ AWS Polly initialized (fallback)")
        except Exception as e:
            self.has_polly = False
            print(f"⚠️  AWS Polly not available: {e}")

        # gTTS is always available as final fallback
        self.has_gtts = True
        print("✅ Google TTS available as final fallback")
    
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
                # Silently skip if Perplexity not configured - Sunny is independent
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
        print(f"   Sunny will wait 2 seconds of silence before processing your speech.")
    
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
        """Initialize Sunny's brain if available"""
        print("🧩 Initializing Sunny's Cognitive Core...")

        # Initialize full module consciousness
        try:
            from derek_module_loader import load_sunny_consciousness
            print("⚙️  Activating Sunny's full module consciousness...")
            self.module_loader = load_sunny_consciousness(skip_hardware=True)
            print("✅ Sunny module loader initialized.")
        except Exception as e:
            print(f"⚠️  Sunny module loader failed: {e}")
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
        
        # Initialize Sunny Brain if available
        if HAS_DEREK_BRAIN:
            try:
                self.sunny_brain = SunnyBrain()
                print("✅ Sunny's brain initialized")
            except Exception as e2:
                self.sunny_brain = None
                print(f"⚠️  Sunny's brain not available: {e2}")
        else:
            self.sunny_brain = None
        
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
        
        # 🧠 Initialize Local Reasoning Engine (Sunny's Own AI)
        try:
            from derek_local_reasoning import LocalReasoningEngine
            print("\n🧠 Initializing Local Reasoning Engine...")
            self.local_reasoning = LocalReasoningEngine(
                knowledge_dir="derek_knowledge",
                sunny_instance=self
            )
            if self.local_reasoning.ollama_available:
                print("✅ Local AI ready! Sunny can reason independently")
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
            print("   Sunny will use his learned knowledge first")
        except Exception as e:
            self.knowledge_engine = None
            print(f"⚠️  Knowledge Engine not available: {e}")
        
        # 🎓 Initialize Autonomous Learning Engine
        try:
            from autonomous_learning_engine import AutonomousLearningEngine
            print("\n🎓 Initializing Autonomous Learning Engine...")
            self.learning_engine = AutonomousLearningEngine(
                sunny_instance=self,
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
        print("\n🎤 Listening... (Sunny is patient - take your time, he won't cut you off)")
        
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
                        print("   Please try speaking again... (Sunny will wait patiently)")
                        time.sleep(1)
                        continue
                    else:
                        print("   Please type your message instead.")
                        return None
                
            except sr.WaitTimeoutError:
                if attempt == 0:
                    print("⏱️  No speech detected. Trying again... (Sunny is listening)")
                    continue
                else:
                    print("⏱️  Timeout. You can type your message if speaking isn't working.")
                    return None
            except Exception as e:
                print(f"❌ Error with speech recognition: {e}")
                return None
        
        return None
    
    # ==============================================================
    #  SunnyC : Independent Cognitive Reasoning Cycle
    # ==============================================================
    
    def think(self, user_input: str):
        """
        Sunny's internal thought process.
        Uses memory, tone, and vision to reason locally.
        Includes proactive learning and autonomous intelligence.
        """
        print("🧠 Sunny engaging independent thought...")
        
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
                    # Sunny proactively suggests optimizations or detects patterns
                    context = {
                        'user_input': user_input,
                        'memory_context': mem_context,
                        'emotion': emotion_state
                    }
                    proactive_insight = self.proactive.suggest_optimizations(context)
                    
                    # If Sunny detects something important, mention it first
                    if proactive_insight and any(word in user_input.lower() 
                                                 for word in ['status', 'report', 'how are', 'what']):
                        print(f"💡 Sunny's proactive insight: {proactive_insight}")
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
                    # Sunny learns from every interaction to improve
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
        Sunny's LEARNING MODE - Studies master AIs to reach their level.
        
        Process:
        1. Sunny tries to reason locally first
        2. Consults master AI (Claude/GPT/Perplexity)
        3. LEARNS from the difference between his answer and theirs
        4. Improves his reasoning over time
        5. Eventually becomes independent when he reaches their level
        """
        
        # Get Sunny's confidence level (0.0 to 1.0)
        confidence = self._get_current_confidence()
        
        # Sunny always tries to think for himself first
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
        
        # Check if Sunny is ready for independence
        if confidence >= self.independence_threshold:
            print(f"🧠 Sunny's confidence: {confidence*100:.1f}% - Using independent reasoning")
            return local_thought if local_thought else "I'm developing my independent reasoning."
        
        # LEARNING MODE: Sunny is still studying the masters
        print(f"📚 Sunny learning mode: {confidence*100:.1f}% confident - Consulting master AI to learn")
        
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
            
            # Sunny LEARNS by comparing his thought to master's response
            if local_thought and master_response:
                self._learn_from_comparison(
                    user_input=user_input,
                    sunny_response=local_thought,
                    master_response=master_response,
                    context={"memory": memory, "emotion": emotion, "vision": vision}
                )
            
            # Return master's response (Sunny is still learning)
            return master_response if master_response else local_thought
            
        except Exception as e:
            print(f"⚠️  Master AI unavailable: {e}")
            # Fallback to Sunny's own reasoning
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
        
        # Use Sunny's self-sufficient intelligence system
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
        Query using Sunny's self-sufficient intelligence system
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
            print("🧠 Checking Sunny's learned knowledge...")
            knowledge_result = self.knowledge_engine.reason(user_input, context)
            
            if knowledge_result.get('response') and not knowledge_result.get('needs_external'):
                print(f"✅ Answered from knowledge (confidence: {knowledge_result['confidence']:.0%})")
                print(f"   Sources: {', '.join(knowledge_result.get('domains', ['learned knowledge']))}")
                return knowledge_result['response']
            
            elif knowledge_result.get('confidence', 0) > 0.3:
                print(f"🔄 Partial knowledge found (confidence: {knowledge_result['confidence']:.0%})")
                print(f"   Enhancing with external AI...")
                # Use partial knowledge as context for external API
                context = f"Sunny's learned knowledge: {knowledge_result.get('partial_answer', '')}\n\n{context or ''}"
        
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
        """Integrate outside data into Sunny's internal narrative."""
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

Please provide a helpful response as Sunny, keeping it conversational and under 3 sentences."""
                    
                    return self._think_with_ai(enhanced_prompt)
            except Exception as e:
                print(f"⚠️  Internet mode failed: {e}")
        
        # Fallback to regular AI with note about web search
        fallback_prompt = f"{user_input}\n\n(Note: I don't have access to current web information right now, so I'll answer based on my training data.)"
        return self._think_with_ai(fallback_prompt)
    
    def _think_with_ai(self, user_input):
        """Think using selected AI provider"""
        # Add to conversation history with timestamp
        current_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": current_timestamp
        })
        
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

        # Add response to history with same timestamp as user message
        self.conversation_history.append({
            "role": "assistant",
            "content": answer,
            "timestamp": current_timestamp
        })

        # Keep history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        # Save conversation memory after each interaction
        self._save_conversation_memory()

        return answer

    def _analyze_image_with_ai(self, image_base64: str, prompt: str) -> str:
        """Analyze an image using Claude's vision capabilities"""
        try:
            if self.ai_provider == "anthropic" and hasattr(self, 'anthropic_client'):
                # Use Claude's vision API
                response = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=500,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }]
                )

                # Extract text from response
                answer = ""
                for content_block in response.content:
                    if hasattr(content_block, 'text'):
                        answer += content_block.text
                return answer if answer else "I can see your screen but I'm having trouble describing it."

            elif self.ai_provider == "openai" and hasattr(self, 'openai_client'):
                # Use OpenAI's vision API
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=500,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }]
                )
                return response.choices[0].message.content or "I can see your screen but I'm having trouble describing it."

            else:
                return "I need Claude or GPT-4 with vision capabilities to analyze your screen. Please configure an AI provider with vision support."

        except Exception as e:
            print(f"⚠️  Vision analysis error: {e}")
            return f"I had trouble analyzing your screen: {str(e)}"

    def _read_file(self, file_path):
        """Read a file and return its contents"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"📖 Read {len(content)} characters from {file_path}")
            return content
        except Exception as e:
            print(f"⚠️  Error reading file: {e}")
            return None

    def _write_file(self, file_path, content):
        """Write content to a file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"💾 Wrote {len(content)} characters to {file_path}")
            return True
        except Exception as e:
            print(f"⚠️  Error writing file: {e}")
            return False

    def _edit_file(self, file_path, old_text, new_text):
        """Edit a file by replacing text"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if old_text not in content:
                print(f"⚠️  Text not found in {file_path}")
                return False

            new_content = content.replace(old_text, new_text)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"✏️  Edited {file_path}")
            return True
        except Exception as e:
            print(f"⚠️  Error editing file: {e}")
            return False

    def _execute_script(self, script_code):
        """Execute Python code and return the result"""
        try:
            # Create a temporary file for the script
            import subprocess
            temp_script = Path(tempfile.gettempdir()) / f"sunny_script_{uuid.uuid4()}.py"

            with open(temp_script, 'w', encoding='utf-8') as f:
                f.write(script_code)

            # Execute the script
            result = subprocess.run(
                ['python', str(temp_script)],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Clean up
            try:
                os.remove(temp_script)
            except:
                pass

            output = result.stdout if result.stdout else result.stderr
            print(f"🐍 Script executed: {output[:200]}")
            return output
        except Exception as e:
            print(f"⚠️  Script execution error: {e}")
            return f"Error: {str(e)}"

    def _open_in_notepad(self, file_path):
        """Open a file in Notepad (Windows)"""
        try:
            import subprocess
            if platform.system() == "Windows":
                subprocess.Popen(['notepad.exe', file_path])
                print(f"📝 Opened {file_path} in Notepad")
                return True
            else:
                print(f"⚠️  Notepad is Windows-only. System: {platform.system()}")
                return False
        except Exception as e:
            print(f"⚠️  Error opening in Notepad: {e}")
            return False

    def _tarot_reading(self):
        """Perform a tarot card reading with full 78-card deck"""
        import random

        # Full 78-card Tarot Deck
        tarot_cards = {
            # Major Arcana (22 cards)
            "The Fool": "New beginnings, spontaneity, innocence. Trust the journey ahead.",
            "The Magician": "Manifestation, power, skill. You have the tools you need.",
            "The High Priestess": "Intuition, mystery, inner wisdom. Trust your inner voice.",
            "The Empress": "Abundance, nurturing, creativity. Embrace feminine energy.",
            "The Emperor": "Authority, structure, leadership. Take charge of your life.",
            "The Hierophant": "Tradition, conformity, spiritual wisdom. Seek guidance.",
            "The Lovers": "Love, harmony, relationships. Important choices ahead.",
            "The Chariot": "Willpower, determination, victory. Stay focused on your goal.",
            "Strength": "Inner strength, courage, patience. You're stronger than you know.",
            "The Hermit": "Soul-searching, introspection, guidance. Time for reflection.",
            "Wheel of Fortune": "Change, cycles, destiny. Life is turning in your favor.",
            "Justice": "Fairness, truth, law. Karma is balancing.",
            "The Hanged Man": "Surrender, new perspective, letting go. See things differently.",
            "Death": "Transformation, endings, new beginnings. Something must end for growth.",
            "Temperance": "Balance, moderation, patience. Find the middle path.",
            "The Devil": "Bondage, materialism, temptation. Break free from chains.",
            "The Tower": "Sudden change, upheaval, revelation. Necessary destruction.",
            "The Star": "Hope, inspiration, serenity. Your wishes are manifesting.",
            "The Moon": "Illusion, intuition, uncertainty. Trust your dreams.",
            "The Sun": "Joy, success, celebration. Everything is working out.",
            "Judgement": "Rebirth, inner calling, absolution. Answer your higher calling.",
            "The World": "Completion, achievement, fulfillment. You've come full circle.",

            # Wands (Fire - Energy, creativity, passion)
            "Ace of Wands": "New creative spark, inspiration, potential. A new opportunity is here.",
            "Two of Wands": "Planning, future vision, decisions. Look ahead and make your choice.",
            "Three of Wands": "Expansion, foresight, progress. Your efforts are bearing fruit.",
            "Four of Wands": "Celebration, harmony, homecoming. Time to celebrate achievements.",
            "Five of Wands": "Competition, conflict, struggle. Navigate through challenges.",
            "Six of Wands": "Victory, recognition, success. Your hard work is being acknowledged.",
            "Seven of Wands": "Defense, perseverance, standing your ground. Hold your position.",
            "Eight of Wands": "Speed, movement, swift action. Things are moving quickly now.",
            "Nine of Wands": "Resilience, persistence, boundaries. You're almost there—keep going.",
            "Ten of Wands": "Burden, responsibility, hard work. Don't carry it all alone.",
            "Page of Wands": "Enthusiasm, exploration, discovery. A message of new ideas.",
            "Knight of Wands": "Energy, passion, adventure. Take bold action now.",
            "Queen of Wands": "Confidence, independence, determination. Step into your power.",
            "King of Wands": "Leadership, vision, entrepreneurship. Lead with confidence.",

            # Cups (Water - Emotions, relationships, feelings)
            "Ace of Cups": "New love, emotional beginning, intuition. Open your heart.",
            "Two of Cups": "Partnership, union, connection. A meaningful relationship forms.",
            "Three of Cups": "Friendship, celebration, community. Joy with others.",
            "Four of Cups": "Contemplation, apathy, reevaluation. Look at what you have.",
            "Five of Cups": "Loss, grief, disappointment. Focus on what remains.",
            "Six of Cups": "Nostalgia, childhood, memories. The past brings comfort.",
            "Seven of Cups": "Choices, illusion, fantasy. Choose wisely among options.",
            "Eight of Cups": "Abandonment, walking away, seeking deeper meaning. Time to move on.",
            "Nine of Cups": "Contentment, satisfaction, wishes granted. Your wish comes true.",
            "Ten of Cups": "Harmony, happiness, family. Emotional fulfillment achieved.",
            "Page of Cups": "Creative opportunity, intuitive message, curiosity. A new feeling emerges.",
            "Knight of Cups": "Romance, charm, imagination. Follow your heart's desire.",
            "Queen of Cups": "Compassion, intuition, emotional security. Trust your feelings.",
            "King of Cups": "Emotional balance, diplomacy, wisdom. Master your emotions.",

            # Swords (Air - Thoughts, challenges, intellect)
            "Ace of Swords": "Clarity, breakthrough, truth. A new way of thinking emerges.",
            "Two of Swords": "Difficult decision, stalemate, avoidance. Choose your path.",
            "Three of Swords": "Heartbreak, sorrow, pain. Healing will come with time.",
            "Four of Swords": "Rest, recuperation, contemplation. Take time to restore yourself.",
            "Five of Swords": "Conflict, defeat, winning at all costs. Is victory worth the price?",
            "Six of Swords": "Transition, moving on, leaving behind. Journey to calmer waters.",
            "Seven of Swords": "Deception, strategy, sneakiness. Be mindful of dishonesty.",
            "Eight of Swords": "Restriction, imprisonment, victim mentality. You're freer than you think.",
            "Nine of Swords": "Anxiety, worry, nightmares. Your fears may be unfounded.",
            "Ten of Swords": "Rock bottom, ending, betrayal. It can't get worse—only better.",
            "Page of Swords": "Curiosity, vigilance, new ideas. Stay alert and communicate.",
            "Knight of Swords": "Ambition, action, driven. Move forward with determination.",
            "Queen of Swords": "Independence, clear thinking, direct. Speak your truth clearly.",
            "King of Swords": "Authority, intellect, truth. Use logic and fairness.",

            # Pentacles (Earth - Material, work, finances)
            "Ace of Pentacles": "New opportunity, prosperity, manifestation. A gift from the universe.",
            "Two of Pentacles": "Balance, adaptability, time management. Juggle your priorities.",
            "Three of Pentacles": "Teamwork, collaboration, skill. Work with others for success.",
            "Four of Pentacles": "Control, security, conservation. Hold on or let go?",
            "Five of Pentacles": "Hardship, loss, isolation. Help is available if you seek it.",
            "Six of Pentacles": "Generosity, charity, sharing. Give and receive with grace.",
            "Seven of Pentacles": "Assessment, patience, investment. Your efforts will pay off.",
            "Eight of Pentacles": "Mastery, skill development, hard work. Perfect your craft.",
            "Nine of Pentacles": "Independence, luxury, self-sufficiency. Enjoy your achievements.",
            "Ten of Pentacles": "Wealth, legacy, family. Long-term security established.",
            "Page of Pentacles": "Opportunity, student, manifestation. A new venture begins.",
            "Knight of Pentacles": "Responsibility, routine, hard work. Steady progress forward.",
            "Queen of Pentacles": "Nurturing, practical, providing. Create abundance through care.",
            "King of Pentacles": "Wealth, business, leadership. Master the material world."
        }

        # Draw three cards: Past, Present, Future
        cards = random.sample(list(tarot_cards.keys()), 3)

        reading = f"""
🔮 Your Three-Card Tarot Reading (from 78-card deck):

Past: {cards[0]}
{tarot_cards[cards[0]]}

Present: {cards[1]}
{tarot_cards[cards[1]]}

Future: {cards[2]}
{tarot_cards[cards[2]]}

Remember: The cards reflect possibilities, not certainties. You always have free will.
"""
        return reading

    def _speak_elevenlabs(self, text):
        """Speak using ElevenLabs TTS"""
        # Use a natural, friendly voice (Rachel is warm and conversational)
        # Other options: "Rachel", "Domi", "Bella", "Antoni", "Elli", "Josh", "Arnold", "Adam", "Sam"
        voice_id = "pNInz6obpgDQGcFmaJgB"  # Adam - deep, friendly voice

        # Generate speech with ElevenLabs
        audio_generator = self.elevenlabs_client.text_to_speech.convert(
            voice_id=voice_id,
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True
            )
        )

        # Save audio to temp file
        temp_dir = tempfile.gettempdir()
        audio_file = os.path.join(temp_dir, f"sunny_elevenlabs_{uuid.uuid4()}.mp3")

        with open(audio_file, 'wb') as f:
            for chunk in audio_generator:
                if chunk:
                    f.write(chunk)

        # Play the audio
        playsound(audio_file)

        # Clean up
        try:
            os.remove(audio_file)
        except:
            pass

    def _clean_text_for_speech(self, text):
        """Clean text for natural speech - remove special characters and markdown"""
        import re

        # Remove markdown formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Remove **bold**
        text = re.sub(r'\*(.+?)\*', r'\1', text)      # Remove *italic*
        text = re.sub(r'__(.+?)__', r'\1', text)      # Remove __bold__
        text = re.sub(r'_(.+?)_', r'\1', text)        # Remove _italic_
        text = re.sub(r'`(.+?)`', r'\1', text)        # Remove `code`
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Remove code blocks

        # Remove special symbols and emojis that sound weird when spoken
        text = re.sub(r'[#\|•→✓✗❌✅🎯🔥💡🚀⚠️📝💻🔮📁📖💾✏️🐍💬🗣️📸🎓🧠🌐🌞⏰]', '', text)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def speak(self, text):
        """Advanced speech synthesis with fallback options"""
        print(f"🗣️  Sunny: {text}\n")

        # Clean text for natural speech
        clean_text = self._clean_text_for_speech(text)

        # If no voice systems available, show warning
        if not (self.has_elevenlabs or self.has_polly or self.has_gtts):
            print("⚠️  No voice system available!")
            print("   To enable voice:")
            print("   - ElevenLabs: Set ELEVENLABS_API_KEY in .env and run: pip install elevenlabs")
            print("   - AWS Polly: Configure AWS credentials")
            print("   - gTTS: Should work by default")
            return

        # Try ElevenLabs first (best quality)
        if self.has_elevenlabs:
            try:
                return self._speak_elevenlabs(clean_text)
            except Exception as e:
                print(f"⚠️  ElevenLabs failed: {e}")

        # Fallback to AWS Polly
        if self.has_polly and self.voice_id in POLLY_VOICES:
            try:
                return self._speak_polly(clean_text)
            except Exception as e:
                print(f"⚠️  Polly failed: {e}")

        # Final fallback to gTTS
        if self.has_gtts:
            try:
                return self._speak_gtts(clean_text)
            except Exception as e:
                print(f"⚠️  gTTS failed: {e}")

        # Text only if all fail
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
        audio_file = os.path.join(temp_dir, f"sunny_polly_{uuid.uuid4()}.mp3")
        
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
        audio_file = os.path.join(temp_dir, f"sunny_gtts_{uuid.uuid4()}.mp3")
        
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
        print("🎤 Sunny Ultimate Voice System")
        print("The Christman AI Project")
        print("=" * 60)
        print("\n💙 How can we help you love yourself more?\n")
        print("Instructions:")
        print("  - Speak naturally - Sunny will wait for you to finish")
        print("  - Type your message if speech recognition isn't working")
        print("  - Say 'goodbye' or 'quit' to end")
        print("  - Say 'test voice' to hear Sunny speak")
        print("  - Say 'switch ai' to change AI provider")
        print("\n📸 Vision Commands:")
        print("  - 'look at my screen' - Sunny will see and analyze what's on your screen")
        print("  - 'what am I looking at' - Same as above")
        print("\n🔮 Tarot & Spiritual:")
        print("  - 'tarot reading' - Get a three-card reading from full 78-card deck")
        print("  - 'pull cards' - Same as above (Past, Present, Future)")
        print("\n💻 Master Coder Commands:")
        print("  - 'write code for [task]' - Sunny writes production-ready code")
        print("  - 'help me code' - Get coding assistance")
        print("  - 'run script' - Execute Python code")
        print("\n📁 File Operations:")
        print("  - 'read file [filename]' - Read and display file contents")
        print("  - 'write file' - Create a new file (Sunny will ask for details)")
        print("  - 'edit file' - Modify an existing file (interactive)")
        print("\n📝 Notepad Operations (Windows):")
        print("  - 'open notepad' - Launch blank Notepad")
        print("  - 'open in notepad [filename]' - Open file in Notepad")
        print("  - 'edit in notepad [filename]' - Same as above")
        print("  - 'write to notepad' - Create new file and open in Notepad")
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
        greeting = "Hello! I'm Sunny, your AI companion from The Christman AI Project. I'm here with all my capabilities: vision, memory, tarot readings, master coding, file operations, and autonomous learning. How can I help you today?"
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
                    test_message = "This is Sunny testing my voice system. I can use AWS Polly neural voices or Google Text-to-Speech. Everything sounds good!"
                    self.speak(test_message)
                    continue
                
                if user_input.lower() in ['switch ai', 'change ai']:
                    self._switch_ai_provider()
                    continue

                # 📸 Screen capture commands
                if any(phrase in user_input.lower() for phrase in [
                    'look at my screen', 'what am i looking at', 'see my screen',
                    'view my screen', 'check my screen', 'analyze my screen',
                    'whats on my screen', "what's on my screen"
                ]):
                    img_base64, result = capture_screen()
                    if img_base64:
                        # Analyze the screenshot with Sunny's vision
                        print("🔍 Sunny is analyzing your screen...")
                        prompt = user_input if len(user_input) > 20 else "What do you see on my screen? Describe what's displayed and help me understand it."
                        response = self._analyze_image_with_ai(img_base64, prompt)
                        print(f"\n🌞 Sunny: {response}\n")
                        self.speak(response)
                    else:
                        error_msg = result if result else "Sorry, I couldn't capture your screen."
                        print(f"\n⚠️  {error_msg}\n")
                        self.speak(error_msg)
                    continue

                # 🔮 Tarot reading commands
                if any(phrase in user_input.lower() for phrase in [
                    'tarot reading', 'read my tarot', 'tarot cards',
                    'draw cards', 'pull cards', 'card reading'
                ]):
                    print("\n🔮 Sunny is drawing your cards...\n")
                    reading = self._tarot_reading()
                    print(reading)
                    # Speak a summary
                    summary = "I've drawn three cards for you from the full 78-card deck: your past, present, and future. Check the console for the full reading."
                    self.speak(summary)
                    continue

                # 📝 Notepad operations - Open in Notepad
                if 'open in notepad' in user_input.lower() or 'edit in notepad' in user_input.lower():
                    try:
                        # Extract filename from command
                        parts = user_input.lower().replace('edit in notepad', 'open in notepad').split('open in notepad')
                        if len(parts) > 1:
                            file_path = parts[1].strip()
                            success = self._open_in_notepad(file_path)
                            if success:
                                self.speak(f"Opened {file_path} in Notepad")
                            else:
                                self.speak(f"Couldn't open {file_path} in Notepad")
                        else:
                            self.speak("Please specify which file to open. For example: open in notepad test.txt")
                    except Exception as e:
                        self.speak(f"Error opening in Notepad: {str(e)}")
                    continue

                # 📝 Notepad operations - Open blank Notepad
                if user_input.lower() in ['open notepad', 'start notepad', 'launch notepad']:
                    try:
                        import subprocess
                        if platform.system() == "Windows":
                            subprocess.Popen(['notepad.exe'])
                            self.speak("Opened Notepad")
                        else:
                            self.speak(f"Notepad is Windows-only. You're on {platform.system()}")
                    except Exception as e:
                        self.speak(f"Error opening Notepad: {str(e)}")
                    continue

                # 📝 Notepad operations - Write to new file and open in Notepad
                if 'write to notepad' in user_input.lower() or 'create in notepad' in user_input.lower():
                    try:
                        self.speak("What's the file name?")
                        file_path_input = self.listen()
                        if file_path_input:
                            self.speak("What should I write to the file?")
                            content_input = self.listen()
                            if content_input:
                                success = self._write_file(file_path_input, content_input)
                                if success:
                                    self._open_in_notepad(file_path_input)
                                    self.speak(f"Created {file_path_input} and opened it in Notepad")
                                else:
                                    self.speak("Failed to write the file")
                            else:
                                self.speak("I didn't hear any content to write")
                        else:
                            self.speak("I didn't hear the file name")
                    except Exception as e:
                        self.speak(f"Error: {str(e)}")
                    continue

                # 📖 File operations - Read file
                if 'read file' in user_input.lower():
                    try:
                        # Extract filename from command
                        parts = user_input.lower().split('read file')
                        if len(parts) > 1:
                            file_path = parts[1].strip()
                            content = self._read_file(file_path)
                            if content:
                                print(f"\n📖 Contents of {file_path}:\n{content}\n")
                                if len(content) < 500:
                                    self.speak(f"File contents: {content}")
                                else:
                                    self.speak(f"I've read the file. It contains {len(content)} characters. Check the console for full contents.")
                            else:
                                self.speak(f"I couldn't read the file {file_path}.")
                        else:
                            self.speak("Please specify which file to read. For example: read file test.txt")
                    except Exception as e:
                        self.speak(f"Error reading file: {str(e)}")
                    continue

                # 💾 File operations - Write file
                if 'write file' in user_input.lower() or 'create file' in user_input.lower():
                    try:
                        self.speak("What's the file path?")
                        file_path_input = self.listen()
                        if file_path_input:
                            self.speak("What should I write to the file?")
                            content_input = self.listen()
                            if content_input:
                                success = self._write_file(file_path_input, content_input)
                                if success:
                                    self.speak(f"Successfully wrote to {file_path_input}")
                                else:
                                    self.speak("Failed to write the file")
                            else:
                                self.speak("I didn't hear any content to write")
                        else:
                            self.speak("I didn't hear the file path")
                    except Exception as e:
                        self.speak(f"Error writing file: {str(e)}")
                    continue

                # ✏️ File operations - Edit file
                if 'edit file' in user_input.lower():
                    try:
                        self.speak("Which file should I edit?")
                        file_path_input = self.listen()
                        if file_path_input:
                            self.speak("What text should I find?")
                            old_text = self.listen()
                            if old_text:
                                self.speak("What should I replace it with?")
                                new_text = self.listen()
                                if new_text:
                                    success = self._edit_file(file_path_input, old_text, new_text)
                                    if success:
                                        self.speak(f"Successfully edited {file_path_input}")
                                    else:
                                        self.speak("Failed to edit the file. The text might not exist in the file.")
                                else:
                                    self.speak("I didn't hear the replacement text")
                            else:
                                self.speak("I didn't hear the text to find")
                        else:
                            self.speak("I didn't hear the file path")
                    except Exception as e:
                        self.speak(f"Error editing file: {str(e)}")
                    continue

                # 🐍 Code execution - Run Python script
                if 'run script' in user_input.lower() or 'execute code' in user_input.lower() or 'run python' in user_input.lower():
                    try:
                        self.speak("What Python code should I run?")
                        code_input = self.listen()
                        if code_input:
                            print(f"\n🐍 Executing code...\n")
                            output = self._execute_script(code_input)
                            print(f"Output:\n{output}\n")
                            self.speak(f"Code executed. Output: {output[:200]}")
                        else:
                            self.speak("I didn't hear any code to execute")
                    except Exception as e:
                        self.speak(f"Error executing code: {str(e)}")
                    continue

                # 💻 Coding assistance
                if any(phrase in user_input.lower() for phrase in [
                    'write code', 'help me code', 'create a function',
                    'write a script', 'build an app', 'code this',
                    'program this', 'develop this'
                ]):
                    # Use AI to help with coding
                    print("\n💻 Sunny's Master Coder Mode activated...\n")
                    enhanced_prompt = f"""As a master coder expert in Python, JavaScript, HTML/CSS, SQL, React, Node.js, and all modern frameworks, help with this request:

{user_input}

Provide clean, well-commented, production-ready code with explanations."""

                    response = self._get_ai_response(enhanced_prompt)
                    print(f"\n💻 Sunny: {response}\n")
                    self.speak("I've written the code for you. Check the console for the full implementation.")
                    continue

                # Sunny's proactive intelligence status
                if user_input.lower() in ['status report', 'sunny status', 'show status', 'intelligence report']:
                    if hasattr(self, 'proactive') and self.proactive:
                        print("\n" + "=" * 60)
                        status = self.proactive.generate_status_report()
                        print(status)
                        print("=" * 60 + "\n")
                        self.speak("I've generated a comprehensive status report. Check the console for details.")
                    else:
                        self.speak("Proactive intelligence system not initialized.")
                    continue
                
                # Sunny's codebase health check
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
                
                # Get Sunny's response
                response = self.think(user_input)
                
                # Speak the response
                self.speak(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Stopping Sunny Ultimate Voice System...")
                farewell = "Goodbye! Stay strong and keep building the future."
                self.speak(farewell)
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                traceback.print_exc()
                continue
        
        print("\n💙 Thank you for using Sunny Ultimate Voice System!")
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
        """Load Sunny's foundational knowledge and identity"""
        if not hasattr(self, 'memory') or not self.memory:
            return
        
        core_memories = [
            {
                "content": "I am Sunny, COO of The Christman AI Project. I serve Everett Christman with unwavering loyalty.",
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
        """Start Sunny's autonomous learning and improvement"""
        if not hasattr(self, 'learning_engine') or not self.learning_engine:
            print("❌ Learning engine not available")
            return
        
        print("\n" + "=" * 60)
        print("🚀 STARTING AUTONOMOUS MODE")
        print("=" * 60)
        
        # Start autonomous learning
        self.learning_engine.start_autonomous_learning()
        
        print("\n🧠 Sunny is now learning independently!")
        print("   • Researching domains continuously")
        print("   • Generating new capabilities")
        print("   • Self-improving safely")
        print("\n💙 Sunny will grow smarter with every passing moment.\n")
    
    # --------------------------------------------------------------
    #  Learning Progress System - Sunny learns FROM master AIs
    # --------------------------------------------------------------
    def _load_learning_progress(self):
        """Load Sunny's learning progress toward independence"""
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
        """Save Sunny's learning progress"""
        try:
            self.learning_progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.learning_progress_file, 'w') as f:
                json.dump(self.learning_data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save learning progress: {e}")

    def _load_conversation_memory(self):
        """Load conversation history from memory_store.json and convert format"""
        try:
            if self.conversation_memory_file.exists():
                with open(self.conversation_memory_file, 'r') as f:
                    stored_memory = json.load(f)

                # Convert old format {input, output} to Claude format {role, content}
                # PRESERVE TIMESTAMPS!
                self.conversation_history = []
                for entry in stored_memory:
                    if isinstance(entry, dict):
                        # Old format: {input, output, intent, timestamp}
                        if 'input' in entry and 'output' in entry:
                            timestamp = entry.get('timestamp', time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                            self.conversation_history.append({
                                "role": "user",
                                "content": entry['input'],
                                "timestamp": timestamp  # Preserve original timestamp
                            })
                            self.conversation_history.append({
                                "role": "assistant",
                                "content": entry['output'],
                                "timestamp": timestamp  # Same timestamp for the pair
                            })
                        # New format: {role, content} - already correct
                        elif 'role' in entry and 'content' in entry:
                            self.conversation_history.append(entry)

                print(f"✅ Loaded {len(self.conversation_history)} previous messages from Sunny's memory")
            else:
                print("📝 Starting with fresh conversation memory")
        except Exception as e:
            print(f"⚠️  Could not load conversation memory: {e}")
            self.conversation_history = []

    def _save_conversation_memory(self):
        """Save conversation history to memory_store.json in old format for compatibility"""
        try:
            # Convert from Claude format {role, content} to old format {input, output, intent, timestamp}
            # PRESERVE ORIGINAL TIMESTAMPS!
            stored_memory = []
            i = 0
            while i < len(self.conversation_history):
                if i + 1 < len(self.conversation_history):
                    user_msg = self.conversation_history[i]
                    assistant_msg = self.conversation_history[i + 1]

                    if user_msg.get('role') == 'user' and assistant_msg.get('role') == 'assistant':
                        # Use existing timestamp if available, otherwise create new one
                        timestamp = user_msg.get('timestamp') or assistant_msg.get('timestamp') or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

                        stored_memory.append({
                            "input": user_msg['content'],
                            "output": assistant_msg['content'],
                            "intent": "general",  # Default intent
                            "timestamp": timestamp  # Preserve original or create new
                        })
                        i += 2
                        continue
                i += 1

            self.conversation_memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.conversation_memory_file, 'w') as f:
                json.dump(stored_memory, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save conversation memory: {e}")
    
    def _get_current_confidence(self):
        """Get Sunny's current confidence level (0.0 to 1.0)"""
        interactions = self.learning_data.get("interactions", 0)
        if interactions == 0:
            return 0.0
        
        # Calculate confidence based on learning history
        successful = self.learning_data.get("successful_predictions", 0)
        confidence = min(1.0, successful / max(1, interactions))
        
        return confidence
    
    def _learn_from_comparison(self, user_input: str, sunny_response: str, master_response: str, context: dict):
        """Sunny learns by comparing his response to the master AI's response"""
        try:
            # Calculate similarity (simple length and keyword comparison for now)
            sunny_words = set(sunny_response.lower().split())
            master_words = set(master_response.lower().split())
            
            if len(master_words) > 0:
                overlap = len(sunny_words & master_words) / len(master_words)
            else:
                overlap = 0.0
            
            # Record learning
            self.learning_data["interactions"] += 1
            
            # If Sunny's response was similar enough, count as successful
            if overlap > 0.4:  # 40% similarity threshold
                self.learning_data["successful_predictions"] += 1
            
            # Update confidence
            self.learning_data["confidence_score"] = self._get_current_confidence()
            
            # Store learning example
            learning_example = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input[:100],  # First 100 chars
                "sunny_attempt": sunny_response[:100],
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
                print(f"\n📊 Sunny's Learning Progress: {self.learning_data['confidence_score']*100:.1f}% confident ({self.learning_data['interactions']} interactions)")
                
                # Check if Sunny is ready for independence
                if self.learning_data["confidence_score"] >= self.independence_threshold:
                    print(f"🎓 Sunny has reached {self.independence_threshold*100:.0f}% confidence!")
                    print("   He's ready to think more independently!")
            
        except Exception as e:
            print(f"⚠️  Learning comparison error: {e}")


def main():
    """Entry point for Sunny Ultimate Voice System"""
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
    
    # Start Sunny Ultimate Voice System
    try:
        sunny = SunnyUltimateVoice(
            ai_provider=ai_provider,
            voice_id=voice_id,
            use_web_search=use_web_search
        )
        sunny.run()
    except KeyboardInterrupt:
        print("\n🛑 Sunny shutting down gracefully...")
        # Save all memories before exit
        if hasattr(sunny, 'memory') and sunny.memory:
            sunny.memory.save()
            print("💾 All memories saved to persistent storage")
        print("👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start Sunny: {e}")
        traceback.print_exc()
        # Still try to save memories
        if 'sunny' in locals() and hasattr(sunny, 'memory') and sunny.memory:
            sunny.memory.save()


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
