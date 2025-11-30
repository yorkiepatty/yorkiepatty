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

# Audio playback function that works cross-platform
def playsound(audio_file):
    """Play audio file using system-appropriate method"""
    try:
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", audio_file], check=True)
        elif system == "Linux":
            subprocess.run(["aplay", audio_file], check=True)
        elif system == "Windows":
            # Use pygame for Windows (supports MP3)
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.quit()
        else:
            print(f"⚠️  Audio playback not supported on {system}")
    except Exception as e:
        print(f"⚠️  Audio playback failed: {e}")
        import traceback
        traceback.print_exc()

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
    # ================================
    # MEMORY MANAGEMENT
    # ================================
    def _save_conversation_memory(self):
        """Save in-session conversation history to disk (for continuity)"""
        try:
            mem_path = Path(self.conversation_memory_file)
            mem_path.parent.mkdir(exist_ok=True)
            with open(mem_path, "w", encoding="utf-8") as f:
                json.dump(self.conversation_history, f, indent=2)
            print(f"💾 Conversation history saved ({len(self.conversation_history)} messages)")
        except Exception as e:
            print(f"⚠️  Saving conversation memory failed: {e}")

    def _load_conversation_memory(self):
        """Load previous conversation history from disk"""
        try:
            mem_path = Path(self.conversation_memory_file)
            if mem_path.exists():
                with open(mem_path, "r", encoding="utf-8") as f:
                    self.conversation_history = json.load(f)
                print(f"🔄 Loaded conversation history ({len(self.conversation_history)} messages)")
            else:
                print("🗂️  No previous conversation memory found. Starting new session.")
        except Exception as e:
            print(f"⚠️  Loading conversation memory failed: {e}")

    def _load_learning_progress(self):
        """Load Sunny's autonomy/confidence progress from disk"""
        try:
            path = Path(self.learning_progress_file)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    self.autonomy_progress = obj.get("autonomy_progress", 0.0)
            else:
                self.autonomy_progress = 0.0
        except Exception as e:
            print(f"⚠️  Error loading autonomy progress: {e}")
            self.autonomy_progress = 0.0

    def _get_current_confidence(self) -> float:
        """Sunny's autonomy/confidence level (0.0 = learning, 1.0 = independent)"""
        try:
            return max(0.0, min(1.0, getattr(self, 'autonomy_progress', 0.0)))
        except:
            return 0.0

    def _learn_from_comparison(self, user_input, sunny_response, master_response, context):
        """Sunny improves by learning from master's answer"""
        if hasattr(self, 'learning_engine') and self.learning_engine:
            self.learning_engine.compare_and_learn(
                user_input,
                sunny_response,
                master_response,
                context
            )
            print("🧑‍🎓 Sunny is learning from master AI response.")
        else:
            print("⚠️  Autonomous learning engine not available. Skipping learning cycle.")

    def speak(self, response_text: str):
        """Say the response using selected TTS provider."""
        print(f"🗣️  Sunny says: {response_text}")
        audio_file = None
        
        # Use ElevenLabs first if available
        if self.has_elevenlabs:
            try:
                audio_file = self._elevenlabs_tts(response_text)
            except Exception as e:
                print(f"⚠️  ElevenLabs TTS error: {e}")
        
        # Fallback to AWS Polly
        if not audio_file and self.has_polly:
            try:
                audio_file = self._polly_tts(response_text, voice_id=self.voice_id)
            except Exception as e:
                print(f"⚠️  Polly TTS error: {e}")
        
        # Fallback to Google TTS
        if not audio_file and self.has_gtts:
            audio_file = self._gtts_tts(response_text)
        
        # Play back (if audio generated)
        if audio_file and Path(audio_file).exists():
            playsound(audio_file)
        else:
            print("❌ Voice playback failed: No valid audio generated.")

    def _elevenlabs_tts(self, text: str) -> Optional[str]:
        """Generate speech using ElevenLabs API and save to file."""
        voice = self.voice_id if self.voice_id else "matthew"
        try:
            out_path = Path(tempfile.gettempdir()) / f"sunny11_{uuid.uuid4()}.mp3"
            content = self.elevenlabs_client.generate(
                text=text,
                voice=voice,
                voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.8)
            )
            with open(out_path, "wb") as f:
                f.write(content)
            print(f"🔊 ElevenLabs TTS saved to: {out_path}")
            return str(out_path)
        except Exception as e:
            print(f"⚠️  ElevenLabs error: {e}")
            return None

    def _polly_tts(self, text: str, voice_id: str = "matthew") -> Optional[str]:
        """Generate speech using AWS Polly and save to file."""
        try:
            response = self.polly.synthesize_speech(
                Text=text,
                VoiceId=voice_id,
                OutputFormat='mp3',
                Engine='neural'
            )
            out_path = Path(tempfile.gettempdir()) / f"sunnypolly_{uuid.uuid4()}.mp3"
            with open(out_path, "wb") as f:
                f.write(response['AudioStream'].read())
            print(f"🔊 Polly TTS saved to: {out_path}")
            return str(out_path)
        except Exception as e:
            print(f"⚠️  Polly TTS error: {e}")
            return None

    def _gtts_tts(self, text: str) -> Optional[str]:
        """Generate speech using Google TTS and save to file"""
        try:
            out_path = Path(tempfile.gettempdir()) / f"sunnygtts_{uuid.uuid4()}.mp3"
            tts = gTTS(text=text, lang='en')
            tts.save(str(out_path))
            print(f"🔊 gTTS saved to: {out_path}")
            return str(out_path)
        except Exception as e:
            print(f"⚠️  gTTS TTS error: {e}")
            return None

    def save_to_disk(self):
        """Save Sunny's full system state/memory to disk"""
        self._save_conversation_memory()
        # Could save more components here as needed
        print("💾 Sunny's current state saved.")

    def capture_and_analyze_screen(self):
        """Capture the user's screen and ask Sunny to analyze"""
        if not HAS_SCREEN_CAPTURE:
            print("⚠️  Screen capture not available in this install.")
            return None, None

        img_base64, img_file = capture_screen()
        if not img_base64:
            print("❌ Screen capture failed.")
            return None, None

        # Optionally use image as input for vision/Claude
        if self.vision and hasattr(self.vision, "analyze_image"):
            analysis = self.vision.analyze_image(img_base64)
            print(f"👀 Screen analysis: {analysis}")
            return img_file, analysis
        else:
            print("⚠️  Vision engine not available to analyze image.")
            return img_file, None
    # ================================
    # SYSTEM EXIT & ERROR HANDLING
    # ================================
    def handle_error(self, error):
        """Central error handler for all Sunny operations"""
        print(f"❌ Sunny encountered an error: {error}")
        logging.error(traceback.format_exc())
        if hasattr(self, "memory") and self.memory:
            self.memory.store("error", str(error))

    def exit(self):
        """Gracefully shut down Sunny and save state"""
        print("👋 Shutting down Sunny Ultimate Voice System...")
        self.save_to_disk()
        print("💾 Memory and state saved - goodbye!")

# MAIN APP ENTRYPOINT
if __name__ == "__main__":
    print("="*60)
    print("☀️  Sunny Ultimate Voice System: Startup")
    print("="*60)
    try:
        sunny = SunnyUltimateVoice(
            ai_provider="auto",
            voice_id="matthew",
            use_web_search=True,
            enable_speech=True
        )
        while True:
            print("\n👤 Awaiting input (say something or type 'exit')...")
            try:
                user_input = sunny.listen()
                if not user_input:
                    user_input = input("⏳ Please type your message: ")
                if user_input.strip().lower() in ["exit", "quit", "bye"]:
                    break
                response = sunny.think(user_input)
                sunny.speak(response)
            except KeyboardInterrupt:
                break
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        logging.error(traceback.format_exc())
    finally:
        try:
            sunny.exit()
        except Exception:
            print("⚠️  Error during shutdown.")