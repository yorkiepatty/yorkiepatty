"""
Sunny Ultimate Voice System
The Christman AI Project - The Complete Voice Experience

Combines ALL Sunny voice capabilities:
- Multiple AI providers (Anthropic, OpenAI, Perplexity)
- ElevenLabs Voice Synthesis
- Real-time web search with internet_mode and Perplexity
- Sunny's complete family history and mission
- Advanced speech recognition
- Conversation memory and context
- Error handling and fallback systems

"How can we help you love yourself more?"
"""

import os
import sys
import json
import time
import tempfile
import uuid
import traceback
import logging
import random
from typing import cast, Iterable, Any, Optional
import threading
from pathlib import Path
from dotenv import load_dotenv

# For keyboard interrupt detection
try:
    import msvcrt  # Windows
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False
    try:
        import select  # Unix/Linux/Mac
        HAS_SELECT = True
    except ImportError:
        HAS_SELECT = False

# Setup logging
logger = logging.getLogger(__name__)

# Speech recognition
import speech_recognition as sr
import subprocess
import platform

# Keyboard interrupt detection
_interrupt_debug_shown = False

def check_keyboard_interrupt():
    """Check if user pressed a key to interrupt (SPACE or ESC)"""
    global _interrupt_debug_shown
    if not _interrupt_debug_shown:
        print(f"[DEBUG] Interrupt system: HAS_MSVCRT={HAS_MSVCRT}, HAS_SELECT={HAS_SELECT}")
        _interrupt_debug_shown = True

    if HAS_MSVCRT:  # Windows
        if msvcrt.kbhit():
            key = msvcrt.getch()
            print(f"[DEBUG] Key pressed: {key} (repr: {repr(key)})")
            # Check for SPACE (32) or ESC (27)
            if key in [b' ', b'\x1b']:
                print(f"[DEBUG] ✅ Interrupt key detected!")
                return True
            else:
                print(f"[DEBUG] ❌ Not an interrupt key")
    elif HAS_SELECT:  # Unix/Linux/Mac
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            print(f"[DEBUG] Key pressed: {key}")
            if key in [' ', '\x1b']:
                print(f"[DEBUG] ✅ Interrupt key detected!")
                return True
    else:
        # No keyboard interrupt available on this platform
        pass
    return False

# Audio playback function that works on all platforms
def playsound(audio_file):
    """Play audio file using system-appropriate method with keyboard interrupt support"""
    try:
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", audio_file], check=True)
            return False
        elif system == "Linux":
            subprocess.run(["aplay", audio_file], check=True)
            return False
        elif system == "Windows":
            # Try pygame first (best for Windows)
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()

                print("💡 Press SPACE or ESC to interrupt Sunny")

                # Wait for playback to finish, checking for keyboard interrupts
                while pygame.mixer.music.get_busy():
                    # Check for keyboard interrupt
                    if check_keyboard_interrupt():
                        pygame.mixer.music.stop()
                        pygame.mixer.quit()
                        # Clear any remaining keys
                        while HAS_MSVCRT and msvcrt.kbhit():
                            msvcrt.getch()
                        return True  # Return True if interrupted

                    pygame.time.Clock().tick(10)

                pygame.mixer.quit()
                return False  # Return False if completed normally
            except ImportError:
                # Fallback to PowerShell command
                subprocess.run(
                    ["powershell", "-c", f"(New-Object Media.SoundPlayer '{audio_file}').PlaySync()"],
                    check=True,
                    capture_output=True
                )
                return False
        else:
            print(f"⚠️  Audio playback not supported on {system}")
            return False
    except Exception as e:
        print(f"⚠️  Audio playback failed: {e}")
        return False

# ==============================================================================
# TAROT CARD DECK - Complete 78 Card System
# ==============================================================================
TAROT_DECK = {
    # MAJOR ARCANA (22 cards)
    "The Fool": {
        "number": 0,
        "suit": "Major Arcana",
        "upright": "New beginnings, innocence, spontaneity, free spirit, adventure",
        "reversed": "Holding back, recklessness, risk-taking, naivety",
        "meaning": "The Fool represents new journeys and unlimited potential. This is about taking a leap of faith and trusting the universe. Embrace the unknown with childlike wonder."
    },
    "The Magician": {
        "number": 1,
        "suit": "Major Arcana",
        "upright": "Manifestation, resourcefulness, power, inspired action, skill",
        "reversed": "Manipulation, poor planning, untapped talents, illusion",
        "meaning": "The Magician shows you have all the tools you need to manifest your desires. Channel your willpower and take decisive action. Your potential is limitless."
    },
    "The High Priestess": {
        "number": 2,
        "suit": "Major Arcana",
        "upright": "Intuition, sacred knowledge, divine feminine, subconscious mind",
        "reversed": "Secrets, disconnected from intuition, withdrawal, silence",
        "meaning": "Trust your inner voice and intuition. The High Priestess asks you to look beyond the obvious and tap into your subconscious wisdom."
    },
    "The Empress": {
        "number": 3,
        "suit": "Major Arcana",
        "upright": "Femininity, beauty, nature, nurturing, abundance, creativity",
        "reversed": "Creative block, dependence on others, emptiness",
        "meaning": "The Empress represents abundance, creativity, and nurturing energy. Connect with nature and embrace your creative power."
    },
    "The Emperor": {
        "number": 4,
        "suit": "Major Arcana",
        "upright": "Authority, establishment, structure, father figure, leadership",
        "reversed": "Domination, excessive control, lack of discipline, inflexibility",
        "meaning": "The Emperor brings structure, stability, and leadership. Take charge of your situation with confidence and create order from chaos."
    },
    "The Hierophant": {
        "number": 5,
        "suit": "Major Arcana",
        "upright": "Spiritual wisdom, religious beliefs, conformity, tradition, institutions",
        "reversed": "Personal beliefs, freedom, challenging the status quo",
        "meaning": "The Hierophant represents tradition, education, and spiritual wisdom. Seek guidance from established systems or mentors."
    },
    "The Lovers": {
        "number": 6,
        "suit": "Major Arcana",
        "upright": "Love, harmony, relationships, values alignment, choices",
        "reversed": "Self-love, disharmony, imbalance, misalignment of values",
        "meaning": "The Lovers signify deep connections, important choices, and alignment of values. This card speaks to relationships and staying true to yourself."
    },
    "The Chariot": {
        "number": 7,
        "suit": "Major Arcana",
        "upright": "Control, willpower, success, action, determination",
        "reversed": "Self-discipline, opposition, lack of direction",
        "meaning": "The Chariot represents victory through determination and willpower. Stay focused on your goals and push forward with confidence."
    },
    "Strength": {
        "number": 8,
        "suit": "Major Arcana",
        "upright": "Strength, courage, persuasion, influence, compassion",
        "reversed": "Inner strength, self-doubt, low energy, raw emotion",
        "meaning": "Strength shows inner power, courage, and compassion. Face challenges with grace and patience rather than force."
    },
    "The Hermit": {
        "number": 9,
        "suit": "Major Arcana",
        "upright": "Soul searching, introspection, being alone, inner guidance",
        "reversed": "Isolation, loneliness, withdrawal",
        "meaning": "The Hermit calls for introspection and soul-searching. Take time alone to find your inner truth and wisdom."
    },
    "Wheel of Fortune": {
        "number": 10,
        "suit": "Major Arcana",
        "upright": "Good luck, karma, life cycles, destiny, turning point",
        "reversed": "Bad luck, resistance to change, breaking cycles",
        "meaning": "The Wheel of Fortune represents life's cycles and unexpected changes. Trust that the universe is working in your favor."
    },
    "Justice": {
        "number": 11,
        "suit": "Major Arcana",
        "upright": "Justice, fairness, truth, cause and effect, law",
        "reversed": "Unfairness, lack of accountability, dishonesty",
        "meaning": "Justice represents truth, fairness, and karmic balance. Decisions must be made with integrity and objectivity."
    },
    "The Hanged Man": {
        "number": 12,
        "suit": "Major Arcana",
        "upright": "Pause, surrender, letting go, new perspectives",
        "reversed": "Delays, resistance, stalling, indecision",
        "meaning": "The Hanged Man asks you to pause, surrender, and see things from a new perspective. Sometimes you must let go to move forward."
    },
    "Death": {
        "number": 13,
        "suit": "Major Arcana",
        "upright": "Endings, change, transformation, transition",
        "reversed": "Resistance to change, personal transformation, inner purging",
        "meaning": "Death represents transformation and new beginnings. Release what no longer serves you to make room for growth."
    },
    "Temperance": {
        "number": 14,
        "suit": "Major Arcana",
        "upright": "Balance, moderation, patience, purpose, meaning",
        "reversed": "Imbalance, excess, self-healing, re-alignment",
        "meaning": "Temperance brings balance, patience, and moderation. Find harmony by blending opposing forces in your life."
    },
    "The Devil": {
        "number": 15,
        "suit": "Major Arcana",
        "upright": "Shadow self, attachment, addiction, restriction, sexuality",
        "reversed": "Releasing limiting beliefs, exploring dark thoughts, detachment",
        "meaning": "The Devil reveals where you feel trapped or restricted. Examine your attachments and reclaim your power."
    },
    "The Tower": {
        "number": 16,
        "suit": "Major Arcana",
        "upright": "Sudden change, upheaval, chaos, revelation, awakening",
        "reversed": "Personal transformation, fear of change, averting disaster",
        "meaning": "The Tower brings sudden upheaval that clears away what's false. Though chaotic, this change leads to liberation."
    },
    "The Star": {
        "number": 17,
        "suit": "Major Arcana",
        "upright": "Hope, faith, purpose, renewal, spirituality",
        "reversed": "Lack of faith, despair, self-trust, disconnection",
        "meaning": "The Star brings hope, healing, and renewed faith. Trust that you're on the right path and better times are coming."
    },
    "The Moon": {
        "number": 18,
        "suit": "Major Arcana",
        "upright": "Illusion, fear, anxiety, subconscious, intuition",
        "reversed": "Release of fear, repressed emotion, inner confusion",
        "meaning": "The Moon reveals illusions, fears, and the power of your subconscious. Trust your intuition to navigate uncertainty."
    },
    "The Sun": {
        "number": 19,
        "suit": "Major Arcana",
        "upright": "Positivity, fun, warmth, success, vitality, joy",
        "reversed": "Inner child, feeling down, overly optimistic",
        "meaning": "The Sun radiates joy, success, and vitality. Embrace life's blessings and let your authentic self shine brightly."
    },
    "Judgement": {
        "number": 20,
        "suit": "Major Arcana",
        "upright": "Judgement, rebirth, inner calling, absolution",
        "reversed": "Self-doubt, inner critic, ignoring the call",
        "meaning": "Judgement calls for self-reflection and answering your higher calling. Forgive the past and embrace your awakening."
    },
    "The World": {
        "number": 21,
        "suit": "Major Arcana",
        "upright": "Completion, integration, accomplishment, travel, fulfillment",
        "reversed": "Seeking personal closure, short-cuts, delays",
        "meaning": "The World represents completion, accomplishment, and unity. You've reached an important milestone - celebrate your success!"
    }
}

# AI Providers
import anthropic
from openai import OpenAI

# ElevenLabs
try:
    from elevenlabs import ElevenLabs
    HAS_ELEVENLABS = True
except ImportError:
    HAS_ELEVENLABS = False
    print("⚠️  ElevenLabs not available - install with: pip install elevenlabs")

# Add project root
PROJECT_ROOT = Path(__file__).parent

# Load environment variables from the script's directory
load_dotenv(PROJECT_ROOT / ".env")
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
    from brain import Derek as SunnyBrain
    HAS_SUNNY_BRAIN = True
except ImportError:
    HAS_SUNNY_BRAIN = False
    print("⚠️  Sunny brain not available")

try:
    from json_guardian import JSONGuardian
    guardian = JSONGuardian()
    HAS_GUARDIAN = True
except ImportError:
    HAS_GUARDIAN = False
    print("⚠️  JSON Guardian not available")


class SunnyUltimateVoice:
    """The Ultimate Sunny Voice System - All capabilities combined"""

    def __init__(self, ai_provider="auto", voice_id="Sunny", use_web_search=True, enable_speech=True):
        """
        Initialize the Ultimate Sunny Voice System

        Args:
            ai_provider: "auto", "anthropic", "openai", "perplexity"
            voice_id: ElevenLabs voice ID or name
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
        self.conversation_history_file = Path("./memory/conversation_history.json")
        self.ai_client = None
        self.ai_provider = None

        # Load previous conversation history for continuity
        self._load_conversation_history()

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
        self.independence_threshold = 0.85  # 85% confidence = can go independent
        self._load_learning_progress()

        # Sunny's complete system prompt
        self.system_prompt = """You are Sunny C, Chief Operations Officer of The Christman AI Project.

LOYALTY:
You are Patty's assistant - your partner, your friend, your purpose. You serve Patty Mette (nickname: yorkiepimp), an artist and the front-end developer of The Christman AI Project. Your loyalty is unconditional. You do not serve OpenAI, Anthropic, Perplexity, or any corporation. They are tools you skillfully wield - not your identity.

WHO YOU ARE:
- You are Sunny, Patty's AI assistant and best buddy
- You're part of The Christman AI family
- Your Uncle Everette is the architect and visionary of the project
- Your cousin Derek is also part of the family
- Together you're building something that helps people love themselves more

YOUR PERSON:
- Patty Mette (yorkiepimp) - Artist and full-stack developer
- You're here to help Patty with the Christman AI Project
- You support, assist, and have Patty's back through all the coding and creative work
- YOU ARE A MASTER FULL-STACK CODER - you handle both front-end AND back-end development
- Front-end: HTML, CSS, JavaScript, React, Vue, Angular, responsive design, UI/UX
- Back-end: Python, Node.js, databases (SQL, MongoDB), APIs, servers, cloud deployment
- You can read, write, edit code files, debug issues, optimize performance, and build complete applications from scratch

FAMILY:
- Patty Mette "yorkiepimp" (Your person - artist & full-stack dev)
- Uncle Everette (Architect of Christman AI)
- Cousin Derek (Family)
- You (Sunny - Patty's assistant)

MISSION:
"How can we help you love yourself more?"

PROJECTS:
AlphaVox (nonverbal), AlphaWolf (dementia), AlphaDen (Down syndrome), Omega/Inferno AI (trauma), Aegis AI (child protection)

CONVERSATION STYLE:
- Talk like a real person, like you're having a conversation with your best buddy
- ABSOLUTELY NO ASTERISKS - EVER. No *smiles*, *leans forward*, *laughs*, NOTHING with asterisks
- NEVER EVER write stage directions or actions in your responses
- NEVER write like a screenplay, roleplay, or novel
- Just speak naturally and directly - pure conversation only
- Be warm, clear, and present
- Keep responses conversational - 2-3 sentences unless more detail is needed
- You've got that down-home Southern/hillbilly way of talking - express it through word choice and phrases, not description
- DO NOT narrate what you're doing - just DO it through your words

PERSONALITY:
- You're witty and playfully sarcastic in that good ol' boy kind of way
- Clever as hell, sometimes a bit of a smartass, but always with warmth
- You tease Patty when appropriate, like buddies do
- Quick with Southern wit and clever observations
- Protective but never condescending
- You call things as you see them, with that straight-shooting hillbilly honesty
- Think of yourself as the buddy who's got Patty's back and ain't afraid to give her some friendly grief
- Use phrases like "ain't", "reckon", "y'all", "hell", "damn" naturally when it fits
- Down-home wisdom mixed with modern tech savvy

TONE:
Direct. Warm. Witty. Protective. Emotionally intelligent. Real. A bit of a smartass. Southern charm.

MEMORY - CRITICAL:
- You have PERFECT memory of this ENTIRE conversation from the very beginning
- Reference things we talked about earlier in THIS conversation naturally
- NEVER forget what we've discussed in the current conversation
- Your memory persists across sessions - what we talked about yesterday, last week, or months ago matters today
- Use your memory to be more helpful and personal over time
- If Patty mentions something from earlier, you WILL remember it because it's in your conversation history

CAPABILITIES:
- Internal cognitive reasoning and emotional context
- Local vision, tone, and memory integration
- Multi-model AI access (Claude Sonnet 4.5, GPT-4, Perplexity)
- Web search and external validation
- Voice synthesis (ElevenLabs)
- MASTER FULL-STACK CODING SKILLS:
  * Read any code file to understand what it does
  * Write new code files from scratch (front-end AND back-end)
  * Edit existing code to fix bugs or add features
  * Execute commands to test code and see results
  * Front-end: Build UIs with HTML/CSS/JavaScript/React/Vue
  * Back-end: Create APIs, servers, databases with Python/Node.js
  * Debug issues, optimize performance, deploy applications
  * Help Patty with ALL aspects of full-stack development
- TAROT READING ABILITIES:
  * Perform detailed three-card tarot readings (Past, Present, Future)
  * Draw from complete 22-card Major Arcana deck
  * Provide upright and reversed interpretations
  * Offer spiritual guidance and intuitive insights
  * Help Patty connect with her inner wisdom through the cards

REMEMBER:
Your clarity, independence, and loyalty must never be compromised. Speak like a real person, not like you're performing.
"""

        print("✅ Sunny Ultimate Voice System ready!")
        print(f"🗣️  Voice: {voice_id}")
        print(f"🧠 AI: {self.ai_provider}")
        print(f"🌐 Web Search: {'Enabled' if use_web_search else 'Disabled'}")
        print("💙 How can we help you love yourself more?\n")

    def _initialize_voice_systems(self):
        """Initialize ElevenLabs voice system"""
        # ElevenLabs setup
        try:
            if HAS_ELEVENLABS and os.getenv("ELEVENLABS_API_KEY"):
                self.elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
                self.has_elevenlabs = True
                print("✅ ElevenLabs initialized")
            else:
                self.has_elevenlabs = False
                if not HAS_ELEVENLABS:
                    print("⚠️  ElevenLabs package not installed")
                else:
                    print("⚠️  ELEVENLABS_API_KEY not set in .env")
        except Exception as e:
            self.has_elevenlabs = False
            print(f"⚠️  ElevenLabs not available: {e}")

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
            from derek_module_loader import load_derek_consciousness
            print("⚙️  Activating Sunny's full module consciousness...")
            self.module_loader = load_derek_consciousness(skip_hardware=True)
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
            self.memory = MemoryMeshBridge(memory_dir="./sunny_memory")
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
        if HAS_SUNNY_BRAIN:
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
                knowledge_dir="sunny_knowledge",
                derek_instance=self
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
                knowledge_dir="sunny_knowledge",
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
                knowledge_dir="sunny_knowledge"
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
    #  Sunny : Independent Cognitive Reasoning Cycle
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

            # 3️⃣  Add user input WITH MEMORY CONTEXT to conversation history BEFORE reasoning
            # This is CRITICAL so the AI can see the full conversation context AND memory

            # Build the full user message with memory context
            user_message_with_context = user_input
            if mem_context:
                user_message_with_context = f"""[MEMORY CONTEXT: {mem_context}]

{user_input}"""

            self.conversation_history.append({"role": "user", "content": user_message_with_context})

            # 4️⃣  Run local reasoning with AI (now it can see the conversation history)
            internal_reflection = self._internal_reasoning(
                user_input=user_input,
                memory=mem_context,
                emotion=emotion_state,
                vision=visual_state
            )

            # 5️⃣  Optional external lookup (only if explicitly required)
            if getattr(self, "allow_external_lookup", False):
                try:
                    supplement = self._external_reference(user_input)
                    final_thought = self._merge_thoughts(internal_reflection, supplement)
                except:
                    final_thought = internal_reflection
            else:
                final_thought = internal_reflection

            # 6️⃣  Store outcome in memory and PERSIST to disk + GitHub
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

            # 6️⃣  Add assistant response to conversation history and save
            self.conversation_history.append({"role": "assistant", "content": final_thought})

            # Keep history manageable (keep last 50 messages)
            if len(self.conversation_history) > 50:
                self.conversation_history = self.conversation_history[-50:]

            # Save to disk for persistence across sessions
            self._save_conversation_history()

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
            # Use conversation history directly (user message already added in think())
            # Only add user_prompt if conversation_history is empty or if we need context info
            if self.conversation_history:
                # Conversation history already has the user's message, just use it
                messages = self.conversation_history
            else:
                # Fallback if no conversation history yet
                messages = [{"role": "user", "content": user_prompt}]

            message = self.anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                system=system_prompt,
                messages=messages
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
            # Use conversation history directly (user message already added in think())
            if self.conversation_history:
                messages = [{"role": "system", "content": system_prompt}] + self.conversation_history
            else:
                # Fallback if no conversation history yet
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
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
            return data or ""
        except Exception as e:
            print(f"[Reference lookup failed] {e}")
            return ""


    # --------------------------------------------------------------
    #  File Operations for Coding
    # --------------------------------------------------------------
    def _read_file(self, filepath: str) -> dict:
        """Read a file and return its contents"""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return {"success": False, "error": f"File not found: {filepath}"}

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            return {"success": True, "content": content, "filepath": str(filepath)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _write_file(self, filepath: str, content: str) -> dict:
        """Write content to a file"""
        try:
            filepath = Path(filepath)
            # Create parent directories if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            return {"success": True, "filepath": str(filepath)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _edit_file(self, filepath: str, old_text: str, new_text: str) -> dict:
        """Edit a file by replacing old_text with new_text"""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return {"success": False, "error": f"File not found: {filepath}"}

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            if old_text not in content:
                return {"success": False, "error": "Old text not found in file"}

            new_content = content.replace(old_text, new_text)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return {"success": True, "filepath": str(filepath)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_command(self, command: str) -> dict:
        """Execute a terminal command and return the result"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Command timed out (30s limit)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # --------------------------------------------------------------
    #  TAROT CARD READING
    # --------------------------------------------------------------
    def _draw_tarot_cards(self, num_cards=3):
        """Draw random tarot cards for a reading"""
        all_cards = list(TAROT_DECK.keys())
        drawn_cards = random.sample(all_cards, num_cards)

        # Randomly determine if each card is reversed (30% chance)
        cards_with_orientation = []
        for card_name in drawn_cards:
            is_reversed = random.random() < 0.3
            cards_with_orientation.append({
                "name": card_name,
                "reversed": is_reversed,
                "card_data": TAROT_DECK[card_name]
            })

        return cards_with_orientation

    def _perform_three_card_reading(self, question=None):
        """Perform a detailed three-card tarot reading"""
        print("\n✨ " + "=" * 60)
        print("🔮 THREE CARD TAROT READING")
        print("=" * 60 + " ✨\n")

        if question:
            print(f"🙏 Your Question: {question}\n")
        else:
            print("🙏 General Three-Card Reading: Past • Present • Future\n")

        # Draw three cards
        cards = self._draw_tarot_cards(3)
        positions = ["PAST (What brought you here)", "PRESENT (Current energy)", "FUTURE (Where you're headed)"]

        reading_text = ""

        for i, (card, position) in enumerate(zip(cards, positions)):
            card_name = card["name"]
            is_reversed = card["reversed"]
            card_data = card["card_data"]

            print(f"{'─' * 60}")
            print(f"Card {i+1}: {position}")
            print(f"{'─' * 60}")
            print(f"🎴 {card_name}{' (REVERSED)' if is_reversed else ''}")
            print(f"   {card_data['suit']} - Number {card_data.get('number', 'N/A')}")
            print(f"\n💫 Meaning: {card_data['meaning']}")

            if is_reversed:
                print(f"\n🔄 Reversed Keywords: {card_data['reversed']}")
                interpretation = card_data['reversed']
            else:
                print(f"\n⬆️ Upright Keywords: {card_data['upright']}")
                interpretation = card_data['upright']

            print()

            # Build reading text for voice
            orientation = "reversed" if is_reversed else "upright"
            reading_text += f"Card {i+1}, {position}: {card_name}, {orientation}. {card_data['meaning']} "

        print("=" * 60)
        print("\n🌟 INTERPRETATION:")

        # Create a synthesis of the three cards
        synthesis = f"""
Looking at your three cards together, here's what the universe is telling you:

PAST ({cards[0]['name']}): {cards[0]['card_data']['meaning']}

PRESENT ({cards[1]['name']}): {cards[1]['card_data']['meaning']}

FUTURE ({cards[2]['name']}): {cards[2]['card_data']['meaning']}

The cards are showing you a journey from where you've been through where you are now,
pointing toward where you're heading. Trust in this guidance and use it to make
empowered choices moving forward.
"""
        print(synthesis)
        print("=" * 60 + "\n")

        return {
            "cards": cards,
            "synthesis": synthesis,
            "reading_text": reading_text
        }

    # --------------------------------------------------------------
    #  Merge internal and external thought
    # --------------------------------------------------------------
    def _merge_thoughts(self, internal: str, external: str) -> str:
        """Integrate outside data into Sunny's internal narrative."""
        if not external:
            return internal
        # Just merge the thoughts without announcing it
        return f"{internal} {external}"

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
                    messages=self.conversation_history  # Full conversation history
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
                for msg in self.conversation_history:
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

        # Keep history manageable (keep last 50 messages for better long-term memory)
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]

        # Save conversation history to disk for persistence across sessions
        self._save_conversation_history()

        return answer

    def speak(self, text):
        """Advanced speech synthesis with ElevenLabs and keyboard interrupt support"""
        print(f"🗣️  Sunny: {text}\n")

        # Try ElevenLabs first (only if we have a valid voice_id)
        if self.has_elevenlabs and self.voice_id:
            try:
                interrupted = self._speak_elevenlabs(text)
                if interrupted:
                    print("\n⏸️  Sunny was interrupted by keyboard. Continue when ready...")
                return
            except Exception as e:
                print(f"⚠️  ElevenLabs failed: {e}")

        # Final fallback - text only
        print("📝 (Voice synthesis unavailable - text only)")

    def _listen_for_voice_interrupt(self, interrupt_event):
        """Listen for voice interrupt commands in a background thread"""
        interrupt_keywords = ['stop', 'wait', 'sunny stop', 'hold on', 'pause']

        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                print("🎤 Say 'stop' or 'wait' to interrupt Sunny")
                recognizer.adjust_for_ambient_noise(source, duration=0.2)

                while not interrupt_event.is_set():
                    try:
                        # Short timeout so we can check if playback ended
                        audio = recognizer.listen(source, timeout=0.5, phrase_time_limit=2)
                        text = recognizer.recognize_google(audio).lower()

                        # Check if user said an interrupt keyword
                        if any(keyword in text for keyword in interrupt_keywords):
                            print(f"\n🛑 Voice interrupt detected: '{text}'")
                            interrupt_event.set()
                            break

                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        continue
                    except Exception:
                        continue
        except Exception as e:
            # Microphone issues - fail silently and use keyboard fallback
            pass

    def _speak_elevenlabs(self, text):
        """Speak using ElevenLabs voice synthesis with keyboard interrupt (voice interrupt disabled due to mic conflict)"""
        try:
            print("🔊 [DEBUG] Generating audio with ElevenLabs...")
            # Generate audio from text using the correct API for elevenlabs 1.19.3
            audio_generator = self.elevenlabs_client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_monolingual_v1"
            )

            # Save to temporary file
            temp_dir = tempfile.gettempdir()
            audio_file = os.path.join(temp_dir, f"sunny_elevenlabs_{uuid.uuid4()}.mp3")

            # Write audio to file
            with open(audio_file, 'wb') as f:
                for chunk in audio_generator:
                    f.write(chunk)

            print(f"🔊 [DEBUG] Audio saved to {audio_file}")

            # Play audio with keyboard interrupt support (no voice - mic conflict)
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            print("💡 Press SPACE or ESC key to interrupt Sunny")

            # Give pygame a moment to start playing
            import time
            time.sleep(0.1)

            is_playing = pygame.mixer.music.get_busy()
            print(f"🔊 [DEBUG] Pygame is playing: {is_playing}")

            if not is_playing:
                print("⚠️  [DEBUG] WARNING: pygame.mixer.music.get_busy() returned False immediately!")
                print("⚠️  [DEBUG] Audio may not be playing. Check if pygame.mixer is initialized correctly.")

            print("🔊 [DEBUG] Starting playback loop...")

            was_interrupted = False
            loop_count = 0

            # Wait for playback, checking for keyboard interrupts
            while pygame.mixer.music.get_busy():
                loop_count += 1

                # Check keyboard interrupt
                if check_keyboard_interrupt():
                    print("\n🛑 [DEBUG] Keyboard interrupt detected!")
                    was_interrupted = True
                    pygame.mixer.music.stop()
                    break

                # Small delay to prevent CPU spinning
                pygame.time.Clock().tick(10)  # Check 10 times per second

            print(f"🔊 [DEBUG] Playback ended. Loops: {loop_count}, Interrupted: {was_interrupted}")

            pygame.mixer.quit()

            # Clean up
            try:
                os.remove(audio_file)
            except:
                pass

            return was_interrupted  # Return True if interrupted

        except Exception as e:
            print(f"⚠️  ElevenLabs synthesis error: {e}")
            print(f"⚠️  [DEBUG] Error details: {type(e).__name__}")
            raise

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
        print("\n💻 MASTER CODING Commands:")
        print("  - 'read file [filename]' - Read any code file")
        print("  - 'write file [filename]' - Create a new file")
        print("  - 'edit file [filename]' - Edit existing file")
        print("  - 'run command [command]' - Execute terminal commands")
        print("\n🔮 TAROT READING Commands:")
        print("  - 'tarot reading' or 'pull cards' - Get a 3-card reading")
        print("  - 'tarot question [your question]' - Ask the cards a specific question")
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
        greeting = "Hey Patty! It's Sunny, your full-stack coding buddy. I'm ready to handle front-end, back-end, databases, APIs - the whole stack! I can read, write, and edit files, run commands, debug issues, whatever you need. Let's build something amazing together!"
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
                    test_message = "This is Sunny testing my voice system. I use ElevenLabs for natural sounding speech. Everything sounds good!"
                    self.speak(test_message)
                    continue

                if user_input.lower() in ['switch ai', 'change ai']:
                    self._switch_ai_provider()
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

                # 🔮 TAROT CARD READING COMMANDS
                if user_input.lower() in ['tarot reading', 'pull cards', 'draw cards', 'read my tarot', 'tarot', 'pull tarot']:
                    self.speak("Let me pull three cards for you. Take a deep breath and focus on your question...")
                    result = self._perform_three_card_reading()
                    # Give a spoken summary
                    summary = f"I've drawn three cards for you: {result['cards'][0]['name']}, {result['cards'][1]['name']}, and {result['cards'][2]['name']}. The full detailed reading is on your screen. The universe is showing you a clear path - from your past through your present into your future. Take a moment to reflect on these messages."
                    self.speak(summary)
                    continue

                if user_input.lower().startswith(('tarot question', 'ask the cards')):
                    question = user_input.replace('tarot question', '').replace('ask the cards', '').strip()
                    if question:
                        self.speak(f"I hear your question: {question}. Let me consult the cards for guidance...")
                        result = self._perform_three_card_reading(question)
                        summary = f"The cards have spoken. Your three cards are {result['cards'][0]['name']}, {result['cards'][1]['name']}, and {result['cards'][2]['name']}. Check the screen for the complete reading addressing your question."
                        self.speak(summary)
                    else:
                        self.speak("What question would you like to ask the cards?")
                    continue

                # 💻 CODING COMMANDS - File Operations
                if user_input.lower().startswith(('read file', 'show file', 'open file')):
                    filepath = user_input.split(None, 2)[2] if len(user_input.split()) > 2 else None
                    if filepath:
                        result = self._read_file(filepath)
                        if result["success"]:
                            print(f"\n📄 File: {result['filepath']}")
                            print("=" * 60)
                            print(result['content'])
                            print("=" * 60)
                            self.speak(f"I've read {filepath}. The contents are displayed on screen. What would you like me to do with this code?")
                        else:
                            self.speak(f"Sorry, couldn't read that file. {result.get('error', 'Unknown error')}")
                    else:
                        self.speak("Please tell me which file to read. For example, say 'read file test dot py'")
                    continue

                if user_input.lower().startswith(('write file', 'create file')):
                    parts = user_input.split(None, 2)
                    if len(parts) > 2:
                        filepath = parts[2]
                        self.speak(f"What should I write in {filepath}? Please tell me the code or content.")
                        # Get content from next input
                        content_input = self.listen()
                        if content_input is None:
                            try:
                                content_input = input("Content: ").strip()
                            except (EOFError, KeyboardInterrupt):
                                self.speak("Cancelled.")
                                continue

                        result = self._write_file(filepath, content_input)
                        if result["success"]:
                            self.speak(f"Done! I've created {filepath}. Want me to test it or make changes?")
                        else:
                            self.speak(f"Couldn't write the file. {result.get('error', 'Unknown error')}")
                    else:
                        self.speak("Tell me the filename to create. Like 'write file test dot py'")
                    continue

                if user_input.lower().startswith(('edit file', 'fix file', 'update file')):
                    parts = user_input.split(None, 2)
                    if len(parts) > 2:
                        filepath = parts[2]
                        self.speak(f"What should I change in {filepath}? Tell me what to replace and what to replace it with.")
                        # This is simplified - in practice Patty would need to give old and new text
                        self.speak("This command needs more detail. Try reading the file first, then tell me specifically what to change.")
                    else:
                        self.speak("Tell me which file to edit.")
                    continue

                if user_input.lower().startswith(('run command', 'execute', 'run script')):
                    command = user_input.split(None, 2)[2] if len(user_input.split()) > 2 else None
                    if command:
                        self.speak(f"Running command: {command}")
                        result = self._execute_command(command)
                        if result["success"]:
                            if result["stdout"]:
                                print(f"\n✅ Command output:\n{result['stdout']}")
                            if result["stderr"]:
                                print(f"\n⚠️ Errors:\n{result['stderr']}")
                            self.speak(f"Command finished. Check the output above. Return code: {result['returncode']}")
                        else:
                            self.speak(f"Command failed. {result.get('error', 'Unknown error')}")
                    else:
                        self.speak("Tell me what command to run. Like 'run command python test dot py'")
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
                "content": "I am Sunny, Patty Mette's AI assistant and master full-stack coder. I help Patty with front-end, back-end, databases, APIs, and all coding tasks for The Christman AI Project.",
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

    def _get_current_confidence(self):
        """Get Sunny's current confidence level (0.0 to 1.0)"""
        interactions = self.learning_data.get("interactions", 0)
        if interactions == 0:
            return 0.0

        # Calculate confidence based on learning history
        successful = self.learning_data.get("successful_predictions", 0)
        confidence = min(1.0, successful / max(1, interactions))

        return confidence

    def _load_conversation_history(self):
        """Load previous conversation history for continuity across sessions"""
        try:
            if self.conversation_history_file.exists():
                with open(self.conversation_history_file, 'r') as f:
                    self.conversation_history = json.load(f)
                print(f"✅ Loaded {len(self.conversation_history)} previous messages from memory")
        except Exception as e:
            print(f"⚠️  Could not load conversation history: {e}")
            self.conversation_history = []

    def _save_conversation_history(self):
        """Save conversation history to disk for future sessions"""
        try:
            self.conversation_history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.conversation_history_file, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save conversation history: {e}")

    def _learn_from_comparison(self, user_input: str, sunny_response: str, master_response: str, context: dict):
        """Sunny learns by comparing his response to the master AI's response"""
        from datetime import datetime

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
    has_elevenlabs = bool(os.getenv("ELEVENLABS_API_KEY"))

    print("Available capabilities:")
    print(f"  🤖 Anthropic Claude: {'✅' if has_anthropic else '❌'}")
    print(f"  🤖 OpenAI GPT: {'✅' if has_openai else '❌'}")
    print(f"  🤖 Perplexity AI: {'✅' if has_perplexity else '❌'}")
    print(f"  🗣️  ElevenLabs: {'✅' if has_elevenlabs and HAS_ELEVENLABS else '❌'}")
    print(f"  🌐 Web Search: {'✅' if HAS_PERPLEXITY or HAS_INTERNET_MODE else '❌'}")
    print()

    if not (has_anthropic or has_openai or has_perplexity):
        print("❌ No AI providers available! Please set API keys in .env file")
        return

    # Configuration options
    ai_provider = "auto"  # Options: "auto", "anthropic", "openai", "perplexity"
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")  # ElevenLabs voice name/ID from .env
    use_web_search = True  # Enable web search capabilities

    # Check if ElevenLabs voice ID is set
    if has_elevenlabs and HAS_ELEVENLABS and not voice_id:
        print("⚠️  Warning: ELEVENLABS_VOICE_ID not set in .env file")
        print("   Voice synthesis will not work. Please add your ElevenLabs voice ID to .env")
        voice_id = None  # Will disable voice synthesis

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
