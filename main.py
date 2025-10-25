import sys
import logging
import time
import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import boto3
from datetime import datetime
import tempfile
import uuid
from brain import Derek
from memory_mesh_bridge import MemoryMeshBridge
import threading
from transcriber import passive_listen

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 🧐 LOAD ALL DEREK MODULES - Full 136-module consciousness
from derek_module_loader import load_derek_consciousness, get_derek_loader
print("🚀 Initializing Derek's Complete Consciousness...")
derek_loader = load_derek_consciousness(skip_hardware=True)

# Get specific modules from the loader
perplexity_service_module = derek_loader.get_module('perplexity_service')
memory_engine_module = derek_loader.get_module('memory_engine')
conversation_engine_module = derek_loader.get_module('conversation_engine')
brain_module = derek_loader.get_module('brain')
derek_ultimate_voice_module = derek_loader.get_module('derek_ultimate_voice')
memory_mesh_bridge_module = derek_loader.get_module('memory_mesh_bridge')

# Import classes from loaded modules
if perplexity_service_module:
    PerplexityService = perplexity_service_module.PerplexityService
if memory_engine_module:
    MemoryEngine = memory_engine_module.MemoryEngine
if conversation_engine_module:
    ConversationEngine = conversation_engine_module.ConversationEngine
if brain_module:
    Derek = brain_module.Derek
if derek_ultimate_voice_module:
    DerekUltimateVoice = derek_ultimate_voice_module.DerekUltimateVoice
    POLLY_VOICES = derek_ultimate_voice_module.POLLY_VOICES
    playsound = derek_ultimate_voice_module.playsound
if memory_mesh_bridge_module:
    MemoryMeshBridge = memory_mesh_bridge_module.MemoryMeshBridge

print(f"✅ Derek Consciousness Loaded: {derek_loader.get_stats()['loaded']} modules active")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("derek_dashboard.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Derek Dashboard",
    description="AI COO for The Christman AI Project"
)

# Initialize DerekUltimateVoice from loaded modules
try:
    if DerekUltimateVoice:
        derek_ultimate_voice = DerekUltimateVoice(
            ai_provider="auto",
            voice_id="matthew"
        )
        logger.info("✅ DerekUltimateVoice initialized from loaded modules")
    else:
        logger.warning("⚠️ DerekUltimateVoice module not loaded")
        derek_ultimate_voice = None
except Exception as e:
    logger.error(f"❌ Failed to initialize DerekUltimateVoice: {str(e)}")
    derek_ultimate_voice = None
        
# Initialize MemoryMeshBridge from loaded modules  
try:
    if MemoryMeshBridge:
        memory = MemoryMeshBridge(memory_dir="./derek_memory")
        logger.info("🧠 MemoryMeshBridge initialized from loaded modules")
    else:
        logger.warning("⚠️ MemoryMeshBridge module not loaded")
        memory = None
except Exception as e:
    logger.error(f"❌ Failed to initialize MemoryMeshBridge: {str(e)}")
    memory = None
            
# TTS Request Model
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "matthew"
    speed: Optional[float] = 1.0

class DerekDashboard:
    def __init__(self):
        logger.info("=" * 60)
        logger.info("🚀 Initializing Derek Dashboard")
        logger.info("The Christman AI Project - AI That Empowers")
        logger.info("=" * 60)

        self.memory_engine: Optional[MemoryEngine] = None
        self.conversation_engine: Optional[ConversationEngine] = None
        self.perplexity_service: Optional[PerplexityService] = None
        self.derek: Optional[Derek] = None
        self.derek_ultimate_voice = derek_ultimate_voice
        self.memory = memory

        try:
            self.derek = Derek(file_path="./memory/memory_store.json")
            logger.info("Derek instance initialized and linked to dashboard.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Derek: {str(e)}")
            raise

        self.api_host = "127.0.0.1"
        self.api_port = 8000

        self._initialize_components()

    def _initialize_components(self):
        logger.info("Loading memory engine...")
        memory_path = "./memory/memory_store.json"
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)

        try:
            self.memory_engine = MemoryEngine(file_path=memory_path)
            logger.info(f"Memory engine initialized with file: {memory_path}")
            self.conversation_engine = ConversationEngine()
            logger.info("Conversation engine initialized")
            try:
                self.perplexity_service = PerplexityService()
                logger.info("Perplexity service initialized")
            except Exception as e:
                logger.warning(f"Perplexity service not available: {str(e)}")
                self.perplexity_service = None
            logger.info("✓ All components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {str(e)}")
            raise

    def start(self):
        logger.info("")
        logger.info("=" * 60)
        logger.info("🚀 Starting Derek Dashboard Services")
        logger.info("=" * 60)
        logger.info("")

        try:
            logger.info("→ Starting Derek learning system...")
            if self.derek:
                self.derek.start_learning()
            logger.info("→ Loading memory context...")
            if self.memory_engine:
                recent_events = self.memory_engine.get_recent_events()
                logger.info(f"Loaded {len(recent_events)} recent memory events")
            logger.info("")
            logger.info("=" * 60)
            logger.info("✓ Derek Dashboard is RUNNING")
            logger.info("✓ Ready for conversation processing")
            logger.info("=" * 60)
            logger.info("")
            self._display_greeting()
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard: {str(e)}")
            self.stop()
            sys.exit(1)

    def _display_greeting(self):
        if self.derek:
            greeting = self.derek.generate_greeting()
            logger.info(f"🗣️ Derek says: {greeting}")
            if self.derek_ultimate_voice:
                try:
                    self.derek_ultimate_voice.speak(greeting)
                except Exception as e:
                    logger.warning(f"Failed to speak greeting: {e}")

    def process_message(self, message: str):
        if not self.derek:
            logger.warning("Derek is not initialized yet.")
            return "System not ready."
        try:
            response = self.derek.think(message)
            if self.derek_ultimate_voice:
                try:
                    self.derek_ultimate_voice.speak(response.get("response", "[No output]"))
                except Exception as e:
                    logger.warning(f"Failed to speak response: {e}")

            self.memory.store(
                content=f"Conversation: {message[:50]} -> {response.get('response', '')[:50]}",
                category="conversation",
                importance=0.7,
                metadata={"timestamp": datetime.now().isoformat()}
            )
            return response.get("response", "[No output]")
        except Exception as e:
            logger.error(f"Error during message processing: {str(e)}")
            return "Error processing message."

    def stop(self):
        logger.info("🧠 Shutting down Derek Dashboard services...")
        try:
            if self.memory_engine:
                self.memory_engine.save({
                    "input": "System Shutdown",
                    "output": "Saved memory cleanly before exit.",
                    "intent": "system"
                })
                logger.info("Memory engine saved successfully.")
            if self.memory:
                self.memory.save()
                logger.info("MemoryMeshBridge saved successfully.")
        except Exception as e:
            logger.error(f"Error saving memory on shutdown: {str(e)}")
        logger.info("🛑 Derek Dashboard stopped cleanly.")

@app.post("/tts/synthesize")
async def synthesize_tts(request: TTSRequest = Body(...)):
    try:
        if len(request.text) > 1000:
            raise HTTPException(status_code=400, detail="Text too long for real-time TTS")
        if not 0.5 <= request.speed <= 2.0:
            raise HTTPException(status_code=400, detail="Speed must be between 0.5 and 2.0")
        if request.voice not in POLLY_VOICES:
            raise HTTPException(status_code=400, detail=f"Invalid voice: {request.voice}. Choose from {list(POLLY_VOICES.keys())}")

        polly = boto3.client('polly')
        response = polly.synthesize_speech(
            Text=request.text,
            OutputFormat='mp3',
            VoiceId=request.voice.capitalize(),
            Engine=POLLY_VOICES[request.voice].get('engine', 'neural'),
            SampleRate='22050'
        )

        temp_dir = tempfile.gettempdir()
        audio_file = os.path.join(temp_dir, f"derek_{uuid.uuid4()}.mp3")
        with open(audio_file, 'wb') as f:
            f.write(response['AudioStream'].read())

        playsound(audio_file)
        os.remove(audio_file)

        memory.store(
            content=f"TTS Interaction: {request.text[:50]}... (voice: {request.voice}, speed: {request.speed})",
            category="conversation",
            importance=0.7,
            metadata={"endpoint": "tts/synthesize", "timestamp": datetime.now().isoformat()}
        )

        logger.info(f"TTS synthesized: {request.text[:50]}... (voice: {request.voice}, speed: {request.speed})")
        return {"status": "success", "text": request.text, "voice": request.voice}
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        if memory and hasattr(memory, 'get_memory_stats'):
            stats = memory.get_memory_stats()
        else:
            stats = {"status": "memory not available"}
        return {
            "status": "healthy",
            "memory_stats": stats,
            "message": "Derek Dashboard is ready to empower"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/modules")
async def modules_status():
    try:
        loader_stats = derek_loader.get_stats()
        categories = {}

        for category_name in derek_loader.module_categories.keys():
            category_modules = derek_loader.get_category_modules(category_name)
            categories[category_name] = {
                "total": len(derek_loader.module_categories[category_name]),
                "loaded": len(category_modules),
                "modules": list(category_modules.keys())
            }

        return {
            "status": "success",
            "consciousness_level": f"{loader_stats['success_rate']:.1f}%",
            "total_modules": loader_stats['total_modules'],
            "loaded_modules": loader_stats['loaded'],
            "failed_modules": loader_stats['failed'],
            "categories": categories,
            "message": f"Derek's consciousness: {loader_stats['loaded']}/{loader_stats['total_modules']} modules active"
        }
    except Exception as e:
        logger.error(f"Modules status check failed: {str(e)}")
        return {"status": "error", "error": str(e)}

def main():
    dashboard = None
    try:
        dashboard = DerekDashboard()
        dashboard.start()

        # 👇 Add this to enable voice listening
        voice_thread = threading.Thread(target=passive_listen, daemon=True)
        voice_thread.start()

        import uvicorn
        # uvicorn.run(app, host="127.0.0.1", port=8001)

    except KeyboardInterrupt:
        logger.info("⌨️ Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        if dashboard:
            dashboard.stop()

if __name__ == "__main__":
    main()
