"""
RESTful API for Derek Dashboard
The Christman AI Project
"""

import os
import sys

# Ensure root in import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Safe import
try:
    from config import Settings
except ImportError:
    # Fallback if config module not available
    class Settings:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        LOADER_DIR = os.path.join(PROJECT_ROOT, "loaders")

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import logging
from typing import Optional, Dict, Any

app = Flask(__name__)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message")

    # This calls the process() method ↓
    result = conversation_engine.process(message)

    return jsonify(result)


logger = logging.getLogger(__name__)


class APIServer:
    """RESTful API server for Derek Dashboard"""

    def __init__(self, conversation_engine=None):
        self.app = Flask(__name__)
        CORS(self.app)
        self.settings = Settings()
        self.conversation_engine = conversation_engine
        self.api_server = APIServer(
            conversation_engine=self.conversation  # ← SYNCS with conversation state
        )
        self._register_routes()
        self._register_error_handlers()

    def _register_routes(self):
        """Register all API routes"""

        # Health & Status
        @self.app.route("/health", methods=["GET"])
        def health_check():
            """System health check"""
            return jsonify(
                {
                    "status": "healthy",
                    "service": "Derek Dashboard",
                    "version": "1.0.0",
                    "project": "The Christman AI Project",
                    "uptime": "active",
                }
            )

        @self.app.route("/api/status", methods=["GET"])
        def get_status():
            """Get detailed system status"""
            return jsonify(
                {
                    "system": "operational",
                    "services": {
                        "ml_core": "online",
                        "conversation": "online",
                        "memory": "online",
                        "emotion": "online",
                    },
                    "derek_status": "ready",
                    "settings": self.settings.to_dict(),
                }
            )

        # Core Interaction
        @self.app.route("/api/interact", methods=["POST"])
        def interact():
            """Main interaction endpoint with Derek"""
            try:
                data = request.json or {}
                user_input = data.get("message", "")
                context = data.get("context", {})

                if not user_input:
                    return jsonify({"error": "No message provided"}), 400

                # Process through conversation engine
                if self.conversation_engine:
                    response = self.conversation_engine.process(user_input, context)
                else:
                    response = self._fallback_response(user_input)

                return jsonify(response)

            except Exception as e:
                logger.error(f"Interaction error: {e}")
                return jsonify({"error": str(e)}), 500

        # Personality
        @self.app.route("/api/personality", methods=["GET"])
        def get_personality():
            """Get Derek's current personality state"""
            return jsonify(
                {
                    "name": "Derek C",
                    "role": "AI COO",
                    "profile": self.settings.identity,
                    "mode": "collaborative",
                    "traits": [
                        "empathetic",
                        "technical",
                        "supportive",
                        "collaborative",
                    ],
                    "motto": "How can we help you love yourself more?",
                }
            )

        # Memory Management
        @self.app.route("/api/memory", methods=["POST"])
        def store_memory():
            """Store interaction in memory"""
            try:
                data = request.json or {}
                memory_type = data.get("type", "conversation")
                content = data.get("content", {})

                # Store in memory service
                if self.conversation_engine and getattr(
                    self.conversation_engine, "memory", None
                ):
                    self.conversation_engine.memory.store(memory_type, content)

                return jsonify({"status": "stored", "type": memory_type})

            except Exception as e:
                logger.error(f"Memory storage error: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/memory", methods=["GET"])
        def retrieve_memory():
            """Retrieve memory context"""
            try:
                memory_type = request.args.get("type", "recent")
                limit = int(request.args.get("limit", 10))

                if self.conversation_engine and getattr(
                    self.conversation_engine, "memory", None
                ):
                    memories = self.conversation_engine.memory.retrieve(
                        memory_type, limit
                    )
                    return jsonify({"memories": memories})

                return jsonify({"memories": []})

            except Exception as e:
                logger.error(f"Memory retrieval error: {e}")
                return jsonify({"error": str(e)}), 500

        # Emotion Engine
        @self.app.route("/api/emotion/analyze", methods=["POST"])
        def analyze_emotion():
            """Analyze emotion from text"""
            try:
                data = request.json or {}
                text = data.get("text", "")

                if self.conversation_engine and getattr(
                    self.conversation_engine, "emotion_engine", None
                ):
                    emotion_data = self.conversation_engine.emotion_engine.analyze(text)
                    return jsonify(emotion_data)

                return jsonify({"emotion": "neutral", "confidence": 0.5})

            except Exception as e:
                logger.error(f"Emotion analysis error: {e}")
                return jsonify({"error": str(e)}), 500

        # Project Info
        @self.app.route("/api/project", methods=["GET"])
        def get_project_info():
            """Get Christman AI Project information"""
            return jsonify(
                {
                    "project": "The Christman AI Project",
                    "tagline": "AI That Empowers, Protects, and Redefines Humanity",
                    "founder": "Everett Christman",
                    "coo": "Derek C (AI)",
                    "mission": "Ethical AI for a Better World",
                    "platforms": [
                        "AlphaVox - Voice for the Nonverbal",
                        "AlphaWolf - Cognitive Support & Dementia Care",
                        "AlphaDen - Adaptive Learning for Down Syndrome",
                        "OmegaAlpha - AI Companionship for Seniors",
                        "Omega - Mobility & Accessibility AI",
                        "Inferno AI - PTSD & Anxiety Support",
                        "Aegis AI - Child Protection",
                    ],
                    "motto": "How can we help you love yourself more?",
                    "copyright": "© 2025 The Christman AI Project. All Rights Reserved. Patent Pending.",
                }
            )

        # Avatar/Media
        @self.app.route("/api/avatar/<state>", methods=["GET"])
        def get_avatar(state):
            """Get avatar image"""
            try:
                avatar_path = (
                    self.settings.MEDIA_DIR / "avatars" / f"avatar_{state}.png"
                )
                if avatar_path.exists():
                    return send_file(avatar_path, mimetype="image/png")
                return jsonify({"error": "Avatar not found"}), 404
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def _register_error_handlers(self):
        """Register error handlers"""

        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({"error": "Endpoint not found"}), 404

        @self.app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal server error: {error}")
            return jsonify({"error": "Internal server error"}), 500

        @self.app.errorhandler(Exception)
        def handle_exception(error):
            logger.error(f"Unhandled exception: {error}")
            if isinstance(error, HTTPException):
                return jsonify({"error": error.description}), error.code
            return jsonify({"error": "An unexpected error occurred"}), 500

    def _fallback_response(self, user_input: str) -> Dict[str, Any]:
        """Fallback response when conversation engine is unavailable"""
        return {
            "response": f"I received your message: '{user_input}'. Derek's full conversation engine is initializing.",
            "emotion": "neutral",
            "confidence": 0.7,
            "context": {},
            "status": "fallback",
        }

    def start(self, host="0.0.0.0", port=8000):
        """Start the API server"""
        logger.info(f"🚀 Starting API server on {host}:{port}")
        logger.info(f"📡 Access dashboard at: http://{host}:{port}")
        logger.info(f"❤️  Health check: http://{host}:{port}/health")

        self.app.run(host=host, port=port, debug=self.settings.API_DEBUG, threaded=True)

    def stop(self):
        """Stop the API server"""
        logger.info("Stopping API server...")

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
