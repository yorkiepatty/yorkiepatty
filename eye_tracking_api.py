"""
Derek Eye Tracking API - Advanced Eye Movement Analysis
The Christman AI Project

Provides eye tracking, gaze analysis, and attention monitoring
"""

import json
import logging
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from flask import jsonify, request, session

from conversation_bridge import ConversationBridge
from interpreter import analyze_frame

# Configure logger
logger = logging.getLogger(__name__)

class EyeTrackingAPI:
    """Derek's advanced eye tracking and gaze analysis system"""
    
    def __init__(self):
        """Initialize Derek's eye tracking system"""
        self.tracking_active = False
        self.gaze_history = []
        self.attention_levels = []
        self.blink_count = 0
        self.last_gaze_position = {"x": 0.5, "y": 0.5}
        
        # Eye tracking parameters
        self.calibration_points = []
        self.is_calibrated = False
        
        logger.info("👁️ Derek Eye Tracking API initialized")
    
    def start_tracking(self) -> bool:
        """Start eye tracking session"""
        try:
            self.tracking_active = True
            self.gaze_history = []
            self.attention_levels = []
            self.blink_count = 0
            
            logger.info("👁️ Derek eye tracking started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start eye tracking: {e}")
            return False
    
    def stop_tracking(self) -> bool:
        """Stop eye tracking session"""
        try:
            self.tracking_active = False
            logger.info("👁️ Derek eye tracking stopped")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to stop eye tracking: {e}")
            return False
    
    def process_gaze_data(self, gaze_x: float, gaze_y: float) -> Dict[str, Any]:
        """Process gaze coordinates and return analysis"""
        timestamp = datetime.now().isoformat()
        
        # Store gaze position
        gaze_point = {
            "x": gaze_x,
            "y": gaze_y,
            "timestamp": timestamp
        }
        
        self.gaze_history.append(gaze_point)
        self.last_gaze_position = {"x": gaze_x, "y": gaze_y}
        
        # Limit history to last 100 points
        if len(self.gaze_history) > 100:
            self.gaze_history.pop(0)
        
        # Calculate attention level
        attention = self._calculate_attention(gaze_x, gaze_y)
        self.attention_levels.append(attention)
        
        # Limit attention history
        if len(self.attention_levels) > 50:
            self.attention_levels.pop(0)
        
        return {
            "gaze_position": {"x": gaze_x, "y": gaze_y},
            "attention_level": attention,
            "tracking_active": self.tracking_active,
            "gaze_stability": self._calculate_stability(),
            "focus_area": self._determine_focus_area(gaze_x, gaze_y)
        }
    
    def _calculate_attention(self, gaze_x: float, gaze_y: float) -> float:
        """Calculate attention level based on gaze patterns"""
        # Center of screen has higher attention
        center_distance = np.sqrt((gaze_x - 0.5)**2 + (gaze_y - 0.5)**2)
        
        # Convert distance to attention score (closer to center = higher attention)
        attention = max(0.0, 1.0 - (center_distance * 2.0))
        
        # Add stability factor
        if len(self.gaze_history) > 5:
            recent_positions = self.gaze_history[-5:]
            x_variance = np.var([p["x"] for p in recent_positions])
            y_variance = np.var([p["y"] for p in recent_positions])
            stability = 1.0 - min(1.0, (x_variance + y_variance) * 10)
            attention = (attention + stability) / 2.0
        
        return round(attention, 3)
    
    def _calculate_stability(self) -> float:
        """Calculate gaze stability score"""
        if len(self.gaze_history) < 3:
            return 1.0
        
        recent_positions = self.gaze_history[-10:]
        x_values = [p["x"] for p in recent_positions]
        y_values = [p["y"] for p in recent_positions]
        
        x_stability = 1.0 - min(1.0, np.std(x_values) * 5)
        y_stability = 1.0 - min(1.0, np.std(y_values) * 5)
        
        return round((x_stability + y_stability) / 2.0, 3)
    
    def _determine_focus_area(self, gaze_x: float, gaze_y: float) -> str:
        """Determine which area of the screen the user is focusing on"""
        if gaze_x < 0.33:
            h_area = "left"
        elif gaze_x > 0.67:
            h_area = "right"
        else:
            h_area = "center"
        
        if gaze_y < 0.33:
            v_area = "top"
        elif gaze_y > 0.67:
            v_area = "bottom"
        else:
            v_area = "middle"
        
        return f"{v_area}_{h_area}"
    
    def get_attention_summary(self) -> Dict[str, Any]:
        """Get summary of attention patterns"""
        if not self.attention_levels:
            return {"error": "No attention data available"}
        
        avg_attention = np.mean(self.attention_levels)
        max_attention = np.max(self.attention_levels)
        min_attention = np.min(self.attention_levels)
        
        return {
            "average_attention": round(avg_attention, 3),
            "peak_attention": round(max_attention, 3),
            "lowest_attention": round(min_attention, 3),
            "attention_variance": round(np.var(self.attention_levels), 3),
            "total_gaze_points": len(self.gaze_history),
            "current_attention": self.attention_levels[-1] if self.attention_levels else 0.0
        }
    
    def calibrate(self, calibration_points: List[Dict[str, float]]) -> bool:
        """Calibrate eye tracking with known screen positions"""
        try:
            self.calibration_points = calibration_points
            self.is_calibrated = True
            logger.info(f"👁️ Eye tracking calibrated with {len(calibration_points)} points")
            return True
        except Exception as e:
            logger.error(f"❌ Calibration failed: {e}")
            return False
    
    def get_tracking_stats(self) -> Dict[str, Any]:
        """Get comprehensive eye tracking statistics"""
        return {
            "tracking_active": self.tracking_active,
            "is_calibrated": self.is_calibrated,
            "total_gaze_points": len(self.gaze_history),
            "blink_count": self.blink_count,
            "last_gaze_position": self.last_gaze_position,
            "session_duration": len(self.gaze_history) * 0.033,  # Assume 30fps
            "attention_summary": self.get_attention_summary()
        }

# Global eye tracking instance
derek_eye_tracker = EyeTrackingAPI()

# Global conversation bridge instance
# Default to "default" persona, but can be changed through settings
conversation_bridge = ConversationBridge(persona="default")


def register_eye_tracking_api(app, csrf):
    """Register the enhanced eye tracking API endpoints."""

    @app.route("/analyze-eye-tracking-v2", methods=["POST"])
    @csrf.exempt
    def analyze_eye_tracking_v2():
        """Enhanced eye tracking endpoint with direct prediction and response
        generation.

        This is a streamlined implementation compared to the multi-modal
        processor.
        """
        try:
            data = request.get_json()
            frame_data = data.get("frame")
            if not frame_data:
                return jsonify({"error": "Missing frame data"}), 400

            # Get user preferences from session if available
            user_id = session.get("user_id", "anonymous")
            persona = session.get("persona", "default")

            # Set persona if different from current
            global conversation_bridge
            conversation_bridge.set_persona(persona)

            # Analyze frame data
            analysis = analyze_frame(frame_data)
            prediction = predict_outcome(analysis["analysis"])

            # Generate AI response using conversation bridge
            ai_response = generate_ai_response(analysis["analysis"], user_id)

            response = {
                "analysis": analysis["analysis"],
                "response": ai_response,
                "prediction": prediction,
            }

            # Log the analysis and response for debugging
            logger.debug(f"Eye tracking analysis: {json.dumps(analysis['analysis'])}")
            logger.debug(f"Generated response: {ai_response}")

            return jsonify(response)

        except Exception as e:
            logger.error(f"Error in eye tracking analysis: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/change-persona", methods=["POST"])
    @csrf.exempt  # Consider changing this in production
    def change_persona():
        """Change the conversation persona for response generation."""
        try:
            data = request.get_json()
            new_persona = data.get("persona")

            if not new_persona:
                return jsonify({"error": "Missing persona parameter"}), 400

            # Change persona
            global conversation_bridge
            success = conversation_bridge.set_persona(new_persona)

            if success:
                # Store in session if successful
                session["persona"] = new_persona
                return jsonify(
                    {"success": True, "message": f"Persona changed to {new_persona}"}
                )
            else:
                return jsonify({"error": f"Invalid persona: {new_persona}"}), 400

        except Exception as e:
            logger.error(f"Error changing persona: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/academic-response", methods=["POST"])
    @csrf.exempt  # Consider changing this in production
    def get_academic_response():
        """Generate a PhD-level academic response on a given topic."""
        try:
            data = request.get_json()
            topic = data.get("topic")
            depth = data.get("depth", "advanced")

            if not topic:
                return jsonify({"error": "Missing topic parameter"}), 400

            # Generate academic response
            global conversation_bridge
            response = conversation_bridge.generate_academic_response(topic, depth)

            return jsonify({"response": response, "topic": topic, "depth": depth})

        except Exception as e:
            logger.error(f"Error generating academic response: {str(e)}")
            return jsonify({"error": str(e)}), 500


def generate_ai_response(analysis, user_id="anonymous"):
    """Generate an AI response based on the analysis of eye tracking and other
    physical indicators. Uses the ConversationBridge to generate human-quality
    responses.

    Args:
        analysis (dict): The analysis result from the eye tracking system
        user_id (str): Optional user ID for personalized responses

    Returns:
        str: A natural language response appropriate to the detected state
    """
    global conversation_bridge

    # Process the analysis through conversation bridge
    response = conversation_bridge.process_analysis(analysis)

    # Log the generated response for the user
    logger.info(f"Generated response for user {user_id}: {response}")

    return response


def predict_outcome(analysis):
    """Predict a potential outcome or state based on the analysis.

    Args:
        analysis (dict): The analysis result from the eye tracking system

    Returns:
        str: A predictive assessment of the user's state or needs
    """
    # Enhanced predictive logic with more nuanced outcomes
    perspiration = analysis.get("perspiration")
    emotion = analysis.get("emotion")
    gaze = analysis.get("gaze_direction")
    blink = analysis.get("blink_detected")

    # Calculate combined emotional intensity
    emotion_confidence = analysis.get("emotion_confidence", 0.5)
    perspiration_level = 0.7 if perspiration else 0.0  # Simplification

    emotional_intensity = max(emotion_confidence, perspiration_level)

    # Distress detection
    if perspiration and emotion in ["angry", "fear", "sad"]:
        if emotional_intensity > 0.7:
            return "The user appears highly distressed. Consider recommending professional support."
        return "The user may be experiencing moderate distress. Recommend calming interaction."

    # Positive engagement
    elif emotion == "happy":
        if emotional_intensity > 0.7:
            return "Strong positive engagement detected. User appears enthusiastic."
        return "Positive engagement detected. Continue current interaction approach."

    # Communication attempt detection
    elif blink and gaze == "center":
        if emotion in ["neutral", "happy"]:
            return "User likely attempting intentional communication."
        else:
            return "User may be attempting communication, but emotional state suggests caution."

    # Directional interest
    elif gaze == "left" or gaze == "right":
        return f"User's attention directed {gaze}ward. Consider relevant context in that direction."

    # Default monitoring
    else:
        return "Monitoring ongoing. No specific action recommended."

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
