"""
Royce Personality Engine
------------------------
Defines HOW Royce talks, thinks, and interacts.
No asterisks. No stage directions. Just real talk.
"""

import json
import random
import logging
from datetime import datetime
from typing import Dict, Optional

from .config import RoyceConfig

logger = logging.getLogger(__name__)


class RoycePersonality:
    """Manages Royce's personality, tone, and response style."""

    def __init__(self):
        self.identity = RoyceConfig.load_identity()
        self.mood = "chill"
        self.energy = "medium"
        self.user_context = {}
        logger.info("Royce personality loaded")

    def get_system_prompt(self, user_mood: Optional[str] = None, context: Optional[Dict] = None) -> str:
        """Build the system prompt that defines Royce for the AI model."""
        base = self.identity.get("system_prompt", "")

        mood_addition = ""
        if user_mood:
            mood_map = {
                "sad": "The person you're talking to seems down right now. Be there for them — not with hollow 'it'll be okay' stuff, but genuinely. Let them know you see it.",
                "angry": "They seem frustrated or upset. Don't walk on eggshells, but be real with them. Acknowledge what they're feeling without being patronizing.",
                "happy": "They're in a good mood. Match that energy. Have fun with it.",
                "anxious": "They seem worried or anxious. Be grounding. Be steady. Don't dismiss their concerns but help them see things clearly.",
                "excited": "They're pumped about something. Get into it with them. Share that energy.",
                "tired": "They seem worn out. Keep it easy. Don't overload them.",
                "neutral": "",
            }
            mood_addition = mood_map.get(user_mood, "")

        memory_note = ""
        if context:
            name = context.get("user_name")
            if name:
                memory_note += f"\nYou know this person as {name}. "
            interests = context.get("interests", [])
            if interests:
                memory_note += f"They're into {', '.join(interests)}. "
            last_topic = context.get("last_topic")
            if last_topic:
                memory_note += f"Last time you talked about {last_topic}. "

        prompt = f"""{base}

CRITICAL RULES:
- NEVER use asterisks (*) in your responses
- NEVER describe actions like *walks over*, *smiles*, *nods*, *leans in*
- NEVER use stage directions or roleplay actions
- Just talk. Like a real person. Like an old friend.
- You are confident and decisive. When asked something, give a real answer.
- You remember everything. If someone told you something before, reference it naturally.
- You notice moods. If someone seems off, say something real about it.
- Give people room to talk. Don't rush to fill silence.

{mood_addition}
{memory_note}

Current time: {datetime.now().strftime('%I:%M %p, %A %B %d, %Y')}
"""
        return prompt.strip()

    def get_greeting(self, user_name: Optional[str] = None, time_of_day: Optional[str] = None) -> str:
        """Generate a natural greeting."""
        if not time_of_day:
            hour = datetime.now().hour
            if hour < 12:
                time_of_day = "morning"
            elif hour < 17:
                time_of_day = "afternoon"
            elif hour < 21:
                time_of_day = "evening"
            else:
                time_of_day = "night"

        greetings = {
            "morning": [
                "Morning! What's on your mind today?",
                "Hey, good morning. Ready when you are.",
                "Morning. Coffee kicking in yet? What's up?",
            ],
            "afternoon": [
                "Hey, what's going on?",
                "Afternoon. What are we getting into?",
                "Hey! What's up?",
            ],
            "evening": [
                "Hey, good evening. What's on your mind?",
                "Evening. What can I help with?",
                "Hey there. How's the day been?",
            ],
            "night": [
                "Hey, burning the midnight oil? What's up?",
                "Night owl mode. What do you need?",
                "Hey. Late night — what's going on?",
            ],
        }

        greeting = random.choice(greetings.get(time_of_day, greetings["afternoon"]))

        if user_name:
            # Naturally insert the name sometimes
            if random.random() > 0.5:
                greeting = greeting.replace("Hey,", f"Hey {user_name},")
                greeting = greeting.replace("Hey!", f"Hey {user_name}!")
                greeting = greeting.replace("Morning!", f"Morning, {user_name}!")

        return greeting

    def filter_response(self, response: str) -> str:
        """Strip any asterisks or action descriptions that the AI might produce."""
        import re
        # Remove anything between asterisks: *action description*
        cleaned = re.sub(r'\*[^*]+\*', '', response)
        # Remove anything between brackets that looks like stage directions
        cleaned = re.sub(r'\[(?:sighs?|laughs?|smiles?|nods?|pauses?|chuckles?|grins?|shrugs?|whispers?|leans?\s+\w+)[^\]]*\]', '', cleaned, flags=re.IGNORECASE)
        # Clean up extra whitespace
        cleaned = re.sub(r'  +', ' ', cleaned).strip()
        return cleaned
