"""
Royce Conversation Engine
-------------------------
The brain that ties everything together.
Handles AI responses, intent detection, memory integration,
and routing to specialized modules (music, news, research, etc.)
"""

import os
import re
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from .config import RoyceConfig
from .personality import RoycePersonality
from .memory import RoyceMemory
from .mood import RoyceMoodDetector
from .music import RoyceMusic
from .research import RoyceResearch
from .horoscope import RoyceHoroscope
from .news import RoyceNews
from .images import RoyceImages

load_dotenv()

logger = logging.getLogger(__name__)


class RoyceConversation:
    """
    The core conversation engine. Royce's brain.

    Orchestrates:
    - AI response generation (Claude via Bedrock or API)
    - Memory (stores and recalls everything)
    - Mood detection
    - Routing to music, research, news, horoscope, image modules
    - Personality enforcement (no asterisks, old friend tone)
    """

    def __init__(self):
        # Core systems
        self.personality = RoycePersonality()
        self.memory = RoyceMemory()
        self.mood_detector = RoyceMoodDetector()

        # Feature modules
        self.music = RoyceMusic()
        self.research = RoyceResearch()
        self.horoscope = RoyceHoroscope()
        self.news = RoyceNews()
        self.images = RoyceImages()

        # AI client
        self.ai_provider = RoyceConfig.AI_PROVIDER
        self._setup_ai_client()

        # State
        self.current_user_mood = "neutral"
        self.previous_user_mood = "neutral"
        self.conversation_active = True

        logger.info("Royce Conversation Engine initialized — all systems go")

    def _setup_ai_client(self):
        """Initialize the AI provider (Anthropic direct API or AWS Bedrock)."""
        self.anthropic_client = None
        self.bedrock_client = None

        if self.ai_provider == "bedrock":
            try:
                import boto3
                self.bedrock_client = boto3.client(
                    "bedrock-runtime",
                    region_name=RoyceConfig.AWS_REGION,
                )
                logger.info("AWS Bedrock AI client initialized")
            except Exception as e:
                logger.error(f"Bedrock init failed: {e}")
        else:
            try:
                from anthropic import Anthropic
                self.anthropic_client = Anthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
                logger.info("Anthropic API client initialized")
            except Exception as e:
                logger.error(f"Anthropic init failed: {e}")

    # ─── Main Processing ───────────────────────────────────────────

    def process(self, user_input: str) -> str:
        """
        Process a user message and return Royce's response.

        This is the main entry point. It:
        1. Detects mood
        2. Checks for special commands (music, news, etc.)
        3. Builds context from memory
        4. Gets AI response
        5. Filters the response (no asterisks!)
        6. Stores everything in memory
        """
        if not user_input or not user_input.strip():
            return ""

        user_input = user_input.strip()

        # Store user message in memory
        self.memory.store_message("user", user_input, mood=self.current_user_mood)

        # Detect mood
        mood, confidence = self.mood_detector.detect(user_input)
        self.previous_user_mood = self.current_user_mood
        self.current_user_mood = mood
        self.memory.log_mood(mood, confidence, trigger=user_input[:100])

        # Check for mood shift
        mood_comment = None
        if self.mood_detector.has_mood_changed() and confidence > 0.5:
            mood_comment = self.mood_detector.get_mood_shift_comment(
                mood, self.previous_user_mood
            )

        # Check for special commands first
        special_response = self._handle_special_commands(user_input)
        if special_response:
            response = special_response
        else:
            # Normal AI conversation
            response = self._get_ai_response(user_input)

        # Prepend mood observation if there was a shift
        if mood_comment:
            response = f"{mood_comment}\n\n{response}"

        # Filter out any asterisks or action descriptions
        response = self.personality.filter_response(response)

        # Store Royce's response in memory
        self.memory.store_message("royce", response, mood=mood)

        # Extract and remember any personal details mentioned
        self._extract_user_info(user_input)

        return response

    # ─── Special Command Routing ───────────────────────────────────

    def _handle_special_commands(self, text: str) -> Optional[str]:
        """Check if the input matches a special command and handle it."""

        # Music commands
        music_cmd = self.music.parse_music_command(text)
        if music_cmd:
            return self._handle_music(music_cmd)

        # News commands
        news_cmd = self.news.parse_news_command(text)
        if news_cmd:
            return self._handle_news(news_cmd)

        # Horoscope commands
        horo_cmd = self.horoscope.parse_horoscope_command(text)
        if horo_cmd:
            return self._handle_horoscope(horo_cmd)

        # Research commands
        research_cmd = self.research.parse_research_command(text)
        if research_cmd:
            return self._handle_research(research_cmd)

        # Image commands
        image_cmd = self.images.parse_image_command(text)
        if image_cmd:
            return self._handle_image(image_cmd)

        return None

    def _handle_music(self, cmd: Dict) -> str:
        action = cmd.get("action")
        if action == "play":
            result = self.music.play_from_youtube(cmd["query"])
            return result.get("message", "Having trouble with that song.")
        elif action == "stop":
            return self.music.stop().get("message", "Stopped.")
        elif action == "pause":
            return self.music.pause().get("message", "Paused.")
        elif action == "resume":
            return self.music.resume().get("message", "Resuming.")
        elif action == "next":
            return self.music.play_next().get("message", "Queue's empty.")
        elif action == "now_playing":
            result = self.music.now_playing()
            if result["status"] == "playing":
                return f"Currently playing: {result.get('title', 'something')}"
            return "Nothing's playing right now."
        elif action == "volume":
            return self.music.set_volume(cmd.get("level", 0.5)).get("message", "Volume adjusted.")
        return "Not sure what you want me to do with the music."

    def _handle_news(self, cmd: Dict) -> str:
        action = cmd.get("action")
        if action == "headlines":
            category = cmd.get("category")
            result = self.news.get_breaking_news(category=category, count=7)
            if result["status"] == "success":
                prefix = f"Here's what's happening"
                if category:
                    prefix += f" in {category}"
                return f"{prefix}:\n\n{result['summary']}"
            return result.get("message", "Couldn't pull up the news right now.")
        elif action == "search":
            result = self.news.search_news(cmd["query"], count=5)
            if result["status"] == "success":
                return f"Here's what I found about {cmd['query']}:\n\n{result['summary']}"
            return result.get("message", "Couldn't find news on that.")
        return "Not sure what news you're looking for."

    def _handle_horoscope(self, cmd: Dict) -> str:
        sign = cmd.get("sign")
        if not sign:
            # Try to get from memory
            user_sign = self.memory.recall_about_user("zodiac_sign")
            birthday = self.memory.recall_about_user("birthday")
            if user_sign:
                sign = user_sign
            elif birthday:
                try:
                    parts = birthday.split("-")
                    month, day = int(parts[1]), int(parts[2])
                    sign = RoyceHoroscope.get_sign_from_birthday(month, day)
                except (ValueError, IndexError):
                    pass

            if not sign:
                return "I don't know your zodiac sign yet. What's your birthday? Or just tell me your sign."

        result = self.horoscope.get_horoscope(sign)
        if result["status"] == "success":
            return f"Here's your {result['sign']} horoscope for {result['date']}:\n\n{result['horoscope']}"
        return result.get("message", "Couldn't get the horoscope right now.")

    def _handle_research(self, cmd: Dict) -> str:
        subject = cmd.get("subject", "")
        result = self.research.deep_search(subject, detail_level="thorough")
        if result["status"] == "success":
            # Store researched knowledge
            self.memory.store_knowledge(
                topic=subject,
                content=result["content"][:2000],
                source=", ".join(result.get("sources", [])[:3]),
            )
            return result["content"]
        return result.get("message", f"Couldn't find much on '{subject}'. Try being more specific.")

    def _handle_image(self, cmd: Dict) -> str:
        description = cmd.get("description", "")
        result = self.images.generate(description)
        if result["status"] == "success":
            return result.get("message", f"Done. Image saved to {result.get('file_path', 'your images folder')}.")
        return result.get("message", "Couldn't generate that image.")

    # ─── AI Response Generation ────────────────────────────────────

    def _get_ai_response(self, user_input: str) -> str:
        """Get a response from the AI model with full personality and context."""

        # Build context from memory
        user_context = self.memory.get_full_user_profile()
        memory_context = self.memory.build_context_for_ai(user_input)

        # Build system prompt with personality
        system_prompt = self.personality.get_system_prompt(
            user_mood=self.current_user_mood,
            context=user_context,
        )

        # Add memory context
        full_system = f"""{system_prompt}

--- MEMORY CONTEXT ---
{memory_context}
--- END MEMORY ---"""

        # Build conversation history for the AI
        recent = self.memory.get_recent_conversations(limit=30)
        messages = []
        for msg in recent:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})

        # Make sure current message is the last one
        if not messages or messages[-1]["content"] != user_input:
            messages.append({"role": "user", "content": user_input})

        # Call the AI
        if self.anthropic_client:
            return self._call_anthropic(full_system, messages)
        elif self.bedrock_client:
            return self._call_bedrock(full_system, messages)
        else:
            return "I'm having trouble connecting to my AI backend right now. Check the API keys in the .env file."

    def _call_anthropic(self, system: str, messages: List[Dict]) -> str:
        """Call Anthropic API directly."""
        try:
            response = self.anthropic_client.messages.create(
                model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"),
                max_tokens=2000,
                system=system,
                messages=messages,
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return "Having a bit of trouble thinking right now. Give me a second and try again."

    def _call_bedrock(self, system: str, messages: List[Dict]) -> str:
        """Call Claude via AWS Bedrock."""
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "system": system,
                "messages": messages,
            })

            response = self.bedrock_client.invoke_model(
                modelId=RoyceConfig.CLAUDE_MODEL,
                body=body,
            )

            result = json.loads(response["body"].read())
            return result["content"][0]["text"]
        except Exception as e:
            logger.error(f"Bedrock API error: {e}")
            return "Having a bit of trouble connecting right now. Check the AWS config."

    # ─── User Info Extraction ──────────────────────────────────────

    def _extract_user_info(self, text: str):
        """Try to extract personal details from what the user says and remember them."""
        text_lower = text.lower()

        # Name detection
        name_patterns = [
            r"(?:my name is|i'm|i am|call me|they call me)\s+([A-Z][a-z]+)",
            r"(?:my name is|i'm|i am|call me)\s+(\w+)",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 1 and name.lower() not in ("a", "an", "the", "just", "really", "so", "very"):
                    self.memory.learn_about_user("name", name)
                    break

        # Birthday detection
        bday_patterns = [
            r"(?:my birthday is|born on|birthday(?:'s| is))\s+(\w+ \d{1,2}(?:,?\s*\d{4})?)",
            r"(?:my birthday is|born on)\s+(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)",
        ]
        for pattern in bday_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                self.memory.learn_about_user("birthday", match.group(1))
                break

        # Zodiac sign detection
        for sign in ["aries", "taurus", "gemini", "cancer", "leo", "virgo",
                      "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"]:
            if re.search(rf"i(?:'m| am) (?:a |an? )?{sign}", text_lower):
                self.memory.learn_about_user("zodiac_sign", sign)
                break

        # Location detection
        loc_patterns = [
            r"i (?:live|am|stay) in\s+([A-Z][\w\s,]+?)(?:\.|$|,\s*(?:and|but))",
            r"i'm from\s+([A-Z][\w\s,]+?)(?:\.|$|,\s*(?:and|but))",
        ]
        for pattern in loc_patterns:
            match = re.search(pattern, text)
            if match:
                self.memory.learn_about_user("location", match.group(1).strip())
                break

        # Interest/hobby detection
        interest_patterns = [
            r"i (?:love|enjoy|like|am into|am passionate about)\s+(.+?)(?:\.|$|,\s*(?:and|but))",
        ]
        for pattern in interest_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                interest = match.group(1).strip()[:100]
                existing = self.memory.recall_about_user("interests")
                if existing:
                    if interest.lower() not in existing.lower():
                        self.memory.learn_about_user("interests", f"{existing}, {interest}")
                else:
                    self.memory.learn_about_user("interests", interest)
                break

    def get_greeting(self) -> str:
        """Generate a greeting, personalized if we know the user."""
        user_name = self.memory.recall_about_user("name")
        return self.personality.get_greeting(user_name=user_name)

    def shutdown(self):
        """Clean shutdown — save everything."""
        self.memory.close()
        self.music.stop()
        logger.info("Royce conversation engine shut down cleanly")
