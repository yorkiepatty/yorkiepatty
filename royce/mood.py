"""
Royce Mood Detection
--------------------
Picks up on how people are feeling from their words.
No fancy sentiment models needed — pattern matching + AI confirmation.
"""

import re
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Mood indicators — words and phrases that signal emotional states
MOOD_SIGNALS = {
    "happy": {
        "words": ["happy", "great", "awesome", "amazing", "love", "fantastic", "wonderful",
                  "excited", "stoked", "pumped", "thrilled", "blessed", "grateful", "yay",
                  "perfect", "beautiful", "incredible", "best", "lol", "haha", "lmao"],
        "phrases": ["so good", "feeling good", "can't wait", "made my day", "on top of the world",
                    "couldn't be better", "let's go", "hell yeah"],
        "weight": 1.0
    },
    "sad": {
        "words": ["sad", "depressed", "down", "upset", "crying", "miss", "lonely", "hurts",
                  "heartbroken", "terrible", "awful", "worst", "sucks", "devastated", "hopeless",
                  "lost", "empty", "broken"],
        "phrases": ["feeling down", "not okay", "can't do this", "don't care anymore",
                    "what's the point", "i'm done", "falling apart", "not feeling it"],
        "weight": 1.2
    },
    "angry": {
        "words": ["angry", "mad", "furious", "pissed", "hate", "annoyed", "frustrated",
                  "sick of", "tired of", "bullshit", "ridiculous", "unbelievable",
                  "stupid", "damn", "hell"],
        "phrases": ["pissed off", "fed up", "had enough", "sick and tired", "drives me crazy",
                    "can't believe", "so done", "ticked off"],
        "weight": 1.1
    },
    "anxious": {
        "words": ["worried", "anxious", "nervous", "scared", "afraid", "panic", "stress",
                  "stressed", "overwhelmed", "freaking", "terrified", "dread", "uneasy"],
        "phrases": ["freaking out", "can't stop thinking", "what if", "losing my mind",
                    "so worried", "can't sleep", "on edge", "losing it"],
        "weight": 1.1
    },
    "excited": {
        "words": ["excited", "hyped", "stoked", "pumped", "thrilled", "amped",
                  "psyched", "fired up", "ecstatic", "omg", "wow"],
        "phrases": ["can't wait", "so excited", "this is it", "let's gooo", "finally",
                    "about to", "just found out"],
        "weight": 1.0
    },
    "tired": {
        "words": ["tired", "exhausted", "drained", "sleepy", "burnt", "burnout",
                  "wiped", "spent", "fatigued", "done"],
        "phrases": ["so tired", "need sleep", "running on empty", "burnt out",
                    "long day", "can barely", "dead tired", "wiped out"],
        "weight": 0.9
    },
    "confused": {
        "words": ["confused", "lost", "unsure", "idk", "huh", "what", "dunno",
                  "bewildered", "baffled"],
        "phrases": ["don't understand", "makes no sense", "what do you mean",
                    "i'm lost", "not sure", "help me understand", "no idea"],
        "weight": 0.8
    },
}


class RoyceMoodDetector:
    """Detects mood from text using pattern matching and context."""

    def __init__(self):
        self.previous_mood = "neutral"
        self.mood_history = []
        self.mood_shift_threshold = 2  # Number of signals needed to confirm a shift

    def detect(self, text: str) -> Tuple[str, float]:
        """
        Analyze text and return (mood, confidence).

        Returns:
            Tuple of (mood_name, confidence_score 0.0-1.0)
        """
        text_lower = text.lower()
        scores: Dict[str, float] = {}

        for mood, signals in MOOD_SIGNALS.items():
            score = 0.0

            # Check individual words
            for word in signals["words"]:
                if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                    score += signals["weight"]

            # Check phrases (weighted higher)
            for phrase in signals["phrases"]:
                if phrase in text_lower:
                    score += signals["weight"] * 1.5

            if score > 0:
                scores[mood] = score

        if not scores:
            return "neutral", 0.3

        # Get the top mood
        top_mood = max(scores, key=scores.get)
        max_score = scores[top_mood]

        # Normalize confidence (cap at 1.0)
        confidence = min(max_score / 5.0, 1.0)

        # Detect mood shifts
        mood_shifted = top_mood != self.previous_mood and confidence > 0.4
        self.previous_mood = top_mood
        self.mood_history.append(top_mood)

        return top_mood, confidence

    def get_mood_shift_comment(self, new_mood: str, old_mood: str) -> Optional[str]:
        """
        Generate a natural comment when Royce notices a mood change.
        Returns None if no comment is warranted.
        """
        if old_mood == new_mood or old_mood == "neutral" or new_mood == "neutral":
            return None

        shifts = {
            ("happy", "sad"): "Hey, you were sounding pretty upbeat before. Everything alright?",
            ("happy", "angry"): "Whoa, something just shifted. What happened?",
            ("sad", "happy"): "That's more like it. Good to hear you sounding better.",
            ("angry", "happy"): "Nice, sounds like the storm passed. Glad you're doing better.",
            ("neutral", "sad"): "Hey, I'm picking up that something's weighing on you. Want to talk about it?",
            ("neutral", "angry"): "Something's got you heated. What's going on?",
            ("neutral", "anxious"): "You seem a little on edge. What's on your mind?",
            ("tired", "excited"): "There's the energy! What happened?",
            ("anxious", "happy"): "Hey, you seem a lot more relaxed now. That's good.",
        }

        comment = shifts.get((old_mood, new_mood))
        if comment:
            return comment

        # Generic shift comments
        if new_mood == "sad":
            return "Hey, I'm noticing you seem a bit down. I'm here if you want to talk about it."
        if new_mood == "angry":
            return "Sounds like something's bothering you. What's up?"
        if new_mood == "anxious":
            return "You seem a bit worried. Want to talk through it?"

        return None

    def has_mood_changed(self) -> bool:
        """Check if the mood recently changed."""
        if len(self.mood_history) < 2:
            return False
        return self.mood_history[-1] != self.mood_history[-2]
