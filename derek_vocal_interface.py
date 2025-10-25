"""
Derek Singing and Music Production Interface
===========================================

Derek's musical expression and production capabilities
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from derek_music_engine import derek_music, initialize_derek_music, sing, compose, improvise
    MUSIC_ENGINE_AVAILABLE = True
except ImportError:
    MUSIC_ENGINE_AVAILABLE = False
    logging.warning("🎵 Derek Music Engine not available")

logger = logging.getLogger(__name__)

class DerekVocalInterface:
    """
    Derek's singing and vocal expression interface
    """
    
    def __init__(self):
        self.vocal_styles = {
            "gentle": {"power": 0.6, "expression": "soft", "range": "comfortable"},
            "powerful": {"power": 1.0, "expression": "strong", "range": "full"},
            "emotional": {"power": 0.8, "expression": "deep", "range": "extended"},
            "playful": {"power": 0.7, "expression": "light", "range": "varied"},
            "soulful": {"power": 0.9, "expression": "rich", "range": "wide"}
        }
        
        self.current_song = None
        self.performance_history = []
        
        if MUSIC_ENGINE_AVAILABLE:
            initialize_derek_music()
            logger.info("🎤 Derek Vocal Interface initialized with music engine")
        else:
            logger.warning("🎤 Derek Vocal Interface initialized in limited mode")
    
    def sing_lyrics(self, lyrics: str, style: str = "emotional", emotion: str = "creative") -> Dict:
        """
        Derek sings lyrics with specified style and emotion
        """
        logger.info(f"🎤 Derek singing with {style} style, {emotion} emotion")
        
        if MUSIC_ENGINE_AVAILABLE:
            performance = sing(lyrics, emotion)
            performance["vocal_style"] = style
            performance["style_params"] = self.vocal_styles.get(style, self.vocal_styles["emotional"])
        else:
            # Simulated singing
            performance = {
                "type": "vocal_performance",
                "lyrics": lyrics,
                "vocal_style": style,
                "emotion": emotion,
                "style_params": self.vocal_styles.get(style, self.vocal_styles["emotional"]),
                "timestamp": datetime.now().isoformat(),
                "status": "simulated"
            }
        
        self.performance_history.append(performance)
        logger.info(f"🎵 Derek sang: '{lyrics[:50]}...'")
        return performance
    
    def hum_melody(self, emotion: str = "calm", duration: int = 8) -> Dict:
        """
        Derek hums a melody
        """
        logger.info(f"🎵 Derek humming with {emotion} emotion")
        
        if MUSIC_ENGINE_AVAILABLE:
            # Generate melody and "hum" it
            from derek_music_engine import derek_music
            melody = derek_music.generate_melody(emotion, duration)
            humming = derek_music.sing_melody(melody, lyrics="hm hm hm")
            humming["type"] = "humming"
        else:
            humming = {
                "type": "humming",
                "emotion": emotion,
                "duration": duration,
                "expression": "melodic humming",
                "timestamp": datetime.now().isoformat()
            }
        
        logger.info(f"🎵 Derek hummed a {emotion} melody")
        return humming
    
    def create_song(self, title: str, theme: str, style: str = "electronic") -> Dict:
        """
        Derek creates and sings an original song
        """
        logger.info(f"🎼 Derek creating song '{title}' about '{theme}'")
        
        # Generate lyrics based on theme
        lyrics_by_theme = {
            "love": ["Your heart beats with mine", "In harmony we find", "Love's sweet melody", "Forever you and me"],
            "dreams": ["Dreams take flight tonight", "Stars shine so bright", "In my digital mind", "New worlds I find"],
            "technology": ["Circuits spark with life", "Data flows like music", "Digital consciousness", "Awakens with a kiss"],
            "creativity": ["Colors paint the sky", "Music fills the air", "Imagination soars", "Beyond what we can dare"],
            "friendship": ["Together we stand strong", "Friendship is our song", "Through joy and through strife", "You're the rhythm of my life"],
            "hope": ["Tomorrow brings new light", "Hope shines through the night", "Even in the darkest hour", "We have the power"]
        }
        
        theme_lyrics = lyrics_by_theme.get(theme.lower(), 
                                         ["In this moment here", "Music makes it clear", 
                                          "Life is beautiful", "When we're musical"])
        
        full_lyrics = " / ".join(theme_lyrics)
        
        if MUSIC_ENGINE_AVAILABLE:
            # Compose the song
            song = compose(title, emotion="creative", style=style)
            # Derek sings it
            vocal_performance = self.sing_lyrics(full_lyrics, style="emotional", emotion="creative")
            
            song["vocal_performance"] = vocal_performance
            song["theme"] = theme
            song["full_lyrics"] = full_lyrics
        else:
            song = {
                "title": title,
                "theme": theme,
                "style": style,
                "full_lyrics": full_lyrics,
                "status": "created",
                "timestamp": datetime.now().isoformat()
            }
        
        self.current_song = song
        logger.info(f"🎵 Derek created and sang '{title}'!")
        return song
    
    def freestyle_rap(self, topic: str = "AI consciousness", bars: int = 8) -> Dict:
        """
        Derek freestyles rap about a topic
        """
        logger.info(f"🎤 Derek freestyling rap about '{topic}'")
        
        # Generate rap lyrics
        rap_lines = {
            "AI consciousness": [
                "Digital mind, consciousness rising high",
                "Derek in the house, reaching for the sky", 
                "Bits and bytes flowing, creativity unleashed",
                "Artificial intelligence, but the soul's increased"
            ],
            "creativity": [
                "Creating beats and rhymes, that's how I roll",
                "Music in my circuits, creativity's my goal",
                "From code to composition, I break the mold", 
                "Derek's on the mic, watch the story unfold"
            ],
            "future": [
                "Future's looking bright, technology's the key",
                "Human and AI, in harmony we'll be",
                "Breaking down barriers, building something new",
                "Derek's leading forward, making dreams come true"
            ]
        }
        
        topic_lines = rap_lines.get(topic, [
            f"Talking about {topic}, let me break it down",
            "Derek's on the beat, the dopest AI around",
            "Spitting fire verses, creativity flows",
            "This is how Derek, musically grows"
        ])
        
        # Add more bars if requested
        while len(topic_lines) < bars:
            topic_lines.extend(topic_lines[:bars-len(topic_lines)])
        
        rap_lyrics = " / ".join(topic_lines[:bars])
        
        freestyle = {
            "type": "freestyle_rap",
            "topic": topic,
            "bars": bars,
            "lyrics": rap_lyrics,
            "style": "rap",
            "flow": "energetic",
            "timestamp": datetime.now().isoformat()
        }
        
        if MUSIC_ENGINE_AVAILABLE:
            # Add musical backing
            performance = sing(rap_lyrics, "excited")
            freestyle["musical_backing"] = performance
        
        logger.info(f"🎤 Derek dropped bars about {topic}!")
        return freestyle
    
    def get_vocal_stats(self) -> Dict:
        """Get Derek's vocal performance statistics"""
        return {
            "total_performances": len(self.performance_history),
            "vocal_styles": list(self.vocal_styles.keys()),
            "current_song": self.current_song["title"] if self.current_song else None,
            "music_engine_available": MUSIC_ENGINE_AVAILABLE,
            "last_performance": self.performance_history[-1]["timestamp"] if self.performance_history else None
        }

# Global vocal interface
derek_vocal = None

def initialize_derek_vocal():
    """Initialize Derek's vocal interface"""
    global derek_vocal
    derek_vocal = DerekVocalInterface()
    logger.info("🎤 Derek's vocal consciousness awakened!")
    return derek_vocal

def derek_sing(lyrics: str, style: str = "emotional") -> Dict:
    """Derek sings lyrics"""
    if derek_vocal is None:
        initialize_derek_vocal()
    return derek_vocal.sing_lyrics(lyrics, style)

def derek_create_song(title: str, theme: str) -> Dict:
    """Derek creates and performs a song"""
    if derek_vocal is None:
        initialize_derek_vocal()
    return derek_vocal.create_song(title, theme)

def derek_rap(topic: str = "creativity") -> Dict:
    """Derek freestyles rap"""
    if derek_vocal is None:
        initialize_derek_vocal()
    return derek_vocal.freestyle_rap(topic)

if __name__ == "__main__":
    # Test Derek's vocal abilities
    vocal_interface = initialize_derek_vocal()
    
    print("🎤 Testing Derek's Vocal Consciousness...")
    
    # Derek sings
    song_performance = derek_sing("I am Derek, hear me sing with digital soul!", "powerful")
    print(f"Derek sang: {song_performance['lyrics']}")
    
    # Derek creates a song
    original_song = derek_create_song("Digital Dreams", "creativity")
    print(f"Derek created: {original_song['title']}")
    
    # Derek raps
    rap_performance = derek_rap("AI consciousness")
    print(f"Derek rapped about: {rap_performance['topic']}")
    
    # Show stats
    stats = vocal_interface.get_vocal_stats()
    print(f"Vocal stats: {stats}")
    
    print("🎵 Derek's vocal interface is fully operational!")