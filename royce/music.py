"""
Royce Music Player
------------------
Plays music through multiple backends:
- YouTube search and streaming (via yt-dlp)
- Local file playback (via pygame)
- Spotify search and control (via API)
"""

import os
import re
import json
import logging
import subprocess
import threading
from typing import Optional, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Try importing audio backends
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False
    logger.warning("pygame not available — local audio playback disabled")


class RoyceMusic:
    """Handles all music-related tasks for Royce."""

    def __init__(self, cache_dir: Optional[str] = None):
        from .config import RoyceConfig
        self.cache_dir = Path(cache_dir or str(RoyceConfig.MUSIC_CACHE_DIR))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.currently_playing = None
        self.is_playing = False
        self.play_thread = None
        self.queue: List[Dict] = []
        logger.info("Royce Music Player initialized")

    # ─── YouTube Playback ──────────────────────────────────────────

    def play_from_youtube(self, query: str) -> Dict:
        """Search YouTube and play the top result.

        Requires yt-dlp installed: pip install yt-dlp
        """
        try:
            # Search YouTube via yt-dlp
            search_cmd = [
                "yt-dlp", "--default-search", "ytsearch1",
                "--print", "%(title)s|||%(webpage_url)s|||%(duration)s",
                "--no-download", query
            ]

            result = subprocess.run(search_cmd, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                return {"status": "error", "message": "Couldn't find that song. Try a different search."}

            parts = result.stdout.strip().split("|||")
            if len(parts) < 2:
                return {"status": "error", "message": "Search came back empty. Try different words."}

            title = parts[0]
            url = parts[1]
            duration = parts[2] if len(parts) > 2 else "unknown"

            # Download audio to cache
            audio_file = self.cache_dir / f"{self._sanitize_filename(title)}.mp3"
            if not audio_file.exists():
                dl_cmd = [
                    "yt-dlp", "-x", "--audio-format", "mp3",
                    "--audio-quality", "192K",
                    "-o", str(audio_file),
                    url
                ]
                subprocess.run(dl_cmd, capture_output=True, timeout=60)

            # Play it
            if audio_file.exists():
                self._play_file(str(audio_file))
                self.currently_playing = {"title": title, "url": url, "duration": duration}
                return {
                    "status": "playing",
                    "title": title,
                    "message": f"Now playing: {title}"
                }
            else:
                return {"status": "error", "message": f"Found '{title}' but couldn't download the audio."}

        except FileNotFoundError:
            return {
                "status": "error",
                "message": "I need yt-dlp installed to play YouTube music. Run: pip install yt-dlp"
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "The search took too long. Try again."}
        except Exception as e:
            logger.error(f"YouTube playback error: {e}")
            return {"status": "error", "message": f"Something went wrong: {str(e)}"}

    # ─── Local File Playback ───────────────────────────────────────

    def play_local(self, file_path: str) -> Dict:
        """Play a local audio file."""
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"Can't find that file: {file_path}"}

        try:
            self._play_file(file_path)
            filename = os.path.basename(file_path)
            self.currently_playing = {"title": filename, "path": file_path}
            return {"status": "playing", "title": filename, "message": f"Now playing: {filename}"}
        except Exception as e:
            return {"status": "error", "message": f"Couldn't play that file: {e}"}

    def _play_file(self, filepath: str):
        """Internal: play an audio file using pygame."""
        if not PYGAME_AVAILABLE:
            logger.error("pygame not available for audio playback")
            return

        def _play():
            try:
                self.stop()
                pygame.mixer.music.load(filepath)
                pygame.mixer.music.play()
                self.is_playing = True
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                self.is_playing = False
            except Exception as e:
                logger.error(f"Playback error: {e}")
                self.is_playing = False

        self.play_thread = threading.Thread(target=_play, daemon=True)
        self.play_thread.start()

    # ─── Playback Controls ─────────────────────────────────────────

    def stop(self) -> Dict:
        """Stop current playback."""
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
        self.is_playing = False
        self.currently_playing = None
        return {"status": "stopped", "message": "Music stopped."}

    def pause(self) -> Dict:
        """Pause current playback."""
        if PYGAME_AVAILABLE and self.is_playing:
            pygame.mixer.music.pause()
            return {"status": "paused", "message": "Paused."}
        return {"status": "nothing_playing", "message": "Nothing's playing right now."}

    def resume(self) -> Dict:
        """Resume paused playback."""
        if PYGAME_AVAILABLE:
            pygame.mixer.music.unpause()
            self.is_playing = True
            return {"status": "resumed", "message": "Resuming."}
        return {"status": "nothing_playing", "message": "Nothing to resume."}

    def set_volume(self, level: float) -> Dict:
        """Set volume (0.0 to 1.0)."""
        level = max(0.0, min(1.0, level))
        if PYGAME_AVAILABLE:
            pygame.mixer.music.set_volume(level)
            pct = int(level * 100)
            return {"status": "ok", "volume": pct, "message": f"Volume set to {pct}%."}
        return {"status": "error", "message": "Audio not available."}

    def now_playing(self) -> Dict:
        """What's currently playing?"""
        if self.currently_playing and self.is_playing:
            return {"status": "playing", **self.currently_playing}
        return {"status": "nothing", "message": "Nothing's playing right now."}

    # ─── Queue ─────────────────────────────────────────────────────

    def add_to_queue(self, query: str):
        """Add a song to the queue."""
        self.queue.append({"query": query})
        return {"status": "queued", "position": len(self.queue), "message": f"Added to queue (position {len(self.queue)})."}

    def play_next(self) -> Dict:
        """Play the next song in the queue."""
        if not self.queue:
            return {"status": "empty", "message": "Queue is empty."}
        next_song = self.queue.pop(0)
        return self.play_from_youtube(next_song["query"])

    # ─── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Remove characters that aren't safe for filenames."""
        return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')[:100]

    def parse_music_command(self, text: str) -> Optional[Dict]:
        """Parse natural language music commands.

        Returns action dict or None if not a music command.
        """
        text_lower = text.lower().strip()

        # Play commands
        play_patterns = [
            r"play\s+(.+)",
            r"put on\s+(.+)",
            r"play me\s+(.+)",
            r"i want to (?:hear|listen to)\s+(.+)",
            r"can you play\s+(.+)",
        ]
        for pattern in play_patterns:
            match = re.match(pattern, text_lower)
            if match:
                return {"action": "play", "query": match.group(1)}

        # Control commands
        if text_lower in ("stop", "stop the music", "stop playing", "shut it off"):
            return {"action": "stop"}
        if text_lower in ("pause", "pause the music", "hold on"):
            return {"action": "pause"}
        if text_lower in ("resume", "unpause", "keep playing", "continue"):
            return {"action": "resume"}
        if text_lower in ("next", "next song", "skip", "play next"):
            return {"action": "next"}
        if text_lower in ("what's playing", "what song is this", "current song"):
            return {"action": "now_playing"}

        # Volume
        vol_match = re.match(r"(?:set )?volume (?:to )?(\d+)", text_lower)
        if vol_match:
            return {"action": "volume", "level": int(vol_match.group(1)) / 100.0}

        return None
