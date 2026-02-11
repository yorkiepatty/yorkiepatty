"""
Royce Memory System
-------------------
Remembers EVERYTHING. Always. Forever.
Uses SQLite for rock-solid persistence.
"""

import sqlite3
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from .config import RoyceConfig

logger = logging.getLogger(__name__)


class RoyceMemory:
    """Persistent memory that never forgets.

    Stores:
    - Every conversation exchange
    - User preferences and personal details
    - Learned facts and knowledge
    - Emotional patterns and mood history
    - Important dates and events
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or RoyceConfig.MEMORY_DB
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        logger.info(f"Royce memory initialized: {self.db_path}")

    def _create_tables(self):
        """Set up memory storage tables."""
        c = self.conn.cursor()

        # Conversation history — every exchange, timestamped
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                mood TEXT,
                topic TEXT,
                importance REAL DEFAULT 0.5
            )
        """)

        # User profile — things Royce learns about the person
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                learned_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source TEXT DEFAULT 'conversation'
            )
        """)

        # Knowledge base — facts and information Royce has researched
        c.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                learned_at TEXT NOT NULL,
                relevance REAL DEFAULT 0.5
            )
        """)

        # Mood tracking — emotional patterns over time
        c.execute("""
            CREATE TABLE IF NOT EXISTS mood_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                detected_mood TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                trigger TEXT
            )
        """)

        # Important dates — birthdays, events, etc.
        c.execute("""
            CREATE TABLE IF NOT EXISTS important_dates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                date TEXT NOT NULL,
                recurring INTEGER DEFAULT 0,
                notes TEXT
            )
        """)

        self.conn.commit()

    # ─── Conversation Memory ────────────────────────────────────────

    def store_message(self, role: str, content: str, mood: Optional[str] = None,
                      topic: Optional[str] = None, importance: float = 0.5):
        """Store a conversation message."""
        self.conn.execute(
            "INSERT INTO conversations (timestamp, role, content, mood, topic, importance) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), role, content, mood, topic, importance)
        )
        self.conn.commit()

    def get_recent_conversations(self, limit: int = 50) -> List[Dict]:
        """Get the most recent conversation exchanges."""
        rows = self.conn.execute(
            "SELECT timestamp, role, content, mood, topic FROM conversations ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [
            {"timestamp": r[0], "role": r[1], "content": r[2], "mood": r[3], "topic": r[4]}
            for r in reversed(rows)
        ]

    def search_conversations(self, query: str, limit: int = 20) -> List[Dict]:
        """Search past conversations for relevant content."""
        rows = self.conn.execute(
            "SELECT timestamp, role, content, mood, topic FROM conversations WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
            (f"%{query}%", limit)
        ).fetchall()
        return [
            {"timestamp": r[0], "role": r[1], "content": r[2], "mood": r[3], "topic": r[4]}
            for r in rows
        ]

    def get_conversation_count(self) -> int:
        """How many messages has Royce stored?"""
        row = self.conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
        return row[0] if row else 0

    # ─── User Profile Memory ───────────────────────────────────────

    def learn_about_user(self, key: str, value: str, source: str = "conversation"):
        """Store or update something learned about the user."""
        now = datetime.now().isoformat()
        self.conn.execute(
            """INSERT INTO user_profile (key, value, learned_at, updated_at, source)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value=?, updated_at=?, source=?""",
            (key, value, now, now, source, value, now, source)
        )
        self.conn.commit()
        logger.info(f"Learned about user: {key} = {value}")

    def recall_about_user(self, key: str) -> Optional[str]:
        """Recall something specific about the user."""
        row = self.conn.execute(
            "SELECT value FROM user_profile WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def get_full_user_profile(self) -> Dict[str, str]:
        """Get everything Royce knows about the user."""
        rows = self.conn.execute("SELECT key, value FROM user_profile").fetchall()
        return {r[0]: r[1] for r in rows}

    # ─── Knowledge Memory ──────────────────────────────────────────

    def store_knowledge(self, topic: str, content: str, source: Optional[str] = None, relevance: float = 0.5):
        """Store a piece of knowledge Royce has learned."""
        self.conn.execute(
            "INSERT INTO knowledge (topic, content, source, learned_at, relevance) VALUES (?, ?, ?, ?, ?)",
            (topic, content, source, datetime.now().isoformat(), relevance)
        )
        self.conn.commit()

    def recall_knowledge(self, topic: str, limit: int = 10) -> List[Dict]:
        """Recall knowledge about a topic."""
        rows = self.conn.execute(
            "SELECT topic, content, source, learned_at FROM knowledge WHERE topic LIKE ? ORDER BY relevance DESC LIMIT ?",
            (f"%{topic}%", limit)
        ).fetchall()
        return [{"topic": r[0], "content": r[1], "source": r[2], "learned_at": r[3]} for r in rows]

    # ─── Mood Tracking ─────────────────────────────────────────────

    def log_mood(self, mood: str, confidence: float = 0.5, trigger: Optional[str] = None):
        """Log a detected mood."""
        self.conn.execute(
            "INSERT INTO mood_log (timestamp, detected_mood, confidence, trigger) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), mood, confidence, trigger)
        )
        self.conn.commit()

    def get_mood_history(self, limit: int = 20) -> List[Dict]:
        """Get recent mood history."""
        rows = self.conn.execute(
            "SELECT timestamp, detected_mood, confidence, trigger FROM mood_log ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [
            {"timestamp": r[0], "mood": r[1], "confidence": r[2], "trigger": r[3]}
            for r in reversed(rows)
        ]

    def get_dominant_mood(self, last_n: int = 5) -> Optional[str]:
        """What mood has been most common recently?"""
        rows = self.conn.execute(
            "SELECT detected_mood FROM mood_log ORDER BY id DESC LIMIT ?", (last_n,)
        ).fetchall()
        if not rows:
            return None
        moods = [r[0] for r in rows]
        return max(set(moods), key=moods.count)

    # ─── Important Dates ───────────────────────────────────────────

    def remember_date(self, label: str, date: str, recurring: bool = False, notes: Optional[str] = None):
        """Store an important date (birthday, event, etc.)."""
        self.conn.execute(
            "INSERT INTO important_dates (label, date, recurring, notes) VALUES (?, ?, ?, ?)",
            (label, date, int(recurring), notes)
        )
        self.conn.commit()
        logger.info(f"Remembered date: {label} on {date}")

    def get_upcoming_dates(self, days_ahead: int = 7) -> List[Dict]:
        """Check for any important dates coming up."""
        from datetime import timedelta
        today = datetime.now().date()
        upcoming = []

        rows = self.conn.execute("SELECT label, date, recurring, notes FROM important_dates").fetchall()
        for label, date_str, recurring, notes in rows:
            try:
                event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                if recurring:
                    # Check if this date's month/day falls within the window
                    this_year = event_date.replace(year=today.year)
                    if today <= this_year <= today + timedelta(days=days_ahead):
                        upcoming.append({"label": label, "date": str(this_year), "notes": notes})
                else:
                    if today <= event_date <= today + timedelta(days=days_ahead):
                        upcoming.append({"label": label, "date": date_str, "notes": notes})
            except ValueError:
                continue

        return upcoming

    # ─── Context Building ──────────────────────────────────────────

    def build_context_for_ai(self, current_input: str) -> str:
        """Build a context string from memory to feed into the AI prompt."""
        parts = []

        # User profile
        profile = self.get_full_user_profile()
        if profile:
            profile_str = ", ".join(f"{k}: {v}" for k, v in list(profile.items())[:20])
            parts.append(f"What you know about this person: {profile_str}")

        # Recent conversation
        recent = self.get_recent_conversations(limit=30)
        if recent:
            conv_str = "\n".join(f"{m['role']}: {m['content'][:150]}" for m in recent[-15:])
            parts.append(f"Recent conversation:\n{conv_str}")

        # Relevant past conversations
        relevant = self.search_conversations(current_input, limit=5)
        if relevant:
            rel_str = "\n".join(f"[{r['timestamp'][:10]}] {r['role']}: {r['content'][:100]}" for r in relevant)
            parts.append(f"Related past conversations:\n{rel_str}")

        # Mood history
        dominant_mood = self.get_dominant_mood()
        if dominant_mood:
            parts.append(f"Their recent mood has been: {dominant_mood}")

        # Upcoming dates
        upcoming = self.get_upcoming_dates()
        if upcoming:
            dates_str = ", ".join(f"{d['label']} on {d['date']}" for d in upcoming)
            parts.append(f"Upcoming dates to remember: {dates_str}")

        # Total conversation count
        count = self.get_conversation_count()
        if count > 0:
            parts.append(f"You've had {count} total messages with this person.")

        return "\n\n".join(parts)

    def close(self):
        """Clean shutdown."""
        self.conn.close()
        logger.info("Royce memory saved and closed")
