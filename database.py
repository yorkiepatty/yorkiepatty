"""
Database Module
---------------
Database abstraction layer for Derek's persistent storage.
"""

import os
import json
import sqlite3
import logging
from typing import Any, List, Dict, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database")


class Database:
    """SQLite database wrapper with JSON support"""
    
    def __init__(self, db_path: str = "./memory/derek.db"):
        """Initialize database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self.connect()
        logger.info(f"Database initialized at {db_path}")
    
    def connect(self):
        """Establish database connection"""
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        logger.debug("Database connection established")
    
    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a database query"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            self.connection.commit()
            return cursor
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            self.connection.rollback()
            raise
    
    def fetch_all(self, query: str, params: tuple = ()) -> List[Dict]:
        """Fetch all results from query"""
        cursor = self.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def fetch_one(self, query: str, params: tuple = ()) -> Optional[Dict]:
        """Fetch single result from query"""
        cursor = self.execute(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def create_tables(self):
        """Create standard Derek database tables"""
        logger.info("Creating database tables...")
        
        # Memory table
        self.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                content TEXT NOT NULL,
                type TEXT,
                metadata TEXT
            )
        """)
        
        # User interactions table
        self.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                input TEXT,
                response TEXT,
                emotion TEXT,
                metadata TEXT
            )
        """)
        
        # Learning progress table
        self.execute("""
            CREATE TABLE IF NOT EXISTS learning_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                skill TEXT,
                progress REAL,
                notes TEXT
            )
        """)
        
        logger.info("Database tables created")
    
    def store_memory(self, content: str, memory_type: str = "general", metadata: dict = None):
        """Store a memory entry"""
        metadata_json = json.dumps(metadata) if metadata else None
        self.execute(
            "INSERT INTO memories (content, type, metadata) VALUES (?, ?, ?)",
            (content, memory_type, metadata_json)
        )
        logger.debug(f"Memory stored: {memory_type}")
    
    def store_interaction(self, user_id: str, user_input: str, response: str, 
                         emotion: str = None, metadata: dict = None):
        """Store a user interaction"""
        metadata_json = json.dumps(metadata) if metadata else None
        self.execute(
            "INSERT INTO interactions (user_id, input, response, emotion, metadata) VALUES (?, ?, ?, ?, ?)",
            (user_id, user_input, response, emotion, metadata_json)
        )
        logger.debug(f"Interaction stored for user: {user_id}")
    
    def get_recent_memories(self, limit: int = 10, memory_type: str = None) -> List[Dict]:
        """Retrieve recent memories"""
        if memory_type:
            return self.fetch_all(
                "SELECT * FROM memories WHERE type = ? ORDER BY timestamp DESC LIMIT ?",
                (memory_type, limit)
            )
        return self.fetch_all(
            "SELECT * FROM memories ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
    
    def get_user_interactions(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve user interaction history"""
        return self.fetch_all(
            "SELECT * FROM interactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit)
        )
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


# Global database instance
_db = None

def get_db() -> Database:
    """Get or create global database instance"""
    global _db
    if _db is None:
        _db = Database()
        _db.create_tables()
    return _db


# Flask-SQLAlchemy compatibility stub
class db:
    """Flask-SQLAlchemy compatibility layer"""
    Model = object
    Column = lambda *args, **kwargs: None
    Integer = "INTEGER"
    String = "TEXT"
    DateTime = "DATETIME"
    Text = "TEXT"
    ForeignKey = lambda x: x
    relationship = lambda *args, **kwargs: None
    
    @staticmethod
    def create_all():
        """Create all tables"""
        get_db().create_tables()


__all__ = ['Database', 'get_db', 'db']

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
