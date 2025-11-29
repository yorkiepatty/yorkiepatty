from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Use the database module we created
try:
    from database import Database as DatabaseManager
except ImportError:
    # Fallback if database module not available
    class DatabaseManager:
        def connect(self): pass
        def sync(self): pass
        def close(self): pass

# Example usage:
# database = DatabaseManager()
# database.connect()  # In start()
# database.sync()     # Periodic sync
# database.close()    # In stop()


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
