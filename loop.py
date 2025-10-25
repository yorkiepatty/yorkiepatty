"""
Event loop handler for Derek Dashboard
Manages continuous processes and background tasks
"""

import asyncio
import logging
from typing import Optional, List, Callable
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("loop.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EventLoop:  # Changed from LearningLoop
    def __init__(self, knowledge_engine=None):
        self.engine = knowledge_engine
        self.running = False
        self.loop = None
        self.tasks = []
    
    def learning_loop(self):
        """Run learning loop if engine is available"""
        if self.engine and self.engine.derek.memory is None:
            logger.error("Cannot start learning loop: memory not initialized")
            raise RuntimeError("Memory system not initialized")
        try:
            if self.engine:
                self.engine._store_knowledge("test_topic", "Test knowledge")
                logger.info("Learning loop running")
        except Exception as e:
            logger.error(f"Learning loop error: {e}")
            raise

    async def heartbeat(self):
        """System heartbeat - runs every 30 seconds"""
        while self.running:
            try:
                logger.debug(f"💓 Heartbeat at {datetime.now().isoformat()}")
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def memory_consolidation(self):
        """Periodic memory consolidation - runs every 5 minutes"""
        while self.running:
            try:
                logger.debug("🧠 Running memory consolidation...")
                await asyncio.sleep(300)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory consolidation error: {e}")

    async def analytics_processing(self):
        """Process analytics - runs every 10 minutes"""
        while self.running:
            try:
                logger.debug("📊 Processing analytics...")
                await asyncio.sleep(600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Analytics processing error: {e}")

    def register_task(self, coro: Callable):
        """Register a custom async task"""
        if self.loop and self.running:
            task = self.loop.create_task(coro())
            self.tasks.append(task)
            logger.info(f"Registered new task: {coro.__name__}")

    def start(self):
        """Start the event loop"""
        if self.running:
            logger.warning("Event loop already running")
            return

        logger.info("Starting event loop...")
        self.running = True

        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            self.tasks = [
                self.loop.create_task(self.heartbeat()),
                self.loop.create_task(self.memory_consolidation()),
                self.loop.create_task(self.analytics_processing()),
            ]

            logger.info("✓ Event loop started with background tasks")
            self.loop.run_forever()

        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the event loop"""
        if not self.running:
            return

        logger.info("Stopping event loop...")
        self.running = False

        for task in self.tasks:
            task.cancel()

        if self.loop:
            self.loop.stop()
            self.loop.close()

        logger.info("✓ Event loop stopped")


# Singleton instance
event_loop = EventLoop()

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# ==============================================================================
