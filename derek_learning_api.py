from fastapi import FastAPI, HTTPException
from autonomous_learning_engine import AutonomousLearningEngine
from models import LearningTopic
from utils.logging import setup_logging
import logging

app = FastAPI(title="Derek's Learning Engine API")

# Setup HIPAA-compliant logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize learning engine
engine = AutonomousLearningEngine()

@app.get("/health")
async def health_check():
    """Health check endpoint for ECS"""
    return {"status": "healthy"}

@app.get("/learning/status")
async def get_learning_status():
    """Get current learning status and progress"""
    try:
        status = engine.get_learning_status()
        logger.info("Fetched learning status")
        return status
    except Exception as e:
        logger.error(f"Error fetching status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learning/queue")
async def queue_learning_topic(topic: LearningTopic):
    """Queue a specific topic for learning"""
    try:
        if topic.domain not in engine.knowledge_domains:
            raise ValueError(f"Invalid domain: {topic.domain}")
        if topic.subtopic not in engine.knowledge_domains[topic.domain]["subtopics"]:
            raise ValueError(f"Invalid subtopic: {topic.subtopic}")
        engine.queue_learning_topic(topic.domain, topic.subtopic)
        logger.info(f"Queued topic: {topic.domain} - {topic.subtopic}")
        return {"status": "queued", "domain": topic.domain, "subtopic": topic.subtopic}
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error queuing topic: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)