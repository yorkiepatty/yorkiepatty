"""
Data models for Derek's Learning Engine API
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List
from datetime import datetime


class LearningTopic(BaseModel):
    """Model for learning topic requests with validation"""
    domain: str = Field(..., description="Knowledge domain (e.g., 'neurodivergency', 'ai_development')")
    subtopic: str = Field(..., description="Specific subtopic to learn")
    priority: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Learning priority (0.0-1.0)")

    @validator("domain")
    def validate_domain(cls, value):
        """Validate that the domain exists in the knowledge domains"""
        # Import here to avoid circular imports
        import os
        from pathlib import Path
        
        # We'll validate this in the API endpoint instead to avoid 
        # creating engine instances during validation
        if not value or len(value.strip()) == 0:
            raise ValueError("Domain cannot be empty")
        return value.strip().lower()

    @validator("subtopic")
    def validate_subtopic(cls, value, values):
        """Validate that the subtopic is not empty"""
        if not value or len(value.strip()) == 0:
            raise ValueError("Subtopic cannot be empty")
        return value.strip().lower()

    class Config:
        """Pydantic config"""
        schema_extra = {
            "example": {
                "domain": "neurodivergency",
                "subtopic": "autism_spectrum",
                "priority": 1.0
            }
        }


class LearningStatus(BaseModel):
    """Model for learning status response"""
    learning_active: bool
    current_topic: Optional[Dict]
    total_topics: int
    learned_topics: int
    progress: float = Field(..., ge=0.0, le=1.0)
    domain_mastery: Dict[str, float]
    generated_modules: int
    improvements_made: int


class KnowledgeEntry(BaseModel):
    """Model for stored knowledge entries"""
    domain: str
    subtopic: str
    content: str
    key_concepts: List[str]
    practical_applications: List[str]
    learned_at: datetime
    confidence: float = Field(..., ge=0.0, le=1.0)
    mastery: float = Field(..., ge=0.0, le=1.0)


class HealthStatus(BaseModel):
    """Health check response model"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    engine_active: bool = True
    memory_available: bool = True


class QueueResponse(BaseModel):
    """Response for queue operations"""
    status: str
    domain: str
    subtopic: str
    queued_at: datetime = Field(default_factory=datetime.now)