"""
API Tests for Derek's Learning Engine FastAPI application
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the FastAPI app
from derek_learning_api import app


class TestAPI:
    """Test suite for Derek Learning Engine API"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test the health check endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_learning_status_endpoint(self):
        """Test the learning status endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/learning/status")
            
            assert response.status_code == 200
            data = response.json()
            
            # Check required fields
            assert "learning_active" in data
            assert "total_topics" in data
            assert "learned_topics" in data
            assert "progress" in data
            assert "domain_mastery" in data
            
            # Check data types
            assert isinstance(data["learning_active"], bool)
            assert isinstance(data["total_topics"], int)
            assert isinstance(data["learned_topics"], int)
            assert isinstance(data["progress"], (int, float))
            assert isinstance(data["domain_mastery"], dict)
    
    @pytest.mark.asyncio
    async def test_queue_learning_topic_endpoint_success(self):
        """Test successfully queuing a valid learning topic"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "domain": "autism", 
                "subtopic": "sensory_processing"
            }
            
            response = await client.post("/learning/queue", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "queued"
            assert data["domain"] == "autism"
            assert data["subtopic"] == "sensory_processing"
    
    @pytest.mark.asyncio
    async def test_queue_learning_topic_invalid_domain(self):
        """Test queuing with invalid domain"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "domain": "invalid_domain_that_does_not_exist", 
                "subtopic": "invalid_subtopic"
            }
            
            response = await client.post("/learning/queue", json=payload)
            
            # Should return 400 for invalid domain
            assert response.status_code == 400
            data = response.json()
            assert "Invalid domain" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_queue_learning_topic_invalid_subtopic(self):
        """Test queuing with valid domain but invalid subtopic"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "domain": "autism",  # Valid domain
                "subtopic": "invalid_subtopic_that_does_not_exist"
            }
            
            response = await client.post("/learning/queue", json=payload)
            
            # Should return 400 for invalid subtopic
            assert response.status_code == 400
            data = response.json()
            assert "Invalid subtopic" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_queue_learning_topic_missing_fields(self):
        """Test queuing with missing required fields"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Missing subtopic
            payload = {"domain": "autism"}
            
            response = await client.post("/learning/queue", json=payload)
            
            # Should return 422 for validation error
            assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_queue_learning_topic_empty_payload(self):
        """Test queuing with empty payload"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/learning/queue", json={})
            
            # Should return 422 for validation error
            assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_queue_learning_topic_malformed_json(self):
        """Test queuing with malformed JSON"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/learning/queue", 
                content="invalid json",
                headers={"content-type": "application/json"}
            )
            
            # Should return 422 for JSON decode error
            assert response.status_code == 422


class TestAPIWithMockedEngine:
    """Test API with mocked learning engine for error scenarios"""
    
    @pytest.mark.asyncio
    async def test_learning_status_endpoint_engine_error(self):
        """Test learning status endpoint when engine throws error"""
        with patch('derek_learning_api.engine') as mock_engine:
            mock_engine.get_learning_status.side_effect = Exception("Engine error")
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/learning/status")
                
                assert response.status_code == 500
                data = response.json()
                assert "Engine error" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_queue_topic_engine_error(self):
        """Test queue endpoint when engine throws error"""
        with patch('derek_learning_api.engine') as mock_engine:
            mock_engine.knowledge_domains = {"autism": {"subtopics": ["sensory_processing"]}}
            mock_engine.queue_learning_topic.side_effect = Exception("Queue error")
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                payload = {"domain": "autism", "subtopic": "sensory_processing"}
                response = await client.post("/learning/queue", json=payload)
                
                assert response.status_code == 500
                data = response.json()
                assert "Queue error" in data["detail"]


class TestAPIValidation:
    """Test API input validation"""
    
    @pytest.mark.asyncio
    async def test_valid_domains_and_subtopics(self):
        """Test that all defined domains and subtopics can be queued"""
        # Import engine to get valid domains
        from autonomous_learning_engine import AutonomousLearningEngine
        
        # Create a temporary engine instance to get valid domains
        with patch.dict(os.environ, {'ENABLE_ENCRYPTION': 'false'}):
            temp_engine = AutonomousLearningEngine()
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                # Test a few key domain/subtopic combinations
                test_cases = [
                    ("neurodivergency", "autism_spectrum"),
                    ("autism", "sensory_sensitivities"),
                    ("ai_development", "machine_learning"),
                    ("mathematics", "linear_algebra")
                ]
                
                for domain, subtopic in test_cases:
                    if (domain in temp_engine.knowledge_domains and 
                        subtopic in temp_engine.knowledge_domains[domain]["subtopics"]):
                        
                        payload = {"domain": domain, "subtopic": subtopic}
                        response = await client.post("/learning/queue", json=payload)
                        
                        assert response.status_code == 200, f"Failed for {domain}.{subtopic}"
                        data = response.json()
                        assert data["status"] == "queued"
                        assert data["domain"] == domain
                        assert data["subtopic"] == subtopic


# Pytest configuration for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])