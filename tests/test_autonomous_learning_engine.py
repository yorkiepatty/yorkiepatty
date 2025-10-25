import unittest
import os
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from autonomous_learning_engine import AutonomousLearningEngine


class TestAutonomousLearningEngine(unittest.TestCase):
    """Test suite for AutonomousLearningEngine"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Use a temporary test directory
        self.test_dir = Path("test_derek_knowledge")
        self.test_dir.mkdir(exist_ok=True)
        
        # Mock environment variables for testing
        with patch.dict(os.environ, {
            'ENABLE_ENCRYPTION': 'false',  # Disable encryption for tests
            'LOG_LEVEL': 'ERROR'  # Reduce logging noise
        }):
            self.engine = AutonomousLearningEngine(knowledge_dir=str(self.test_dir))
    
    def tearDown(self):
        """Clean up after each test"""
        # Clean up test files
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test engine initializes properly"""
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.knowledge_domains)
        self.assertFalse(self.engine.learning_active)
        self.assertEqual(len(self.engine.knowledge_domains), 9)
    
    def test_knowledge_domains_structure(self):
        """Test knowledge domains have correct structure"""
        for domain, info in self.engine.knowledge_domains.items():
            self.assertIn('subtopics', info)
            self.assertIn('priority', info)
            self.assertIn('mastery_level', info)
            self.assertIsInstance(info['subtopics'], list)
            self.assertGreaterEqual(info['priority'], 0)
            self.assertLessEqual(info['priority'], 1)
    
    @patch('autonomous_learning_engine.Derek')
    def test_learn_topic_with_mock_derek(self, mock_derek_class):
        """Test learning a topic with mocked Derek"""
        # Mock Derek instance
        mock_derek = MagicMock()
        mock_derek.anthropic_client = None
        mock_derek.openai_client = None
        mock_derek.perplexity = None
        mock_derek_class.return_value = mock_derek
        
        # Create engine with mocked Derek
        engine = AutonomousLearningEngine(knowledge_dir=str(self.test_dir))
        
        topic = {"domain": "autism", "subtopic": "sensory_processing"}
        knowledge = engine._learn_topic(topic)
        
        # Check knowledge structure
        self.assertIn("content", knowledge)
        self.assertIn("key_concepts", knowledge)
        self.assertIn("practical_applications", knowledge)
        self.assertEqual(knowledge["domain"], "autism")
        self.assertEqual(knowledge["subtopic"], "sensory_processing")
    
    def test_queue_learning_topic(self):
        """Test queueing a learning topic"""
        initial_queue_size = self.engine.learning_queue.qsize()
        
        self.engine.queue_learning_topic("autism", "sensory_processing")
        
        self.assertEqual(self.engine.learning_queue.qsize(), initial_queue_size + 1)
        
        # Get the queued topic
        queued_topic = self.engine.learning_queue.get()
        self.assertEqual(queued_topic["domain"], "autism")
        self.assertEqual(queued_topic["subtopic"], "sensory_processing")
    
    def test_invalid_domain_validation(self):
        """Test that invalid domains are handled properly"""
        # This should work without raising an error since we don't validate in queue_learning_topic
        self.engine.queue_learning_topic("invalid_domain", "invalid_subtopic")
        
        # But the validation would happen during learning
        topic = {"domain": "invalid_domain", "subtopic": "invalid_subtopic"}
        knowledge = self.engine._learn_topic(topic)
        
        # Should still create knowledge structure even with invalid domain
        self.assertIn("domain", knowledge)
        self.assertEqual(knowledge["domain"], "invalid_domain")
    
    def test_get_next_learning_topic(self):
        """Test getting next learning topic"""
        # Queue a specific topic
        self.engine.queue_learning_topic("neurodivergency", "autism_spectrum")
        
        next_topic = self.engine._get_next_learning_topic()
        
        self.assertIsNotNone(next_topic)
        self.assertEqual(next_topic["domain"], "neurodivergency")
        self.assertEqual(next_topic["subtopic"], "autism_spectrum")
    
    def test_learning_status(self):
        """Test getting learning status"""
        status = self.engine.get_learning_status()
        
        self.assertIn("learning_active", status)
        self.assertIn("total_topics", status)
        self.assertIn("learned_topics", status)
        self.assertIn("progress", status)
        self.assertIn("domain_mastery", status)
        
        self.assertIsInstance(status["learning_active"], bool)
        self.assertIsInstance(status["total_topics"], int)
        self.assertIsInstance(status["progress"], float)
        self.assertIsInstance(status["domain_mastery"], dict)
    
    def test_learning_stats_alias(self):
        """Test that get_learning_stats is an alias for get_learning_status"""
        status = self.engine.get_learning_status()
        stats = self.engine.get_learning_stats()
        
        self.assertEqual(status, stats)
    
    def test_save_and_load_knowledge_base(self):
        """Test saving and loading knowledge base"""
        # Add some test knowledge
        test_knowledge = {
            "domain": "test_domain",
            "subtopic": "test_subtopic",
            "content": "test content",
            "key_concepts": ["concept1", "concept2"],
            "practical_applications": ["app1", "app2"],
            "confidence": 0.8,
            "mastery": 0.6
        }
        
        topic_key = "test_domain.test_subtopic"
        self.engine.knowledge_base[topic_key] = test_knowledge
        
        # Save knowledge base
        self.engine.save_knowledge_base()
        
        # Clear knowledge base and reload
        self.engine.knowledge_base = {}
        self.engine.load_knowledge_base()
        
        # Check if knowledge was loaded
        self.assertIn(topic_key, self.engine.knowledge_base)
        loaded_knowledge = self.engine.knowledge_base[topic_key]
        self.assertEqual(loaded_knowledge["domain"], "test_domain")
        self.assertEqual(loaded_knowledge["content"], "test content")
    
    def test_extract_key_concepts(self):
        """Test extracting key concepts from content"""
        content = """
        1. First concept here
        2. Second important concept
        - Third bullet point concept
        * Fourth starred concept
        **Fifth bold concept**
        """
        
        concepts = self.engine._extract_key_concepts(content)
        
        self.assertIsInstance(concepts, list)
        self.assertGreater(len(concepts), 0)
        self.assertLessEqual(len(concepts), 10)  # Max 10 concepts
    
    def test_extract_applications(self):
        """Test extracting practical applications"""
        content = """
        This content discusses applications of autism support.
        The main application is communication enhancement.
        We can implement better strategies for users.
        Another practical use case is sensory accommodation.
        """
        
        applications = self.engine._extract_applications("autism", "support", content)
        
        self.assertIsInstance(applications, list)
        # Should find lines with application keywords
        self.assertGreater(len([app for app in applications if app]), 0)
    
    def test_update_mastery(self):
        """Test updating domain mastery levels"""
        # Add some knowledge with different mastery levels
        self.engine.knowledge_base["autism.test1"] = {"mastery": 0.8}
        self.engine.knowledge_base["autism.test2"] = {"mastery": 0.6}
        
        initial_mastery = self.engine.knowledge_domains["autism"]["mastery_level"]
        
        self.engine._update_mastery("autism")
        
        updated_mastery = self.engine.knowledge_domains["autism"]["mastery_level"]
        
        # Mastery should be updated based on learned topics
        # (It calculates average mastery / total subtopics)
        self.assertGreaterEqual(updated_mastery, 0)
    
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_learning_loop_start_stop(self, mock_sleep):
        """Test starting and stopping learning loop"""
        self.assertFalse(self.engine.learning_active)
        
        # Start learning
        self.engine.start_autonomous_learning()
        self.assertTrue(self.engine.learning_active)
        
        # Stop learning
        self.engine.stop_autonomous_learning()
        self.assertFalse(self.engine.learning_active)
    
    def test_generate_learning_curriculum(self):
        """Test curriculum generation"""
        curriculum = self.engine._generate_learning_curriculum()
        
        self.assertIsInstance(curriculum, list)
        self.assertGreater(len(curriculum), 0)
        
        # Check curriculum structure
        for item in curriculum:
            self.assertIn("domain", item)
            self.assertIn("subtopic", item)
            self.assertIn("priority", item)
    
    def test_research_with_local_fallback(self):
        """Test local fallback research method"""
        prompt = "Learn about autism communication strategies"
        
        result = self.engine._research_with_local_fallback(prompt)
        
        self.assertIn("content", result)
        self.assertIn("confidence", result)
        self.assertIsInstance(result["content"], str)
        self.assertGreater(len(result["content"]), 0)
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 1)


class TestEncryptionIntegration(unittest.TestCase):
    """Test encryption functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = Path("test_derek_knowledge_encrypted")
        self.test_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up after test"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @patch.dict(os.environ, {'ENABLE_ENCRYPTION': 'true', 'ENCRYPTION_KEY': ''})
    def test_encrypted_save_load(self):
        """Test saving and loading with encryption enabled"""
        # This test requires cryptography to be installed
        try:
            from cryptography.fernet import Fernet
            
            # Generate test key
            test_key = Fernet.generate_key().decode()
            
            with patch.dict(os.environ, {'ENCRYPTION_KEY': test_key}):
                engine = AutonomousLearningEngine(knowledge_dir=str(self.test_dir))
                
                # Add test knowledge
                test_knowledge = {"test": "encrypted_data"}
                engine.knowledge_base["test.encrypted"] = test_knowledge
                
                # Save (should be encrypted)
                engine.save_knowledge_base()
                
                # Check that encrypted files exist
                encrypted_file = self.test_dir / "knowledge_base.enc"
                self.assertTrue(encrypted_file.exists())
                
                # Clear and reload
                engine.knowledge_base = {}
                engine.load_knowledge_base()
                
                # Check data was loaded correctly
                self.assertIn("test.encrypted", engine.knowledge_base)
                
        except ImportError:
            self.skipTest("Cryptography not available for encryption test")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)