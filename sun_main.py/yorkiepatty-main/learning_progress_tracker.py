"""
Learning Progress Tracker for Yorkie
Tracks learning sessions, progress, and provides analytics
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class LearningProgressTracker:
    def __init__(self, progress_file: str = "yorkie_progress.json"):
        self.progress_file = progress_file
        self.progress_data = self._load_progress()
        
    def _load_progress(self) -> Dict:
        """Load existing progress data"""
        try:
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_initial_progress()
    
    def _create_initial_progress(self) -> Dict:
        """Create initial progress structure"""
        return {
            "student": "Yorkie",
            "started_date": datetime.now().isoformat(),
            "total_sessions": 0,
            "total_learning_time": 0,
            "current_streak": 0,
            "longest_streak": 0,
            "modules": {},
            "sessions": [],
            "achievements": [],
            "confidence_levels": [],
            "last_updated": datetime.now().isoformat()
        }
    
    def start_session(self) -> str:
        """Start a new learning session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_data = {
            "id": session_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "topics_covered": [],
            "questions_asked": 0,
            "concepts_learned": [],
            "exercises_completed": 0,
            "errors_encountered": 0,
            "errors_resolved": 0,
            "confidence_before": None,
            "confidence_after": None,
            "notes": []
        }
        
        self.progress_data["sessions"].append(session_data)
        self.progress_data["total_sessions"] += 1
        self._save_progress()
        
        logger.info(f"Started learning session: {session_id}")
        return session_id
    
    def end_session(self, session_id: str, confidence_level: int = None) -> Dict:
        """End a learning session and calculate statistics"""
        session = self._find_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        session["end_time"] = datetime.now().isoformat()
        
        # Calculate session duration
        start_time = datetime.fromisoformat(session["start_time"])
        end_time = datetime.fromisoformat(session["end_time"])
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        session["duration_minutes"] = duration_minutes
        self.progress_data["total_learning_time"] += duration_minutes
        
        # Update confidence level if provided
        if confidence_level:
            session["confidence_after"] = confidence_level
            self.progress_data["confidence_levels"].append({
                "date": datetime.now().isoformat(),
                "level": confidence_level,
                "session_id": session_id
            })
        
        # Update streak
        self._update_streak()
        
        # Check for achievements
        new_achievements = self._check_achievements()
        
        self._save_progress()
        
        return {
            "duration_minutes": duration_minutes,
            "topics_covered": len(session["topics_covered"]),
            "exercises_completed": session["exercises_completed"],
            "new_achievements": new_achievements,
            "total_sessions": self.progress_data["total_sessions"],
            "current_streak": self.progress_data["current_streak"]
        }
    
    def log_concept_learned(self, session_id: str, concept: str) -> None:
        """Log that a concept was learned in this session"""
        session = self._find_session(session_id)
        if session and concept not in session["concepts_learned"]:
            session["concepts_learned"].append({
                "concept": concept,
                "timestamp": datetime.now().isoformat()
            })
            self._save_progress()
    
    def log_question_asked(self, session_id: str, question: str) -> None:
        """Log a question asked during the session"""
        session = self._find_session(session_id)
        if session:
            session["questions_asked"] += 1
            session["notes"].append({
                "type": "question",
                "content": question,
                "timestamp": datetime.now().isoformat()
            })
            self._save_progress()
    
    def log_exercise_completed(self, session_id: str, exercise: str, success: bool) -> None:
        """Log completion of a coding exercise"""
        session = self._find_session(session_id)
        if session:
            if success:
                session["exercises_completed"] += 1
            session["notes"].append({
                "type": "exercise",
                "content": exercise,
                "success": success,
                "timestamp": datetime.now().isoformat()
            })
            self._save_progress()
    
    def log_error_encountered(self, session_id: str, error_type: str, resolved: bool = False) -> None:
        """Log an error encountered and whether it was resolved"""
        session = self._find_session(session_id)
        if session:
            session["errors_encountered"] += 1
            if resolved:
                session["errors_resolved"] += 1
            session["notes"].append({
                "type": "error",
                "error_type": error_type,
                "resolved": resolved,
                "timestamp": datetime.now().isoformat()
            })
            self._save_progress()
    
    def get_progress_summary(self) -> Dict:
        """Get a summary of learning progress"""
        recent_sessions = self._get_recent_sessions(days=7)
        
        return {
            "total_sessions": self.progress_data["total_sessions"],
            "total_hours": round(self.progress_data["total_learning_time"] / 60, 1),
            "current_streak": self.progress_data["current_streak"],
            "longest_streak": self.progress_data["longest_streak"],
            "recent_activity": len(recent_sessions),
            "achievements_earned": len(self.progress_data["achievements"]),
            "current_confidence": self._get_latest_confidence(),
            "learning_velocity": self._calculate_learning_velocity(),
            "favorite_topics": self._get_favorite_topics(),
            "next_milestone": self._get_next_milestone()
        }
    
    def get_encouraging_stats(self) -> str:
        """Get encouraging statistics for motivation"""
        stats = self.get_progress_summary()
        
        messages = []
        
        if stats["total_sessions"] > 0:
            messages.append(f"🎯 You've completed {stats['total_sessions']} learning sessions!")
        
        if stats["total_hours"] > 1:
            messages.append(f"⏰ That's {stats['total_hours']} hours of dedicated learning!")
        
        if stats["current_streak"] > 1:
            messages.append(f"🔥 You're on a {stats['current_streak']}-day learning streak!")
        
        if stats["achievements_earned"] > 0:
            messages.append(f"🏆 You've earned {stats['achievements_earned']} achievements!")
        
        if not messages:
            messages.append("🌟 Every expert was once a beginner - you're off to a great start!")
        
        return " ".join(messages)
    
    def _find_session(self, session_id: str) -> Optional[Dict]:
        """Find a session by ID"""
        for session in self.progress_data["sessions"]:
            if session["id"] == session_id:
                return session
        return None
    
    def _update_streak(self) -> None:
        """Update learning streak based on recent activity"""
        today = datetime.now().date()
        streak = 0
        
        # Get unique session dates in reverse order
        session_dates = []
        for session in self.progress_data["sessions"]:
            if session["end_time"]:  # Only completed sessions
                session_date = datetime.fromisoformat(session["end_time"]).date()
                if session_date not in session_dates:
                    session_dates.append(session_date)
        
        session_dates.sort(reverse=True)
        
        # Calculate current streak
        for i, date in enumerate(session_dates):
            expected_date = today - timedelta(days=i)
            if date == expected_date:
                streak += 1
            else:
                break
        
        self.progress_data["current_streak"] = streak
        if streak > self.progress_data["longest_streak"]:
            self.progress_data["longest_streak"] = streak
    
    def _check_achievements(self) -> List[str]:
        """Check for new achievements"""
        new_achievements = []
        existing_achievements = [a["title"] for a in self.progress_data["achievements"]]
        
        # Define achievement criteria
        achievements = {
            "First Steps": {"sessions": 1, "description": "Completed your first coding session!"},
            "Getting Started": {"sessions": 5, "description": "Completed 5 learning sessions!"},
            "Dedicated Learner": {"sessions": 10, "description": "Completed 10 learning sessions!"},
            "Consistent Coder": {"streak": 3, "description": "3-day learning streak!"},
            "Code Warrior": {"streak": 7, "description": "1-week learning streak!"},
            "Hour of Code": {"hours": 1, "description": "Completed 1 hour of learning!"},
            "Marathon Learner": {"hours": 10, "description": "Completed 10 hours of learning!"},
            "Problem Solver": {"exercises": 5, "description": "Completed 5 coding exercises!"}
        }
        
        for title, criteria in achievements.items():
            if title not in existing_achievements:
                earned = False
                
                if "sessions" in criteria and self.progress_data["total_sessions"] >= criteria["sessions"]:
                    earned = True
                elif "streak" in criteria and self.progress_data["current_streak"] >= criteria["streak"]:
                    earned = True
                elif "hours" in criteria and (self.progress_data["total_learning_time"] / 60) >= criteria["hours"]:
                    earned = True
                elif "exercises" in criteria:
                    total_exercises = sum(s.get("exercises_completed", 0) for s in self.progress_data["sessions"])
                    if total_exercises >= criteria["exercises"]:
                        earned = True
                
                if earned:
                    achievement = {
                        "title": title,
                        "description": criteria["description"],
                        "date_earned": datetime.now().isoformat()
                    }
                    self.progress_data["achievements"].append(achievement)
                    new_achievements.append(title)
        
        return new_achievements
    
    def _get_recent_sessions(self, days: int = 7) -> List[Dict]:
        """Get sessions from the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_sessions = []
        
        for session in self.progress_data["sessions"]:
            if session["end_time"]:
                session_date = datetime.fromisoformat(session["end_time"])
                if session_date >= cutoff_date:
                    recent_sessions.append(session)
        
        return recent_sessions
    
    def _get_latest_confidence(self) -> Optional[int]:
        """Get the most recent confidence level"""
        if self.progress_data["confidence_levels"]:
            return self.progress_data["confidence_levels"][-1]["level"]
        return None
    
    def _calculate_learning_velocity(self) -> str:
        """Calculate how quickly concepts are being learned"""
        recent_sessions = self._get_recent_sessions(days=14)
        if not recent_sessions:
            return "Just getting started!"
        
        total_concepts = sum(len(s.get("concepts_learned", [])) for s in recent_sessions)
        concepts_per_session = total_concepts / len(recent_sessions) if recent_sessions else 0
        
        if concepts_per_session >= 3:
            return "Rapid learner! 🚀"
        elif concepts_per_session >= 2:
            return "Steady progress! 📈"
        elif concepts_per_session >= 1:
            return "Building knowledge! 🧱"
        else:
            return "Taking it step by step! 👣"
    
    def _get_favorite_topics(self) -> List[str]:
        """Get the most frequently covered topics"""
        topic_counts = {}
        for session in self.progress_data["sessions"]:
            for topic in session.get("topics_covered", []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Return top 3 topics
        return sorted(topic_counts.keys(), key=lambda x: topic_counts[x], reverse=True)[:3]
    
    def _get_next_milestone(self) -> str:
        """Suggest the next milestone to work toward"""
        sessions = self.progress_data["total_sessions"]
        streak = self.progress_data["current_streak"]
        hours = self.progress_data["total_learning_time"] / 60
        
        if sessions < 5:
            return f"Complete {5 - sessions} more sessions to earn 'Getting Started'"
        elif streak < 3:
            return "Keep learning daily to start a 3-day streak!"
        elif hours < 10:
            return f"Learn for {10 - hours:.1f} more hours to earn 'Marathon Learner'"
        else:
            return "You're crushing it! Keep up the amazing work!"
    
    def _save_progress(self) -> None:
        """Save progress data to file"""
        self.progress_data["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

# Singleton instance for import
progress_tracker = LearningProgressTracker()

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================