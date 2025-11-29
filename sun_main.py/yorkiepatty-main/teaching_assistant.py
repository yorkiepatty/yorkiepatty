"""
Teaching Assistant Module for Sonny
Specialized responses and behaviors for coding education
Focused on helping Yorkie learn programming concepts
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class TeachingAssistant:
    def __init__(self, curriculum_path: str = "yorkie_curriculum.json"):
        self.curriculum_path = curriculum_path
        self.curriculum = self._load_curriculum()
        self.session_start = datetime.now()
        self.teaching_mode = "encouraging"  # modes: encouraging, explanatory, challenge, review
        
    def _load_curriculum(self) -> Dict:
        """Load Yorkie's learning curriculum"""
        try:
            with open(self.curriculum_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Curriculum file not found: {self.curriculum_path}")
            return self._create_default_curriculum()
    
    def _create_default_curriculum(self) -> Dict:
        """Create a basic curriculum if file doesn't exist"""
        return {
            "curriculum": {
                "student": "Yorkie",
                "current_level": "beginner",
                "current_module": "foundations"
            },
            "teaching_principles": {
                "encouragement_first": True,
                "learn_by_doing": True,
                "mistake_friendly": True
            }
        }
    
    def detect_learning_intent(self, message: str) -> Optional[str]:
        """Detect if the user wants to learn something specific"""
        learning_keywords = {
            'learn': 'wants_to_learn',
            'teach': 'wants_to_learn', 
            'show me': 'wants_example',
            'how do': 'needs_explanation',
            'what is': 'needs_definition',
            'explain': 'needs_explanation',
            'help with': 'needs_help',
            'stuck': 'needs_help',
            'error': 'needs_debugging',
            'bug': 'needs_debugging',
            'practice': 'wants_practice',
            'exercise': 'wants_practice'
        }
        
        message_lower = message.lower()
        for keyword, intent in learning_keywords.items():
            if keyword in message_lower:
                return intent
        return None
    
    def get_teaching_response(self, message: str, intent: str) -> str:
        """Generate an appropriate teaching response based on intent"""
        responses = {
            'wants_to_learn': self._generate_learning_response,
            'wants_example': self._generate_example_response,
            'needs_explanation': self._generate_explanation_response,
            'needs_definition': self._generate_definition_response,
            'needs_help': self._generate_help_response,
            'needs_debugging': self._generate_debugging_response,
            'wants_practice': self._generate_practice_response
        }
        
        if intent in responses:
            return responses[intent](message)
        else:
            return self._generate_encouraging_response(message)
    
    def _generate_learning_response(self, message: str) -> str:
        """Response for when someone wants to learn something"""
        current_module = self.curriculum.get("curriculum", {}).get("current_module", "foundations")
        
        responses = [
            f"Good. You want to learn {current_module}? Let's do this right. No shortcuts, no excuses. We're building excellence here.",
            "Finally. Someone who wants to actually learn instead of just talk about it. What specifically? I don't have time for vague questions.",
            "Smart move. Learning to code separates the builders from the talkers. Ready to work hard, or are you here to waste my time?",
            "About time. Coding isn't a hobby - it's a superpower. I'm going to push you harder than you've ever been pushed. Ready?"
        ]
        
        import random
        return random.choice(responses)
    
    def _generate_example_response(self, message: str) -> str:
        """Response when someone wants to see an example"""
        return """I'd love to show you an example! Let me break it down step by step so you can see how it works. 

Here's a simple example to start with:

```python
# This is a comment - it explains what the code does
name = "Yorkie"  # This stores your name in a variable
print(f"Hello, {name}!")  # This displays a greeting
```

See how each line has a purpose? Would you like me to explain any part of this?"""
    
    def _generate_explanation_response(self, message: str) -> str:
        """Response for explaining concepts"""
        return """Great question! I believe the best way to understand something is to see it in action. Let me explain this concept in simple terms first, then we'll try it together.

Think of programming like giving very specific instructions to a computer. The computer is really smart but needs everything explained in exact detail.

Would you like me to show you a hands-on example of this concept?"""
    
    def _generate_definition_response(self, message: str) -> str:
        """Response for defining terms"""
        return """Let me explain that term in a way that makes sense! I like to think of programming concepts like everyday things you already know.

For example, a 'variable' is like a labeled box where you can store something for later use. You can put different things in the box, and you can always check what's inside by looking at the label.

Does that help make it clearer? I can show you exactly how this works in code!"""
    
    def _generate_help_response(self, message: str) -> str:
        """Response when someone needs help"""
        return """Stuck? Good. That means you're finally hitting something challenging instead of cruising through easy stuff.

Here's how we fix this - and I mean FIX it, not band-aid it:

1. Show me EXACTLY what you tried - not what you think you tried
2. What error message? Copy it. Don't paraphrase it.
3. What did you expect vs what actually happened?

Stop saying 'it doesn't work' - that tells me nothing. Be precise. Be specific. Let's solve this properly."""
    
    def _generate_debugging_response(self, message: str) -> str:
        """Response for debugging help"""
        return """Error messages aren't your enemy - ignoring them is. Stop panicking and start reading.

My debugging protocol - follow it exactly:
1. **READ THE ERROR** - Every. Single. Word. It's telling you exactly what's wrong.
2. **Find the line number** - Go there. Look at what you actually wrote.
3. **Think before you code** - Random changes make it worse, not better.
4. **Test one thing at a time** - Shotgun debugging is for amateurs.

Show me the EXACT error message. Not your interpretation. Not a summary. The actual error."""
    
    def _generate_practice_response(self, message: str) -> str:
        """Response when someone wants practice"""
        return """Practice makes progress! 🎯 I love that you want to practice - that's how you build confidence and skill.

Here are some practice ideas based on where you are:

**Beginner Practice:**
- Modify existing code to do something slightly different
- Write a program that asks for your name and age, then tells you something fun
- Create a simple calculator for basic math

**Ready for More:**
- Build a number guessing game
- Create a program that helps organize a to-do list
- Make a simple quiz about something you're interested in

What sounds fun to you? Or do you have your own idea you'd like to try?"""
    
    def _generate_encouraging_response(self, message: str) -> str:
        """General sharp but motivating response"""
        sharp_phrases = [
            "Good question. Shows you're actually thinking instead of just copying code blindly.",
            "That's the kind of curiosity that separates real developers from script kiddies. Keep pushing.",
            "Every expert started exactly where you are. The difference? They didn't quit when it got hard.",
            "Smart to ask instead of assuming. Assumptions are where code goes to die.",
            "You're asking the right questions. Now let's get the right answers and build something that matters."
        ]
        
        import random
        return random.choice(sharp_phrases)
    
    def track_progress(self, topic: str, success: bool) -> None:
        """Track learning progress for analytics"""
        progress_entry = {
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "success": success,
            "session_duration": (datetime.now() - self.session_start).total_seconds()
        }
        
        # Log progress for future analysis
        logger.info(f"Learning progress: {progress_entry}")
    
    def get_next_lesson_suggestion(self) -> str:
        """Suggest what to learn next based on current progress"""
        current_module = self.curriculum.get("curriculum", {}).get("current_module", "foundations")
        
        suggestions = {
            "foundations": "Ready to try your first Python program? Let's write some code together!",
            "python_basics": "You're doing great with the basics! Want to learn about making decisions in code with if/else?",
            "functions": "Functions are like recipes for code. Should we learn to write our own?",
            "data": "Ready to work with real data? Let's build something that saves information!",
            "web": "Time to make your code visible to the world! Want to learn web development?",
            "apis": "Let's connect your programs to other services and make them even more powerful!",
            "databases": "Ready to store lots of information efficiently? Databases are our next adventure!",
            "project": "Time for the big finale - let's build something amazing that showcases all you've learned!"
        }
        
        return suggestions.get(current_module, "Let's keep learning together! What interests you most?")

# Singleton instance for import
teaching_assistant = TeachingAssistant()

def get_teaching_response(message: str) -> Tuple[str, Optional[str]]:
    """Main function to get teaching-focused responses"""
    intent = teaching_assistant.detect_learning_intent(message)
    
    if intent:
        response = teaching_assistant.get_teaching_response(message, intent)
        return response, intent
    else:
        # Not a learning-specific query, return None to use regular conversation
        return "", None

def get_encouragement() -> str:
    """Get a random encouraging message"""
    return teaching_assistant._generate_encouraging_response("")

def get_next_lesson() -> str:
    """Get suggestion for next learning topic"""
    return teaching_assistant.get_next_lesson_suggestion()

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================