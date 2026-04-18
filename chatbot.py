"""
chatbot.py - AI Chatbot for PlantCure
Handles chat interactions for plant disease treatment advice
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# Try to import Gemini API (optional, for advanced responses)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from chatbot_helper import chatbot_reply, get_chat_response

# Configure Gemini if API key is available
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY and GEMINI_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    gemini_model = None


class PlantChatbot:
    """Chatbot for plant disease queries"""
    
    def __init__(self):
        self.conversation_history = {}
        self.fallback_responses = {
            "greeting": "Hello! I'm your plant care assistant. I can help you with:\n• Disease identification\n• Treatment steps\n• Prevention tips\n• Organic and chemical remedies\n\nWhat would you like to know about your plant?",
            "thanks": "You're welcome! Happy gardening! 🌿 Is there anything else I can help you with?",
            "unknown": "I'm not sure about that. Could you please ask about:\n• Treatment for a specific disease\n• Prevention tips\n• Symptoms to look for\n• Organic or chemical remedies",
            "help": "I can help you with:\n• Treatment steps\n• Prevention methods\n• Organic remedies (neem oil, etc.)\n• Chemical options (fungicides, etc.)\n• Disease symptoms\n\nJust ask me anything about your plant's condition!"
        }
    
    def get_response(self, user_message: str, disease: str = "Unknown") -> Tuple[str, str, Dict]:
        """
        Get response from chatbot
        
        Args:
            user_message: User's question or message
            disease: Current detected disease
        
        Returns:
            Tuple of (response_text, source, metadata)
        """
        user_message = user_message.strip()
        
        # Check for simple commands
        if not user_message:
            return self.fallback_responses["help"], "local_kb", {}
        
        # Handle greetings
        if self._is_greeting(user_message):
            return self.fallback_responses["greeting"], "local_kb", {"disease": disease}
        
        # Handle thanks
        if self._is_thanks(user_message):
            return self.fallback_responses["thanks"], "local_kb", {}
        
        # Handle help
        if self._is_help_request(user_message):
            return self.fallback_responses["help"], "local_kb", {}
        
        # Try to get response from knowledge base first
        response, source, meta = chatbot_reply(user_message, disease)
        
        # If response is too generic or from fallback, try Gemini
        if self._needs_better_response(response, source) and gemini_model:
            gemini_response = self._get_gemini_response(user_message, disease)
            if gemini_response:
                return gemini_response, "gemini", meta
        
        return response, source, meta
    
    def _is_greeting(self, message: str) -> bool:
        """Check if message is a greeting"""
        greetings = ["hi", "hello", "hey", "namaste", "good morning", "good evening", "greetings"]
        return message.lower().strip() in greetings
    
    def _is_thanks(self, message: str) -> bool:
        """Check if message is a thank you"""
        thanks = ["thanks", "thank you", "thx", "appreciate it", "helpful"]
        return any(word in message.lower() for word in thanks)
    
    def _is_help_request(self, message: str) -> bool:
        """Check if user is asking for help"""
        help_terms = ["help", "what can you do", "how to use", "capabilities", "features"]
        return any(term in message.lower() for term in help_terms) and len(message.split()) < 5
    
    def _needs_better_response(self, response: str, source: str) -> bool:
        """Check if response needs improvement from Gemini"""
        if source == "gemini":
            return False
        
        generic_responses = [
            "I am specialized in plant disease diagnosis",
            "No treatment data available",
            "symptoms not available",
            "Unknown"
        ]
        
        return any(generic in response for generic in generic_responses)
    
    def _get_gemini_response(self, user_message: str, disease: str) -> Optional[str]:
        """Get response from Gemini API"""
        if not gemini_model:
            return None
        
        try:
            prompt = f"""You are a plant disease expert assistant. The detected disease is {disease}. 
            User question: {user_message}
            
            Provide a helpful, accurate response about plant disease treatment, prevention, or symptoms.
            Keep it concise (2-3 sentences) and practical.
            If you don't know, suggest consulting a local plant expert.
            """
            
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini error: {e}")
            return None


# Singleton instance
_chatbot = None

def get_chatbot() -> PlantChatbot:
    """Get chatbot singleton instance"""
    global _chatbot
    if _chatbot is None:
        _chatbot = PlantChatbot()
    return _chatbot


def chat_response(user_message: str, disease: str = "Unknown") -> Tuple[str, str, Dict]:
    """
    Get chat response (compatible with existing code)
    
    Args:
        user_message: User's question
        disease: Current detected disease
    
    Returns:
        Tuple of (response_text, source, metadata)
    """
    chatbot = get_chatbot()
    return chatbot.get_response(user_message, disease)


# Alternative simple function for direct import
def get_bot_reply(question: str, disease: str = "Unknown") -> str:
    """Simple function to get bot reply as string"""
    response, source, meta = chat_response(question, disease)
    return response


# Pre-defined responses for common questions
QUICK_RESPONSES = {
    "treatment": {
        "keywords": ["treat", "treatment", "cure", "fix", "medicine", "spray"],
        "template": "Here's how to treat {disease}:\n{steps}"
    },
    "prevention": {
        "keywords": ["prevent", "prevention", "avoid", "stop", "protect"],
        "template": "To prevent {disease}:\n{tips}"
    },
    "organic": {
        "keywords": ["organic", "natural", "home remedy", "neem", "eco-friendly"],
        "template": "Organic options for {disease}:\n{remedies}"
    },
    "chemical": {
        "keywords": ["chemical", "fungicide", "pesticide", "commercial", "bought"],
        "template": "Chemical options for {disease}:\n{options}"
    },
    "symptoms": {
        "keywords": ["symptom", "sign", "look", "appear", "identify"],
        "template": "Common symptoms of {disease}:\n{symptoms}"
    }
}


def quick_reply(question: str, disease: str, disease_info: Dict) -> Optional[str]:
    """
    Get quick reply based on keywords
    
    Args:
        question: User's question
        disease: Disease name
        disease_info: Dictionary with disease information
    
    Returns:
        Quick response string or None
    """
    question_lower = question.lower()
    
    for intent, config in QUICK_RESPONSES.items():
        if any(keyword in question_lower for keyword in config["keywords"]):
            template = config["template"]
            
            if intent == "treatment":
                steps = disease_info.get("treatment_steps", "Not available")
                return template.format(disease=disease, steps=steps)
            elif intent == "prevention":
                tips = disease_info.get("prevention_tips", "Not available")
                return template.format(disease=disease, tips=tips)
            elif intent == "organic":
                remedies = disease_info.get("organic_remedies", "Not available")
                return template.format(disease=disease, remedies=remedies)
            elif intent == "chemical":
                options = disease_info.get("chemical_remedies", "Not available")
                return template.format(disease=disease, options=options)
            elif intent == "symptoms":
                symptoms = disease_info.get("symptoms", [])
                if isinstance(symptoms, list):
                    symptoms = "\n".join(f"• {s}" for s in symptoms)
                return template.format(disease=disease, symptoms=symptoms)
    
    return None


# Export main functions
__all__ = [
    'chatbot_reply',
    'get_chat_response', 
    'chat_response',
    'get_bot_reply',
    'quick_reply',
    'PlantChatbot'
]