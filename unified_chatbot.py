# unified_chatbot.py - With production-grade fallbacks

import os
import re
import threading
import time
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import signal

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

def safe_translate(text, dest):
    if not text or dest == 'en' or not TRANSLATOR_AVAILABLE: return text
    try:
        # Map some common codes
        if dest == 'hi': target_lang = 'hindi'
        elif dest == 'mr': target_lang = 'marathi'
        else: target_lang = dest
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception:
        return text


@dataclass
class ChatSession:
    session_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    current_disease: str = "Unknown"
    disease_info: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)


class PlantCareChatbot:
    """
    Production-ready chatbot with multiple fallback layers:
    Layer 1: Groq API (primary)
    Layer 2: Local Knowledge Base (backup)
    Layer 3: Smart intent-based responses (last resort)
    """
    
    # Fallback responses for when EVERYTHING fails
    ULTIMATE_FALLBACKS = {
        "treatment": "🌿 For treatment, remove affected leaves first. Apply neem oil (5ml per liter) every 7 days. For severe cases, consult your local agricultural officer.",
        "prevention": "🛡️ Prevention: Water at soil level, maintain plant spacing, remove infected debris, and rotate crops seasonally.",
        "organic": "🌱 Organic options: Neem oil spray, baking soda solution (1 tbsp per gallon), or buttermilk spray (1:10 ratio).",
        "chemical": "🧪 Consult your local agricultural store for recommended fungicides. Always follow label instructions and wear protective gear.",
        "symptoms": "🔍 Common symptoms include leaf spots, yellowing, wilting, powdery growth, or curling leaves.",
        "general": "🌿 I'm here to help with plant diseases. You can ask about treatment, prevention, organic remedies, or chemical options.",
    }
    
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        max_tokens: int = 500,
        timeout_seconds: int = 10,  # Groq timeout
        fallback_to_local: bool = True,  # Enable local KB fallback
        session_timeout: int = 3600,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.fallback_to_local = fallback_to_local
        
        # Initialize Groq (primary)
        self.groq_client = self._init_groq(groq_api_key)
        self.groq_available = self.groq_client is not None
        
        # Initialize local knowledge base (first fallback)
        self.knowledge_base = self._init_knowledge_base()
        
        # Session management
        self.sessions: Dict[str, ChatSession] = {}
        self.session_timeout = session_timeout
        self._lock = threading.Lock()
        
        # Rate limiting
        self.request_log: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))
        
        # MongoDB for persistence (optional)
        self.mongo = self._init_mongodb()
        
        # Statistics for monitoring
        self.stats = {
            "groq_success": 0,
            "groq_failure": 0,
            "local_fallback": 0,
            "ultimate_fallback": 0,
            "total_requests": 0,
        }
        
        print(f"[Chatbot] Initialized - Groq: {'ON' if self.groq_available else 'OFF'}, Local KB: {'ON' if self.knowledge_base else 'OFF'}")
    
    def _init_groq(self, api_key: Optional[str]) -> Optional[Groq]:
        """Initialize Groq client with error handling"""
        if not GROQ_AVAILABLE:
            print("[Chatbot] Groq package not installed")
            return None
        
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            print("[Chatbot] No GROQ_API_KEY found")
            return None
        
        try:
            client = Groq(api_key=key)
            # Test the connection with a quick ping
            print(f"[Chatbot] Groq client ready (model: {self.model})")
            return client
        except Exception as e:
            print(f"[Chatbot] Groq init failed: {e}")
            return None
    
    def _init_knowledge_base(self):
        """Initialize local knowledge base from JSON"""
        try:
            from disease_knowledge import DiseaseKnowledgeBase
            kb = DiseaseKnowledgeBase()
            if kb.data:
                print(f"[Chatbot] Local KB loaded: {len(kb.data)} diseases")
                return kb
        except Exception as e:
            print(f"[Chatbot] Local KB load failed: {e}")
        return None
    
    def _init_mongodb(self):
        """Initialize MongoDB for persistence"""
        if not MONGO_AVAILABLE:
            return None
        
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            return None
        
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            db = client[os.environ.get("MONGODB_DB", "plantcure")]
            return db["chat_sessions"]
        except Exception as e:
            print(f"[Chatbot] MongoDB unavailable: {e}")
            return None
    
    def get_response(
        self,
        user_message: str,
        disease: str = "Unknown",
        disease_info: Optional[Dict] = None,
        session_id: Optional[str] = None,
        language: str = "en",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get response with automatic fallback across multiple layers.
        """
        user_message = (user_message or "").strip()
        session_id = session_id or "default"
        disease = disease or "Unknown"
        
        self.stats["total_requests"] += 1
        
        # Handle empty message
        if not user_message:
            greeting = self._get_greeting(disease)
            if language != 'en':
                greeting = safe_translate(greeting, language)
            return greeting, {
                "source": "greeting",
                "fallback_used": False,
                "disease": disease,
            }
        
        # Check rate limit
        allowed, wait = self._check_rate_limit(session_id)
        if not allowed:
            return f"⏳ Please wait {wait} seconds before asking again.", {
                "source": "rate_limited",
                "fallback_used": False,
            }
        
        # Get or create session
        session = self._get_session(session_id, disease, disease_info)
        
        # ----- LAYER 1: Try Groq API -----
        if self.groq_available:
            response, error = self._try_groq(user_message, session, language)
            if response:
                self.stats["groq_success"] += 1
                self._save_to_session(session, user_message, response, "groq")
                return response, {
                    "source": "groq",
                    "fallback_used": False,
                    "model": self.model,
                    "disease": session.current_disease,
                }
            else:
                self.stats["groq_failure"] += 1
                print(f"[Chatbot] Groq failed: {error}")
        
        # ----- LAYER 2: Try Local Knowledge Base -----
        if self.fallback_to_local and self.knowledge_base is not None:
            response = self._try_local_kb(user_message, session)
            if response:
                if language != 'en':
                    response = safe_translate(response, language)
                self.stats["local_fallback"] += 1
                self._save_to_session(session, user_message, response, "local_kb")
                return response, {
                    "source": "local_kb",
                    "fallback_used": True,
                    "fallback_reason": "Groq unavailable",
                    "disease": session.current_disease,
                }
        
        # ----- LAYER 3: Ultimate Fallback (always works) -----
        self.stats["ultimate_fallback"] += 1
        response = self._get_ultimate_fallback(user_message, session)
        if language != 'en':
            response = safe_translate(response, language)
        self._save_to_session(session, user_message, response, "ultimate_fallback")
        
        return response, {
            "source": "ultimate_fallback",
            "fallback_used": True,
            "fallback_reason": "All primary systems failed",
            "disease": session.current_disease,
        }
    
    def _try_groq(self, message: str, session: ChatSession, language: str) -> Tuple[Optional[str], Optional[str]]:
        """Attempt to get response from Groq API"""
        if not self.groq_client:
            return None, "Groq client not initialized"
        
        # Build conversation history
        history = []
        for msg in session.messages[-10:]:  # Last 5 exchanges
            history.append(f"{msg['role']}: {msg['content']}")
        history_text = "\n".join(history) if history else "No previous conversation."
        
        # Language instruction
        lang_instruction = {
            "en": "Respond in English.",
            "hi": "Respond in Hindi (हिंदी) using Devanagari script.",
            "mr": "Respond in Marathi (मराठी) using Devanagari script.",
        }.get(language, "Respond in English.")
        
        prompt = f"""You are PlantSaathi, an expert plant doctor for Indian farmers.

DISEASE CONTEXT:
- Detected disease: {session.current_disease}
- Available info: {json.dumps(session.disease_info, ensure_ascii=False)[:500]}

CONVERSATION HISTORY:
{history_text}

{lang_instruction}

RULES:
1. Be concise (2-4 sentences for simple questions)
2. Use simple, farmer-friendly language
3. Give specific, actionable advice
4. Include safety warnings for chemicals
5. Use emojis naturally (🌿 💊 🛡️)

USER QUESTION: {message}

YOUR RESPONSE:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful plant disease expert. Keep responses practical and farmer-friendly."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            result = response.choices[0].message.content.strip()
            if result:
                return result, None
            return None, "Empty response from Groq"
                
        except Exception as e:
            return None, str(e)
    
    def _try_local_kb(self, message: str, session: ChatSession) -> Optional[str]:
        """Try to get response from local knowledge base"""
        if self.knowledge_base is None:
            return None
        
        message_lower = message.lower()
        disease = session.current_disease
        
        # Get disease info from KB
        info = self.knowledge_base.get_all_info(disease) if disease != "Unknown" else {}
        
        # Detect intent
        intent = self._detect_intent(message_lower)
        
        # Map intent to KB fields
        intent_map = {
            "treatment": ("treatment", "🌿 Treatment"),
            "prevention": ("prevention", "🛡️ Prevention"),
            "organic": ("organic_remedies", "🌱 Organic remedies"),
            "chemical": ("chemical_remedies", "🧪 Chemical options"),
            "symptoms": ("symptoms", "🔍 Symptoms"),
        }
        
        if intent in intent_map:
            field, emoji = intent_map[intent]
            items = info.get(field, [])
            if items:
                items_text = ", ".join(items[:3])
                return f"{emoji} for {disease}: {items_text}"
        
        # General response from KB
        if info.get("symptoms"):
            symptoms = ", ".join(info["symptoms"][:2])
            return f"🌿 {disease}: Common symptoms include {symptoms}. Ask me about treatment or prevention for more details."
        
        return None
    
    def _get_ultimate_fallback(self, message: str, session: ChatSession) -> str:
        """Last resort - always works, never fails"""
        message_lower = message.lower()
        intent = self._detect_intent(message_lower)
        
        disease = session.current_disease
        disease_note = f" for {disease}" if disease and disease != "Unknown" else ""
        
        if intent == "treatment":
            return self.ULTIMATE_FALLBACKS["treatment"] + f" This applies{disease_note}."
        elif intent == "prevention":
            return self.ULTIMATE_FALLBACKS["prevention"] + f" This applies{disease_note}."
        elif intent == "organic":
            return self.ULTIMATE_FALLBACKS["organic"]
        elif intent == "chemical":
            return self.ULTIMATE_FALLBACKS["chemical"]
        elif intent == "symptoms":
            return self.ULTIMATE_FALLBACKS["symptoms"] + f" For{disease_note}, inspect your plant carefully."
        else:
            return self.ULTIMATE_FALLBACKS["general"]
    
    def _detect_intent(self, text: str) -> str:
        """Detect user intent from message"""
        intents = {
            "treatment": ["treat", "cure", "medicine", "spray", "fix", "control"],
            "prevention": ["prevent", "avoid", "stop", "protection", "future"],
            "organic": ["organic", "natural", "neem", "home remedy", "bio"],
            "chemical": ["chemical", "fungicide", "pesticide", "spray", "medicine"],
            "symptoms": ["symptom", "sign", "look like", "identify", "spot"],
        }
        for intent, keywords in intents.items():
            if any(kw in text for kw in keywords):
                return intent
        return "general"
    
    def _get_greeting(self, disease: str) -> str:
        """Get greeting response"""
        if disease and disease != "Unknown":
            return f"🌿 Namaste! I see your plant has {disease}. Ask me about treatment, prevention, or organic remedies!"
        return "🌿 Namaste! I'm PlantSaathi. Upload a leaf photo or tell me your plant's symptoms, and I'll help diagnose and treat it!"
    
    def _check_rate_limit(self, session_id: str) -> Tuple[bool, int]:
        """Check if rate limit exceeded"""
        now = time.time()
        window = now - 60
        with self._lock:
            entries = self.request_log[session_id]
            while entries and entries[0] < window:
                entries.popleft()
            if len(entries) >= 30:
                wait = max(1, int(60 - (now - entries[0])))
                return False, wait
            entries.append(now)
            return True, 0
    
    def _get_session(self, session_id: str, disease: str, disease_info: Optional[Dict]) -> ChatSession:
        """Get or create session"""
        with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                if disease != "Unknown" and disease != session.current_disease:
                    session.current_disease = disease
                    if disease_info:
                        session.disease_info.update(disease_info)
                session.last_activity = time.time()
                return session
            session = ChatSession(session_id=session_id, current_disease=disease, disease_info=disease_info or {})
            self.sessions[session_id] = session
            return session
    
    def _save_to_session(self, session: ChatSession, user_msg: str, bot_msg: str, source: str):
        """Save interaction to session and optionally MongoDB"""
        session.messages.append({"role": "user", "content": user_msg})
        session.messages.append({"role": "assistant", "content": bot_msg, "source": source})
        session.last_activity = time.time()
        if len(session.messages) > 40:
            session.messages = session.messages[-40:]
        if self.mongo is not None:
            try:
                self.mongo.update_one(
                    {"session_id": session.session_id},
                    {"$set": {
                        "messages": session.messages[-20:],
                        "current_disease": session.current_disease,
                        "last_activity": datetime.now(timezone.utc),
                    }},
                    upsert=True
                )
            except: pass
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a session"""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
        if self.mongo is not None:
            try: self.mongo.delete_one({"session_id": session_id})
            except: pass
        return True
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        session = self.sessions.get(session_id)
        if not session: return {"exists": False}
        return {"exists": True, "session_id": session.session_id}

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {**self.stats, "active_sessions": len(self.sessions)}


# Singleton instance
_chatbot: Optional[PlantCareChatbot] = None


def get_chatbot() -> PlantCareChatbot:
    global _chatbot
    if _chatbot is None:
        _chatbot = PlantCareChatbot()
    return _chatbot


def chat_response(user_message: str, disease: str = "Unknown", disease_info: Optional[Dict] = None, session_id: Optional[str] = None, language: str = "en") -> Tuple[str, Dict[str, Any]]:
    """Main entry point for chat responses"""
    bot = get_chatbot()
    return bot.get_response(user_message, disease, disease_info, session_id, language)


def clear_session(session_id: str) -> bool:
    """Clear a chat session"""
    bot = get_chatbot()
    return bot.clear_session(session_id)


def get_stats() -> Dict[str, Any]:
    """Get chatbot statistics"""
    bot = get_chatbot()
    return bot.get_stats()