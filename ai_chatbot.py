"""
AI chatbot for PlantCure.
Uses Groq when available and falls back to the local disease knowledge base.
"""

import os
import re
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

from disease_knowledge import DiseaseKnowledgeBase

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

try:
    from chatbot import Chat
    CHATBOT_LIBRARY_AVAILABLE = True
except ImportError:
    CHATBOT_LIBRARY_AVAILABLE = False


class TTLCache:
    def __init__(self, ttl_seconds: int = 3600, max_items: int = 512):
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._cache: Dict[Any, Tuple[float, Any]] = {}
        self._lock = threading.Lock()
        # ❌ REMOVED the incorrect line: self.groq_model_name = groq_model_name or "llama-3.3-70b-versatile"

    def get(self, key: Any) -> Any:
        with self._lock:
            item = self._cache.get(key)
            if not item:
                return None
            expires_at, value = item
            if expires_at <= time.time():
                self._cache.pop(key, None)
                return None
            return value

    def set(self, key: Any, value: Any) -> None:
        with self._lock:
            if len(self._cache) >= self.max_items:
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key, None)
            self._cache[key] = (time.time() + self.ttl_seconds, value)


class PlantChatbot:
    """Context-aware plant chatbot with Groq and local KB fallback."""

    def __init__(
        self,
        knowledge_file: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        groq_model_name: str = "llama-3.3-70b-versatile",  # ✅ CHANGED from "llama3-70b-8192"
        temperature: float = 0.7,
        max_output_tokens: int = 300,
        cache_ttl_seconds: int = 3600,
        rate_limit_per_minute: int = 60,
    ):
        self.knowledge_base = DiseaseKnowledgeBase(knowledge_file)
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.groq_model_name = groq_model_name  # ✅ This will now use the updated model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.cache = TTLCache(ttl_seconds=cache_ttl_seconds)
        self.rate_limit_per_minute = rate_limit_per_minute
        self.session_history: Dict[str, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=8))
        self.request_log: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self.mongo_collection = self._init_mongo_collection()
        self.groq_client = self._init_groq_client()
        self.engine = self._init_chatbot_engine()

    def _init_chatbot_engine(self):
        if not CHATBOT_LIBRARY_AVAILABLE:
            print("[Chatbot] ChatBotAI library not found. Falling back to legacy regex matching.")
            return None
        
        template_path = os.path.join(os.path.dirname(__file__), "models", "chat_templates", "plant_care.template")
        if not os.path.exists(template_path):
            print(f"[Chatbot] Template file not found at {template_path}")
            return None
            
        try:
            # Initialize with the template
            engine = Chat(template_path)
            print(f"[Chatbot] ChatBotAI engine initialized with template: {os.path.basename(template_path)}")
            return engine
        except Exception as exc:
            print(f"[Chatbot] ChatBotAI initialization failed: {exc}")
            return None

    def _init_groq_client(self):
        if not GROQ_AVAILABLE:
            print("[Chatbot] Groq package not found.")
            return None
        if not self.groq_api_key:
            print("[Chatbot] GROQ_API_KEY environment variable not set.")
            return None
        try:
            client = Groq(api_key=self.groq_api_key)
            print(f"[Chatbot] Groq client initialized with model: {self.groq_model_name}")
            return client
        except Exception as exc:
            print(f"[Chatbot] Groq initialization failed: {exc}")
            return None

    def _init_mongo_collection(self):
        mongo_uri = os.getenv("MONGODB_URI")
        mongo_db = os.getenv("MONGODB_DB", "plantcure")
        mongo_collection = os.getenv("MONGODB_CHAT_COLLECTION", "chatbot_interactions")
        if not (MongoClient and mongo_uri):
            return None
        try:
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=1500)
            collection = client[mongo_db][mongo_collection]
            collection.create_index("session_id")
            collection.create_index("timestamp")
            return collection
        except Exception as exc:
            print(f"[Chatbot] MongoDB unavailable: {exc}")
            return None

    def get_response(
        self,
        user_message: str,
        disease: str = "Unknown",
        session_id: Optional[str] = None,
        confidence_score: Optional[float] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        user_message = (user_message or "").strip()
        disease = (disease or "Unknown").strip()
        resolved_disease = self._resolve_disease_name(disease)
        session_id = session_id or "default"

        if not user_message:
            response = "🌿 Ask me about symptoms, treatment, prevention, or remedies for your plant."
            return response, "local_kb", self._build_meta(resolved_disease, "help", confidence_score, session_id, False)

        allowed, wait_seconds = self._check_rate_limit(session_id)
        if not allowed:
            response = f"⏳ Please wait about {wait_seconds} seconds, then ask again. I handle up to 60 requests per minute per user."
            return response, "local_kb", self._build_meta(resolved_disease, "rate_limited", confidence_score, session_id, False)

        intent = self._detect_intent(user_message)
        cache_key = (session_id, resolved_disease.lower(), intent, user_message.lower())
        cached = self.cache.get(cache_key)
        if cached:
            response, source, meta = cached
            cached_meta = dict(meta)
            cached_meta["cached"] = True
            self._append_history(session_id, user_message, response, resolved_disease)
            return response, source, cached_meta

        kb_info = self.knowledge_base.get_all_info(resolved_disease)
        meta = self._build_meta(resolved_disease, intent, confidence_score, session_id, False)
        meta["knowledge"] = kb_info

        response_text = None
        source = "local_kb"

        # 1. Try ChatBotAI Engine (Template Matching with Context)
        if self.engine:
            try:
                # Sync context attributes to the engine
                self.engine.attr(
                    disease=resolved_disease,
                    symptoms=", ".join(kb_info.get("symptoms", [])[:3]) or "No specific symptoms listed",
                    treatment=", ".join(kb_info.get("treatment", [])[:2]) or "Contact agricultural officer",
                    organic=", ".join(kb_info.get("organic_remedies", [])[:2]) or "Neem oil and bio-pesticides",
                    chemical=", ".join(kb_info.get("chemical_remedies", [])[:2]) or "Consult local KVK for fungicides",
                    prevention=", ".join(kb_info.get("prevention", [])[:2]) or "Field sanitation and crop rotation",
                    watering=", ".join(self.knowledge_base.data.get("PlantCare", {}).get("watering", [])[:2]) or "Water at soil level"
                )
                
                engine_response = self.engine.respond(user_message)
                # If the engine found a specialized match (not the catch-all)
                if engine_response and not engine_response.startswith("I'm not exactly sure"):
                    response_text = engine_response
                    source = "chatbot_engine"
            except Exception as e:
                print(f"[Chatbot] Engine response failed: {e}")

        # 2. Try Groq (AI Fallback for complex queries)
        if not response_text and self._should_use_groq(intent):
            response_text = self._get_groq_response(user_message, resolved_disease, kb_info, session_id, intent)
            if response_text:
                source = "groq"

        # 3. Last Resort: Local KB fallback
        if not response_text:
            response_text = self._get_local_response(user_message, resolved_disease, kb_info, intent)
            source = "local_kb"

        response_text = self._format_response(response_text, resolved_disease, intent)
        self._append_history(session_id, user_message, response_text, resolved_disease)
        self._persist_interaction(
            session_id=session_id,
            user_message=user_message,
            bot_response=response_text,
            detected_disease=resolved_disease,
            response_source=source,
            confidence_score=confidence_score,
        )
        self.cache.set(cache_key, (response_text, source, meta))
        return response_text, source, meta

    def clear_session_history(self, session_id: str) -> None:
        with self._lock:
            if session_id in self.session_history:
                self.session_history.pop(session_id)
            if session_id in self.request_log:
                self.request_log.pop(session_id)

    def _resolve_disease_name(self, disease: str) -> str:
        label = (disease or "Unknown").strip()
        lowered = label.lower()
        
        # Prefer specific match if it exists in data
        if label in self.knowledge_base.data:
            return label
            
        if lowered in {"unknown", "uncertain", "invalid"}:
            return "Unknown"
        
        # If it's a generic "healthy" check, but no specific key found
        if "healthy" in lowered and "healthy" not in self.knowledge_base.data.get(label, {}):
             # Only return generic if no specific key exists
             if label not in self.knowledge_base.data:
                 return "Healthy"
        cleaned = label.replace("___", " ").replace("__", " ").replace("_", " ").strip()
        return cleaned or "Unknown"

    def _build_meta(
        self,
        disease: str,
        intent: str,
        confidence_score: Optional[float],
        session_id: str,
        cached: bool,
    ) -> Dict[str, Any]:
        return {
            "detected_disease": disease,
            "intent": intent,
            "session_id": session_id,
            "confidence_score": confidence_score,
            "cached": cached,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _check_rate_limit(self, session_id: str) -> Tuple[bool, int]:
        now = time.time()
        window_start = now - 60
        with self._lock:
            entries = self.request_log[session_id]
            while entries and entries[0] < window_start:
                entries.popleft()
            if len(entries) >= self.rate_limit_per_minute:
                wait_seconds = max(1, int(60 - (now - entries[0])))
                return False, wait_seconds
            entries.append(now)
        return True, 0

    def _append_history(self, session_id: str, user_message: str, bot_response: str, disease: str) -> None:
        history = self.session_history[session_id]
        history.append({"role": "user", "content": user_message, "disease": disease})
        history.append({"role": "assistant", "content": bot_response, "disease": disease})

    def _get_recent_history(self, session_id: str) -> List[Dict[str, str]]:
        history = self.session_history[session_id]
        if not history and self.mongo_collection is not None:
            try:
                # Load last 6 interactions from persistent storage
                interactions = list(self.mongo_collection.find(
                    {"session_id": session_id}
                ).sort("timestamp", -1).limit(6))
                
                for doc in reversed(interactions):
                    self.session_history[session_id].append({
                        "role": "user", 
                        "content": doc.get("user_message", ""),
                        "disease": doc.get("detected_disease", "Unknown")
                    })
                    self.session_history[session_id].append({
                        "role": "assistant", 
                        "content": doc.get("bot_response", ""),
                        "disease": doc.get("detected_disease", "Unknown")
                    })
            except Exception as e:
                print(f"[Chatbot] Failed to load history from MongoDB: {e}")
                
        return list(self.session_history.get(session_id, []))[-8:]

    def _detect_intent(self, user_message: str) -> str:
        text = user_message.lower()
        patterns = {
            "treatment": [r"\btreat\b", r"\btreatment\b", r"\bcure\b", r"\bmedicine\b", r"\bspray\b", r"\bcontrol\b", r"\bmanage\b", r"\buse kya karu\b"],
            "prevention": [r"\bprevent\b", r"\bprevention\b", r"\bprecaution", r"\bavoid\b", r"\bprotect\b", r"\bcome back\b", r"\bstop\b", r"\bbachav\b"],
            "organic": [r"\borganic\b", r"\bnatural\b", r"\bhome remedy\b", r"\bhome remedies\b", r"\bneem\b", r"\bbuttermilk\b", r"\bcow urine\b", r"\bbio\b"],
            "chemical": [r"\bchemical\b", r"\bfungicide\b", r"\bpesticide\b", r"\bcommercial\b", r"\bbrand\b", r"\bcost\b", r"\bspray\b.*\bname\b", r"\bwhich spray\b"],
            "symptoms": [r"\bsymptoms?\b", r"\blook like\b", r"\bsigns?\b", r"\bcheck for\b", r"\bidentify\b", r"\bspots?\b", r"\byellow\b", r"\bwilt\b", r"\bpata\b"],
            "care": [r"\bwater(ing)?\b", r"\bfertiliz(er|ation|ing)\b", r"\bmanure\b", r"\bsoil\b", r"\bcare\b", r"\bmonsoon\b"],
            "explain": [r"\bwhy\b", r"\bexplain\b", r"\bdetail\b", r"\bkaise\b", r"\bkyun\b"],
            "greeting": [r"\bhi\b", r"\bhello\b", r"\bnamaste\b", r"\bhey\b"],
            "thanks": [r"\bthanks\b", r"\bthank you\b", r"\bshukriya\b"],
        }
        for intent, rules in patterns.items():
            if any(re.search(rule, text) for rule in rules):
                return intent
        if any(word in text for word in ["prevention", "prevent", "avoid", "protection", "technique"]):
            return "prevention"
        if any(word in text for word in ["symptom", "sign", "check", "spot", "lesion"]):
            return "symptoms"
        if any(word in text for word in ["organic", "natural", "neem", "bio"]):
            return "organic"
        if any(word in text for word in ["chemical", "fungicide", "pesticide", "spray", "medicine"]):
            return "chemical"
        return "general"

    def _should_use_groq(self, intent: str) -> bool:
        # Use Groq for everything except simple greetings/thanks/rate limiting
        return bool(self.groq_client) and intent not in {"greeting", "thanks", "rate_limited"}

    def _get_groq_response(
        self,
        user_message: str,
        disease: str,
        kb_info: Dict[str, List[str]],
        session_id: str,
        intent: str,
    ) -> Optional[str]:
        if not self.groq_client:
            return None

        history_lines = []
        for item in self._get_recent_history(session_id)[-6:]:
            history_lines.append(f"{item['role'].upper()}: {item['content']}")
        history_text = "\n".join(history_lines) or "No previous conversation."

        prompt = (
            "You are an AI farming assistant for Indian farmers using a plant disease detection system.\n"
            "Answer only plant-health-related questions.\n"
            "Use simple farmer-friendly language with short practical advice.\n"
            "Keep normal replies to 2 to 4 sentences.\n"
            "If user asks why or explain, provide more detail.\n"
            "Use emojis like 🌿 💊 🛡️ 🌱 🧪 naturally.\n"
            "If disease is Healthy or Unknown, give general plant care advice.\n"
            "If the user symptoms do not fit the disease context, ask for a clearer image.\n"
            "For severe cases, suggest contacting a local agricultural officer or KVK.\n"
            "Prefer Indian conditions, low-cost suggestions, and monsoon-aware prevention.\n"
            "If the user asks about water or fertilization, provide specific, actionable advice (e.g., 'water at soil level' or 'apply balanced NPK') suitable for the detected disease.\n\n"
            f"Detected disease: {disease}\n"
            f"Intent: {intent}\n"
            f"Symptoms: {kb_info.get('symptoms', [])}\n"
            f"Treatment: {kb_info.get('treatment', [])}\n"
            f"Prevention: {kb_info.get('prevention', [])}\n"
            f"Organic remedies: {kb_info.get('organic_remedies', [])}\n"
            f"Chemical remedies: {kb_info.get('chemical_remedies', [])}\n"
            f"Conversation history:\n{history_text}\n\n"
            f"User question: {user_message}"
        )

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a concise AI plant health assistant."},
                    {"role": "user", "content": prompt}
                ],
                model=self.groq_model_name,  # ✅ Now using the updated model name
                temperature=self.temperature,
                max_completion_tokens=self.max_output_tokens,
            )
            text = chat_completion.choices[0].message.content.strip()
            return text or None
        except Exception as exc:
            print(f"[Chatbot] Groq request failed: {exc}")
            return None

    def _get_local_response(
        self,
        user_message: str,
        disease: str,
        kb_info: Dict[str, List[str]],
        intent: str,
    ) -> str:
        disease_key = disease.lower()
        if intent == "greeting":
            return "🌿 Namaste. Ask me about symptoms, treatment, prevention, organic remedies, or sprays for your plant."
        if intent == "thanks":
            return "🌱 You're welcome. Ask if you want the next step for this plant."
        if disease_key == "healthy":
            return self._healthy_response(intent)
        if disease_key == "unknown":
            return self._unknown_response(intent)
        if self._symptom_mismatch(user_message, kb_info):
            return "📷 Your symptom description does not fully match this disease. Please upload a clearer image of the affected leaf from both sides."

        if intent == "treatment":
            return self._list_response("🌿 Treatment", disease, kb_info.get("treatment", []), "Start with sanitation first, then spray in the early morning or evening.")
        if intent == "prevention":
            return self._list_response("🛡️ Prevention", disease, kb_info.get("prevention", []), "Regular spacing, leaf hygiene, and controlled watering help stop repeat infection.", include_suggestions=True)
        if intent == "organic":
            remedies = list(kb_info.get("organic_remedies", []))
            if not remedies or "not available" in remedies[0].lower():
                 remedies = self.knowledge_base.data.get("PlantCare", {}).get("organic_practices", [])
            
            if not remedies:
                remedies = ["Neem oil spray", "Buttermilk spray", "Diluted cow urine solution"]
                
            return self._list_response("🌱 Organic remedies", disease, remedies, "Test on a few leaves first before full spraying.", include_suggestions=True)
        if intent == "chemical":
            chemicals = self._with_cost_hints(kb_info.get("chemical_remedies", []))
            return self._list_response("🧪 Chemical options", disease, chemicals, "Use only label dose and wear protection while spraying.", include_suggestions=True)
        if intent == "symptoms":
            return self._list_response("🔍 Symptoms", disease, kb_info.get("symptoms", []), "If the disease is spreading fast, contact a local agricultural officer or KVK.", include_suggestions=True)
        if intent == "care":
            care_info = self.knowledge_base.data.get("PlantCare", {})
            if any(w in user_message.lower() for w in ["water", "irrigation"]):
                items = care_info.get("watering", [])
            elif any(w in user_message.lower() for w in ["fertiliz", "manure", "npk"]):
                items = care_info.get("fertilizer", [])
            elif any(w in user_message.lower() for w in ["sun", "light"]):
                items = care_info.get("sunlight", [])
            elif any(w in user_message.lower() for w in ["monsoon", "rain"]):
                items = care_info.get("monsoon_care", [])
            else:
                return (
                    f"🌱 For {disease}, water at soil level, keep plants well spaced, and remove infected debris quickly. "
                    "In general, ensure 6-8 hours of sunlight and use well-drained soil."
                )
            return f"🌱 Plant Care Tip: {', '.join(items[:2])}. Always monitor your plants daily."
        if intent == "explain":
            symptoms = ", ".join(kb_info.get("symptoms", [])[:3]) or "visible leaf damage"
            prevention = ", ".join(kb_info.get("prevention", [])[:2]) or "clean field management"
            return (
                f"🌿 {disease} usually shows as {symptoms}. It spreads faster when leaves stay wet or airflow is poor, so {prevention} is important."
            )
        return (
            f"🌿 {disease} needs timely treatment and prevention. Ask me about symptoms, treatment, prevention, organic remedies, or chemical options."
        )

    def _healthy_response(self, intent: str) -> str:
        healthy_prevention = [
            "Monitor leaves weekly for early spots or curling",
            "Water near the soil, not over the leaves",
            "Keep good airflow and remove fallen infected debris nearby",
        ]
        if intent == "care":
            return "🌱 Your plant looks healthy. Water in the morning, use balanced fertilizer, and inspect leaves weekly, especially in monsoon."
        if intent == "prevention":
            return self._list_response("🛡️ Prevention", "this healthy plant", healthy_prevention, "The current uploaded leaf looks healthy, so focus on regular care to keep it disease-free.", include_suggestions=True)
        if intent == "symptoms":
            return "🔍 This uploaded leaf looks healthy. Healthy signs are green color, normal texture, and no active spots, rot, or powdery growth. You can also ask about prevention, treatment, organic remedies, or chemical options."
        return "🌿 The uploaded plant looks healthy. Keep good watering, sunlight, and field hygiene, and monitor leaves regularly. You can also ask about prevention, symptoms, treatment, or remedies."

    def _unknown_response(self, intent: str) -> str:
        if intent == "symptoms":
            return "🔬 Common disease signs are yellow or brown spots, white powder, orange pustules, curling, wilting, or dark wet lesions. Upload a clearer close image for better diagnosis."
        return "📷 I cannot identify the disease confidently yet. Please upload a clearer photo and tell me the crop name and how many days the problem has been visible."

    def _list_response(self, title: str, disease: str, items: List[str], tail: str, include_suggestions: bool = False) -> str:
        clean_items = [item for item in items if item and "not available" not in item.lower()]
        if not clean_items:
            message = f"{title} for {disease} is not available yet. 🌿 Monitor the plant closely and consult a local agricultural officer if the issue becomes severe."
        else:
            message = f"{title} for {disease}: {', '.join(clean_items[:3])}. {tail}"
        if include_suggestions:
            message += " You can also ask about symptoms, treatment, organic remedies, or chemical options."
        return message

    def _symptom_mismatch(self, user_message: str, kb_info: Dict[str, List[str]]) -> bool:
        msg = user_message.lower()
        if not any(word in msg for word in ["spot", "yellow", "brown", "white", "orange", "wilt", "curl", "rot"]):
            return False
        known_text = " ".join(kb_info.get("symptoms", [])).lower()
        words = [word for word in ["spot", "yellow", "brown", "white", "orange", "wilt", "curl", "rot"] if word in msg]
        return bool(words) and not any(word in known_text for word in words)

    def _with_cost_hints(self, chemicals: List[str]) -> List[str]:
        return [f"{item} (check local agri shop price)" for item in chemicals[:3]]

    def _format_response(self, response_text: str, disease: str, intent: str) -> str:
        cleaned = re.sub(r"\s+", " ", response_text or "").strip()
        if disease.lower() not in {"healthy", "unknown"} and intent not in {"greeting", "thanks"}:
            if any(term in cleaned.lower() for term in ["collapse", "severe", "whole plant", "spreading fast"]):
                cleaned += " 🚨 Please contact your local agricultural officer or KVK."
        return cleaned

    def _persist_interaction(
        self,
        session_id: str,
        user_message: str,
        bot_response: str,
        detected_disease: str,
        response_source: str,
        confidence_score: Optional[float],
    ) -> None:
        if self.mongo_collection is None:
            return
        document = {
            "session_id": session_id,
            "user_message": user_message,
            "bot_response": bot_response,
            "detected_disease": detected_disease,
            "response_source": response_source,
            "timestamp": datetime.now(timezone.utc),
            "confidence_score": confidence_score,
        }
        try:
            self.mongo_collection.insert_one(document)
        except Exception as exc:
            print(f"[Chatbot] MongoDB insert failed: {exc}")


_chatbot: Optional[PlantChatbot] = None


def get_chatbot() -> PlantChatbot:
    global _chatbot
    if _chatbot is None:
        _chatbot = PlantChatbot()
    return _chatbot


def chatbot_reply(
    user_message: str,
    disease: str = "Unknown",
    session_id: Optional[str] = None,
    confidence_score: Optional[float] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    return get_chatbot().get_response(user_message, disease, session_id, confidence_score)


def chat_response(
    user_message: str,
    disease: str = "Unknown",
    session_id: Optional[str] = None,
    confidence_score: Optional[float] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    return chatbot_reply(user_message, disease, session_id, confidence_score)


def get_bot_reply(question: str, disease: str = "Unknown", session_id: Optional[str] = None) -> str:
    response, _, _ = chatbot_reply(question, disease, session_id)
    return response
