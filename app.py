"""
app.py  —  PlantCure v3
========================
Python 3.11+ | Flask 3.1+ | SQLite | Anthropic AI chatbot
"""
import os, sys, io
# Fix for Windows terminal UTF-8 output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()
import datetime, json, re
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, flash, send_from_directory, abort)
from pymongo import MongoClient
from bson.objectid import ObjectId
from flask_login import (LoginManager, UserMixin, login_user,
                         logout_user, login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from model_loader import predict_disease, model_health
import json
from googletrans import Translator
from disease_knowledge import DiseaseKnowledgeBase 

translator = Translator()

BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
INSTANCE_DIR  = os.path.join(BASE_DIR, "instance")
ALLOWED_EXT   = {"png", "jpg", "jpeg", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INSTANCE_DIR,  exist_ok=True)

app = Flask(__name__)
app.config.update(
    SECRET_KEY                     = os.environ.get("SECRET_KEY", "plantcure-secret-2025"),
    UPLOAD_FOLDER                  = UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH             = 10 * 1024 * 1024,
    SEND_FILE_MAX_AGE_DEFAULT      = 0,
    TEMPLATES_AUTO_RELOAD          = True,
)

mongo_client = MongoClient(os.environ.get("MONGODB_URI", "mongodb://localhost:27017"))
mongo_db = mongo_client[os.environ.get("MONGODB_DB", "plantcure")]
login_manager = LoginManager(app)
login_manager.login_view             = "login"
login_manager.login_message          = "Please log in to continue."
login_manager.login_message_category = "info"

app.jinja_env.auto_reload = True
knowledge_base = DiseaseKnowledgeBase()
@app.after_request
def add_no_cache_headers(response):
    # Helps when UI/CSS/JS updates seem not applied due browser cache.
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ─── Language Helpers ─────────────────────────────────────────────
def load_lang_json(code):
    try:
        path = os.path.join(os.path.dirname(__file__), "lang", f"{code}.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return load_lang_json("en")

def safe_translate(text, dest):
    if not text or dest == 'en': return text
    try:
        res = translator.translate(text, dest=dest)
        return res.text
    except Exception as e:
        print(f"Translation Error: {e}")
        return text

# ─── Database Models (PyMongo) ────────────────────────────────────────────────
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data.get("_id"))
        self.name = user_data.get("name")
        self.email = user_data.get("email")
        self.password = user_data.get("password")
        self.role = user_data.get("role", "farmer")

    def check_password(self, raw):
        return check_password_hash(self.password, raw)

def prediction_to_dict(pred):
    disease_text = (pred.get("disease") or "").lower()
    normalized_status = pred.get("status")
    if "healthy" in disease_text:
        normalized_status = "healthy"
    
    health = 0
    if normalized_status == 'healthy':
        health = 100.0
    elif normalized_status in ['uncertain', 'invalid']:
        health = 0
    else:
        sev = str(pred.get("severity") or "").lower()
        if 'high' in sev: health = 10.0
        elif 'moderate' in sev or 'medium' in sev: health = 40.0
        elif 'low' in sev: health = 70.0
        else: health = max(5.0, 100.0 - pred.get("confidence", 0.0))
        
    return {
        "id"        : str(pred.get("_id")),
        "disease"   : pred.get("disease"),
        "confidence": round(pred.get("confidence", 0.0), 1),
        "status"    : normalized_status,
        "severity"  : pred.get("severity"),
        "filename"  : pred.get("filename"),
        "timestamp" : pred.get("timestamp").isoformat(timespec="seconds") if pred.get("timestamp") else "",
        "health"    : round(health, 1),
        "diseased"  : round(100.0 - health, 1)
    }

def treatment_to_dict(treat):
    return {
        "id": str(treat.get("_id")),
        "disease_name": treat.get("disease_name"),
        "treatment_steps": treat.get("treatment_steps"),
        "prevention_tips": treat.get("prevention_tips"),
        "chemical_remedies": treat.get("chemical_remedies"),
        "organic_remedies": treat.get("organic_remedies"),
        "common_questions": treat.get("common_questions"),
        "created_at": treat.get("created_at").isoformat() if hasattr(treat.get("created_at"), "isoformat") else ""
    }

@login_manager.user_loader
def load_user(uid):
    try:
        user_data = mongo_db.users.find_one({"_id": ObjectId(uid)})
        if user_data:
            return User(user_data)
    except Exception:
        pass
    return None

def allowed_file(fn):
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def _latest_user_message(messages):
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""

def _is_ambiguous_query(text):
    t = (text or "").strip().lower()
    if not t:
        return True
    care_keywords = [
        "control", "treat", "treatment", "cure", "stop", "save", "kill", "fix",
        "spray", "fungicide", "pesticide", "fertilizer", "water", "watering",
        "symptom", "blight", "spot", "mold", "virus", "mite",
    ]
    if any(k in t for k in care_keywords):
        return False
    if len(t) < 8:
        return True
    vague_tokens = {"help", "why", "what", "how", "problem", "issue", "tell me", "explain"}
    if t in vague_tokens:
        return True
    # Very short fragment without concrete crop/disease words.
    if len(t.split()) <= 2 and not any(k in t for k in ["leaf", "plant", "crop", "disease", "spot", "spray", "fertilizer", "water"]):
        return True
    return False


def _is_treatment_question(text):
    t = (text or "").strip().lower()
    if re.search(r"\bnot\s+spray\b", t) or re.search(r"\bno(t|\s)+spray\b", t):
        return False
    return any(
        k in t
        for k in [
            "how to",
            "how i",
            "how do",
            "control",
            "treat",
            "cure",
            "medicine",
            "spray",
            "fungicide",
            "pesticide",
            "save plant",
            "stop spread",
        ]
    )


def _looks_like_short_answer(text):
    t = (text or "").strip().lower()
    if not t:
        return False
    # Accept short numeric replies like: "2", "3 days", "daily", "weekly".
    if t.isdigit():
        return True
    words = t.split()
    if len(words) <= 3 and any(k in t for k in ["day", "days", "daily", "weekly", "month", "once", "twice"]):
        return True
    if len(words) <= 3 and any(k in t for k in ["yes", "no", "none"]):
        return True
    return False

def _previous_assistant_asked_question(messages):
    for m in reversed(messages or []):
        if m.get("role") == "assistant":
            content = (m.get("content") or "").strip()
            return "?" in content
    return False


def _last_assistant_text(messages):
    for m in reversed(messages or []):
        if m.get("role") == "assistant":
            return (m.get("content") or "").strip()
    return ""

def _is_irrelevant_to_plantcare(text):
    t = (text or "").strip().lower()
    if not t:
        return False
    plant_terms = [
        "plant", "leaf", "crop", "disease", "spot", "fungus", "blight", "spray",
        "fertilizer", "water", "soil", "tomato", "potato", "pepper", "grape",
        "healthy", "symptom", "pesticide", "fungicide"
    ]
    off_topic_terms = [
        "boat", "car", "movie", "song", "cricket", "football", "politics", "stock",
        "crypto", "phone", "laptop", "coding", "exam", "travel"
    ]
    if any(k in t for k in off_topic_terms) and not any(k in t for k in plant_terms):
        return True
    return False


def _is_greeting_only(text):
    t = (text or "").strip().lower()
    if not t:
        return False
    if t in {"hi", "hello", "hey", "hii", "hlo", "namaste", "good morning", "good evening", "gm", "yo"}:
        return True
    return len(t.split()) <= 2 and t in {"hi there", "hey there"}


def _user_says_already_told(text):
    t = (text or "").strip().lower()
    return any(
        p in t
        for p in [
            "already said",
            "i said",
            "i told",
            "previous message",
            "previous chat",
            "i say it",
            "said it before",
            "you asked",
            "i already",
        ]
    )


def _fallback_user_texts(messages):
    return [
        (m.get("content") or "").strip()
        for m in messages
        if m.get("role") == "user" and (m.get("content") or "").strip()
    ]


def _fallback_parse_slots(messages):
    """Read watering / fertilizer / symptom timing from the whole thread (not turn count)."""
    texts = _fallback_user_texts(messages)
    combined = "\n".join(t.lower() for t in texts)
    slots = {"symptom_days": None, "watering": None, "fertilizer": None}

    for t in reversed(texts):
        tl = t.lower()
        if re.search(r"\b(water|watering|irrigation|daily|weekly|hour|hr\b|twice|once|every\s+\d)", tl):
            slots["watering"] = t.strip()
            break

    for t in texts:
        tl = t.lower()
        m = re.search(r"(\d+)\s*(day|days)\b", tl)
        if m:
            slots["symptom_days"] = f"{m.group(1)} days ago"
            break
    if slots["symptom_days"] is None:
        for t in texts:
            if t.strip().isdigit():
                n = int(t.strip())
                if 0 < n < 90:
                    slots["symptom_days"] = f"~{n} days ago (please confirm)"
                    break

    if re.search(r"not\s+spray", combined) or re.search(
        r"no(t|\s)+.*\b(fertil|fertilizer|pesticide)\b", combined
    ):
        slots["fertilizer"] = "no"
    elif re.search(
        r"\b(no|none|never)\b.{0,40}\b(fertil|fertilizer|pesticide|spray)\b",
        combined,
        re.DOTALL,
    ):
        slots["fertilizer"] = "no"
    elif re.search(r"without\s+fertil", combined) or "no fertilizer" in combined:
        slots["fertilizer"] = "no"

    if slots["fertilizer"] != "no":
        if re.search(
            r"\b(yes|applied|used|sprayed)\b.{0,60}\b(fertil|fertilizer|pesticide|npk|urea)\b",
            combined,
        ):
            slots["fertilizer"] = "yes"
        elif re.search(
            r"\b(yes|yeah|yep)\b.{0,20}\b(i\s+used|applied|sprayed)\b",
            combined,
        ):
            slots["fertilizer"] = "yes"

    return slots


def _slots_summary(slots):
    parts = []
    if slots.get("symptom_days"):
        parts.append(f"symptoms since: {slots['symptom_days']}")
    if slots.get("watering"):
        parts.append(f"watering: {slots['watering']}")
    if slots.get("fertilizer") == "no":
        parts.append("fertilizer/pesticide: none recently")
    elif slots.get("fertilizer") == "yes":
        parts.append("fertilizer/pesticide: used (details not captured)")
    return "; ".join(parts) if parts else "nothing saved yet"


def _fallback_next_question(slots, disease, severity):
    if not slots.get("symptom_days"):
        return (
            f"For **{disease}** (severity: {severity}), I need one detail first: "
            "**how many days ago did you first notice spots, yellowing, or wilting?** "
            "(If the plant looks fully healthy, say so.)"
        )
    if not slots.get("watering"):
        return "**How do you water this plant?** (For example: daily, every 2 days, or minutes per day.)"
    if slots.get("fertilizer") is None:
        return "**In the last 2 weeks, have you used any fertilizer or pesticide?** If yes, which one?"
    return None


def _fallback_full_advice(disease, severity, slots):
    w = slots.get("watering") or "your usual schedule"
    f = slots.get("fertilizer")
    fert_line = (
        "You mentioned no recent fertilizer or spray - avoid random chemicals until the diagnosis is clear."
        if f == "no"
        else "If you use fertilizer or spray, follow the product label."
    )
    return (
        f"Understanding: **{disease}** (severity: {severity}), symptoms about **{slots.get('symptom_days')}**, "
        f"watering: {w}. {fert_line} "
        "Practical steps: remove badly affected leaves, avoid wetting foliage, improve airflow, and consider "
        "**copper fungicide or neem oil (about 5 ml per litre)** only if you see lesions that match late blight. "
        "If the plant looks healthy, retake a **single-leaf close-up** (little or no soil) for a safer scan. "
        "Want a simple day-by-day plan next?"
    )


# ── Auth Routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("dashboard") if current_user.is_authenticated else url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email    = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        remember = bool(request.form.get("remember"))
        user_data = mongo_db.users.find_one({"email": email})
        user = User(user_data) if user_data else None
        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash(f"Welcome back, {user.name}! 🌿", "success")
            return redirect(request.args.get("next") or url_for("dashboard"))
        flash("Incorrect email or password.", "danger")
    return render_template("auth/login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name     = request.form.get("name","").strip()
        email    = request.form.get("email","").strip().lower()
        role     = request.form.get("role","farmer")
        password = request.form.get("password","")
        confirm  = request.form.get("confirm_password","")
        errors   = []
        if len(name) < 2:        errors.append("Name must be at least 2 characters.")
        if "@" not in email:     errors.append("Enter a valid email.")
        if len(password) < 6:   errors.append("Password must be at least 6 characters.")
        if password != confirm:  errors.append("Passwords do not match.")
        if mongo_db.users.find_one({"email": email}): errors.append("Email already registered.")
        if errors:
            for e in errors: flash(e, "danger")
        else:
            hashed_pw = generate_password_hash(password)
            mongo_db.users.insert_one({"name": name, "email": email, "role": role, "password": hashed_pw, "created_at": datetime.datetime.utcnow()})
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
    return render_template("auth/register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))

# ── Page Routes ───────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("main_spa.html")

@app.route("/uploads/<path:filename>")
@login_required
def serve_upload(filename):
    """Serve a saved scan image only to the user who owns it (filename prefix = user id)."""
    if "/" in filename or "\\" in filename or ".." in filename:
        abort(404)
    base = os.path.basename(filename.replace("\\", "/"))
    if not base:
        abort(404)
    if not base.startswith(f"{current_user.id}_"):
        abort(403)
    fp = os.path.join(UPLOAD_FOLDER, base)
    if not os.path.isfile(fp):
        abort(404)
    return send_from_directory(UPLOAD_FOLDER, base)

@app.route("/detect")
@login_required
def detect():
    return render_template("detect.html")

@app.route("/history")
@login_required
def history():
    return render_template("history.html")

@app.route("/analytics")
@login_required
def analytics():
    return render_template("analytics.html")

# ── API: Predict ──────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400
    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use JPG or PNG."}), 415

    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_user.id}_{ts}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Get prediction from model
    lang_code = request.form.get('lang', 'en')
    mode = request.form.get('mode', 'mean') # mean, max, or voting
    result = predict_disease(filepath, mode=mode)
    
    # Translate specific fields if requested
    if lang_code != 'en':
        result['translated_disease'] = safe_translate(result.get('disease', ''), lang_code)
        
        if result.get('status') == 'uncertain':
            result['message'] = safe_translate(result.get('description', ''), lang_code)
            result['issues'] = [safe_translate(i, lang_code) for i in result.get('issues', [])]
            result['suggestions'] = [safe_translate(s, lang_code) for s in result.get('suggestions', [])]

    # Add UI translations
    result['lang'] = load_lang_json(lang_code)

    if result.get("model_unavailable"):
        try:
            os.remove(filepath)
        except Exception:
            pass
        return jsonify(result), 503

    if result.get("invalid_image") or result.get("status") in ["uncertain", "invalid", "error"]:
        try:
            os.remove(filepath)
        except Exception:
            pass
        return jsonify({
            "success": False,
            "status": result.get("status", "invalid"),
            "message": result.get("message") or result.get("description") or "Please upload a better photo",
            "issues": result.get("issues", []),
            "suggestions": result.get("suggestions", []),
            "disease": result.get("disease"),
            "confidence": result.get("confidence")
        }), 422

    # Normalize healthy labels but preserve specific plant names if stored in KB
    disease_name = result.get("disease", "Unknown")
    status = result.get("status", "unknown")
    
    # Try to see if specific info exists before falling back to generic "Healthy"
    display_disease = disease_name
    if "healthy" in str(disease_name).lower():
        status = "healthy"
        # If the KB doesn't have the specific "Blueberry___healthy", then we use "Healthy"
        if not knowledge_base._find(disease_name):
            display_disease = "Healthy"

    confidence = float(result.get("confidence", 0))
    severity = result.get("severity", "Medium")
    
    if status == "healthy":
        health = 100.0
    else:
        health = float(result.get("health", 0.0))
        
    # Get dynamic symptoms, treatment, prevention from knowledge base
    disease_info = knowledge_base.get_all_info(display_disease)
    
    # Translate disease info if needed
    if lang_code != 'en':
        for key in ['symptoms', 'treatment', 'prevention', 'organic_remedies', 'chemical_remedies']:
            if key in disease_info and isinstance(disease_info[key], list):
                disease_info[key] = [safe_translate(s, lang_code) for s in disease_info[key]]

    # Save to database
    pred_doc = {
        "user_id": current_user.id,
        "filename": filename,
        "disease": disease_name,
        "confidence": confidence,
        "status": status,
        "severity": severity,
        "timestamp": datetime.datetime.utcnow()
    }
    db_res = mongo_db.predictions.insert_one(pred_doc)
    pred_id = str(db_res.inserted_id)
    
    # Return enriched response
    return jsonify({
        "success": True,
        "record_id": pred_id,
        "disease": disease_name,
        "translated_disease": display_disease,
        "confidence": confidence,
        "health": health,
        "diseased": round(100.0 - health, 1),
        "severity": severity,
        "status": status,
        "description": result.get("description") or (
            "The uploaded leaf looks healthy. Continue regular care and monitoring."
            if status == "healthy"
            else f"Analysis suggests the presence of {display_disease}. Recommended actions are provided below."
        ),
        "symptoms": disease_info.get("symptoms", []),
        "treatment": disease_info.get("treatment", []),
        "prevention": disease_info.get("prevention", []),
        "organic_remedies": disease_info.get("organic_remedies", []),
        "chemical_remedies": disease_info.get("chemical_remedies", [])
    })

# ── API: Train Ensemble ───────────────────────────────────────────
from model_loader import get_model_instance

@app.route("/api/train", methods=["POST"])
@login_required
def api_train():
    """Trigger retraining of the ensemble components (Admin only)."""
    if getattr(current_user, 'role', 'user') != 'admin':
        return jsonify({"error": "Unauthorized. Admin role required."}), 403
    
    loader = get_model_instance()
    # Path to your training dataset
    data_path = os.path.join(BASE_DIR, 'data', 'PlantVillage')
    
    try:
        # In a real app, this should run in a background thread or queue
        loader.train_it(data_path)
        return jsonify({"success": True, "message": "Model ensemble training/fine-tuning initiated."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ── API: AI Chatbot ───────────────────────────────────────────────
from ai_chatbot import chatbot_reply

@app.route('/api/treatment/<disease_name>')
@login_required
def get_treatment(disease_name):
    treatment = mongo_db.plant_treatments.find_one({"disease_name": disease_name})
    if treatment:
        return jsonify(treatment_to_dict(treatment))
    return jsonify({"error": "Treatment not found"}), 404

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    data = request.get_json()
    user_message = data.get("question") or data.get("message")
    disease = data.get("disease", "Unknown")
    lang_code = data.get("lang", 'en')
    session_id = data.get("session_id") or f"user-{current_user.id}"
    confidence = data.get("confidence")

    # Translate input to English if needed
    eng_input = user_message
    if lang_code != 'en':
        eng_input = safe_translate(user_message, 'en')

    # Pass everything to the chatbot reply logic
    response_text, source, meta = chatbot_reply(
        eng_input,
        disease,
        session_id=session_id,
        confidence_score=confidence,
    )
    
    # Translate reply back to user's language
    final_reply = response_text
    if lang_code != 'en':
        final_reply = safe_translate(response_text, lang_code)

    chat_doc = {
        "user_id": current_user.id,
        "disease": disease,
        "user_query": user_message,
        "response": final_reply,
        "confidence": confidence,
        "source": source,
        "timestamp": datetime.datetime.utcnow()
    }
    mongo_db.chat_messages.insert_one(chat_doc)

    return jsonify({
        "response": final_reply,
        "source": source,
        "meta": meta,
        "session_id": session_id
    })

@app.route('/api/chat/save', methods=['POST'])
@login_required
def save_chat_history():
    data = request.get_json()
    chat_doc = {
        "user_id": current_user.id,
        "disease": data.get("disease"),
        "user_query": data.get("question") or data.get("message"),
        "response": data.get("response"),
        "confidence": data.get("confidence"),
        "source": data.get("source"),
        "timestamp": datetime.datetime.utcnow()
    }
    mongo_db.chat_messages.insert_one(chat_doc)
    return jsonify({"status": "success"})

@app.route('/api/lang/<lang_code>', methods=['GET'])
def get_language_pack(lang_code):
    return jsonify(load_lang_json(lang_code))

def get_fallback_response(crop, disease, severity, messages):
    """Rule-based fallback if API is unavailable."""
    latest_raw = _latest_user_message(messages)
    latest = latest_raw.lower()
    slots = _fallback_parse_slots(messages)
    
    # Get disease-specific info from knowledge base
    disease_info = knowledge_base.get_all_info(disease)
    
    if _is_irrelevant_to_plantcare(latest_raw):
        return (
            "I am specialized in plant disease diagnosis and treatment guidance only. "
            "Please ask me about symptoms, sprays, fertilizer, watering, prevention, or recovery plan for your plant."
        )
    
    # If user asks for treatment, return from knowledge base
    if _is_treatment_question(latest_raw):
        treatment_steps = disease_info.get("treatment", [])
        if treatment_steps:
            return f"For **{disease}** (severity: {severity}), here's what to do:\n\n" + \
                   "\n".join([f"• {step}" for step in treatment_steps[:3]])
    
    # Rest of your existing fallback logic...
    if _is_greeting_only(latest_raw):
        return (
            f"Hi. The last scan shows **{disease}** (severity: {severity}). "
            f"Symptoms include: {', '.join(disease_info.get('symptoms', ['leaf changes'])[:2])}. "
            "What would you like to know about treatment or prevention?"
        )
    


    if _user_says_already_told(latest_raw):
        summ = _slots_summary(slots)
        nxt = _fallback_next_question(slots, disease, severity)
        if nxt:
            return f"Sorry I repeated myself. Here is what I understood: {summ}. {nxt}"
        return _fallback_full_advice(disease, severity, slots)

    if any(k in latest for k in ["which plant", "what plant", "which leaf", "what leaf", "plant name", "crop name"]):
        return (
            f"From this diagnosis context, it is most likely a **{crop}** leaf. "
            f"The detected condition is **{disease}** (Severity: {severity}). "
            "Would you like fertilizer and spray guidance for this crop?"
        )

    # Short yes/no: interpret using last assistant question + earlier thread
    la = _last_assistant_text(messages).lower()
    short = latest_raw.strip().lower()
    if short in {"yes", "yeah", "yep", "no", "nope"}:
        if slots.get("fertilizer") == "no" and short in {"yes", "yeah", "yep"}:
            return (
                "Earlier you wrote that you did not spray fertilizer. If **yes** meant something else, please say what. "
                "Otherwise, tell me **how many days ago** you first noticed symptoms (or say the plant looks healthy)."
            )
        if slots.get("fertilizer") is None and short in {"no", "nope"} and (
            "fertilizer" in la or "pesticide" in la
        ):
            slots = {**slots, "fertilizer": "no"}
            nxt = _fallback_next_question(slots, disease, severity)
            if nxt:
                return f"Noted: no recent fertilizer or pesticide. {nxt}"
            return _fallback_full_advice(disease, severity, slots)
        if slots.get("fertilizer") is None and short in {"yes", "yeah", "yep"} and (
            "fertilizer" in la or "pesticide" in la
        ):
            slots = {**slots, "fertilizer": "yes"}
            nxt = _fallback_next_question(slots, disease, severity)
            if nxt:
                return f"Noted. {nxt}"
            return _fallback_full_advice(disease, severity, slots)

    if _is_ambiguous_query(latest_raw):
        if _looks_like_short_answer(latest_raw) and _previous_assistant_asked_question(messages):
            pass
        else:
            return (
                "I want to answer correctly, but your question is not clear yet. "
                "Please share one detail: symptoms, watering routine, fertilizer/spray used, or how many plants are affected."
            )

    if _is_treatment_question(latest_raw):
        nxt = _fallback_next_question(slots, disease, severity)
        if nxt:
            return (
                f"For **{disease}** (severity: {severity}), start by removing badly infected leaves, improving airflow, "
                f"and avoiding wet leaves. {nxt}"
            )
        return _fallback_full_advice(disease, severity, slots)

    nxt = _fallback_next_question(slots, disease, severity)
    if nxt:
        return nxt
    return _fallback_full_advice(disease, severity, slots)

# ── API: History & Analytics ──────────────────────────────────────
@app.route("/api/history")
@login_required
def api_history():
    preds = list(mongo_db.predictions.find({"user_id": current_user.id}).sort("timestamp", -1).limit(50))
    return jsonify([prediction_to_dict(p) for p in preds])

@app.route("/api/history/<pid>", methods=["DELETE"])
@login_required
def api_delete(pid):
    try:
        pid_obj = ObjectId(pid)
    except Exception:
        abort(404)
    pred = mongo_db.predictions.find_one({"_id": pid_obj, "user_id": current_user.id})
    if not pred:
        abort(404)
    mongo_db.predictions.delete_one({"_id": pid_obj})
    return jsonify({"ok": True})

@app.route("/api/analytics")
@login_required
def api_analytics():
    preds = list(mongo_db.predictions.find({"user_id": current_user.id}))
    if not preds:
        return jsonify({
            "total": 0,
            "healthy_count": 0,
            "diseased_count": 0,
            "disease_distribution": {},
            "most_common_disease": None,
            "avg_confidence": 0
        })

    healthy_count = sum(1 for p in preds if p.get("status") == "healthy")
    diseased_count = len(preds) - healthy_count
    avg_conf = round(sum(p.get("confidence", 0) for p in preds) / len(preds), 1)

    disease_distribution = {}
    for p in preds:
        disease = p.get("disease", "Unknown")
        disease_distribution[disease] = disease_distribution.get(disease, 0) + 1

    most_common = max(disease_distribution, key=disease_distribution.get) if disease_distribution else None

    return jsonify({
        "total": len(preds),
        "healthy_count": healthy_count,
        "diseased_count": diseased_count,
        "avg_confidence": avg_conf,
        "disease_distribution": disease_distribution,
        "most_common_disease": most_common
    })

@app.route("/api/health")
def api_health():
    mh = model_health()
    return jsonify({
        "status": "ok" if mh.get("loaded") else "degraded",
        "version": "3.0.0",
        "model": mh
    })

# ── Init ──────────────────────────────────────────────────────────
# MongoDB initialization is handled at global level

if __name__ == "__main__":
    print("\n" + "="*52)
    print("  🌿  PlantCure v3 — AI-Powered Disease Detection")
    print("  →   http://localhost:5000")
    print("="*52 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
