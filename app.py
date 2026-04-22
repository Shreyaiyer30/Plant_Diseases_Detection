"""
app.py  —  PlantCure v3
========================
Python 3.11+ | Flask 3.1+ | MongoDB | Groq AI Chatbot
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
from unified_chatbot import chat_response, clear_session, get_chatbot
from deep_translate_helper import (
    translate_text,
    translate_list,
    translate_disease_name,
    translate_prediction_result,
    get_ui_text,
    SUPPORTED_LANGUAGES,
    clear_cache,
    reload_json_translations
)
from disease_knowledge import DiseaseKnowledgeBase

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INSTANCE_DIR, exist_ok=True)

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "plantcure-secret-2025"),
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    SEND_FILE_MAX_AGE_DEFAULT=0,
    TEMPLATES_AUTO_RELOAD=True,
)

mongo_client = MongoClient(os.environ.get("MONGODB_URI", "mongodb://localhost:27017"))
mongo_db = mongo_client[os.environ.get("MONGODB_DB", "plantcure")]
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to continue."
login_manager.login_message_category = "info"

app.jinja_env.auto_reload = True
knowledge_base = DiseaseKnowledgeBase()

@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ─── Language Helpers ─────────────────────────────────────────────
def load_lang_json(code):
    """Load language JSON file for UI translations"""
    try:
        path = os.path.join(os.path.dirname(__file__), "lang", f"{code}.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {code}.json: {e}")
        # Return basic fallback
        if code == 'hi':
            return {"healthy": "स्वस्थ", "diseased": "रोगग्रस्त"}
        elif code == 'mr':
            return {"healthy": "निरोगी", "diseased": "रोगट"}
        return {"healthy": "Healthy", "diseased": "Diseased"}

# ─── Database Models (PyMongo) ───────────────────────────────────
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
        if 'high' in sev:
            health = 10.0
        elif 'moderate' in sev or 'medium' in sev:
            health = 40.0
        elif 'low' in sev:
            health = 70.0
        else:
            health = max(5.0, 100.0 - pred.get("confidence", 0.0))
        
    return {
        "id": str(pred.get("_id")),
        "disease": pred.get("disease"),
        "confidence": round(pred.get("confidence", 0.0), 1),
        "status": normalized_status,
        "severity": pred.get("severity"),
        "filename": pred.get("filename"),
        "timestamp": pred.get("timestamp").isoformat(timespec="seconds") if pred.get("timestamp") else "",
        "health": round(health, 1),
        "diseased": round(100.0 - health, 1)
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

# ── Auth Routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("dashboard") if current_user.is_authenticated else url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
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
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        role = request.form.get("role", "farmer")
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        errors = []
        if len(name) < 2:
            errors.append("Name must be at least 2 characters.")
        if "@" not in email:
            errors.append("Enter a valid email.")
        if len(password) < 6:
            errors.append("Password must be at least 6 characters.")
        if password != confirm:
            errors.append("Passwords do not match.")
        if mongo_db.users.find_one({"email": email}):
            errors.append("Email already registered.")
        if errors:
            for e in errors:
                flash(e, "danger")
        else:
            hashed_pw = generate_password_hash(password)
            mongo_db.users.insert_one({
                "name": name, "email": email, "role": role,
                "password": hashed_pw, "created_at": datetime.datetime.utcnow()
            })
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

# ── API: Predict (with JSON-based translation) ───────────────────
@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400
    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use JPG or PNG."}), 415

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_user.id}_{ts}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Get prediction from model
    lang_code = request.form.get('lang', 'en')
    mode = request.form.get('mode', 'mean')
    result = predict_disease(filepath, mode=mode)
    
    raw_disease = result.get("disease", "Unknown")
    # Clean the raw disease name
    cleaned_disease = str(raw_disease).replace("___", " ").replace("__", " ").replace("_", " ").strip()
    
    # Handle invalid/uncertain cases
    if result.get("invalid_image") or result.get("status") in ["uncertain", "invalid", "error"]:
        try:
            os.remove(filepath)
        except Exception:
            pass
        
        # Translate error messages using JSON translator
        message = result.get("message") or result.get("description") or "Please upload a better photo"
        issues = result.get("issues", [])
        suggestions = result.get("suggestions", [])
        
        if lang_code != 'en':
            message = translate_text(message, lang_code)
            issues = translate_list(issues, lang_code)
            suggestions = translate_list(suggestions, lang_code)
        
        return jsonify({
            "success": False,
            "status": result.get("status", "invalid"),
            "message": message,
            "issues": issues,
            "suggestions": suggestions,
            "disease": raw_disease,
            "translated_disease": translate_disease_name(cleaned_disease, lang_code) if lang_code != 'en' else cleaned_disease,
            "confidence": result.get("confidence")
        }), 422

    # Normalize healthy labels
    status = result.get("status", "unknown")
    display_disease = cleaned_disease
    
    if "healthy" in cleaned_disease.lower():
        status = "healthy"
        # Check if KB has specific disease entry
        if not knowledge_base._find(raw_disease):
            display_disease = "Healthy"
    
    confidence = float(result.get("confidence", 0))
    severity = result.get("severity", "Medium")
    
    if status == "healthy":
        health = 100.0
    else:
        health = float(result.get("health", max(5.0, 100.0 - confidence)))
    
    # Get disease info from knowledge base
    disease_info = knowledge_base.get_all_info(raw_disease)
    
    # ========== TRANSLATE USING JSON TRANSLATOR ==========
    if lang_code != 'en':
        translated_disease = translate_disease_name(display_disease, lang_code)
        
        # Build description
        if status == "healthy":
            desc_text = "The uploaded leaf looks healthy. Continue regular care and monitoring."
        else:
            desc_text = f"Analysis suggests the presence of {translated_disease}. Recommended actions are provided below."
        translated_description = translate_text(desc_text, lang_code)
        
        translated_symptoms = translate_list(disease_info.get("symptoms", []), lang_code)
        translated_treatment = translate_list(disease_info.get("treatment", []), lang_code)
        translated_prevention = translate_list(disease_info.get("prevention", []), lang_code)
        translated_organic = translate_list(disease_info.get("organic_remedies", []), lang_code)
        translated_chemical = translate_list(disease_info.get("chemical_remedies", []), lang_code)
    else:
        translated_disease = display_disease
        if status == "healthy":
            translated_description = "The uploaded leaf looks healthy. Continue regular care and monitoring."
        else:
            translated_description = f"Analysis suggests the presence of {display_disease}. Recommended actions are provided below."
        translated_symptoms = disease_info.get("symptoms", [])
        translated_treatment = disease_info.get("treatment", [])
        translated_prevention = disease_info.get("prevention", [])
        translated_organic = disease_info.get("organic_remedies", [])
        translated_chemical = disease_info.get("chemical_remedies", [])

    # Save to database
    pred_doc = {
        "user_id": current_user.id,
        "filename": filename,
        "disease": raw_disease,
        "confidence": confidence,
        "status": status,
        "severity": severity,
        "timestamp": datetime.datetime.utcnow()
    }
    db_res = mongo_db.predictions.insert_one(pred_doc)
    pred_id = str(db_res.inserted_id)
    
    # Return fully translated response
    return jsonify({
        "success": True,
        "record_id": pred_id,
        "disease": raw_disease,
        "translated_disease": translated_disease,
        "confidence": confidence,
        "health": round(health, 1),
        "diseased": round(100.0 - health, 1),
        "severity": severity,
        "status": status,
        "description": translated_description,
        "symptoms": translated_symptoms,
        "treatment": translated_treatment,
        "prevention": translated_prevention,
        "organic_remedies": translated_organic,
        "chemical_remedies": translated_chemical,
        "language": lang_code
    })

# ── API: AI Chatbot ───────────────────────────────────────────────
@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """Chatbot API endpoint using Unified LLM Chatbot"""
    try:
        data = request.get_json()
        
        user_message = data.get('message', '') or data.get('question', '')
        disease = data.get('disease', 'Unknown')
        lang_code = data.get('lang', 'en')
        session_id = data.get("session_id") or f"user-{current_user.id}"
        
        # Get disease info from request or knowledge base
        disease_info = {
            "symptoms": data.get('symptoms', []),
            "treatment": data.get('treatments', []),
            "prevention": data.get('preventions', []),
            "organic_remedies": data.get('organic_remedies', []),
            "chemical_remedies": data.get('chemical_remedies', []),
        }
        
        # Fallback to KB if info is empty
        if not any(disease_info.values()):
            kb_info = knowledge_base.get_all_info(disease)
            disease_info = {
                "symptoms": kb_info.get("symptoms", []),
                "treatment": kb_info.get("treatment", []),
                "prevention": kb_info.get("prevention", []),
                "organic_remedies": kb_info.get("organic_remedies", []),
                "chemical_remedies": kb_info.get("chemical_remedies", []),
            }

        # Get AI-generated response
        response_text, metadata = chat_response(
            user_message=user_message,
            disease=disease,
            disease_info=disease_info,
            session_id=session_id,
            language=lang_code,
        )
        
        return jsonify({
            'success': True,
            'response': response_text,
            'metadata': metadata,
            'session_id': session_id,
            'source': metadata.get("source", "groq_ai"),
            'language': lang_code
        })
        
    except Exception as e:
        print(f"Chat API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'response': "Sorry, I'm having trouble right now. Please try again."
        }), 500

@app.route('/api/chat/clear', methods=['POST'])
@login_required
def clear_chat():
    """Clear chat session history"""
    try:
        data = request.get_json() or {}
        session_id = data.get("session_id") or f"user-{current_user.id}"
        clear_session(session_id)
        return jsonify({'success': True, 'message': 'Chat session cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/session', methods=['GET'])
@login_required
def get_session_info_route():
    """Get chat session information"""
    try:
        session_id = request.args.get('session_id') or f"user-{current_user.id}"
        chatbot = get_chatbot()
        info = chatbot.get_session_info(session_id)
        return jsonify({'success': True, 'session': info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ── API: Language Pack ─────────────────────────────────────────────
@app.route('/api/lang/<lang_code>', methods=['GET'])
def get_language_pack(lang_code):
    """Get UI translations for frontend"""
    translations = load_lang_json(lang_code)
    
    # Add additional translations
    if lang_code == 'hi':
        translations.update({
            'chat_header': 'प्लांट सहायक',
            'chat_placeholder': 'अपने पौधे के बारे में पूछें...',
            'detect_loading': 'पत्ती के पैटर्न का विश्लेषण किया जा रहा है...',
        })
    elif lang_code == 'mr':
        translations.update({
            'chat_header': 'प्लांट सहाय्यक',
            'chat_placeholder': 'तुमच्या वनस्पतीबद्दल विचारा...',
            'detect_loading': 'पानाच्या नमुन्यांचे विश्लेषण करत आहे...',
        })
    
    return jsonify(translations)

# ── API: History & Analytics ──────────────────────────────────────
@app.route("/api/history")
@login_required
def api_history():
    lang_code = request.args.get('lang', 'en')
    preds = list(mongo_db.predictions.find({"user_id": current_user.id}).sort("timestamp", -1).limit(50))
    
    results = []
    for p in preds:
        doc = prediction_to_dict(p)
        raw = doc.get("disease", "Unknown")
        clean = str(raw).replace("___", " ").replace("__", " ").replace("_", " ").strip()
        doc['display_name'] = clean
        
        if lang_code != 'en':
            doc['display_name'] = translate_disease_name(clean, lang_code)
        results.append(doc)
        
    return jsonify(results)

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
    lang_code = request.args.get('lang', 'en')
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
        raw = p.get("disease", "Unknown")
        clean = str(raw).replace("___", " ").replace("__", " ").replace("_", " ").strip()
        disp = clean
        if lang_code != 'en':
            disp = translate_disease_name(clean, lang_code)
        disease_distribution[disp] = disease_distribution.get(disp, 0) + 1

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
if __name__ == "__main__":
    print("\n" + "="*52)
    print("  🌿  PlantCure v3 — AI-Powered Disease Detection")
    print("  →   http://localhost:5000")
    print("="*52 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)