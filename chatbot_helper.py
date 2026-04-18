# chatbot_helper.py - Add this at the top if missing
import os
import sqlite3

from model_loader import DISEASE_INFO, KAGGLE_MAP, _display_name
from disease_knowledge import DiseaseKnowledgeBase as _KB

_kb = _KB()   # singleton — loads models/knowledge_base.json once

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "instance", "plantcure.db")


def _normalize_key(value):
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _load_db_treatment(disease):
    if not os.path.exists(DB_PATH):
        return {}

    disease_norm = _normalize_key(disease)
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM plant_treatments").fetchall()
        conn.close()
    except Exception:
        return {}

    for row in rows:
        row_dict = dict(row)
        stored_name = row_dict.get("disease_name", "")
        if _normalize_key(stored_name) == disease_norm:
            return row_dict
        display_name = _display_name(stored_name)
        if _normalize_key(display_name) == disease_norm:
            return row_dict
    return {}


def _disease_payload(disease):
    display = _display_name(disease or "Unknown")
    canonical = KAGGLE_MAP.get(disease, display)
    base = DISEASE_INFO.get(canonical, DISEASE_INFO.get(display, {})).copy()
    db_row = _load_db_treatment(disease) or _load_db_treatment(display)

    # Prefer the rich knowledge base (models/knowledge_base.json) for all text fields
    kb_info = _kb.get_all_info(display) or _kb.get_all_info(canonical)

    # Symptoms: prefer rich KB, fallback to DISEASE_INFO
    kb_symptoms = kb_info.get("symptoms") or base.get("symptoms", [])

    # Treatment steps
    treatment_steps = db_row.get("treatment_steps") or "\n".join(
        f"{idx + 1}. {step}"
        for idx, step in enumerate(kb_info.get("treatment") or base.get("treatment", []))
    )
    # Prevention tips
    prevention_tips = db_row.get("prevention_tips") or "\n".join(
        f"- {step}"
        for step in (kb_info.get("prevention") or base.get("prevention", []))
    )
    # Organic remedies
    organic_remedies = db_row.get("organic_remedies") or "\n".join(
        f"- {r}" for r in (kb_info.get("organic_remedies") or [
            "Neem oil spray once every 5 to 7 days.",
            "Remove the worst affected leaves and discard them.",
            "Improve airflow and keep foliage dry.",
        ])
    )
    # Chemical remedies
    chemical_remedies = db_row.get("chemical_remedies") or "\n".join(
        f"- {r}" for r in (kb_info.get("chemical_remedies") or base.get("treatment", []))
    )

    return {
        "name": display,
        "description": base.get(
            "description",
            "Plant stress is visible. Inspect the leaf closely and isolate the plant if symptoms spread.",
        ),
        "symptoms": kb_symptoms,
        "treatment_steps":   treatment_steps.strip(),
        "prevention_tips":   prevention_tips.strip(),
        "organic_remedies":  organic_remedies.strip(),
        "chemical_remedies": chemical_remedies.strip(),
    }


def _intent(question):
    text = (question or "").lower()
    if any(word in text for word in ["organic", "natural", "home remedy", "neem"]):
        return "organic"
    if any(word in text for word in ["chemical", "fungicide", "pesticide", "spray", "medicine"]):
        return "chemical"
    if any(word in text for word in ["prevent", "prevention", "avoid", "stop again"]):
        return "prevention"
    if any(word in text for word in ["symptom", "sign", "look like", "why"]):
        return "symptoms"
    if any(word in text for word in ["treat", "treatment", "cure", "steps", "what should i do"]):
        return "treatment"
    return "overview"


def _format_response(intent_name, data, question=""):
    disease = data["name"]
    q = (question or "").lower()
    
    # Contradiction check: User sees symptoms on a 'Healthy' plant
    if disease == "Healthy" and any(word in q for word in ["spot", "yellow", "brown", "orange", "wilt", "hole", "rot", "mold"]):
        return (
            "You mentioned seeing spots or discoloration, though the scan results suggest the leaf is healthy. "
            "This can happen if the disease is at a very early stage or if the photo was taken from a distance. "
            "I recommend checking the leaf underside for tiny pustules or searching for 'Rust' or 'Blight' symptoms. "
            "Would you like to know about general prevention for leaf spots?"
        )

    if intent_name == "treatment":
        return (
            f"Treatment steps for {disease}:\n"
            f"{data['treatment_steps']}\n\n"
            "Start with sanitation first, then apply the next control step in the early morning or evening."
        )
    if intent_name == "prevention":
        return (
            f"Prevention tips for {disease}:\n"
            f"{data['prevention_tips']}\n\n"
            "Prevention works best when watering, spacing, and leaf hygiene stay consistent."
        )
    if intent_name == "organic":
        return f"Organic remedies for {disease}:\n{data['organic_remedies']}"
    if intent_name == "chemical":
        return f"Chemical options for {disease}:\n{data['chemical_remedies']}"
    if intent_name == "symptoms":
        symptom_lines = "\n".join(f"- {item}" for item in data["symptoms"]) or "- Visible spotting or stress on the leaf."
        return f"Common signs of {disease}:\n{symptom_lines}\n\nDescription: {data['description']}"
    
    return (
        f"Disease: {disease}\n\n"
        f"Description: {data['description']}\n\n"
        "You can ask me for treatment steps, prevention tips, organic remedies, or chemical options."
    )


def chatbot_reply(question, disease):
    data = _disease_payload(disease)
    response = _format_response(_intent(question), data, question)
    meta = {
        "disease": data["name"],
        "description": data["description"],
        "treatment_steps": data["treatment_steps"],
        "prevention_tips": data["prevention_tips"],
        "organic_remedies": data["organic_remedies"],
        "chemical_remedies": data["chemical_remedies"],
    }
    return response, "local_kb", meta


def get_chat_response(question, disease):
    """Wrapper for chatbot_reply with error handling"""
    try:
        response, source, meta = chatbot_reply(question, disease)
        return response, source, meta
    except Exception as e:
        print(f"Chatbot error: {e}")
        return (
            "I'm having trouble processing your request right now. "
            "Please try asking about treatment, prevention, or symptoms.",
            "error",
            {"error": str(e)}
        )