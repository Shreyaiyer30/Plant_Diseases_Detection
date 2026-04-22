# deep_translate_helper.py
"""
Translation helper for PlantCure using deep-translator API as PRIMARY
and JSON files as FALLBACK. Includes memory caching.
Supports English, Hindi, Marathi.
"""

import json
import os
import threading
import time
from deep_translator import GoogleTranslator

# Path to lang folder
LANG_FOLDER = os.path.join(os.path.dirname(__file__), 'lang')

# Global cache and lock
_translation_cache = {}
_cache_lock = threading.Lock()
_json_translations = {}

# Language mapping for deep-translator
LANG_MAP = {
    'hi': 'hi',  # hindi
    'mr': 'mr',  # marathi
    'en': 'en',  # english
}

SUPPORTED_LANGUAGES = {
    'en': {'name': 'English', 'native': 'English', 'code': 'en'},
    'hi': {'name': 'Hindi', 'native': 'हिन्दी', 'code': 'hi'},
    'mr': {'name': 'Marathi', 'native': 'मराठी', 'code': 'mr'},
}

def load_json_translations(lang_code):
    """Load language JSON file"""
    try:
        file_path = os.path.join(LANG_FOLDER, f"{lang_code}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {lang_code}.json: {e}")
    return {}

def get_json_translations(lang_code):
    """Get cached JSON translations"""
    if lang_code not in _json_translations:
        _json_translations[lang_code] = load_json_translations(lang_code)
    return _json_translations[lang_code]

def reload_json_translations():
    """Reload all JSON translation files"""
    global _json_translations
    _json_translations.clear()
    for lang in ['en', 'hi', 'mr']:
        _json_translations[lang] = load_json_translations(lang)
    print(f"JSON translations reloaded.")

def clear_cache():
    """Clear translation cache"""
    with _cache_lock:
        _translation_cache.clear()

def get_translator(target_lang):
    """Get translator instance for target language"""
    target = LANG_MAP.get(target_lang, target_lang)
    return GoogleTranslator(source='auto', target=target)

def translate_text(text, target_lang='en'):
    """
    Translate text using deep_translator API first, then JSON fallback.
    Caches results to avoid API rate limits.
    """
    if target_lang == 'en' or not text or not str(text).strip():
        return text
    
    text = str(text).strip()
    cache_key = f"{text}_{target_lang}"
    
    # 1. Check cache first
    with _cache_lock:
        if cache_key in _translation_cache:
            return _translation_cache[cache_key]
            
    translated = None
    
    # 2. Try API (Primary)
    try:
        translator = get_translator(target_lang)
        translated = translator.translate(text)
        time.sleep(0.05) # Small delay to respect rate limits
    except Exception as e:
        print(f"API translation error for '{text[:30]}...': {e}")
        translated = None
        
    # 3. Fallback to JSON if API fails
    if not translated:
        json_trans = get_json_translations(target_lang)
        if text in json_trans:
            translated = json_trans[text]
        else:
            text_lower = text.lower()
            for key, value in json_trans.items():
                if key.lower() == text_lower:
                    translated = value
                    break
                    
    # 4. Ultimate Fallback
    if not translated:
        translated = text
        
    # Cache and return
    with _cache_lock:
        _translation_cache[cache_key] = translated
        
    return translated

def translate_list(items, target_lang):
    """Translate a list of strings"""
    if not items or target_lang == 'en':
        return items
    return [translate_text(item, target_lang) for item in items]

def translate_disease_name(disease_name, target_lang):
    """Translate disease name with special formatting"""
    if target_lang == 'en' or not disease_name:
        return disease_name
        
    clean_name = str(disease_name).replace("___", " ").replace("__", " ").replace("_", " ").strip()
    
    # Try API through translate_text
    translated = translate_text(clean_name, target_lang)
    
    # If API failed (returned same text), check JSON for original exact match
    if translated == clean_name:
        json_trans = get_json_translations(target_lang)
        if disease_name in json_trans:
            translated = json_trans[disease_name]
            
    return translated

def translate_prediction_result(result, target_lang):
    """Translate prediction result (for compatibility)"""
    pass

def get_ui_text(key, target_lang, default=None):
    """Get UI text from JSON files (always JSON)"""
    if target_lang == 'en':
        return default or key
    json_trans = get_json_translations(target_lang)
    return json_trans.get(key, default or key)

# Load initial translations
reload_json_translations()
