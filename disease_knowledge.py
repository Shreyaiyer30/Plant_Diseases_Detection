"""
disease_knowledge.py — Dynamic disease information retrieval
============================================================
Loads per-disease symptoms, treatment, prevention, and remedies
from models/knowledge_base.json.  Falls back to a built-in
default only if the file is missing or corrupt.
"""
import json
import os

# Always resolve relative to this file's location, not CWD
_BASE_DIR = os.path.abspath(os.path.dirname(__file__))
_DEFAULT_KB_PATH = os.path.join(_BASE_DIR, "models", "knowledge_base.json")


class DiseaseKnowledgeBase:
    def __init__(self, knowledge_file=None):
        # Accept caller-supplied path or default to models/knowledge_base.json
        if knowledge_file is None:
            self.knowledge_file = _DEFAULT_KB_PATH
        else:
            # If path is relative, anchor it to BASE_DIR so it works from any CWD
            if not os.path.isabs(knowledge_file):
                self.knowledge_file = os.path.join(_BASE_DIR, knowledge_file)
            else:
                self.knowledge_file = knowledge_file

        self.data = {}
        self.load_knowledge()

    # ------------------------------------------------------------------
    def load_knowledge(self):
        """Load knowledge base from JSON file."""
        if os.path.exists(self.knowledge_file):
            try:
                with open(self.knowledge_file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                print(f"[KnowledgeBase] Loaded {len(self.data)} diseases from {self.knowledge_file}")
            except Exception as e:
                print(f"[KnowledgeBase] Error loading knowledge base: {e}")
                self.data = self._get_default_knowledge()
        else:
            print(f"[KnowledgeBase] File not found: {self.knowledge_file} — using built-in defaults")
            self.data = self._get_default_knowledge()

    # ------------------------------------------------------------------
    def _normalize(self, name: str) -> str:
        """Case-insensitive, whitespace-tolerant key lookup."""
        return name.strip().lower()

    def _find(self, disease_name: str) -> dict:
        """Try exact match first, then case-insensitive match."""
        if disease_name in self.data:
            return self.data[disease_name]
        norm = self._normalize(disease_name)
        for key, val in self.data.items():
            if self._normalize(key) == norm:
                return val
        return {}

    # ------------------------------------------------------------------
    def get_symptoms(self, disease_name: str) -> list:
        return self._find(disease_name).get(
            "symptoms", [f"No symptom data available for: {disease_name}"]
        )

    def get_treatment(self, disease_name: str) -> list:
        return self._find(disease_name).get(
            "treatment", [f"No treatment data available for: {disease_name}"]
        )

    def get_prevention(self, disease_name: str) -> list:
        return self._find(disease_name).get(
            "prevention", [f"No prevention data available for: {disease_name}"]
        )

    def get_organic_remedies(self, disease_name: str) -> list:
        return self._find(disease_name).get("organic_remedies", [])

    def get_chemical_remedies(self, disease_name: str) -> list:
        return self._find(disease_name).get("chemical_remedies", [])

    def get_all_info(self, disease_name: str) -> dict:
        """Return all structured information for a given disease name."""
        info = self._find(disease_name)
        return {
            "symptoms":          info.get("symptoms",          [f"Symptoms for '{disease_name}' not available"]),
            "treatment":         info.get("treatment",         [f"Treatment for '{disease_name}' not available"]),
            "prevention":        info.get("prevention",        [f"Prevention for '{disease_name}' not available"]),
            "organic_remedies":  info.get("organic_remedies",  []),
            "chemical_remedies": info.get("chemical_remedies", []),
        }

    # ------------------------------------------------------------------
    def add_or_update_disease(self, disease_name: str, data: dict):
        """Add or update a disease entry and persist to disk."""
        self.data[disease_name] = data
        self.save_knowledge()

    def save_knowledge(self) -> bool:
        """Save current in-memory knowledge back to the JSON file."""
        try:
            with open(self.knowledge_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[KnowledgeBase] Error saving: {e}")
            return False

    # ------------------------------------------------------------------
    @staticmethod
    def _get_default_knowledge() -> dict:
        """Minimal in-code fallback if the JSON file is missing."""
        return {
            "Healthy": {
                "symptoms":          ["Uniform green colour", "No spots or lesions", "Firm leaf texture"],
                "treatment":         ["Maintain regular watering", "Ensure proper sunlight", "Continue balanced fertilization"],
                "prevention":        ["Monitor regularly", "Keep soil well-drained"],
                "organic_remedies":  ["None needed — plant is healthy"],
                "chemical_remedies": ["None needed"],
            },
            "Powdery Mildew": {
                "symptoms":          ["White powdery spots on leaf surface", "Yellowing or browning leaves", "Stunted or distorted growth"],
                "treatment":         ["Apply sulfur-based fungicide", "Spray with neem oil (5 ml/L)", "Remove severely infected leaves"],
                "prevention":        ["Avoid overhead watering", "Ensure good air circulation", "Space plants properly"],
                "organic_remedies":  ["Neem oil", "Baking soda solution (1 tbsp/gallon)", "Milk spray (1:10 ratio)"],
                "chemical_remedies": ["Sulfur 80 WP", "Myclobutanil", "Propiconazole"],
            },
        }