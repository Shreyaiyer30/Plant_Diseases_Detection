import os
import json
import logging
import numpy as np
import cv2
import datetime
from typing import List, Dict, Union, Any, Tuple
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
KNOWLEDGE_BASE_PATH = os.path.join(MODELS_DIR, 'knowledge_base.json')
CLASS_LABELS_PATH = os.path.join(MODELS_DIR, 'class_labels_final.json') # Updated to new labels
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, 'combined_plant_model_final.h5') # New Master Model
IMG_SIZE = 128

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class EnsembleModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EnsembleModelLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.models = []
        self.class_names = self._load_class_names()
        self.load_models()
        self._initialized = True

    def _load_class_names(self) -> List[str]:
        """Loads class labels created during training"""
        try:
            if os.path.exists(CLASS_LABELS_PATH):
                with open(CLASS_LABELS_PATH, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                    logger.info(f"Loaded {len(labels)} labels from {CLASS_LABELS_PATH}")
                    return labels
            
            # Fallback to KB if final labels not found
            if os.path.exists(KNOWLEDGE_BASE_PATH):
                with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
                    kb = json.load(f)
                    return list(kb.keys())

            return ["Healthy", "Diseased"]
        except Exception as e:
            logger.error(f"Error loading class names: {e}")
            return ["Healthy", "Diseased"]

    def load_models(self):
        """Loads the master model"""
        self.models = []
        if os.path.exists(FINAL_MODEL_PATH):
            try:
                logger.info(f"Loading Master Model from {FINAL_MODEL_PATH}...")
                model = load_model(FINAL_MODEL_PATH, compile=False)
                self.models.append(model)
                logger.info("✓ Master Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading master model: {e}")
        else:
            logger.error(f"Master model not found at {FINAL_MODEL_PATH}")

    def preprocess(self, image_path: str) -> np.ndarray:
        """Production image preprocessing."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        # Optional: slight blur to reduce noise
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Convert to float32 and normalize
        img_array = image.astype(np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)

    def predict(self, image_path: str, mode: str = 'mean') -> Dict[str, Any]:
        """Performs ensemble prediction and returns enriched data for app.py."""
        if not self.models:
            logger.error("No models available for prediction")
            return {
                "success": False, 
                "model_unavailable": True, 
                "status": "error",
                "description": "No trained models found. Please ensure the .h5 files are in the 'models' directory."
            }

        try:
            processed = self.preprocess(image_path)
            
            # Get predictions from all models
            all_preds = []
            for idx, m in enumerate(self.models):
                try:
                    pred = m.predict(processed, verbose=0)[0]
                    all_preds.append(pred)
                except Exception as e:
                    logger.warning(f"Model {idx+1} prediction failed: {e}")
                    continue

            if not all_preds:
                return {"success": False, "status": "error", "description": "All models failed to predict"}

            all_preds_np = np.array(all_preds)

            if not all_preds:
                return {"success": False, "status": "error", "description": "Model failed to predict"}

            final_probs = all_preds[0] # Single master model output

            top_idx = int(np.argmax(final_probs))
            confidence = float(final_probs[top_idx])
            label = self.class_names[top_idx] if top_idx < len(self.class_names) else "Unknown"
            
            # Determine status
            is_healthy = "healthy" in label.lower() or label == "Healthy"
            status = "healthy" if is_healthy else "diseased"
            
            # Health score logic
            if is_healthy:
                health = 100.0
            else:
                health = max(5.0, 100.0 - (confidence * 100 * 0.9))

            # Get top 3 predictions
            top_3_indices = np.argsort(final_probs)[-3:][::-1]
            top_3 = []
            for i in top_3_indices:
                if i < len(self.class_names):
                    top_3.append({"label": self.class_names[i], "confidence": round(float(final_probs[i]) * 100, 1)})

            # Confidence thresholds: lower bar for healthy (30%), higher for diseased (70%)
            CONFIDENCE_THRESHOLD = 0.30 if is_healthy else 0.70
            
            result = {
                "success": True,
                "disease": label,
                "confidence": round(confidence * 100, 1),
                "status": status,
                "health": round(health, 1),
                "top_predictions": top_3,
                "ensemble_info": f"Fused {len(all_preds)} models"
            }

            # Load knowledge base info if available
            try:
                if os.path.exists(KNOWLEDGE_BASE_PATH):
                    with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
                        kb = json.load(f)
                    
                    if label in kb:
                        info = kb[label]
                        result["symptoms"] = info.get("symptoms", [])
                        result["treatment"] = info.get("treatment", [])
                        result["prevention"] = info.get("prevention", [])
                        result["organic_remedies"] = info.get("organic_remedies", [])
                        result["chemical_remedies"] = info.get("chemical_remedies", [])
                        
                        # Build description
                        if is_healthy:
                            result["description"] = "Your plant appears healthy. Continue regular care and monitoring."
                        else:
                            result["description"] = f"Detected: {label} with {result['confidence']}% confidence."
            except Exception as e:
                logger.warning(f"Could not load knowledge base: {e}")

            # Confidence thresholds: lower bar for healthy (30%), higher for diseased (70%)
            CONFIDENCE_THRESHOLD = 0.30 if is_healthy else 0.70

            if confidence < CONFIDENCE_THRESHOLD:
                # If confidence is extremely low, it's likely not a plant at all or a very poor image
                if confidence < 0.35:
                    result["status"] = "invalid"
                    result["message"] = "Please provide a good and clear image."
                    result["description"] = "The system could not detect a plant or leaf in this photo. Please ensure you are taking a clear photo of a single leaf."
                else:
                    result["status"] = "uncertain"
                    result["message"] = f"Low confidence detection ({result['confidence']}%)."
                    result["description"] = f"Please provide a good and clear image. The model detected {label} but with low confidence."
                
                result["success"] = False
                thresh_pct = int(CONFIDENCE_THRESHOLD * 100)
                
                result["issues"] = [
                    "The image may not contain a recognizable plant or leaf",
                    f"Confidence Level ({result['confidence']}%) is below the required threshold",
                    "The photo might be blurry, too dark, or too far away"
                ]
                result["suggestions"] = [
                    "Provide a clear close-up of a single leaf",
                    "Ensure the leaf is centered and well-lit",
                    "Use a plain, neutral background if possible",
                    "Avoid taking photos of non-plant objects"
                ]

            return result

        except FileNotFoundError as e:
            logger.error(f"Image file error: {e}")
            return {"success": False, "status": "error", "description": str(e), "invalid_image": True}
        except Exception as e:
            logger.error(f"Ensemble Prediction Failure: {e}")
            return {"success": False, "status": "error", "description": str(e), "invalid_image": True}

    def train_it(self, data_path: str):
        """Skeleton for training logic after combining."""
        logger.info(f"Retraining ensemble components on {data_path}...")
        pass

# --- Exports ---
def get_model_instance():
    return EnsembleModelLoader()

def predict_disease(image_path: str, mode: str = 'mean') -> dict:
    return EnsembleModelLoader().predict(image_path, mode=mode)

def model_health() -> dict:
    loader = EnsembleModelLoader()
    return {
        "loaded": len(loader.models) > 0,
        "total_models": getattr(loader, 'total_loaded', 0),
        "compatible_models": len(loader.models),
        "classes": len(loader.class_names)
    }