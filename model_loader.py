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
import gdown
import h5py
import tempfile
from tensorflow.keras.optimizers import Adam

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
KNOWLEDGE_BASE_PATH = os.path.join(MODELS_DIR, 'knowledge_base.json')
CLASS_LABELS_PATH = os.path.join(MODELS_DIR, 'class_labels.json') # Updated to actual file
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, 'combined_plant_models.h5') # New Master Model
#GD_MODEL_ID = "1-XXXX_YOUR_GOOGLE_DRIVE_ID_HERE" # Update with your actual file ID
GD_MODEL_ID = "1E36WZR_UFKckILyUhQ-l_8wD6UIrrSEO" # Update with your actual file ID
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
        self._ensure_model_exists()
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

    def _ensure_model_exists(self):
        """Downloads the model from Google Drive if it's missing"""
        if not os.path.exists(FINAL_MODEL_PATH):
            logger.info("Master model not found. Attempting to download from Google Drive...")
            try:
                os.makedirs(MODELS_DIR, exist_ok=True)
                url = f'https://drive.google.com/uc?id={GD_MODEL_ID}'
                gdown.download(url, FINAL_MODEL_PATH, quiet=False)
                if not os.path.exists(FINAL_MODEL_PATH):
                    logger.error("Download failed or file not found after download.")
                else:
                    logger.info("✓ Model downloaded successfully from Google Drive.")
            except Exception as e:
                logger.error(f"Failed to download model from Google Drive: {e}")
                logger.info("Please manually place the model file in 'models/' directory.")

    def load_models(self):
        """Loads models, supporting both single files and merged HDF5 bundles"""
        self.models = []
        if not os.path.exists(FINAL_MODEL_PATH):
            logger.error(f"Master model not found at {FINAL_MODEL_PATH}")
            return

        try:
            # Try 1: Standard Keras Load
            logger.info(f"Attempting standard load: {FINAL_MODEL_PATH}")
            model = load_model(FINAL_MODEL_PATH, compile=False)
            self.models.append(model)
            logger.info("✓ Model loaded successfully (Standard)")
        except Exception:
            # Try 2: Merged HDF5 Bundle Load
            logger.info("Standard load failed. Checking if this is a merged HDF5 bundle...")
            try:
                with h5py.File(FINAL_MODEL_PATH, 'r') as f:
                    groups = list(f.keys())
                    if not groups:
                        logger.error("H5 file is empty.")
                        return

                    logger.info(f"Found {len(groups)} groups in H5: {groups}")
                    for group_name in groups:
                        try:
                            # Extract group to a temporary H5 file that Keras can read
                            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                                tmp_path = tmp.name
                            
                            with h5py.File(tmp_path, 'w') as tmp_h5:
                                # Copy everything from the group to the root of the temp file
                                for key in f[group_name].keys():
                                    f[group_name].copy(key, tmp_h5)
                                # Copy attributes
                                for attr_nm, attr_val in f[group_name].attrs.items():
                                    tmp_h5.attrs[attr_nm] = attr_val
                            
                            # Load from temp file
                            m = load_model(tmp_path, compile=False)
                            self.models.append(m)
                            logger.info(f"✓ Loaded nested model from group: {group_name}")
                            
                            # Cleanup
                            os.remove(tmp_path)
                        except Exception as e:
                            logger.warning(f"Could not load nested model '{group_name}': {e}")
                            if os.path.exists(tmp_path): os.remove(tmp_path)
            except Exception as e:
                logger.error(f"Failed to process combined H5 file: {e}")

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
            
            # 1. Advanced Variance Check (Ensemble Disagreement)
            # Real leaves produce consistent predictions across models.
            # Non-leaf images (faces, walls) cause models to 'disagree' wildly.
            prediction_variance = np.var(all_preds_np, axis=0).mean()
            
            if mode == 'mean':
                final_probs = np.mean(all_preds_np, axis=0)
            else:
                final_probs = all_preds[0]
            
            top_idx = int(np.argmax(final_probs))
            confidence = float(final_probs[top_idx])
            
            # 2. Heuristic: Color Distribution Check
            # Convert to HSV to check for 'plant-like' colors and 'skin-like' colors
            img_bgr = cv2.imread(image_path)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            
            # --- Plant Range (Green/Yellow/Dry Green) ---
            lower_plant = np.array([25, 30, 30])
            upper_plant = np.array([95, 255, 255])
            plant_mask = cv2.inRange(hsv, lower_plant, upper_plant)
            plant_ratio = np.count_nonzero(plant_mask) / (hsv.shape[0] * hsv.shape[1])
            
            # --- Human Skin Range (General approximation for hands/faces) ---
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 150, 255])
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.count_nonzero(skin_mask) / (hsv.shape[0] * hsv.shape[1])
            
            # Detection Logic Refinement
            is_invalid = False
            error_reason = ""
            
            if skin_ratio > 0.15: # Significant skin detected (hand/face)
                is_invalid = True
                error_reason = "Human skin detected. Please take a photo of only the leaf, without your hands or face in the frame."
            elif plant_ratio < 0.20: # Not enough plant-like colors
                is_invalid = True
                error_reason = "No leaf detected. Please ensure you are taking a well-lit photo of a green or yellow leaf."
            elif prediction_variance > 0.10: # Tightened variance (was 0.15)
                is_invalid = True
                error_reason = "The object in the photo is not recognized as a leaf. Please provide a clearer, closer image."
            elif confidence < 0.60: # High absolute confidence required (was 0.50)
                is_invalid = True
                error_reason = "Low confidence: Image may be too blurry or not a clear leaf."
            
            if is_invalid:
                return {
                    "success": False,
                    "status": "invalid",
                    "message": "Please provide a good and clear image.",
                    "description": error_reason,
                    "issues": ["Image contains non-plant objects (hands/skin)", "Background too distracting", "Leaf not centered"],
                    "suggestions": ["Hold the leaf by the stem only", "Place the leaf on a flat, neutral surface (ground or paper)", "Ensure no people are in the photo"]
                }
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