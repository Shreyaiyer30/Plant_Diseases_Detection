"""
model_loader.py  —  Enhanced CNN Model Loader with Image Validation
=====================================================================
Added lenient validation, auto-enhancement, and progressive confidence.
"""

import os, json, datetime
import cv2
import numpy as np
from tensorflow.keras.models import load_model

SUPPORTED_CROPS = {
    "Tomato", "Potato", "Grape", "Apple", "Corn", "Pepper",
    "Strawberry", "Cherry", "Peach", "Soybean", "Squash", "Blueberry", "Raspberry", "Orange"
}

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

PLANT_PROFILES = {
    "Tomato": {
        "display_name": "Tomato Plant",
        "about": "This item appears to be a tomato plant leaf, commonly checked for fungal spots, blight, and pest damage during humid weather.",
        "common_issues": "Dark concentric spots, yellow halos, curling, and rapid leaf blight are common warning signs on tomato foliage.",
        "light_requirements": "Tomatoes grow best in full sun with at least 6 to 8 hours of bright light each day.",
        "watering": "Water deeply at the base and let the top layer of soil dry slightly between sessions to avoid leaf wetness.",
        "propagation": "Tomatoes are usually propagated from seed or healthy cuttings in warm conditions.",
    },
    "Potato": {
        "display_name": "Potato Plant",
        "about": "This item appears to be a potato plant leaf, where blight and stress symptoms often start as irregular dark lesions.",
        "common_issues": "Early blight, late blight, nutrient stress, and water imbalance can all cause spotting or rapid leaf decline.",
        "light_requirements": "Potatoes prefer full sun and steady airflow around the canopy.",
        "watering": "Keep soil evenly moist but not soggy, especially during active tuber development.",
        "propagation": "Potatoes are propagated from certified seed tubers rather than leaf cuttings.",
    },
    "Grape": {
        "display_name": "Grape Vine",
        "about": "This item appears to be a grape vine leaf, which is often monitored for black rot, mildew, and leaf blight.",
        "common_issues": "Leaf lesions, mildew growth, and premature drop are the most common foliage problems.",
        "light_requirements": "Grapes prefer full sun and an open canopy with strong airflow.",
        "watering": "Water deeply and avoid frequent overhead irrigation that keeps leaves wet.",
        "propagation": "Grapes are commonly propagated from hardwood cuttings or grafted vines.",
    },
    "Healthy": {
        "display_name": "Healthy Plant",
        "about": "This item appears healthy based on the uploaded leaf image.",
        "common_issues": "No major disease signs are obvious in the scanned leaf.",
        "light_requirements": "Keep the plant in the light range that matches its crop type.",
        "watering": "Maintain a regular watering schedule and avoid standing water around the roots.",
        "propagation": "Use healthy plant material when propagating to reduce future disease pressure.",
    },
}

# ─── Disease Knowledge Base ───────────────────────────────────────
# (Maintaining the existing knowledge base from previous version)
DISEASE_INFO = {
    "Healthy": {
        "status": "healthy", "severity": "None",
        "description": "The plant looks healthy. No disease detected.",
        "symptoms"  : ["Uniform green colour", "No spots or lesions", "Firm leaf texture"],
        "treatment" : ["Maintain regular watering", "Ensure proper sunlight", "Continue balanced fertilization"],
        "prevention": ["Monitor regularly", "Keep soil well-drained", "Avoid overhead watering"],
    },
    "Tomato - Early Blight": {
        "status": "diseased", "severity": "Moderate",
        "description": "Caused by Alternaria solani. Dark concentric rings on older leaves.",
        "symptoms"  : ["Dark brown concentric ring spots", "Yellow halo around lesions", "Lesions on lower leaves first"],
        "treatment" : ["Apply Mancozeb or copper fungicide", "Remove infected leaves", "Use Chlorothalonil spray"],
        "prevention": ["Crop rotation every 2-3 years", "Avoid wetting foliage", "Use disease-free seeds"],
    },
    "Tomato - Late Blight": {
        "status": "diseased", "severity": "High",
        "description": "Caused by Phytophthora infestans. Spreads rapidly in cool, moist conditions.",
        "symptoms"  : ["Water-soaked leaf lesions", "White mold on leaf undersides", "Brown greasy stem lesions"],
        "treatment" : ["Apply Metalaxyl or Cymoxanil immediately", "Remove all infected plants", "Improve air circulation"],
        "prevention": ["Plant resistant varieties", "Avoid overhead irrigation", "Apply preventive copper fungicide"],
    },
    "Tomato - Leaf Curl Virus": {
        "status": "diseased", "severity": "High",
        "description": "Tomato Leaf Curl Virus spread by whiteflies. Causes severe leaf curling.",
        "symptoms"  : ["Upward/downward leaf curling", "Yellowing leaf margins", "Stunted plant growth"],
        "treatment" : ["Control whiteflies with Imidacloprid", "Remove and destroy infected plants", "Apply Thiamethoxam"],
        "prevention": ["Use reflective mulches", "Grow resistant varieties", "Inspect transplants before planting"],
    },
    "Potato - Early Blight": {
        "status": "diseased", "severity": "Moderate",
        "description": "Caused by Alternaria solani. Target-board-like spots on leaves.",
        "symptoms"  : ["Dark brown target-like ring spots", "Yellowing around lesions", "Premature defoliation"],
        "treatment" : ["Apply Mancozeb or Azoxystrobin", "Remove infected debris", "Balance nitrogen fertilization"],
        "prevention": ["Use certified seed potatoes", "Crop rotation", "Maintain adequate plant spacing"],
    },
    "Potato - Late Blight": {
        "status": "diseased", "severity": "Very High",
        "description": "Caused by Phytophthora infestans. One of the most destructive potato diseases.",
        "symptoms"  : ["Dark water-soaked leaf patches", "White sporulation on undersides", "Rapid tuber brown rot"],
        "treatment" : ["Apply Metalaxyl + Mancozeb immediately", "Destroy infected haulm", "Harvest promptly in dry conditions"],
        "prevention": ["Plant resistant varieties (Kufri Jyoti)", "Apply preventive fungicide", "Avoid irrigation in cloudy weather"],
    },
    "Grape - Black Rot": {
        "status": "diseased", "severity": "High",
        "description": "Caused by Guignardia bidwellii. Affects leaves and fruit.",
        "symptoms"  : ["Circular tan lesions with dark borders", "Black mummified fruit", "Black pycnidia in lesions"],
        "treatment" : ["Apply Myclobutanil or Mancozeb", "Remove mummified fruit", "Prune infected canes"],
        "prevention": ["Maintain open vine canopy", "Apply fungicide from bud break", "Remove crop debris after harvest"],
    },
    "Grape - Leaf Blight": {
        "status": "diseased", "severity": "Moderate",
        "description": "Caused by Isariopsis clavispora. Irregular dark spots on grape leaves.",
        "symptoms"  : ["Dark brown irregular spots", "Premature leaf drop", "Reduced fruit quality"],
        "treatment" : ["Apply copper-based or Zineb fungicide", "Remove infected leaves", "Improve canopy management"],
        "prevention": ["Good vine spacing and pruning", "Avoid wetting foliage", "Apply preventive fungicide spray"],
    },
    "Powdery Mildew": {
        "status": "diseased", "severity": "Moderate",
        "description": "Fungal disease causing white powdery coating on leaf surfaces.",
        "symptoms"  : ["White/grey powdery coating", "Leaf distortion and curling", "Yellowing and premature drop"],
        "treatment" : ["Apply Sulphur-based fungicide", "Spray Neem oil (5ml/L)", "Remove heavily infected parts"],
        "prevention": ["Ensure good air circulation", "Avoid excess nitrogen fertilizer", "Water at base not overhead"],
    },
    "Rust": {
        "status": "diseased", "severity": "High",
        "description": "Fungal disease characterized by orange, yellow, or brown pustules.",
        "symptoms"  : ["Orange or brown rusty spots/pustules", "Yellowing around spots", "Premature leaf drop", "Powdery spores on underside"],
        "treatment" : ["Apply copper or sulphur fungicide", "Remove and destroy infected leaves", "Avoid overhead watering"],
        "prevention": ["Plant resistant varieties", "Ensure good spacing/airflow", "Clean garden debris in autumn"],
    },
    "Apple Scab": {
        "status": "diseased", "severity": "Moderate",
        "description": "Fungal disease causing olive-green to black velvety spots.",
        "symptoms"  : ["Olive-green or brown velvety spots", "Yellowing leaves", "Fruit with scabby lesions"],
        "treatment" : ["Apply Myclobutanil or Captan", "Remove fallen leaves", "Prune for better airflow"],
        "prevention": ["Plant resistant varieties", "Clear fallen leaves in winter", "Avoid wet foliage"],
    },
    "Northern Leaf Blight": {
        "status": "diseased", "severity": "High",
        "description": "Fungal disease causing cigar-shaped lesions on corn leaves.",
        "symptoms"  : ["Long, tan cigar-shaped lesions", "Grayish-green spots", "Premature drying of leaves"],
        "treatment" : ["Apply Azoxystrobin or Pyraclostrobin", "Rotate with non-grass crops", "Manage irrigation"],
        "prevention": ["Use resistant hybrids", "Conventional tillage to bury debris", "Rotate crops"],
    },
    "Gray Leaf Spot": {
        "status": "diseased", "severity": "High",
        "description": "Fungal infection causing leaf lesions and chlorosis.",
        "symptoms": ["Lesions often develop into tan or brown spots and streaks", "Reduces photosynthetic area", "Chlorosis around lesions"],
        "treatment": ["Apply fungicide timely at first sign", "Increase plant spacing"],
        "prevention": ["Use resistant plant hybrids", "Practice crop rotation", "Clear debris to avoid overwintering spores"],
    },
    "Bacterial Leaf Spot": {
        "status": "diseased", "severity": "Moderate",
        "description": "Caused by Xanthomonas species. Water-soaked angular lesions.",
        "symptoms"  : ["Water-soaked angular spots", "Dark brown lesions with yellow halo", "Spots merge causing blight"],
        "treatment" : ["Apply copper hydroxide spray", "Remove infected debris immediately", "Avoid working in wet crop"],
        "prevention": ["Use disease-free certified seeds", "Avoid overhead irrigation", "Maintain proper plant spacing"],
    },
}

KAGGLE_MAP = {
    "Apple___Apple_scab": "Apple___Apple_scab",
    "Apple___Black_rot": "Apple___Black_rot",
    "Apple___Cedar_apple_rust": "Apple___Cedar_apple_rust",
    "Apple___healthy": "Apple___healthy",
    "Blueberry___healthy": "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew": "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy": "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Gray Leaf Spot",
    "Corn_(maize)___Common_rust_": "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight": "Northern Leaf Blight",
    "Corn_(maize)___healthy": "Corn_(maize)___healthy",
    "Grape___Black_rot": "Grape___Black_rot",
    "Pepper__bell___Bacterial_spot": "Pepper__bell___Bacterial_spot",
    "Pepper,_bell___Bacterial_spot": "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy": "Pepper__bell___healthy",
    "Pepper,_bell___healthy": "Pepper__bell___healthy",
    "Tomato___Early_blight": "Tomato___Early_Blight",
    "Tomato___Late_blight": "Tomato___Late_Blight",
    "Tomato___healthy": "Healthy",
    "Strawberry___healthy": "Apple___healthy",
}

def _canon(text: str) -> str:
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())

def _get_info(name: str) -> dict:
    """Safe lookup — never raises KeyError."""
    if name in DISEASE_INFO:
        return DISEASE_INFO[name]
    mapped = KAGGLE_MAP.get(name)
    if mapped:
        return DISEASE_INFO.get(mapped, {})
    cn = _canon(name)
    for k in DISEASE_INFO.keys():
        if _canon(k) == cn:
            return DISEASE_INFO[k]
    for k, v in KAGGLE_MAP.items():
        if _canon(k) == cn:
            return DISEASE_INFO.get(v, {})
    return {
        "status": "diseased", "severity": "Unknown",
        "description": f"Detected: {name.replace('___',' - ').replace('_',' ')}. Consult an agricultural expert.",
        "symptoms"  : ["Visible abnormalities on leaf"],
        "treatment" : ["Consult local agricultural officer"],
        "prevention": ["Monitor crop regularly"],
    }

def _display_name(label: str) -> str:
    if label in DISEASE_INFO:
        return label
    if label in KAGGLE_MAP:
        return KAGGLE_MAP[label]
    c = _canon(label)
    for k, v in KAGGLE_MAP.items():
        if _canon(k) == c:
            return v
    for k in DISEASE_INFO.keys():
        if _canon(k) == c:
            return k
    return label.replace("___", " - ").replace("__", " ").replace("_", " ").strip()

def _extract_crop(label_or_disease: str) -> str:
    text = str(label_or_disease or "").strip()
    if not text:
        return "Unknown"
    if "___" in text:
        return text.split("___", 1)[0].replace("_", " ").replace(",", "").strip().title()
    if " - " in text:
        return text.split(" - ", 1)[0].strip().title()
    return text.split(" ", 1)[0].strip().title()


def _build_result_summary(crop: str, disease_name: str, info: dict) -> tuple[str, dict]:
    profile = PLANT_PROFILES.get(crop) or PLANT_PROFILES.get("Healthy")
    disease_text = disease_name if disease_name != "Healthy" else "a healthy plant"
    summary = (
        f"This item appears to be a {profile['display_name']}. "
        f"The detected condition is {disease_text.lower()}. "
        f"{info.get('description', profile['about'])}"
    )
    details = {
        "about": profile["about"],
        "common_issues": profile["common_issues"],
        "light_requirements": profile["light_requirements"],
        "watering": profile["watering"],
        "propagation": profile["propagation"],
    }
    return summary, details

class PlantDiseaseModel:
    def __init__(self, model_path='models/plant_model.h5', labels_path='models/class_labels.json'):
        self.model = None
        self.class_labels = []
        # Resolve paths relative to this file's directory
        _this_dir = os.path.abspath(os.path.dirname(__file__))
        
        self.loaded = False
        self.reason = "Not loaded."
        self._labels_mismatch = False
        self._init_log = os.path.join(_this_dir, "model_init.log")
        
        try:
            with open(self._init_log, "w") as f:
                f.write(f"--- Model Init: {datetime.datetime.now()} ---\n")
        except: pass
        if not os.path.isabs(model_path):
            model_path = os.path.join(_this_dir, model_path)
        if not os.path.isabs(labels_path):
            labels_path = os.path.join(_this_dir, labels_path)

        # Preferred model paths
        _alt_model_path = os.path.join(_this_dir, "models", "plant_model.h5")
        
        def _try_load(path, name):
            if os.path.exists(path):
                self._log(f"Attempting to load {name}: {path}")
                try:
                    self.model = load_model(path, compile=False)
                    self.loaded = True
                    self.reason = f"Model ({name}) loaded successfully."
                    self._log(f"✅ Success: {path} initialized.")
                    return True
                except Exception as e:
                    self._log(f"❌ ERROR loading {name}: {e}")
            else:
                self._log(f"⚠️ {name} not found at {path}")
            return False

        # Attempt primary load
        if not _try_load(model_path, "Primary Model"):
            # Attempt fallback load
            if _alt_model_path != model_path:
                if _try_load(_alt_model_path, "Fallback Model"):
                    self.reason = "Running on fallback model due to primary load failure."
                else:
                    self.reason = "Primary and fallback models both failed to load."
            else:
                self.reason = "Primary model failed and no alternative provided."

        if os.path.exists(labels_path):
            try:
                with open(labels_path, 'r') as f:
                    self.class_labels = json.load(f)
            except Exception as e:
                print(f"[ModelLoader] Error loading labels: {e}")

        # ── Sanity-check: labels vs model output layer size ────────────
        if self.loaded and self.class_labels:
            try:
                model_out = self.model.output_shape[-1]
                labels_count = len(self.class_labels)
                if model_out != labels_count:
                    self._labels_mismatch = True
                    print(
                        f"\n{'='*60}\n"
                        f"[ModelLoader] WARNING: CLASS COUNT MISMATCH DETECTED!\n"
                        f"   Model output neurons : {model_out}\n"
                        f"   Labels in JSON file  : {labels_count}\n"
                        f"   This is why confidence shows as ~100% for only\n"
                        f"   one or two classes — the model and labels file\n"
                        f"   do not match. You must retrain the model against\n"
                        f"   the full PlantVillage 38-class dataset.\n"
                        f"   Run:  python train_model.py --dataset data/PlantVillage --epochs 30 --fine-tune\n"
                        f"{'='*60}\n"
                    )
                    # Truncate labels to model output to avoid index errors
                    if model_out < labels_count:
                        self.class_labels = self.class_labels[:model_out]
                else:
                    print(f"[ModelLoader] OK: Labels match model output ({labels_count} classes)")
            except Exception:
                pass  # non-fatal

        # LENIENT THRESHOLDS
        self.min_confidence = 0.55
        self.low_confidence = 0.35
        self.reject_threshold = 0.15
        self.min_margin = 0.18

        # Auto-detect input image size from the model (e.g. 128 or 224)
        if self.loaded:
            try:
                self.img_size = self.model.input_shape[1]  # (None, H, W, 3) → H
                print(f"[ModelLoader] Input size detected from model: {self.img_size}x{self.img_size}")
            except Exception:
                self.img_size = 128  # safe fallback
        else:
            self.img_size = 128
        
        # Image validation thresholds
        self.min_dimension = 80
        self.min_brightness = 15
        self.min_blur_threshold = 30
        self.min_green_percentage = 3
        
        # Load model metrics from training logs
        self.metrics = self._load_metrics()

    def _load_metrics(self):
        """Find the latest CSV log and extract the last row of metrics"""
        try:
            log_dir = os.path.join(BASE_DIR, "training_logs")
            if not os.path.isdir(log_dir):
                return {}
            
            logs = [f for f in os.listdir(log_dir) if f.endswith(".csv")]
            if not logs:
                return {}
            
            # Sort by date in filename (format: YYYYMMDD_HHMMSS_...)
            latest_log = sorted(logs)[-1]
            log_path = os.path.join(log_dir, latest_log)
            
            import csv
            with open(log_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    return {}
                last_row = rows[-1]
                return {
                    "accuracy": round(float(last_row.get("accuracy", 0)) * 100, 1),
                    "val_accuracy": round(float(last_row.get("val_accuracy", 0)) * 100, 1),
                    "loss": round(float(last_row.get("loss", 0)), 3),
                    "val_loss": round(float(last_row.get("val_loss", 0)), 3),
                    "epoch": int(float(last_row.get("epoch", 0))) + 1,
                    "log_file": latest_log
                }
        except Exception as e:
            print(f"[ModelLoader] Error loading metrics: {e}")
            return {}
        return {}
        
    def auto_enhance_image(self, image):
        """Automatically enhance poor quality images"""
        enhanced = image.copy()
        
        # 1. Fix dark images
        gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 40:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # 2. Fix slightly blurry images
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if 30 < laplacian_var < 80:
            kernel = np.array([[-0.5,-0.5,-0.5],
                              [-0.5, 5,-0.5],
                              [-0.5,-0.5,-0.5]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # 3. Fix color issues
        if brightness < 30:
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.3, 0, 255)
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # 4. Remove noise
        if laplacian_var < 50:
            enhanced = cv2.medianBlur(enhanced, 3)
        
        return enhanced
    
    def validate_and_enhance(self, image):
        """Validate image and attempt to fix issues"""
        issues = []
        warnings = []
        
        h, w = image.shape[:2]
        
        if h < self.min_dimension or w < self.min_dimension:
            issues.append(f"Image too small ({w}x{h}). Please zoom in on the leaf.")
            return False, issues, warnings, image
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        
        if brightness < self.min_brightness:
            image = self.auto_enhance_image(image)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if np.mean(gray) < self.min_brightness:
                issues.append("Image too dark. Please use flash or better lighting.")
            else:
                warnings.append("Auto-brightness applied")
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < self.min_blur_threshold:
            if laplacian_var > 20:
                image = self.auto_enhance_image(image)
                warnings.append("Auto-sharpening applied")
            else:
                issues.append("Image is too blurry. Please hold camera steady.")
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        green_mask = cv2.inRange(hsv, (25, 25, 25), (85, 255, 255))
        green_percentage = np.sum(green_mask > 0) / (h * w) * 100
        
        if green_percentage < self.min_green_percentage:
            issues.append("No clear leaf detected. Please photograph a single leaf.")
        elif green_percentage < 10:
            warnings.append("Low leaf visibility - results may be less accurate")
        
        return len(issues) == 0, issues, warnings, image
    
    def test_time_augmentation(self, image):
        """Perform TTA — input size read from self.img_size (auto-detected from model)"""
        sz = self.img_size  # e.g. 128 or 224 depending on what was trained
        processed_orig = cv2.resize(image, (sz, sz))
        processed_orig = processed_orig.astype('float32') / 255.0
        
        # Logically different versions: Flip, Rotate, Brightness adjust
        augmented = [np.expand_dims(processed_orig, axis=0)]
        
        # Horizontal flip
        flipped = cv2.flip(processed_orig, 1)
        augmented.append(np.expand_dims(flipped, axis=0))
        
        # Slight rotation
        rows, cols = processed_orig.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 5, 1)
        rotated = cv2.warpAffine(processed_orig, M, (cols, rows))
        augmented.append(np.expand_dims(rotated, axis=0))
        
        # Brightness adjustment
        bright = np.clip(processed_orig * 1.1, 0, 1)
        augmented.append(np.expand_dims(bright, axis=0))
        
        # Batch predict
        batch = np.vstack(augmented)
        batch_preds = self.model.predict(batch, verbose=0)
        
        # Average results
        avg_preds = np.mean(batch_preds, axis=0)
        return avg_preds

    def _log(self, msg):
        print(f"[ModelLoader] {msg}")
        try:
            with open(self._init_log, "a") as f:
                f.write(f"{msg}\n")
        except: pass

    def predict_with_feedback(self, image_path):
        """Make prediction with helpful feedback and TTA"""
        if not self.loaded:
            return {"model_unavailable": True}

        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return {"invalid_image": True, "description": "Could not read image file."}
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Validate and enhance
            is_valid, issues, warnings, enhanced_image = self.validate_and_enhance(image)
            
            if not is_valid:
                return {
                    'invalid_image': True,
                    'status': 'invalid',
                    'description': " | ".join(issues),
                    'issues': issues,
                    'warnings': warnings,
                    'suggestions': [
                        'Take photo in natural daylight',
                        'Hold camera steady',
                        'Get closer to a single leaf',
                        'Ensure leaf fills most of the frame'
                    ]
                }
            
            # Predict with TTA
            avg_preds = self.test_time_augmentation(enhanced_image)
            
            # Get top predictions
            top_indices = np.argsort(avg_preds)[::-1]
            top1_idx = int(top_indices[0])
            top1_conf = float(avg_preds[top1_idx])
            top2_conf = float(avg_preds[int(top_indices[1])]) if len(top_indices) > 1 else 0.0
            
            # --- Fix 1 & 3: Crop Awareness and Bias Correction ---
            top1_label = self.class_labels[top1_idx] if top1_idx < len(self.class_labels) else "Unknown"
            top2_idx = int(top_indices[1]) if len(top_indices) > 1 else -1
            top2_label = self.class_labels[top2_idx] if top2_idx >= 0 else "Unknown"
            
            label = top1_label

            # Top 3 for diagnostics
            top3 = []
            for i in range(min(3, len(top_indices))):
                ti = int(top_indices[i])
                raw_l = self.class_labels[ti] if ti < len(self.class_labels) else "Unknown"
                top3.append({
                    "label": _display_name(raw_l),
                    "confidence": round(float(avg_preds[ti]) * 100, 1)
                })

            disp = _display_name(label)
            crop = _extract_crop(disp)
            info = _get_info(label)
            confidence_margin = top1_conf - top2_conf
            
            # --- Fix 6: Unknown Disease Logic ---
            if top1_conf > 0.95 and is_valid and len(issues) == 0:
                pass

            # 70% Confidence Threshold Gate (Fix 6 fallback)
            CONFIDENCE_THRESHOLD = 0.70
            
            if top1_conf < CONFIDENCE_THRESHOLD:
                return {
                    'invalid_image': True,
                    'status': 'uncertain',
                    'disease': "Unknown Disease",
                    'confidence': round(top1_conf * 100, 1),
                    'description': (
                        f"The model detected {disp} but with low confidence ({round(top1_conf * 100, 1)}%). "
                        "This prediction might be inaccurate. Please provide a clearer photo for better results."
                    ),
                    'issues': [
                        f'Prediction confidence ({round(top1_conf * 100, 1)}%) is below the {int(CONFIDENCE_THRESHOLD*100)}% threshold',
                        'The leaf details may be unclear or obscured'
                    ],
                    'warnings': warnings,
                    'suggestions': [
                        'Capture a clear, close-up of a single leaf',
                        'Ensure the characteristic symptoms (spots, wilting) are well-lit',
                        'Avoid busy backgrounds; use a white paper if possible'
                    ],
                    'top_predictions': top3
                }
            
            # Similar Diseases Check (Fix 3)
            similar_diseases = []
            if confidence_margin < 0.15 and len(top3) > 1:
                similar_diseases = [top3[1]]
            
            status = info.get("status", "diseased")
            message = f"Detected: {disp}"
            result_summary, plant_details = _build_result_summary(crop, disp, info)
            
            # Health Calculation (Fix 4 requirement - Severity based logic)
            if status == "healthy":
                health_score = round(top1_conf * 100, 1)
            elif status in ['uncertain', 'invalid']:
                health_score = 0
            else:
                sev = str(info.get("severity", "")).lower()
                if 'high' in sev: health_score = 15.0
                elif 'moderate' in sev or 'medium' in sev: health_score = 45.0
                elif 'low' in sev: health_score = 75.0
                else: health_score = round(max(5.0, 100.0 - (top1_conf * 100)), 1)

            return {
                "disease": disp,
                "crop": crop,
                "confidence": round(top1_conf * 100, 1),
                "health": health_score,
                "top_predictions": top3,
                "similar_diseases": similar_diseases,
                "status": status,
                "message": message,
                "result_summary": result_summary,
                "plant_details": plant_details,
                "warnings": warnings,
                "enhancements_applied": len(warnings) > 0,
                **info,
                "invalid_image": False,
                "model_unavailable": False
            }
        except Exception as e:
            return {"status": "error", "description": str(e), "invalid_image": True}

# Global singleton
_model_instance = PlantDiseaseModel()

def predict_disease(image_path: str) -> dict:
    return _model_instance.predict_with_feedback(image_path)

def model_health() -> dict:
    return {
        "loaded": _model_instance.loaded,
        "reason": _model_instance.reason,
        "class_count": len(_model_instance.class_labels),
        "labels_mismatch": _model_instance._labels_mismatch,
        "img_size": _model_instance.img_size,
        "metrics": _model_instance.metrics
    }
