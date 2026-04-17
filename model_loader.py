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
    "Bacterial Leaf Spot": {
        "status": "diseased", "severity": "Moderate",
        "description": "Caused by Xanthomonas species. Water-soaked angular lesions.",
        "symptoms"  : ["Water-soaked angular spots", "Dark brown lesions with yellow halo", "Spots merge causing blight"],
        "treatment" : ["Apply copper hydroxide spray", "Remove infected debris immediately", "Avoid working in wet crop"],
        "prevention": ["Use disease-free certified seeds", "Avoid overhead irrigation", "Maintain proper plant spacing"],
    },
}

KAGGLE_MAP = {
    "Tomato_Bacterial_spot"                            : "Bacterial Leaf Spot",
    "Tomato_Early_blight"                              : "Tomato - Early Blight",
    "Tomato_Late_blight"                               : "Tomato - Late Blight",
    "Tomato_Leaf_Mold"                                 : "Powdery Mildew",
    "Tomato_Septoria_leaf_spot"                        : "Bacterial Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite"      : "Bacterial Leaf Spot",
    "Tomato__Target_Spot"                              : "Tomato - Early Blight",
    "Tomato__Tomato_YellowLeaf__Curl_Virus"            : "Tomato - Leaf Curl Virus",
    "Tomato__Tomato_mosaic_virus"                      : "Tomato - Leaf Curl Virus",
    "Tomato_healthy"                                   : "Healthy",
    "Tomato___Bacterial_spot"                           : "Bacterial Leaf Spot",
    "Tomato___Early_blight"                             : "Tomato - Early Blight",
    "Tomato___Late_blight"                              : "Tomato - Late Blight",
    "Tomato___Leaf_Mold"                                : "Powdery Mildew",
    "Tomato___Septoria_leaf_spot"                       : "Bacterial Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite"     : "Bacterial Leaf Spot",
    "Tomato___Target_Spot"                              : "Tomato - Early Blight",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"            : "Tomato - Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus"                      : "Tomato - Leaf Curl Virus",
    "Tomato___healthy"                                  : "Healthy",
    "Potato___Early_Blight"                             : "Potato - Early Blight",
    "Potato___Late_Blight"                              : "Potato - Late Blight",
    "Potato___healthy"                                  : "Healthy",
    "Grape___Black_rot"                                 : "Grape - Black Rot",
    "Grape___Esca_(Black_Measles)"                      : "Grape - Leaf Blight",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"        : "Grape - Leaf Blight",
    "Grape___healthy"                                   : "Healthy",
    "Apple___Apple_scab"                                : "Bacterial Leaf Spot",
    "Apple___Black_rot"                                 : "Grape - Black Rot",
    "Apple___Cedar_apple_rust"                          : "Powdery Mildew",
    "Apple___healthy"                                   : "Healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Bacterial Leaf Spot",
    "Corn_(maize)___Common_rust_"                       : "Powdery Mildew",
    "Corn_(maize)___Northern_Leaf_Blight"               : "Tomato - Early Blight",
    "Corn_(maize)___healthy"                            : "Healthy",
    "Pepper,_bell___Bacterial_spot"                     : "Bacterial Leaf Spot",
    "Pepper,_bell___healthy"                            : "Healthy",
    "Strawberry___Leaf_scorch"                          : "Bacterial Leaf Spot",
    "Strawberry___healthy"                              : "Healthy",
    "Cherry_(including_sour)___Powdery_mildew"          : "Powdery Mildew",
    "Cherry_(including_sour)___healthy"                 : "Healthy",
    "Peach___Bacterial_spot"                            : "Bacterial Leaf Spot",
    "Peach___healthy"                                   : "Healthy",
    "Soybean___healthy"                                 : "Healthy",
    "Squash___Powdery_mildew"                           : "Powdery Mildew",
    "Blueberry___healthy"                               : "Healthy",
    "Raspberry___healthy"                               : "Healthy",
    "Orange___Haunglongbing_(Citrus_greening)"          : "Bacterial Leaf Spot",
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
        self.loaded = False
        self.reason = "Not loaded."
        
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path, compile=False)
                self.loaded = True
                self.reason = "Model loaded."
            except Exception as e:
                self.reason = f"Model load failed: {e}"
        else:
            self.reason = f"{model_path} missing."

        if os.path.exists(labels_path):
            try:
                with open(labels_path, 'r') as f:
                    self.class_labels = json.load(f)
            except Exception as e:
                print(f"Error loading labels: {e}")

        # LENIENT THRESHOLDS - Fixes the "invalid image" issue
        self.min_confidence = 0.55
        self.low_confidence = 0.35
        self.reject_threshold = 0.15
        self.min_margin = 0.18
        
        # Image validation thresholds
        self.min_dimension = 80
        self.min_brightness = 15
        self.min_blur_threshold = 30
        self.min_green_percentage = 3
        
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
        """Perform Test-Time Augmentation (TTA) for more stable predictions"""
        # Original
        processed_orig = cv2.resize(image, (224, 224))
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
            
            # Top 3 for diagnostics
            top3 = []
            for i in range(min(3, len(top_indices))):
                ti = int(top_indices[i])
                raw_label = self.class_labels[ti] if ti < len(self.class_labels) else "Unknown"
                top3.append({
                    "label": _display_name(raw_label),
                    "confidence": round(float(avg_preds[ti]) * 100, 1)
                })

            label = self.class_labels[top1_idx] if top1_idx < len(self.class_labels) else "Unknown"
            disp = _display_name(label)
            crop = _extract_crop(disp)
            info = _get_info(label)
            confidence_margin = top1_conf - top2_conf

            # 70% Confidence Threshold Gate
            CONFIDENCE_THRESHOLD = 0.70
            
            if top1_conf < CONFIDENCE_THRESHOLD:
                return {
                    'invalid_image': True,
                    'status': 'uncertain',
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
            
            status = info.get("status", "diseased")
            message = f"Detected: {disp}"
            result_summary, plant_details = _build_result_summary(crop, disp, info)
            
            return {
                "disease": disp,
                "crop": crop,
                "confidence": round(top1_conf * 100, 1),
                "top_predictions": top3,
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
        "class_count": len(_model_instance.class_labels)
    }
