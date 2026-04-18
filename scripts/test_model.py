"""
test_model.py - Simplified Plant Disease Model Tester
Run this in VS Code to test your trained model
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ============================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================

MODEL_PATH = "models/plant_model.h5"
LABELS_PATH = "models/class_labels.json"

# ============================================================
# LOAD MODEL AND LABELS
# ============================================================

def load_model_and_labels():
    """Load the trained model and class labels"""
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("   Please copy your model file to the models/ folder")
        return None, None
    
    # Check if labels exist
    if not os.path.exists(LABELS_PATH):
        print(f"Labels not found at {LABELS_PATH}")
        print("   Please copy class_labels.json to the models/ folder")
        return None, None
    
    try:
        # Load model
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"Model loaded from {MODEL_PATH}")
        
        # Load class labels
        with open(LABELS_PATH, 'r') as f:
            class_names = json.load(f)
        
        print(f"Loaded {len(class_names)} disease classes")
        return model, class_names
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# ============================================================
# PREDICT SINGLE IMAGE
# ============================================================

def predict_image(model, class_names, image_path):
    """Predict disease from a single image"""
    
    if not os.path.exists(image_path):
        return None, None, f"Image not found: {image_path}", None
    
    try:
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get Top 3
        top_indices = np.argsort(predictions)[::-1][:3]
        results = []
        for idx in top_indices:
            results.append({
                "class": class_names[idx],
                "confidence": float(predictions[idx] * 100)
            })
        
        return results, None, img
        
    except Exception as e:
        return None, f"Error: {str(e)}", None

# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(image_path, results, error, img):
    """Display prediction results"""
    
    print("\n" + "="*60)
    print(f"IMAGE: {os.path.basename(image_path)}")
    print("="*60)
    
    if error:
        print(f" ERROR: {error}")
        return
    
    print(f"\nRANKED PREDICTIONS:")
    for i, res in enumerate(results):
        prefix = ">>" if i == 0 else "  "
        print(f"  {prefix} {res['class']:40} | {res['confidence']:6.1f}%")
    
    top = results[0]
    disease_name = top['class']
    
    print(f"\nFINAL DECISION:")
    print(f"   Disease: {disease_name}")
    print(f"   Confidence: {top['confidence']:.1f}%")
    
    # Determine status
    if "healthy" in disease_name.lower():
        print(f"   Status: HEALTHY")
    else:
        print(f"   Status: DISEASED")
    
    # Severity
    if top['confidence'] > 85:
        print(f"   Severity: HIGH")
    elif top['confidence'] > 60:
        print(f"   Severity: MEDIUM")
    else:
        print(f"   Severity: LOW")
    
    return disease_name

# ============================================================
# TEST SINGLE IMAGE
# ============================================================

def test_single_image():
    """Test a single image provided by user"""
    
    model, class_names = load_model_and_labels()
    if model is None:
        return
    
    print("\n" + "="*60)
    print("TEST SINGLE IMAGE")
    print("="*60)
    
    img_path = input("\n📁 Enter image path (or 'quit' to exit): ").strip().strip('"').strip("'")
    
    if img_path.lower() == 'quit':
        return
    
    results, error, img = predict_image(model, class_names, img_path)
    display_result(img_path, results, error, img)

# ============================================================
# QUICK TEST WITH DEFAULT IMAGE
# ============================================================

def quick_test(image_path):
    """Quick test with a specific image"""
    
    model, class_names = load_model_and_labels()
    if model is None:
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    results, error, img = predict_image(model, class_names, image_path)
    display_result(image_path, results, error, img)

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    print("="*60)
    print("PLANT DISEASE MODEL TESTER")
    print("="*60)
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--image' and len(sys.argv) > 2:
        # Test with provided image
        quick_test(sys.argv[2])
    else:
        # Interactive mode
        test_single_image()
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)

# ============================================================
# RUN THE TEST
# ============================================================

if __name__ == "__main__":
    main()