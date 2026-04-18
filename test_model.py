"""
test_model.py - Test your trained plant disease model
Run this in VS Code to verify your model is working correctly
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Paths to your model files
MODEL_PATH = "models/plant_model.h5"
LABELS_PATH = "models/class_labels.json"
TEST_IMAGE_PATH = "test_images"  # Folder containing test images

# Create test_images folder if it doesn't exist
os.makedirs(TEST_IMAGE_PATH, exist_ok=True)

# ============================================================
# LOAD MODEL AND LABELS
# ============================================================

def load_model_and_labels():
    """Load the trained model and class labels"""
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("   Please copy your plant_model.h5 to the models/ folder")
        return None, None
    
    # Check if labels exist
    if not os.path.exists(LABELS_PATH):
        print(f"❌ Labels not found at {LABELS_PATH}")
        print("   Please copy class_labels.json to the models/ folder")
        return None, None
    
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
    
    # Load class labels
    with open(LABELS_PATH, 'r') as f:
        class_names = json.load(f)
    
    print(f"✅ Loaded {len(class_names)} disease classes")
    
    return model, class_names

# ============================================================
# PREDICT SINGLE IMAGE
# ============================================================

def predict_image(model, class_names, image_path):
    """Predict disease from a single image"""
    
    # Check if image exists
    if not os.path.exists(image_path):
        return None, None, f"Image not found: {image_path}"
    
    try:
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3 = [(class_names[i], predictions[0][i] * 100) for i in top_3_indices]
        
        return predicted_class, confidence, top_3, img
        
    except Exception as e:
        return None, None, f"Error: {str(e)}", None

# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(image_path, predicted_class, confidence, top_3, class_names, img):
    """Display prediction results"""
    
    print("\n" + "="*60)
    print(f"📷 IMAGE: {os.path.basename(image_path)}")
    print("="*60)
    
    if predicted_class is None:
        print(f"❌ {confidence}")  # confidence holds error message
        return
    
    print(f"\n🔍 PREDICTION RESULT:")
    print(f"   Disease: {class_names[predicted_class]}")
    print(f"   Confidence: {confidence:.1f}%")
    
    # Determine status
    status = "🟢 HEALTHY" if "healthy" in class_names[predicted_class].lower() else "🔴 DISEASED"
    print(f"   Status: {status}")
    
    # Severity based on confidence
    if confidence > 85:
        severity = "HIGH"
        severity_icon = "🔴"
    elif confidence > 60:
        severity = "MEDIUM"
        severity_icon = "🟡"
    else:
        severity = "LOW"
        severity_icon = "🟢"
    
    print(f"   Severity: {severity_icon} {severity}")
    
    print(f"\n📊 TOP 3 PREDICTIONS:")
    for i, (disease, prob) in enumerate(top_3):
        bar = "█" * int(prob / 5)
        print(f"   {i+1}. {disease[:40]:40}: {prob:5.1f}% {bar}")
    
    # Display image
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[predicted_class]}\nConfidence: {confidence:.1f}%", 
              fontsize=12, color='green' if confidence > 70 else 'orange')
    plt.axis('off')
    plt.show()
    
    return class_names[predicted_class]

# ============================================================
# TEST ON MULTIPLE IMAGES
# ============================================================

def test_multiple_images(model, class_names, image_folder):
    """Test all images in a folder"""
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = []
    for ext in image_extensions:
        images.extend(Path(image_folder).glob(f"*{ext}"))
    
    if not images:
        print(f"\n❌ No images found in '{image_folder}' folder")
        print(f"   Please add some test images to the '{image_folder}' folder")
        return
    
    print(f"\n📂 Testing {len(images)} images from '{image_folder}'...")
    print("="*60)
    
    results = []
    for img_path in images:
        predicted_class, confidence, top_3, img = predict_image(model, class_names, str(img_path))
        if predicted_class is not None:
            results.append({
                'image': img_path.name,
                'disease': class_names[predicted_class],
                'confidence': confidence
            })
            print(f"\n✅ {img_path.name}: {class_names[predicted_class]} ({confidence:.1f}%)")
    
    return results

# ============================================================
# TEST SINGLE IMAGE (User Input)
# ============================================================

def test_single_image(model, class_names):
    """Test a single image provided by user"""
    
    print("\n" + "="*60)
    print("📸 TEST SINGLE IMAGE")
    print("="*60)
    print("Enter the path to an image file (or 'quit' to exit)")
    
    while True:
        img_path = input("\n📁 Image path: ").strip().strip('"').strip("'")
        
        if img_path.lower() == 'quit':
            break
        
        if os.path.exists(img_path):
            predicted_class, confidence, top_3, img = predict_image(model, class_names, img_path)
            display_result(img_path, predicted_class, confidence, top_3, class_names, img)
        else:
            print(f"❌ File not found: {img_path}")
            print("   Tip: Drag and drop an image file into this terminal to get its path")

# ============================================================
# BATCH TEST (Create sample test)
# ============================================================

def create_sample_test():
    """Create a sample test image (for demonstration)"""
    
    print("\n📝 SAMPLE TEST INSTRUCTIONS:")
    print("="*60)
    print("To test your model:")
    print("1. Place test images in the 'test_images' folder")
    print("2. Run: python test_model.py")
    print("\nOr test a single image:")
    print("   python test_model.py --image path/to/your/image.jpg")
    print("\nOr test all images in a folder:")
    print("   python test_model.py --folder path/to/images/folder")

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    print("="*60)
    print("🌿 PLANT DISEASE MODEL TESTER")
    print("="*60)
    
    # Load model and labels
    model, class_names = load_model_and_labels()
    if model is None:
        print("\n⚠️ Please copy your model files to the 'models' folder:")
        print("   - plant_model.h5")
        print("   - class_labels.json")
        return
    
    print(f"\n📊 Model info:")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output classes: {model.output_shape[-1]}")
    
    # Check command line arguments
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--image' and len(sys.argv) > 2:
            # Test single image
            img_path = sys.argv[2]
            if os.path.exists(img_path):
                predicted_class, confidence, top_3, img = predict_image(model, class_names, img_path)
                display_result(img_path, predicted_class, confidence, top_3, class_names, img)
            else:
                print(f"❌ Image not found: {img_path}")
        
        elif sys.argv[1] == '--folder' and len(sys.argv) > 2:
            # Test folder
            folder_path = sys.argv[2]
            if os.path.exists(folder_path):
                test_multiple_images(model, class_names, folder_path)
            else:
                print(f"❌ Folder not found: {folder_path}")
        
        elif sys.argv[1] == '--help':
            print("\nUsage:")
            print("  python test_model.py                      # Interactive mode")
            print("  python test_model.py --image image.jpg    # Test single image")
            print("  python test_model.py --folder images/     # Test all images in folder")
        
        else:
            create_sample_test()
    else:
        # Interactive mode
        print("\n" + "="*60)
        print("📋 OPTIONS:")
        print("="*60)
        print("1. Test a single image")
        print("2. Test all images in 'test_images' folder")
        print("3. Exit")
        
        choice = input("\n👉 Enter your choice (1/2/3): ").strip()
        
        if choice == '1':
            test_single_image(model, class_names)
        elif choice == '2':
            test_multiple_images(model, class_names, TEST_IMAGE_PATH)
        else:
            print("👋 Goodbye!")
    
    print("\n" + "="*60)
    print("✅ Testing complete!")
    print("="*60)

# ============================================================
# RUN THE TEST
# ============================================================

if __name__ == "__main__":
    main()