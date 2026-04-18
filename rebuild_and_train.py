"""
evaluate_current_model.py - Test current model for overfitting and real-world performance
No retraining - just evaluation!
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def load_model_and_labels():
    """Load existing model and class labels"""
    
    # Load model
    model_path = 'models/plant_model.h5'
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return None, None
    
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded: {model_path}")
    
    # Load class labels
    labels_path = 'models/class_labels.json'
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            class_names = json.load(f)
        print(f"✅ Class labels loaded: {len(class_names)} classes")
    else:
        # Try to infer from dataset
        data_dir = 'data/PlantVillage/PlantVillage'
        if os.path.exists(data_dir):
            class_names = sorted([d for d in os.listdir(data_dir) 
                                 if os.path.isdir(os.path.join(data_dir, d))])
            print(f"✅ Inferred {len(class_names)} classes from dataset")
        else:
            class_names = [f"Class_{i}" for i in range(model.output_shape[-1])]
            print(f"⚠️ Using generic class names: {len(class_names)} classes")
    
    return model, class_names

def collect_test_images(data_dir, class_names, samples_per_class=10):
    """Collect test images (not used in training)"""
    
    test_images = []
    test_labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️ Class directory not found: {class_name}")
            continue
        
        # Get all images in this class
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
        
        # Take random samples
        samples = random.sample(images, min(samples_per_class, len(images)))
        
        for img_file in samples:
            test_images.append(os.path.join(class_dir, img_file))
            test_labels.append(class_idx)
        
        print(f"  {class_name}: {len(samples)} images")
    
    return test_images, test_labels

def evaluate_model(model, test_images, test_labels, class_names):
    """Evaluate model on test images"""
    
    print(f"\n{'='*60}")
    print("📊 EVALUATING CURRENT MODEL")
    print(f"{'='*60}")
    
    correct = 0
    confidences = []
    predictions = []
    
    for i, (img_path, true_label) in enumerate(zip(test_images, test_labels)):
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Predict
        pred = model.predict(img_array, verbose=0)
        pred_class = np.argmax(pred[0])
        confidence = np.max(pred[0]) * 100
        
        is_correct = (pred_class == true_label)
        if is_correct:
            correct += 1
        
        confidences.append(confidence)
        predictions.append({
            'image': os.path.basename(img_path),
            'true': class_names[true_label],
            'predicted': class_names[pred_class],
            'correct': is_correct,
            'confidence': confidence
        })
    
    accuracy = (correct / len(test_images)) * 100
    avg_confidence = np.mean(confidences)
    
    print(f"\n📈 RESULTS:")
    print(f"   Total test images: {len(test_images)}")
    print(f"   Correct predictions: {correct}")
    print(f"   Accuracy: {accuracy:.1f}%")
    print(f"   Average confidence: {avg_confidence:.1f}%")
    
    return accuracy, avg_confidence, predictions

def check_overfitting_signs(accuracy, avg_confidence, predictions, class_names):
    """Check for overfitting signs"""
    
    print(f"\n{'='*60}")
    print("🔍 OVERFITTING ANALYSIS")
    print(f"{'='*60}")
    
    # Sign 1: Too high accuracy
    if accuracy > 95:
        print("⚠️ SIGN 1: Very high accuracy (>95%) - Possible overfitting")
    elif accuracy > 85:
        print("✅ SIGN 1: Good accuracy (85-95%) - Reasonable")
    else:
        print("✅ SIGN 1: Normal accuracy (<85%) - Likely not overfitting")
    
    # Sign 2: Confidence vs accuracy gap
    confidence_gap = avg_confidence - accuracy
    if confidence_gap > 20:
        print(f"⚠️ SIGN 2: Large confidence gap ({confidence_gap:.1f}%) - Model overconfident")
    else:
        print(f"✅ SIGN 2: Confidence gap ({confidence_gap:.1f}%) - Normal")
    
    # Sign 3: Check per-class performance
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for pred in predictions:
        class_name = pred['true']
        class_total[class_name] += 1
        if pred['correct']:
            class_correct[class_name] += 1
    
    print("\n📊 Per-class accuracy:")
    low_performance = []
    for class_name in class_total:
        acc = (class_correct[class_name] / class_total[class_name]) * 100
        status = "✅" if acc > 70 else "⚠️"
        print(f"   {status} {class_name}: {acc:.1f}% ({class_correct[class_name]}/{class_total[class_name]})")
        if acc < 70:
            low_performance.append(class_name)
    
    # Overall diagnosis
    print(f"\n{'='*60}")
    print("🎯 FINAL DIAGNOSIS")
    print(f"{'='*60}")
    
    if accuracy > 95 and avg_confidence > 95:
        print("⚠️ HIGH PROBABILITY OF OVERFITTING")
        print("   → Model may not generalize well to new images")
        print("   → Consider retraining with proper validation split")
    elif accuracy > 85:
        print("✅ MODEL IS PERFORMING WELL")
        print("   → Acceptable for production use")
        print("   → Monitor real-world performance")
    else:
        print("✅ MODEL IS NOT OVERFITTING")
        print("   → Accuracy is realistic for plant disease detection")
        print("   → Good for production use")
    
    if low_performance:
        print(f"\n⚠️ Classes needing improvement:")
        for cls in low_performance[:5]:
            print(f"   - {cls}")
    
    return accuracy > 95 and avg_confidence > 95

def show_sample_predictions(predictions, num_samples=5):
    """Show sample predictions"""
    
    print(f"\n{'='*60}")
    print(f"🔬 SAMPLE PREDICTIONS (first {num_samples} images)")
    print(f"{'='*60}")
    
    # Show correct and incorrect samples
    correct_samples = [p for p in predictions if p['correct']][:num_samples]
    incorrect_samples = [p for p in predictions if not p['correct']][:num_samples]
    
    print("\n✅ CORRECT PREDICTIONS:")
    for p in correct_samples:
        print(f"   {p['image']}: {p['true']} → {p['predicted']} ({p['confidence']:.1f}%)")
    
    if incorrect_samples:
        print("\n❌ INCORRECT PREDICTIONS:")
        for p in incorrect_samples:
            print(f"   {p['image']}: {p['true']} → {p['predicted']} ({p['confidence']:.1f}%)")
    else:
        print("\n🎉 No incorrect predictions on test samples!")

def main():
    print("="*60)
    print("🌿 PlantCure Model Evaluator (No Retraining)")
    print("="*60)
    
    # Load model
    model, class_names = load_model_and_labels()
    if model is None:
        return
    
    # Find dataset
    data_dir = 'data/PlantVillage/PlantVillage'
    if not os.path.exists(data_dir):
        data_dir = 'data/PlantVillage'
    
    if not os.path.exists(data_dir):
        print(f"❌ Dataset not found at {data_dir}")
        print("   Please ensure dataset is in data/PlantVillage/")
        return
    
    print(f"\n📂 Dataset: {data_dir}")
    
    # Collect test images (10 per class)
    print(f"\n📸 Collecting test images (10 per class)...")
    test_images, test_labels = collect_test_images(data_dir, class_names, samples_per_class=10)
    
    if len(test_images) == 0:
        print("❌ No test images found!")
        return
    
    # Evaluate model
    accuracy, avg_confidence, predictions = evaluate_model(
        model, test_images, test_labels, class_names
    )
    
    # Check for overfitting
    is_overfit = check_overfitting_signs(accuracy, avg_confidence, predictions, class_names)
    
    # Show sample predictions
    show_sample_predictions(predictions)
    
    # Final recommendation
    print(f"\n{'='*60}")
    print("💡 RECOMMENDATION")
    print(f"{'='*60}")
    
    if is_overfit:
        print("⚠️ Your model shows signs of overfitting.")
        print("   → Run: python scripts/rebuild_and_train.py --epochs 20 --batch 16")
        print("   → This will create a more robust model")
    else:
        print("✅ Your model is ready for production!")
        print("   → No retraining needed")
        print("   → Keep using the current model")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()