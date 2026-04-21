# test_ensemble.py - Enhanced with Image Support
import pickle
import numpy as np
import os
import sys
import cv2

def run_ensemble_test():
    print("="*60)
    print("🧪 ENSEMBLE IMAGE TESTER (Weighted 40/20/20/20)")
    print("="*60)

    # 1. Load Ensemble
    ensemble_path = os.path.join('models', 'ensemble_models.pkl')
    if not os.path.exists(ensemble_path):
        print(f"❌ File not found: {ensemble_path}")
        return

    with open(ensemble_path, 'rb') as f:
        all_models = pickle.load(f)
    print(f"✓ Loaded {len(all_models)} total models from pickle")

    # 2. Filter Compatible Models
    compatible_models = []
    model_indices = []
    for i, m in enumerate(all_models, 1):
        # 128x128 input and 14 output classes
        if m.input_shape[1:3] == (128, 128) and m.output_shape[1] == 14:
            compatible_models.append(m)
            model_indices.append(i)
            print(f"   ✅ Model {i} is compatible")
        else:
            print(f"   ⏭️ Skipping Model {i} (Incompatible shape/classes)")

    if len(compatible_models) != 4:
        print(f"\n⚠️ Warning: Expected 4 compatible models, but found {len(compatible_models)}")
    
    # 3. Load Class Names
    class_names = []
    kb_path = os.path.join('models', 'knowledge_base.json')
    if os.path.exists(kb_path):
        import json
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb = json.load(f)
            class_names = list(kb.keys())
        print(f"✓ Loaded {len(class_names)} class names")
    else:
        print("⚠️ knowledge_base.json not found in models/")

    # 4. Get Image Path
    while True:
        print("\n" + "-"*40)
        img_path = input("📁 Enter Image Path (or 'q' to quit): ").strip().strip('"').strip("'")
        if img_path.lower() == 'q':
            break
            
        if not os.path.exists(img_path):
            print(f"❌ File not found: {img_path}")
            continue

        # 5. Preprocess Image (128x128)
        img = cv2.imread(img_path)
        if img is None:
            print("❌ Error reading image with OpenCV")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img_array = img.astype(np.float32) / 255.0
        img_input = np.expand_dims(img_array, axis=0)

        # 6. Predict with All Compatible Models
        preds = []
        for i, model in enumerate(compatible_models):
            p = model.predict(img_input, verbose=0)[0]
            preds.append(p)
            print(f"   Model {model_indices[i]} prediction complete")

        # 7. Apply Weighting (40% for Model 1, 20% for others)
        if len(preds) == 4:
            weights = np.array([0.4, 0.2, 0.2, 0.2])
            final_pred = np.average(preds, axis=0, weights=weights)
            print("   ⚖️ Applied Weights: 40% (M1), 20% (M4), 20% (M5), 20% (M6)")
        else:
            final_pred = np.mean(preds, axis=0)
            print("   ⚖️ Applied Weights: Equal (Mean)")

        # 8. Show Result
        top_idx = np.argmax(final_pred)
        confidence = final_pred[top_idx] * 100
        
        disease = class_names[top_idx] if top_idx < len(class_names) else f"Unknown (ID: {top_idx})"
        
        print("\n" + "="*40)
        print(f"🏆 ENSEMBLE RESULT:")
        print(f"   Disease:    {disease}")
        print(f"   Confidence: {confidence:.2f}%")
        print("="*40)

if __name__ == "__main__":
    run_ensemble_test()