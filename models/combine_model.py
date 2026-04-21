# combine_models.py - Save this in the 'models' folder
import os
import numpy as np
from tensorflow.keras.models import load_model

print("="*60)
print("🔗 COMBINING MODELS (Running from models folder)")
print("="*60)

# Get all .h5 files in current directory
model_files = [f for f in os.listdir('.') if f.endswith('.h5')]

print(f"\n📁 Found {len(model_files)} model files:")
for f in model_files:
    file_size = os.path.getsize(f) / (1024*1024)
    print(f"   • {f} ({file_size:.1f} MB)")

if len(model_files) < 2:
    print("\n❌ Need at least 2 models to combine")
    exit()

# Load all models
print("\n📥 Loading models...")
models = []
for i, f in enumerate(model_files, 1):
    try:
        model = load_model(f, compile=False)
        models.append(model)
        print(f"   ✅ {i}. Loaded {f}")
    except Exception as e:
        print(f"   ❌ Error loading {f}: {e}")

if len(models) < 2:
    print("\n❌ Failed to load enough models")
    exit()

# Check compatibility
print("\n🔍 Checking compatibility...")
compatible = True
for i, model in enumerate(models[1:], 2):
    if model.input_shape != models[0].input_shape:
        print(f"   ⚠️ Model {i} input mismatch")
        compatible = False
    if model.output_shape != models[0].output_shape:
        print(f"   ⚠️ Model {i} output mismatch")
        compatible = False

if not compatible:
    print("\n   Models have different architectures!")
    print("   Saving as pickle ensemble instead...")
    
    import pickle
    with open('ensemble_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    print(f"\n✅ Saved to ensemble_models.pkl")
    exit()

# Average weights
print("\n🔗 Averaging weights...")
ref_model = models[0]
num_models = len(models)

# Initialize
avg_weights = [np.zeros_like(w) for w in ref_model.get_weights()]

# Add all weights
for idx, model in enumerate(models, 1):
    weights = model.get_weights()
    for j, w in enumerate(weights):
        avg_weights[j] += w
    print(f"   ✓ Added model {idx}/{num_models}")

# Average
for j in range(len(avg_weights)):
    avg_weights[j] /= num_models

# Apply and save
ref_model.set_weights(avg_weights)
output_file = 'averaged_model.h5'
ref_model.save(output_file)

# Results
original_size = sum(os.path.getsize(f) for f in model_files) / (1024*1024)
new_size = os.path.getsize(output_file) / (1024*1024)

print("\n" + "="*60)
print("✅ SUCCESS!")
print("="*60)
print(f"\n📊 Results:")
print(f"   Original {num_models} models: {original_size:.1f} MB")
print(f"   Averaged model: {new_size:.1f} MB")
print(f"   Reduction: {(1 - new_size/original_size)*100:.0f}%")
print(f"\n📁 Output: {output_file}")