import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.models import load_model

print("="*50)
print("🔗 COMBINING TWO MODELS")
print("="*50)

# Set paths
models_dir = 'models'
model1_path = os.path.join(models_dir, 'plant_disease_model_20260421_054458.h5')
model2_path = os.path.join(models_dir, 'combined_plant_disease_model.h5')
output_path = os.path.join(models_dir, 'final_combined_model.h5')

print(f"\n📁 Model 1: {os.path.basename(model1_path)}")
print(f"📁 Model 2: {os.path.basename(model2_path)}")

# Check if files exist
if not os.path.exists(model1_path):
    print(f"❌ Error: {model1_path} not found!")
    exit()
if not os.path.exists(model2_path):
    print(f"❌ Error: {model2_path} not found!")
    exit()

# Load models
print("\n📥 Loading models...")

try:
    # Load with compile=False to avoid compatibility issues
    model1 = load_model(model1_path, compile=False)
    print(f"   ✅ Loaded model 1")
except Exception as e:
    print(f"   ❌ Error loading model 1: {e}")
    exit()

try:
    model2 = load_model(model2_path, compile=False)
    print(f"   ✅ Loaded model 2")
except Exception as e:
    print(f"   ❌ Error loading model 2: {e}")
    exit()

# Check if models have same architecture
print("\n🔍 Checking model compatibility...")
print(f"   Model 1 input shape: {model1.input_shape}")
print(f"   Model 2 input shape: {model2.input_shape}")
print(f"   Model 1 output shape: {model1.output_shape}")
print(f"   Model 2 output shape: {model2.output_shape}")

if model1.input_shape != model2.input_shape:
    print("   ⚠️ Warning: Input shapes don't match!")
if model1.output_shape != model2.output_shape:
    print("   ⚠️ Warning: Output shapes don't match!")

# Combine weights (average)
print("\n🔗 Combining model weights...")
weights1 = model1.get_weights()
weights2 = model2.get_weights()

if len(weights1) != len(weights2):
    print(f"   ❌ Error: Models have different number of layers!")
    print(f"      Model 1: {len(weights1)} layers")
    print(f"      Model 2: {len(weights2)} layers")
    exit()

# Average the weights
combined_weights = []
for i, (w1, w2) in enumerate(zip(weights1, weights2)):
    if w1.shape == w2.shape:
        avg_weight = (w1 + w2) / 2.0
        combined_weights.append(avg_weight)
    else:
        print(f"   ⚠️ Layer {i}: Shape mismatch - using weights from model 1")
        combined_weights.append(w1)

# Apply combined weights to model 1
model1.set_weights(combined_weights)
print(f"   ✅ Weights combined successfully")

# Save combined model
print(f"\n💾 Saving combined model...")
model1.save(output_path)
print(f"   ✅ Saved to: {output_path}")
print(f"   📦 File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

# Also save in .keras format
keras_output = os.path.join(models_dir, 'final_combined_model.keras')
model1.save(keras_output)
print(f"   ✅ Also saved as: {keras_output}")

print("\n" + "="*50)
print("✅ SUCCESS! Models combined")
print("="*50)
print(f"\n📁 Output files:")
print(f"   • {output_path}")
print(f"   • {keras_output}")