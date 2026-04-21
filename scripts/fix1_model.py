import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load the model
model_path = 'models/plant_disease_model_20260421_054458.h5'

print(f"Loading model: {model_path}")

# Try loading with different strategies
try:
    # Attempt 1: Normal load
    model = load_model(model_path)
    print("✅ Loaded with normal method")
except:
    try:
        # Attempt 2: Load without compiling
        model = load_model(model_path, compile=False)
        print("✅ Loaded without compilation")
    except:
        try:
            # Attempt 3: Use tf.keras.models.load_model
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Loaded with tf.keras.models.load_model")
        except Exception as e:
            print(f"❌ Failed to load: {e}")
            exit()

# Save in new compatible format
new_path = 'models/plant_disease_model_compatible.h5'
model.save(new_path, save_format='h5')
print(f"✅ Model re-saved as: {new_path}")

# Also save in .keras format (newer)
new_path_keras = 'models/plant_disease_model_compatible.keras'
model.save(new_path_keras)
print(f"✅ Model re-saved as: {new_path_keras}")