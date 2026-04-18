"""
fix_model.py — Patch-loads a Keras 3 model saved with quantization_config
into TF 2.21 and re-saves it in a compatible h5 format.

Run from project root:
    venv\\Scripts\\python.exe scripts\\fix_model.py
"""

import os, json, sys, h5py
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "plant_model.h5")
FIXED_PATH  = os.path.join(BASE_DIR, "models", "plant_model_fixed.h5")
LABELS_PATH = os.path.join(BASE_DIR, "models", "class_labels.json")
plant_model
print(f"Model  : {MODEL_PATH}")
print(f"Output : {FIXED_PATH}")
print()

# ── Step 1: Load labels ───────────────────────────────────────────────────────
with open(LABELS_PATH) as f:
    labels = json.load(f)
print(f"Labels: {len(labels)} classes")

# ── Step 2: Read model config from HDF5 and strip quantization_config ─────────
print("\nPatching model config in HDF5...")

def strip_quantization_config(obj):
    """Recursively remove quantization_config keys from a nested dict/list."""
    if isinstance(obj, dict):
        return {k: strip_quantization_config(v)
                for k, v in obj.items()
                if k != "quantization_config"}
    if isinstance(obj, list):
        return [strip_quantization_config(i) for i in obj]
    return obj

import tensorflow as tf

# Read raw model config from H5
with h5py.File(MODEL_PATH, "r") as f:
    model_config_str = f.attrs.get("model_config", None)
    if model_config_str is None:
        print("ERROR: No model_config in HDF5 — this is not a standard Keras H5 file.")
        sys.exit(1)
    if isinstance(model_config_str, bytes):
        model_config_str = model_config_str.decode("utf-8")

model_config = json.loads(model_config_str)
model_config_clean = strip_quantization_config(model_config)
print("quantization_config keys stripped OK")

# ── Step 3: Reconstruct model from clean config ───────────────────────────────
print("\nReconstructing model from patched config...")
try:
    model = tf.keras.models.model_from_json(json.dumps(model_config_clean))
    print(f"Architecture OK | Input: {model.input_shape} | Output: {model.output_shape}")
except Exception as e:
    print(f"ERROR reconstructing: {e}")
    sys.exit(1)

# ── Step 4: Load weights ──────────────────────────────────────────────────────
print("\nLoading weights from original H5...")
try:
    model.load_weights(MODEL_PATH)
    print("Weights loaded OK")
except Exception as e:
    print(f"ERROR loading weights: {e}")
    sys.exit(1)

# ── Step 5: Verify output neuron count ────────────────────────────────────────
output_classes = model.output_shape[-1]
print(f"\nOutput neurons : {output_classes}")
print(f"Labels in JSON : {len(labels)}")

if output_classes != len(labels):
    print(f"\n⚠️  MISMATCH: model has {output_classes} outputs but labels.json has {len(labels)}")
    print("   Trimming labels to match model output...")
    labels = labels[:output_classes]
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"   Labels trimmed and saved → {len(labels)} classes")
else:
    print("✓ Labels match model output")

# ── Step 6: Quick sanity prediction ───────────────────────────────────────────
print("\nRunning test prediction on random input...")
dummy = np.random.rand(1, *model.input_shape[1:]).astype("float32")
pred = model.predict(dummy, verbose=0)
top_idx = int(np.argmax(pred[0]))
top_conf = float(pred[0][top_idx]) * 100
print(f"Test pred: class[{top_idx}] = {labels[top_idx]} @ {top_conf:.1f}%")
print("(Random input → result is meaningless but confirms model runs)")

# ── Step 7: Re-save in TF2-compatible format ──────────────────────────────────
print(f"\nSaving fixed model → {FIXED_PATH}")
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.save(FIXED_PATH, save_format="h5")
print("Saved OK")

# ── Step 8: Swap — overwrite original with fixed ──────────────────────────────
import shutil
backup = MODEL_PATH.replace(".h5", "_backup.h5")
shutil.copy2(MODEL_PATH, backup)
print(f"Backup: {backup}")
shutil.copy2(FIXED_PATH, MODEL_PATH)
print(f"Fixed model copied → {MODEL_PATH}")

print()
print("=" * 58)
print("✅  Done! Model is now compatible with TF 2.21")
print(f"   Classes  : {output_classes}")
print(f"   Input    : {model.input_shape}")
print(f"   Labels   : {LABELS_PATH}")
print(f"   Backup   : {backup}")
print()
print("Next: restart the Flask app with:  python app.py")
print("=" * 58)
